#!/usr/bin/env python3
"""Training dashboard — web UI for the full YOLOv8 training pipeline.

Manages training steps (prepare dataset, train, export ONNX, convert HEF)
as subprocesses with live output, metrics charts, and config editing.

Usage:
    source venv_train/bin/activate
    python training_dashboard.py
    python training_dashboard.py --config configs/project.yaml --port 5004
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request, send_file

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, load_config, resolve_path
from utils.display import print_info, print_success, print_error, print_warning, print_header

import yaml


# ═══════════════════════════════════════════════════════════════════════════
# Training Step Manager
# ═══════════════════════════════════════════════════════════════════════════

STEP_ORDER = ["pull", "annotate", "prepare", "train", "export", "convert", "deploy"]

STEP_LABELS = {
    "pull": "Pull Data from Pi",
    "annotate": "Review Annotations",
    "prepare": "Prepare Dataset",
    "train": "Train Model",
    "export": "Export ONNX",
    "convert": "Convert HEF",
    "deploy": "Deploy to Pi",
}

# Steps that are interactive (not subprocess-based)
INTERACTIVE_STEPS = {"annotate"}


@dataclass
class TrainingStep:
    name: str
    label: str
    state: str = "idle"  # idle, running, done, error
    output: collections.deque = field(default_factory=lambda: collections.deque(maxlen=500))
    process: subprocess.Popen | None = None
    start_time: float | None = None
    end_time: float | None = None
    exit_code: int | None = None
    log_file: object = None  # file handle for persisting full output


class TrainingStepManager:
    """Run training pipeline steps as subprocesses."""

    def __init__(self, project_root: Path, config_path: Path | None, config: dict):
        self.project_root = project_root
        self.config_path = config_path
        self._config = config
        self._lock = threading.Lock()
        self._auto_pipeline = False
        self._current_step: str | None = None
        self._train_python = str(project_root / "venv_train" / "bin" / "python")
        self._active_version: str = "v1"
        self._version_size_cache: dict[str, float] = {}
        self._version_size_cache_time: float = 0

        self._steps: dict[str, TrainingStep] = {
            name: TrainingStep(name=name, label=STEP_LABELS[name])
            for name in STEP_ORDER
        }

        # Auto-detect active version from existing versions
        versions = self.get_versions()
        if versions:
            self._active_version = versions[-1]["name"]

        # Ensure the active version directory exists
        (self.project_root / "versions" / self._active_version).mkdir(parents=True, exist_ok=True)

    def get_version_dir(self) -> Path:
        """Return the version directory for the active version."""
        return self.project_root / "versions" / self._active_version

    def set_active_version(self, name: str) -> None:
        """Switch to a different version, creating its directory if needed."""
        with self._lock:
            if name == self._active_version:
                return

            self._active_version = name
            ver_dir = self.project_root / "versions" / name
            ver_dir.mkdir(parents=True, exist_ok=True)

            # Clear stale step output from previous version
            if self._current_step is None:
                for step in self._steps.values():
                    step.state = "idle"
                    step.output.clear()
                    step.exit_code = None
                    step.start_time = None
                    step.end_time = None
                    step.process = None

    def get_versions(self) -> list[dict]:
        """List available training versions."""
        versions_dir = self.project_root / "versions"
        versions = []
        if not versions_dir.is_dir():
            return versions

        # Refresh size cache if stale (> 60s)
        now = time.time()
        if now - self._version_size_cache_time > 60:
            self._version_size_cache.clear()
            self._version_size_cache_time = now

        for d in sorted(versions_dir.iterdir()):
            if d.is_dir():
                has_best = (d / "training" / "weights" / "best.pt").exists()
                has_onnx = (d / "training" / "weights" / "best.onnx").exists()
                has_hef = (d / "cat_detector.hef").exists()
                name = d.name
                if name not in self._version_size_cache:
                    size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    self._version_size_cache[name] = round(size_bytes / 1024 / 1024, 1)
                versions.append({
                    "name": name,
                    "has_model": has_best,
                    "has_onnx": has_onnx,
                    "has_hef": has_hef,
                    "size_mb": self._version_size_cache[name],
                })
        return versions

    def _invalidate_size_cache(self, name: str | None = None) -> None:
        """Clear cached version sizes. Pass a name to clear one, or None to clear all."""
        if name is None:
            self._version_size_cache.clear()
        else:
            self._version_size_cache.pop(name, None)

    @staticmethod
    def _fix_dataset_yaml_path(version_dir: Path) -> None:
        """Update the absolute path in dataset.yaml to match the version directory."""
        dataset_yaml = version_dir / "dataset" / "dataset.yaml"
        if not dataset_yaml.exists():
            return
        with open(dataset_yaml) as f:
            content = yaml.safe_load(f)
        if content and "path" in content:
            content["path"] = str((version_dir / "dataset").resolve())
            with open(dataset_yaml, "w") as f:
                yaml.dump(content, f, default_flow_style=False, sort_keys=False)

    def delete_version(self, name: str) -> dict:
        """Delete a version directory."""
        import shutil
        with self._lock:
            if self._current_step is not None and name == self._active_version:
                return {"status": "error", "message": "Cannot delete version while a step is running"}

            ver_dir = self.project_root / "versions" / name
            if not ver_dir.is_dir():
                return {"status": "error", "message": f"Version '{name}' not found"}

            shutil.rmtree(ver_dir)
            self._invalidate_size_cache(name)

            # If we deleted the active version, switch to another
            if name == self._active_version:
                versions = self.get_versions()
                if versions:
                    self._active_version = versions[-1]["name"]
                else:
                    self._active_version = "v1"
                    (self.project_root / "versions" / "v1").mkdir(parents=True, exist_ok=True)

                # Clear step state for the new version
                for step in self._steps.values():
                    step.state = "idle"
                    step.output.clear()
                    step.exit_code = None
                    step.start_time = None
                    step.end_time = None
                    step.process = None

        return {"status": "ok", "active": self._active_version}

    def rename_version(self, name: str, new_name: str) -> dict:
        """Rename a version directory."""
        with self._lock:
            if self._current_step is not None and name == self._active_version:
                return {"status": "error", "message": "Cannot rename version while a step is running"}

            ver_dir = self.project_root / "versions" / name
            new_dir = self.project_root / "versions" / new_name

            if not ver_dir.is_dir():
                return {"status": "error", "message": f"Version '{name}' not found"}
            if new_dir.exists():
                return {"status": "error", "message": f"Version '{new_name}' already exists"}

            ver_dir.rename(new_dir)
            self._fix_dataset_yaml_path(new_dir)
            # Transfer cached size to new name
            old_size = self._version_size_cache.pop(name, None)
            if old_size is not None:
                self._version_size_cache[new_name] = old_size

            if name == self._active_version:
                self._active_version = new_name

        return {"status": "ok", "active": self._active_version}

    def duplicate_version(self, name: str, new_name: str) -> dict:
        """Duplicate a version directory."""
        import shutil
        with self._lock:
            ver_dir = self.project_root / "versions" / name
            new_dir = self.project_root / "versions" / new_name

            if not ver_dir.is_dir():
                return {"status": "error", "message": f"Version '{name}' not found"}
            if new_dir.exists():
                return {"status": "error", "message": f"Version '{new_name}' already exists"}

        # Copy outside lock (can be slow for large versions)
        shutil.copytree(ver_dir, new_dir)
        self._fix_dataset_yaml_path(new_dir)

        self.set_active_version(new_name)
        return {"status": "ok", "active": new_name}

    def _cleanup_train_dir(self) -> None:
        """Remove partial training artifacts so the next run starts fresh."""
        train_dir = self.get_version_dir() / "training"
        if train_dir.exists():
            import shutil
            shutil.rmtree(train_dir)

    def _build_command(self, step_name: str) -> list[str]:
        """Build subprocess command for a given step."""
        config_args = ["--config", str(self.config_path)] if self.config_path else []
        cats = self._config.get("cats", [])
        deploy_cfg = self._config.get("deployment", {})
        ver_dir = self.get_version_dir()

        if step_name == "pull":
            return ["bash", "scripts/pull_dataset.sh", "", "", str(ver_dir / "raw")]
        elif step_name == "prepare":
            return [
                self._train_python, "train/prepare_dataset.py",
                "--input", str(ver_dir / "raw"),
                "--output", str(ver_dir / "dataset"),
            ] + config_args
        elif step_name == "train":
            return [
                self._train_python, "train/train.py",
                "--run-name", "training",
                "--project", str(ver_dir),
                "--dataset-yaml", str(ver_dir / "dataset" / "dataset.yaml"),
            ] + config_args
        elif step_name == "export":
            return [
                self._train_python, "train/export_onnx.py",
                "--model", str(ver_dir / "training" / "weights" / "best.pt"),
            ] + config_args
        elif step_name == "convert":
            onnx_path = str(ver_dir / "training" / "weights" / "best.onnx")
            num_classes = str(len(cats))
            calib_dir = str(ver_dir / "dataset" / "calibration")
            hw_arch = deploy_cfg.get("hw_arch", "hailo10h")
            model_size = self._config.get("training", {}).get("model_size", "yolov8s")
            return ["bash", "train/convert_hef.sh", onnx_path, num_classes, calib_dir, hw_arch, model_size, str(ver_dir)]
        elif step_name == "deploy":
            hef_path = str(ver_dir / "cat_detector.hef")
            return ["bash", "scripts/transfer_model.sh", "", hef_path]
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def run_step(self, step_name: str) -> dict:
        """Launch a single training step."""
        if step_name not in self._steps:
            return {"status": "error", "message": f"Unknown step: {step_name}"}

        # Interactive steps don't run as subprocesses
        if step_name in INTERACTIVE_STEPS:
            return {"status": "interactive", "action": "open_annotate_tab"}

        with self._lock:
            if self._current_step is not None:
                return {"status": "error", "message": f"Step '{self._current_step}' is already running"}

            step = self._steps[step_name]
            cmd = self._build_command(step_name)

            try:
                step.process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                step.state = "error"
                step.output.append(f"[ERROR] Failed to start: {exc}")
                return {"status": "error", "message": str(exc)}

            step.state = "running"
            step.start_time = time.time()
            step.end_time = None
            step.exit_code = None
            step.output.clear()
            self._current_step = step_name

            # Open log file for full output persistence
            self._close_step_log(step)
            log_dir = self.project_root / "logs" / "training" / self._active_version
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            step.log_file = open(log_dir / f"{step_name}_{ts}.log", "w")

            # Clean up partial training artifacts before starting a new run
            if step_name == "train":
                self._cleanup_train_dir()
                step.output.append("[INFO] Cleaned up previous training run")

        # Reader + completion thread
        thread = threading.Thread(
            target=self._monitor_step, args=(step,), daemon=True
        )
        thread.start()

        return {"status": "ok", "step": step_name, "pid": step.process.pid}

    def run_all(self) -> dict:
        """Start the full pipeline (auto-chain steps)."""
        with self._lock:
            if self._current_step is not None:
                return {"status": "error", "message": f"Step '{self._current_step}' is already running"}

            # Reset all steps
            for step in self._steps.values():
                step.state = "idle"
                step.output.clear()
                step.exit_code = None
                step.start_time = None
                step.end_time = None

            self._auto_pipeline = True

        # Start first step
        return self.run_step(STEP_ORDER[0])

    def stop(self) -> dict:
        """Stop the currently running step and cancel auto-pipeline."""
        with self._lock:
            self._auto_pipeline = False
            if self._current_step is None:
                return {"status": "ok", "message": "Nothing running"}

            step = self._steps[self._current_step]
            if step.process and step.process.poll() is None:
                step.process.terminate()
                try:
                    step.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    step.process.kill()
                    step.process.wait(timeout=2)

            step.state = "error"
            step.end_time = time.time()
            step.output.append("[STOPPED] Step terminated by user")
            self._close_step_log(step)
            name = self._current_step
            self._current_step = None
            return {"status": "ok", "step": name}

    @staticmethod
    def _close_step_log(step: TrainingStep) -> None:
        if step.log_file and not step.log_file.closed:
            try:
                step.log_file.close()
            except Exception:
                pass
        step.log_file = None

    def _monitor_step(self, step: TrainingStep) -> None:
        """Read output and detect completion (runs in daemon thread)."""
        if step.process and step.process.stdout:
            try:
                for line in step.process.stdout:
                    step.output.append(line.rstrip())
                    if step.log_file and not step.log_file.closed:
                        try:
                            step.log_file.write(line)
                            step.log_file.flush()
                        except Exception:
                            pass
            except (ValueError, OSError):
                pass

        self._close_step_log(step)

        # Process finished
        exit_code = step.process.wait() if step.process else -1
        step.exit_code = exit_code
        step.end_time = time.time()

        with self._lock:
            self._invalidate_size_cache(self._active_version)
            if exit_code == 0:
                step.state = "done"
                # Copy annotations from previous version after pull
                if step.name == "pull":
                    msg = self._copy_annotations_from_previous()
                    if msg:
                        step.output.append(msg)
                self._current_step = None
                # Auto-chain to next step
                if self._auto_pipeline:
                    next_step = self._next_step(step.name)
                    if next_step:
                        # Release lock before recursing
                        threading.Thread(
                            target=lambda: self.run_step(next_step), daemon=True
                        ).start()
                    else:
                        self._auto_pipeline = False
            else:
                step.state = "error"
                step.output.append(f"[ERROR] Exit code: {exit_code}")
                self._current_step = None
                self._auto_pipeline = False

    def _next_step(self, current: str) -> str | None:
        """Return the next step name after current, or None if done.

        Skips interactive steps (like annotate) during auto-pipeline.
        """
        idx = STEP_ORDER.index(current)
        for i in range(idx + 1, len(STEP_ORDER)):
            if STEP_ORDER[i] not in INTERACTIVE_STEPS:
                return STEP_ORDER[i]
        return None

    def get_annotation_stats(self) -> dict:
        """Count annotated vs total images in the active version's raw dir."""
        ver_dir = self.get_version_dir()
        raw_dir = ver_dir / "raw"
        cats = self._config.get("cats", [])
        total = 0
        annotated = 0
        for cat_name in cats:
            cat_dir = raw_dir / cat_name
            if not cat_dir.is_dir():
                continue
            for img in cat_dir.glob("*.jpg"):
                total += 1
                if img.with_suffix(".txt").exists():
                    annotated += 1
        return {"total": total, "annotated": annotated, "missing": total - annotated}

    def _copy_annotations_from_previous(self) -> str:
        """Copy .txt annotation files from the most recent previous version.

        Matches by filename so images that carry over between versions
        keep their annotations automatically.
        """
        import shutil

        ver_dir = self.get_version_dir()
        raw_dir = ver_dir / "raw"
        cats = self._config.get("cats", [])

        # Find previous version with raw data
        versions = self.get_versions()
        prev_raw = None
        prev_name = None
        for v in reversed(versions):
            if v["name"] == self._active_version:
                continue
            candidate = self.project_root / "versions" / v["name"] / "raw"
            if candidate.is_dir():
                prev_raw = candidate
                prev_name = v["name"]
                break

        if not prev_raw:
            return ""

        copied = 0
        for cat_name in cats:
            cur_cat = raw_dir / cat_name
            prev_cat = prev_raw / cat_name
            if not cur_cat.is_dir() or not prev_cat.is_dir():
                continue
            for img in cur_cat.glob("*.jpg"):
                txt = img.with_suffix(".txt")
                if txt.exists():
                    continue  # already annotated
                prev_txt = prev_cat / txt.name
                if prev_txt.exists():
                    shutil.copy2(prev_txt, txt)
                    copied += 1

        if copied:
            return f"[INFO] Copied {copied} annotations from {prev_name}"
        return ""

    def reload_config(self, config: dict) -> None:
        """Update stored config (after settings change)."""
        with self._lock:
            self._config = config

    def get_status(self) -> dict:
        """Return full pipeline status."""
        with self._lock:
            steps = {}
            for name, step in self._steps.items():
                elapsed = None
                if step.start_time:
                    end = step.end_time or time.time()
                    elapsed = end - step.start_time
                steps[name] = {
                    "label": step.label,
                    "state": step.state,
                    "elapsed": elapsed,
                    "exit_code": step.exit_code,
                }

            # Compute annotate step state from annotation completeness
            ann_stats = self.get_annotation_stats()
            if ann_stats["total"] == 0:
                steps["annotate"]["state"] = "idle"
            elif ann_stats["missing"] == 0:
                steps["annotate"]["state"] = "done"
            else:
                steps["annotate"]["state"] = "idle"
            steps["annotate"]["annotation_stats"] = ann_stats

            return {
                "steps": steps,
                "current_step": self._current_step,
                "auto_pipeline": self._auto_pipeline,
            }

    def get_step_output(self, step_name: str) -> list[str]:
        """Return output lines for the active or most recent step."""
        step = self._steps.get(step_name)
        if step is None:
            return []
        return list(step.output)

    def get_active_output(self) -> list[str]:
        """Return output for the currently running or most recently run step."""
        with self._lock:
            # Show current running step, or last completed/errored step
            target = self._current_step
            if target is None:
                # Find the last step that has output
                for name in reversed(STEP_ORDER):
                    if self._steps[name].output:
                        target = name
                        break
            if target is None:
                return []
            return list(self._steps[target].output)


# ═══════════════════════════════════════════════════════════════════════════
# Training Config Manager
# ═══════════════════════════════════════════════════════════════════════════

class TrainingConfigManager:
    """Read/write training settings in project.yaml."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._lock = threading.Lock()

    def read(self) -> dict:
        with self._lock:
            with open(self.config_path) as f:
                return yaml.safe_load(f)

    def get_training_settings(self) -> dict:
        cfg = self.read()
        return {
            "training": cfg.get("training", {}),
            "deployment": {
                "hw_arch": cfg.get("deployment", {}).get("hw_arch", "hailo10h"),
            },
            "cats": cfg.get("cats", []),
        }

    def update_training_settings(self, settings: dict) -> None:
        with self._lock:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)

            if "training" in settings:
                for key in ("model_size", "epochs", "batch_size", "image_size",
                            "patience", "lr0", "lrf", "optimizer",
                            "val_split", "test_split"):
                    if key in settings["training"]:
                        cfg["training"][key] = settings["training"][key]

            with open(self.config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics reader
# ═══════════════════════════════════════════════════════════════════════════

def read_training_metrics(run_dir: Path) -> dict:
    """Parse results.csv from a training run directory into chart-friendly JSON."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"available": False}

    epochs = []
    train_box_loss = []
    train_cls_loss = []
    val_box_loss = []
    val_cls_loss = []
    map50 = []
    map50_95 = []

    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # ultralytics CSV has whitespace-padded headers
                row = {k.strip(): v.strip() for k, v in row.items()}
                epoch = int(row.get("epoch", len(epochs) + 1))
                epochs.append(epoch)
                train_box_loss.append(float(row.get("train/box_loss", 0)))
                train_cls_loss.append(float(row.get("train/cls_loss", 0)))
                val_box_loss.append(float(row.get("val/box_loss", 0)))
                val_cls_loss.append(float(row.get("val/cls_loss", 0)))
                map50.append(float(row.get("metrics/mAP50(B)", 0)))
                map50_95.append(float(row.get("metrics/mAP50-95(B)", 0)))
    except Exception:
        return {"available": False}

    best_map50 = max(map50) if map50 else 0
    best_epoch = map50.index(best_map50) + 1 if map50 else 0

    return {
        "available": True,
        "epochs": epochs,
        "train_box_loss": train_box_loss,
        "train_cls_loss": train_cls_loss,
        "val_box_loss": val_box_loss,
        "val_cls_loss": val_cls_loss,
        "map50": map50,
        "map50_95": map50_95,
        "best_map50": best_map50,
        "best_epoch": best_epoch,
        "total_epochs": len(epochs),
    }


def get_dataset_stats(version_dir: Path, cats: list[str]) -> dict:
    """Scan raw and dataset directories for image counts (per-version)."""
    raw_dir = version_dir / "raw"
    dataset_dir = version_dir / "dataset"

    raw_counts = {}
    for cat in cats:
        cat_dir = raw_dir / cat
        raw_counts[cat] = len(list(cat_dir.glob("*.jpg"))) if cat_dir.is_dir() else 0
    unlabeled_dir = raw_dir / "unlabeled"
    raw_counts["unlabeled"] = len(list(unlabeled_dir.glob("*.jpg"))) if unlabeled_dir.is_dir() else 0

    # Dataset split counts
    split_counts = {}
    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        split_counts[split] = len(list(img_dir.glob("*.jpg"))) if img_dir.is_dir() else 0

    calib_dir = dataset_dir / "calibration"
    split_counts["calibration"] = len(list(calib_dir.glob("*.jpg"))) if calib_dir.is_dir() else 0

    dataset_yaml = dataset_dir / "dataset.yaml"

    return {
        "raw_counts": raw_counts,
        "split_counts": split_counts,
        "raw_total": sum(raw_counts.values()),
        "dataset_ready": dataset_yaml.exists(),
    }


def get_output_files(project_root: Path, version_dir: Path) -> dict:
    """Check which output artifacts exist."""
    files = {
        "dataset_yaml": version_dir / "dataset" / "dataset.yaml",
        "best_pt": version_dir / "training" / "weights" / "best.pt",
        "best_onnx": version_dir / "training" / "weights" / "best.onnx",
        "cat_detector_hef": version_dir / "cat_detector.hef",
        "labels_json": project_root / "deploy" / "labels.json",
    }
    result = {}
    for key, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            result[key] = {"exists": True, "size_mb": round(size_mb, 1)}
        else:
            result[key] = {"exists": False}
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Flask application factory
# ═══════════════════════════════════════════════════════════════════════════

def create_app(
    step_mgr: TrainingStepManager,
    config_mgr: TrainingConfigManager,
    cats: list[str],
) -> Flask:
    template_dir = Path(__file__).resolve().parent / "templates"
    flask_app = Flask(__name__, template_folder=str(template_dir))
    project_root = step_mgr.project_root

    # ── Page ──────────────────────────────────────────────────────
    @flask_app.route("/")
    def index():
        return render_template(
            "training_dashboard.html",
            cat_names=cats,
        )

    # ── Pipeline Control API ──────────────────────────────────────
    @flask_app.route("/api/run/<step_name>", methods=["POST"])
    def api_run_step(step_name):
        result = step_mgr.run_step(step_name)
        return jsonify(result)

    @flask_app.route("/api/run_all", methods=["POST"])
    def api_run_all():
        result = step_mgr.run_all()
        return jsonify(result)

    @flask_app.route("/api/stop", methods=["POST"])
    def api_stop():
        result = step_mgr.stop()
        return jsonify(result)

    # ── Status API ────────────────────────────────────────────────
    @flask_app.route("/api/status")
    def api_status():
        result = step_mgr.get_status()
        result["output"] = step_mgr.get_active_output()
        ver_dir = step_mgr.get_version_dir()
        result["outputs"] = get_output_files(project_root, ver_dir)
        result["active_version"] = step_mgr._active_version
        return jsonify(result)

    # ── Metrics API ───────────────────────────────────────────────
    @flask_app.route("/api/metrics")
    def api_metrics():
        return jsonify(read_training_metrics(step_mgr.get_version_dir() / "training"))

    # ── Version API ───────────────────────────────────────────────
    @flask_app.route("/api/versions")
    def api_versions():
        return jsonify({
            "active": step_mgr._active_version,
            "versions": step_mgr.get_versions(),
        })

    @flask_app.route("/api/versions", methods=["POST"])
    def api_set_version():
        data = request.get_json()
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"status": "error", "message": "No version name"}), 400
        step_mgr.set_active_version(name)
        return jsonify({"status": "ok", "active": name})

    @flask_app.route("/api/versions", methods=["DELETE"])
    def api_delete_version():
        data = request.get_json()
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"status": "error", "message": "No version name"}), 400
        result = step_mgr.delete_version(name)
        if result["status"] == "error":
            return jsonify(result), 409
        return jsonify(result)

    @flask_app.route("/api/versions", methods=["PATCH"])
    def api_rename_version():
        data = request.get_json()
        name = data.get("name", "").strip()
        new_name = data.get("new_name", "").strip()
        new_name = re.sub(r"[^a-zA-Z0-9_-]", "_", new_name)
        if not name or not new_name:
            return jsonify({"status": "error", "message": "Name required"}), 400
        result = step_mgr.rename_version(name, new_name)
        if result["status"] == "error":
            return jsonify(result), 409
        return jsonify(result)

    @flask_app.route("/api/versions/duplicate", methods=["POST"])
    def api_duplicate_version():
        data = request.get_json()
        name = data.get("name", "").strip()
        new_name = data.get("new_name", "").strip()
        new_name = re.sub(r"[^a-zA-Z0-9_-]", "_", new_name)
        if not name or not new_name:
            return jsonify({"status": "error", "message": "Name required"}), 400
        result = step_mgr.duplicate_version(name, new_name)
        if result["status"] == "error":
            return jsonify(result), 409
        return jsonify(result)

    # ── Dataset Stats API ─────────────────────────────────────────
    @flask_app.route("/api/dataset_stats")
    def api_dataset_stats():
        return jsonify(get_dataset_stats(step_mgr.get_version_dir(), cats))

    # ── Settings API ──────────────────────────────────────────────
    @flask_app.route("/api/settings", methods=["GET"])
    def api_settings_get():
        return jsonify(config_mgr.get_training_settings())

    @flask_app.route("/api/settings", methods=["POST"])
    def api_settings_post():
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400
        config_mgr.update_training_settings(data)
        # Reload config in step manager
        new_cfg = config_mgr.read()
        step_mgr.reload_config(new_cfg)
        return jsonify({"status": "ok"})

    # ── Annotation API ─────────────────────────────────────────
    @flask_app.route("/api/annotate/images")
    def api_annotate_images():
        ver_dir = step_mgr.get_version_dir()
        raw_dir = ver_dir / "raw"
        images = []
        for cat_name in cats:
            cat_dir = raw_dir / cat_name
            if not cat_dir.is_dir():
                continue
            for img in sorted(cat_dir.glob("*.jpg")):
                has_bbox = img.with_suffix(".txt").exists()
                images.append({
                    "cat": cat_name,
                    "filename": img.name,
                    "has_bbox": has_bbox,
                })
        return jsonify({"images": images})

    @flask_app.route("/api/annotate/image/<cat>/<filename>")
    def api_annotate_image(cat, filename):
        ver_dir = step_mgr.get_version_dir()
        img_path = ver_dir / "raw" / cat / filename
        if not img_path.exists():
            return "Not found", 404
        return send_file(str(img_path), mimetype="image/jpeg")

    @flask_app.route("/api/annotate/bbox/<cat>/<filename>", methods=["GET"])
    def api_annotate_bbox_get(cat, filename):
        ver_dir = step_mgr.get_version_dir()
        label_name = Path(filename).with_suffix(".txt").name
        label_path = ver_dir / "raw" / cat / label_name
        if not label_path.exists():
            return jsonify({"exists": False})
        text = label_path.read_text().strip()
        if not text:
            return jsonify({"exists": False})
        parts = text.split("\n")[0].split()
        if len(parts) < 5:
            return jsonify({"exists": False})
        return jsonify({
            "exists": True,
            "class_id": int(parts[0]),
            "cx": float(parts[1]),
            "cy": float(parts[2]),
            "w": float(parts[3]),
            "h": float(parts[4]),
        })

    @flask_app.route("/api/annotate/bbox/<cat>/<filename>", methods=["POST"])
    def api_annotate_bbox_save(cat, filename):
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400
        ver_dir = step_mgr.get_version_dir()
        label_name = Path(filename).with_suffix(".txt").name
        label_path = ver_dir / "raw" / cat / label_name
        class_id = data.get("class_id", 0)
        cx = data["cx"]
        cy = data["cy"]
        w = data["w"]
        h = data["h"]
        label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        return jsonify({"status": "ok"})

    @flask_app.route("/api/annotate/bbox/<cat>/<filename>", methods=["DELETE"])
    def api_annotate_bbox_delete(cat, filename):
        ver_dir = step_mgr.get_version_dir()
        label_name = Path(filename).with_suffix(".txt").name
        label_path = ver_dir / "raw" / cat / label_name
        if label_path.exists():
            label_path.unlink()
        return jsonify({"status": "ok"})

    @flask_app.route("/api/annotate/auto-detect", methods=["POST"])
    def api_annotate_auto_detect():
        data = request.get_json()
        cat = data.get("cat", "")
        filename = data.get("filename", "")
        ver_dir = step_mgr.get_version_dir()
        img_path = ver_dir / "raw" / cat / filename
        if not img_path.exists():
            return jsonify({"status": "error", "message": "Image not found"}), 404

        try:
            from ultralytics import YOLO
            # Cache model in module-level variable
            if not hasattr(api_annotate_auto_detect, "_coco_model"):
                api_annotate_auto_detect._coco_model = YOLO("yolov8n.pt")
            model = api_annotate_auto_detect._coco_model
            results = model.predict(str(img_path), verbose=False, conf=0.15)
            best_box = None
            best_conf = 0.0
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 15:  # COCO cat class
                        conf = float(box.conf)
                        if conf > best_conf:
                            best_conf = conf
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            img_h, img_w = r.orig_shape
                            best_box = {
                                "cx": ((x1 + x2) / 2) / img_w,
                                "cy": ((y1 + y2) / 2) / img_h,
                                "w": (x2 - x1) / img_w,
                                "h": (y2 - y1) / img_h,
                                "confidence": best_conf,
                            }
            if best_box:
                return jsonify({"status": "ok", "detected": True, **best_box})
            return jsonify({"status": "ok", "detected": False})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @flask_app.route("/api/annotate/stats")
    def api_annotate_stats():
        return jsonify(step_mgr.get_annotation_stats())

    # ── Logs ───────────────────────────────────────────────────────
    @flask_app.route("/api/logs")
    def api_logs():
        log_base = project_root / "logs" / "training"
        if not log_base.is_dir():
            return jsonify({"logs": []})
        logs = []
        for f in sorted(log_base.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
            st = f.stat()
            # Extract version from path (logs/training/<version>/file.log)
            version = f.parent.name
            logs.append({
                "name": f.name,
                "version": version,
                "path": f"{version}/{f.name}",
                "size_kb": round(st.st_size / 1024, 1),
                "modified": st.st_mtime,
            })
        return jsonify({"logs": logs})

    @flask_app.route("/api/logs/<version>/<filename>")
    def api_log_file(version, filename):
        log_base = project_root / "logs" / "training"
        log_path = log_base / version / filename
        if not log_path.exists() or not log_path.is_file():
            return "Not found", 404
        try:
            log_path.resolve().relative_to(log_base.resolve())
        except ValueError:
            return "Forbidden", 403
        return send_file(str(log_path), mimetype="text/plain")

    return flask_app


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training dashboard — web UI for the YOLOv8 training pipeline"
    )
    add_common_args(parser)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5004, help="Port (default: 5004)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config_with_args(args)

    cats: list[str] = config["cats"]
    config_path = args.config

    print_header("Training Dashboard")
    print_info(f"Cats: {', '.join(cats)}")

    # Step manager
    step_mgr = TrainingStepManager(PROJECT_ROOT, config_path, config)

    # Config manager
    yaml_path = config_path or (PROJECT_ROOT / "configs" / "project.yaml")
    config_mgr = TrainingConfigManager(yaml_path)

    # Check dataset / output state
    stats = get_dataset_stats(step_mgr.get_version_dir(), cats)
    outputs = get_output_files(PROJECT_ROOT, step_mgr.get_version_dir())
    print_info(f"Raw images: {stats['raw_total']}")
    if stats["dataset_ready"]:
        sc = stats["split_counts"]
        print_info(f"Dataset ready: train={sc['train']}, val={sc['val']}, test={sc['test']}")
    else:
        print_warning("Dataset not prepared yet (run 'Prepare Dataset' step)")

    for name, info in outputs.items():
        if info["exists"]:
            print_success(f"  {name}: {info['size_mb']} MB")

    # Graceful shutdown
    def shutdown(*_args):
        print_info("\nShutting down...")
        step_mgr.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print_success(f"Dashboard: http://{args.host}:{args.port}")

    flask_app = create_app(
        step_mgr=step_mgr,
        config_mgr=config_mgr,
        cats=cats,
    )
    flask_app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
