#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Training dashboard — web UI for the cat classifier training pipeline.

Manages training steps (pull, label, crop, prepare, train, deploy)
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

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, load_config, resolve_path, get_cat_colors_hex
from utils.display import print_info, print_success, print_error, print_warning, print_header

import yaml


# ═══════════════════════════════════════════════════════════════════════════
# Training Step Manager
# ═══════════════════════════════════════════════════════════════════════════

STEP_ORDER = ["pull", "label", "crop", "prepare", "train", "deploy"]

STEP_LABELS = {
    "pull":    "Pull Captures from Pi",
    "label":   "Label Captures",
    "crop":    "Crop Images",
    "prepare": "Prepare Dataset",
    "train":   "Train Classifier",
    "deploy":  "Deploy to Pi",
    "crossval": "Cross-Validate",
}

# Steps that exist outside the main auto-pipeline
EXTRA_STEPS = ["crossval"]

# Steps that are interactive (not subprocess-based)
INTERACTIVE_STEPS = {"label"}


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
        self._ann_stats_cache: dict | None = None
        self._ann_stats_cache_time: float = 0

        self._steps: dict[str, TrainingStep] = {
            name: TrainingStep(name=name, label=STEP_LABELS[name])
            for name in STEP_ORDER + EXTRA_STEPS
        }

        # Auto-detect active version (lightweight — no size computation)
        versions_dir = self.project_root / "versions"
        if versions_dir.is_dir():
            dirs = sorted(d.name for d in versions_dir.iterdir() if d.is_dir())
            if dirs:
                self._active_version = dirs[-1]

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
            self._ann_stats_cache = None
            self._ann_stats_cache_time = 0
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

        for d in sorted(versions_dir.iterdir()):
            if d.is_dir():
                has_dataset = (d / "classifier_dataset" / "train").exists()
                has_model = (d / "cat_classifier.pt").exists()
                name = d.name
                versions.append({
                    "name": name,
                    "has_model": has_model,
                    "has_dataset": has_dataset,
                    "size_mb": self._version_size_cache.get(name),
                })
        return versions

    def _precompute_version_sizes(self) -> None:
        """Compute version sizes in background thread."""
        versions_dir = self.project_root / "versions"
        if not versions_dir.is_dir():
            return
        for d in sorted(versions_dir.iterdir()):
            if d.is_dir() and d.name not in self._version_size_cache:
                size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                self._version_size_cache[d.name] = round(size_bytes / 1024 / 1024, 1)

    def _invalidate_size_cache(self, name: str | None = None) -> None:
        """Clear cached version sizes. Pass a name to clear one, or None to clear all."""
        if name is None:
            self._version_size_cache.clear()
        else:
            self._version_size_cache.pop(name, None)

    def _deploy_state_path(self) -> Path:
        return self.project_root / "versions" / ".deploy_state.json"

    def get_deployed_version(self) -> str | None:
        """Read which version is currently deployed from disk."""
        p = self._deploy_state_path()
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f).get("version")
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _record_deploy(self, version: str) -> None:
        """Write deploy state to disk after successful deployment."""
        p = self._deploy_state_path()
        with open(p, "w") as f:
            json.dump({"version": version, "timestamp": datetime.utcnow().isoformat() + "Z"}, f)

    @staticmethod
    def _fix_dataset_yaml_path(version_dir: Path) -> None:
        """No-op — retained for compatibility with duplicate/rename. Classifier uses symlinks."""
        pass

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

            # Clean up deploy state if we deleted the deployed version
            if name == self.get_deployed_version():
                self._deploy_state_path().unlink(missing_ok=True)

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

            # Update deploy state if we renamed the deployed version
            if name == self.get_deployed_version():
                self._record_deploy(new_name)

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
        ver_dir = self.get_version_dir()

        if step_name == "pull":
            import_cmd = (
                f"{self._train_python} train/import_captures.py"
                f" --output-dir {ver_dir / 'raw_images'}"
            )
            if self.config_path:
                import_cmd += f" --config {self.config_path}"
            return ["bash", "-c", f"bash scripts/pull_dataset.sh && {import_cmd}"]
        elif step_name == "crop":
            padding = self._config.get("classifier", {}).get("crop_padding", 0.20)
            return [
                self._train_python, "train/crop_images.py",
                "--input-dir", str(ver_dir / "raw_images"),
                "--output-dir", str(ver_dir / "cropped_images"),
                "--padding", str(padding),
            ] + config_args
        elif step_name == "prepare":
            return [
                self._train_python, "train/prepare_classifier_dataset.py",
                "--source-dir", str(ver_dir / "cropped_images"),
                "--output-dir", str(ver_dir / "classifier_dataset"),
            ] + config_args
        elif step_name == "train":
            return [
                self._train_python, "train/train_classifier.py",
                "--dataset-dir", str(ver_dir / "classifier_dataset"),
                "--output-dir", str(ver_dir),
            ] + config_args
        elif step_name == "deploy":
            model_path = str(ver_dir / "cat_classifier.pt")
            return ["bash", "scripts/transfer_model.sh", model_path]
        elif step_name == "crossval":
            n_folds = self._config.get("classifier_training", {}).get("cv_folds", 5)
            return [
                self._train_python, "train/cross_validate.py",
                "--source-dir", str(ver_dir / "cropped_images"),
                "--output-dir", str(ver_dir / "cv_results"),
                "--n-folds", str(n_folds),
            ] + config_args
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
                if step.name == "pull":
                    self.invalidate_label_cache()
                if step.name == "deploy":
                    self._record_deploy(self._active_version)
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

    def get_label_stats(self) -> dict:
        """Count labeled vs unlabeled images in the active version's raw_images/ directory."""
        now = time.time()
        if self._ann_stats_cache and now - self._ann_stats_cache_time < 5:
            return self._ann_stats_cache

        raw_dir = self.get_version_dir() / "raw_images"
        cats = self._config.get("cats", [])
        labeled = 0
        per_cat: dict[str, int] = {}
        for cat_name in cats:
            cat_dir = raw_dir / cat_name
            count = sum(1 for _ in cat_dir.glob("*.jpg")) if cat_dir.is_dir() else 0
            per_cat[cat_name] = count
            labeled += count
        unlabeled_dir = raw_dir / "unlabeled"
        unlabeled = sum(1 for _ in unlabeled_dir.glob("*.jpg")) if unlabeled_dir.is_dir() else 0

        result = {
            "labeled": labeled,
            "unlabeled": unlabeled,
            "per_cat": per_cat,
        }
        self._ann_stats_cache = result
        self._ann_stats_cache_time = now
        return result

    def invalidate_label_cache(self) -> None:
        """Clear cached label stats."""
        self._ann_stats_cache = None
        self._ann_stats_cache_time = 0

    # Keep for backward compatibility (annotate tab still functional)
    def get_annotation_stats(self) -> dict:
        return self.get_label_stats()

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

            # Compute label step state from raw_images/ labeling progress.
            label_stats = self.get_label_stats()
            if label_stats["labeled"] == 0 and label_stats["unlabeled"] == 0:
                steps["label"]["state"] = "idle"
            elif label_stats["unlabeled"] == 0 and label_stats["labeled"] > 0:
                steps["label"]["state"] = "done"
            else:
                steps["label"]["state"] = "idle"
            steps["label"]["label_stats"] = label_stats

            # Restore deploy state from persistent file
            deployed_version = self.get_deployed_version()
            if steps["deploy"]["state"] == "idle" and deployed_version == self._active_version:
                steps["deploy"]["state"] = "done"

            return {
                "steps": steps,
                "current_step": self._current_step,
                "auto_pipeline": self._auto_pipeline,
                "deployed_version": deployed_version,
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
            "classifier_training": cfg.get("classifier_training", {}),
            "cats": cfg.get("cats", []),
        }

    def update_training_settings(self, settings: dict) -> None:
        with self._lock:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)

            if "classifier_training" in settings:
                if "classifier_training" not in cfg:
                    cfg["classifier_training"] = {}
                for key in ("phase1_epochs", "phase2_epochs", "phase3_epochs",
                            "phase1_lr", "phase2_backbone_lr", "phase2_head_lr",
                            "weight_decay", "batch_size", "patience", "optimizer",
                            "session_gap_minutes", "mixup_alpha", "balance_classes",
                            "cv_folds", "energy_ood_percentile"):
                    if key in settings["classifier_training"]:
                        cfg["classifier_training"][key] = settings["classifier_training"][key]

                if "augmentation" in settings["classifier_training"]:
                    if "augmentation" not in cfg["classifier_training"]:
                        cfg["classifier_training"]["augmentation"] = {}
                    for aug_key in ("horizontal_flip", "brightness_contrast",
                                    "brightness_limit", "contrast_limit",
                                    "shift_scale_rotate", "rotate_limit",
                                    "gaussian_blur", "gaussian_noise",
                                    "coarse_dropout", "hue_saturation", "clahe"):
                        if aug_key in settings["classifier_training"]["augmentation"]:
                            cfg["classifier_training"]["augmentation"][aug_key] = \
                                settings["classifier_training"]["augmentation"][aug_key]

            with open(self.config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    def update_crop_padding(self, padding: float) -> None:
        with self._lock:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)
            if "classifier" not in cfg:
                cfg["classifier"] = {}
            cfg["classifier"]["crop_padding"] = round(padding, 4)
            with open(self.config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics reader
# ═══════════════════════════════════════════════════════════════════════════

def read_training_metrics(run_dir: Path, cats: list[str] | None = None) -> dict:
    """Parse results.csv from a classifier training run into chart-friendly JSON.

    Columns: epoch, phase, train_loss, val_loss, train_acc, val_acc, acc_<cat>...
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"available": False}

    epochs: list[int] = []
    phases: list[str] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    train_acc: list[float] = []
    val_acc: list[float] = []
    per_class: dict[str, list[float]] = {}

    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row.get("epoch", len(epochs) + 1)))
                phases.append(row.get("phase", ""))
                train_loss.append(float(row.get("train_loss", 0)))
                val_loss.append(float(row.get("val_loss", 0)))
                train_acc.append(float(row.get("train_acc", 0)))
                val_acc.append(float(row.get("val_acc", 0)))
                for key, val in row.items():
                    if key.startswith("acc_"):
                        cat_name = key[4:]
                        per_class.setdefault(cat_name, []).append(float(val or 0))
    except Exception:
        return {"available": False}

    if not epochs:
        return {"available": False}

    best_val_acc = max(val_acc) if val_acc else 0
    best_epoch = val_acc.index(best_val_acc) + 1 if val_acc else 0

    # Compute phase boundaries (start epoch of each phase)
    phase_boundaries: list[dict] = []
    prev_phase = ""
    for i, ph in enumerate(phases):
        if ph != prev_phase:
            phase_boundaries.append({"phase": ph, "epoch_idx": i})
            prev_phase = ph

    # Build per-phase breakdown
    phase_labels = {"phase1": "Head Warmup", "phase2": "Fine-tune", "phase3": "QAT"}
    phases_data: dict[str, dict] = {}
    for i, ph in enumerate(phases):
        if ph not in phases_data:
            phases_data[ph] = {
                "label": phase_labels.get(ph, ph),
                "local_epochs": [], "train_loss": [], "val_loss": [],
                "train_acc": [], "val_acc": [], "per_class": {},
                "best_val_acc": 0, "best_epoch_local": 0,
            }
        pd = phases_data[ph]
        local_ep = len(pd["local_epochs"]) + 1
        pd["local_epochs"].append(local_ep)
        pd["train_loss"].append(train_loss[i])
        pd["val_loss"].append(val_loss[i])
        pd["train_acc"].append(train_acc[i])
        pd["val_acc"].append(val_acc[i])
        for cat in per_class:
            pd["per_class"].setdefault(cat, []).append(per_class[cat][i])
        if val_acc[i] > pd["best_val_acc"]:
            pd["best_val_acc"] = round(val_acc[i], 4)
            pd["best_epoch_local"] = local_ep

    return {
        "available": True,
        "epochs": epochs,
        "phases": phases,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "per_class": per_class,
        "phase_boundaries": phase_boundaries,
        "phases_data": phases_data,
        "best_val_acc": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "total_epochs": len(epochs),
        "current_phase": phases[-1] if phases else "",
    }


def get_dataset_stats(version_dir: Path, cats: list[str], project_root: Path | None = None) -> dict:
    """Scan raw_images/, cropped_images/, and classifier_dataset/ for image counts.

    raw_images/ and cropped_images/ are version-specific.
    The prepared dataset lives in version_dir/classifier_dataset/{train,val}/<cat>/.
    """
    def count_jpgs(d: Path) -> int:
        return sum(1 for _ in d.glob("*.jpg")) if d.is_dir() else 0

    # Version-specific raw_images/
    raw_dir = version_dir / "raw_images"
    raw_counts: dict[str, int] = {}
    for cat in cats:
        raw_counts[cat] = count_jpgs(raw_dir / cat)
    raw_counts["unlabeled"] = count_jpgs(raw_dir / "unlabeled")

    # Version-specific cropped_images/
    crop_dir = version_dir / "cropped_images"
    crop_counts: dict[str, int] = {}
    for cat in cats:
        crop_counts[cat] = count_jpgs(crop_dir / cat)

    # Version-specific classifier dataset
    dataset_dir = version_dir / "classifier_dataset"
    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        total = 0
        for cat in cats:
            total += count_jpgs(dataset_dir / split / cat)
        split_counts[split] = total

    return {
        "raw_counts": raw_counts,
        "crop_counts": crop_counts,
        "split_counts": split_counts,
        "raw_total": sum(v for k, v in raw_counts.items() if k != "unlabeled"),
        "crop_total": sum(crop_counts.values()),
        "dataset_ready": (dataset_dir / "train").is_dir(),
    }


def get_output_files(project_root: Path, version_dir: Path) -> dict:
    """Check which output artifacts exist."""
    files = {
        "cat_classifier_pt": version_dir / "cat_classifier.pt",
        "global_model": project_root / "models" / "cat_classifier.pt",
    }
    result = {}
    for key, path in files.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / 1024 / 1024
                result[key] = {"exists": True, "size_mb": round(size_mb, 1)}
            else:
                result[key] = {"exists": True, "size_mb": None}
        else:
            result[key] = {"exists": False}

    # Directory artifact counts
    def count_jpgs(d: Path) -> int:
        return sum(1 for _ in d.glob("*.jpg")) if d.is_dir() else 0

    raw_dir = version_dir / "raw_images"
    result["raw_images"] = {"exists": raw_dir.is_dir(), "count": count_jpgs(raw_dir)}
    crop_dir = version_dir / "cropped_images"
    result["cropped_images"] = {"exists": crop_dir.is_dir(), "count": count_jpgs(crop_dir)}
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
        cat_colors = get_cat_colors_hex(cats)
        return render_template(
            "training_dashboard.html",
            cat_names=cats,
            cat_colors=cat_colors,
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

    @flask_app.route("/api/metrics/<version_name>")
    def api_metrics_version(version_name):
        ver_dir = project_root / "versions" / version_name / "training"
        return jsonify(read_training_metrics(ver_dir))

    @flask_app.route("/api/cv_results")
    def api_cv_results():
        cv_dir = step_mgr.get_version_dir() / "cv_results"
        summary_path = cv_dir / "cv_summary.json"
        if not summary_path.exists():
            return jsonify({"available": False})
        import json as _json
        with open(summary_path) as f:
            data = _json.load(f)
        data["available"] = True
        return jsonify(data)

    # ── Version API ───────────────────────────────────────────────
    @flask_app.route("/api/versions")
    def api_versions():
        return jsonify({
            "active": step_mgr._active_version,
            "versions": step_mgr.get_versions(),
            "deployed_version": step_mgr.get_deployed_version(),
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
        return jsonify(get_dataset_stats(step_mgr.get_version_dir(), cats, project_root))

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

    @flask_app.route("/api/settings/crop", methods=["GET"])
    def api_settings_crop_get():
        cfg = config_mgr.read()
        padding = cfg.get("classifier", {}).get("crop_padding", 0.20)
        return jsonify({"crop_padding": padding})

    @flask_app.route("/api/settings/crop", methods=["POST"])
    def api_settings_crop_post():
        data = request.get_json()
        if not data or "crop_padding" not in data:
            return jsonify({"status": "error", "message": "crop_padding required"}), 400
        padding = float(data["crop_padding"])
        if not (0.0 <= padding <= 1.0):
            return jsonify({"status": "error", "message": "crop_padding must be 0.0–1.0"}), 400
        config_mgr.update_crop_padding(padding)
        new_cfg = config_mgr.read()
        step_mgr.reload_config(new_cfg)
        return jsonify({"status": "ok", "crop_padding": padding})

    # ── Label API (classifier labeling on raw_images/) ────────────
    _label_undo_stack: list[tuple[Path, Path]] = []  # (src, dst) pairs for undo

    def _label_raw_dir() -> Path:
        return step_mgr.get_version_dir() / "raw_images"

    def _move_with_sidecar(src: Path, dst: Path) -> None:
        """Move a JPEG and its .json sidecar (if present) from src to dst."""
        src.rename(dst)
        src_json = src.with_suffix(".json")
        if src_json.exists():
            src_json.rename(dst.with_suffix(".json"))

    @flask_app.route("/api/label/images")
    def api_label_images():
        raw_dir = _label_raw_dir()
        unlabeled_dir = raw_dir / "unlabeled"
        if not unlabeled_dir.is_dir():
            return jsonify({"count": 0, "first": None, "total_labeled": 0})
        images = sorted(unlabeled_dir.glob("*.jpg"), key=lambda p: p.name)
        labeled_total = sum(
            sum(1 for _ in (raw_dir / c).glob("*.jpg"))
            for c in cats
            if (raw_dir / c).is_dir()
        )
        first = images[0].name if images else None
        filenames = [p.name for p in images[:30]]
        return jsonify({"count": len(images), "first": first, "total_labeled": labeled_total, "filenames": filenames})

    @flask_app.route("/api/label/image/<filename>")
    def api_label_image(filename):
        img_path = _label_raw_dir() / "unlabeled" / filename
        if not img_path.is_file():
            return "Not found", 404
        return send_file(str(img_path), mimetype="image/jpeg")

    @flask_app.route("/api/label/image-meta/<filename>")
    def api_label_image_meta(filename):
        """Return sidecar JSON for a given unlabeled image (for bbox overlay)."""
        sidecar_path = _label_raw_dir() / "unlabeled" / Path(filename).with_suffix(".json").name
        if not sidecar_path.is_file():
            return jsonify({"exists": False})
        try:
            meta = json.loads(sidecar_path.read_text())
            return jsonify({"exists": True, **meta})
        except Exception:
            return jsonify({"exists": False})

    @flask_app.route("/api/label/assign", methods=["POST"])
    def api_label_assign():
        data = request.get_json()
        filename = data.get("filename", "").strip()
        cat_name = data.get("cat_name", "").strip()
        if not filename or cat_name not in cats:
            return jsonify({"status": "error", "message": "Invalid filename or cat_name"}), 400
        raw_dir = _label_raw_dir()
        src = raw_dir / "unlabeled" / filename
        if not src.is_file():
            return jsonify({"status": "error", "message": "File not found"}), 404
        dst_dir = raw_dir / cat_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / filename
        _move_with_sidecar(src, dst)
        _label_undo_stack.append((dst, src))
        if len(_label_undo_stack) > 20:
            _label_undo_stack.pop(0)
        step_mgr.invalidate_label_cache()
        # Return next image
        unlabeled_dir = raw_dir / "unlabeled"
        remaining = sorted(unlabeled_dir.glob("*.jpg"), key=lambda p: p.name)
        return jsonify({"status": "ok", "next": remaining[0].name if remaining else None,
                        "remaining": len(remaining)})

    @flask_app.route("/api/label/skip", methods=["POST"])
    def api_label_skip():
        data = request.get_json()
        filename = data.get("filename", "").strip()
        if not filename:
            return jsonify({"status": "error", "message": "No filename"}), 400
        raw_dir = _label_raw_dir()
        src = raw_dir / "unlabeled" / filename
        if not src.is_file():
            return jsonify({"status": "error", "message": "File not found"}), 404
        skip_dir = raw_dir / "unlabeled" / "skipped"
        skip_dir.mkdir(parents=True, exist_ok=True)
        dst = skip_dir / filename
        _move_with_sidecar(src, dst)
        _label_undo_stack.append((dst, src))
        if len(_label_undo_stack) > 20:
            _label_undo_stack.pop(0)
        unlabeled_dir = raw_dir / "unlabeled"
        remaining = sorted(unlabeled_dir.glob("*.jpg"), key=lambda p: p.name)
        return jsonify({"status": "ok", "next": remaining[0].name if remaining else None,
                        "remaining": len(remaining)})

    @flask_app.route("/api/label/undo", methods=["POST"])
    def api_label_undo():
        if not _label_undo_stack:
            return jsonify({"status": "error", "message": "Nothing to undo"}), 400
        dst, src = _label_undo_stack.pop()
        if not dst.is_file():
            return jsonify({"status": "error", "message": "File no longer at expected location"}), 404
        src.parent.mkdir(parents=True, exist_ok=True)
        _move_with_sidecar(dst, src)
        step_mgr.invalidate_label_cache()
        return jsonify({"status": "ok", "restored": src.name})

    # ── Annotation API (bbox sidecars in raw_images/, read by crop step) ──
    # Boxes are stored in the same .json sidecar format the Pi's detection
    # capture writes, so train/crop_images.py crops annotated images exactly
    # like Pi-captured ones (highest-confidence detection, bbox_norm =
    # [xmin, ymin, w, h] normalized). The UI works in center format
    # (cx, cy, w, h); conversion happens here.

    def _ann_image_path(folder: str, filename: str) -> Path | None:
        """Resolve raw_images/<folder>/<filename>; None if folder/name invalid."""
        if folder not in cats and folder != "unlabeled":
            return None
        if not filename or Path(filename).name != filename or not filename.lower().endswith(".jpg"):
            return None
        return _label_raw_dir() / folder / filename

    def _ann_read_sidecar(img_path: Path) -> dict | None:
        sidecar = img_path.with_suffix(".json")
        if not sidecar.is_file():
            return None
        try:
            return json.loads(sidecar.read_text())
        except Exception:
            return None

    def _ann_write_sidecar(img_path: Path, meta: dict) -> None:
        """Write the sidecar atomically (temp file + rename)."""
        sidecar = img_path.with_suffix(".json")
        tmp = sidecar.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(sidecar)

    def _ann_best_detection(meta: dict | None) -> dict | None:
        """Highest-confidence detection with a valid bbox (same rule as crop_images.py)."""
        if not meta:
            return None
        valid = [
            d for d in meta.get("detections", [])
            if isinstance(d.get("bbox_norm"), list) and len(d["bbox_norm"]) == 4
        ]
        return max(valid, key=lambda d: d.get("confidence", 0)) if valid else None

    def _ann_image_size(img_path: Path) -> list[int] | None:
        try:
            import cv2
            img = cv2.imread(str(img_path))
            return [int(img.shape[1]), int(img.shape[0])] if img is not None else None
        except Exception:
            return None

    @flask_app.route("/api/annotate/images")
    def api_annotate_images():
        raw_dir = _label_raw_dir()
        images = []
        for folder in [*cats, "unlabeled"]:
            folder_dir = raw_dir / folder
            if not folder_dir.is_dir():
                continue
            for img in sorted(folder_dir.glob("*.jpg")):
                images.append({
                    "cat": folder,
                    "filename": img.name,
                    "has_bbox": _ann_best_detection(_ann_read_sidecar(img)) is not None,
                })
        return jsonify({"images": images})

    @flask_app.route("/api/annotate/image/<cat>/<filename>")
    def api_annotate_image(cat, filename):
        img_path = _ann_image_path(cat, filename)
        if img_path is None or not img_path.is_file():
            return "Not found", 404
        return send_file(str(img_path), mimetype="image/jpeg")

    @flask_app.route("/api/annotate/bbox/<cat>/<filename>", methods=["GET"])
    def api_annotate_bbox_get(cat, filename):
        img_path = _ann_image_path(cat, filename)
        if img_path is None:
            return jsonify({"exists": False})
        best = _ann_best_detection(_ann_read_sidecar(img_path))
        if best is None:
            return jsonify({"exists": False})
        xmin, ymin, w, h = (float(v) for v in best["bbox_norm"])
        return jsonify({
            "exists": True,
            "cx": xmin + w / 2,
            "cy": ymin + h / 2,
            "w": w,
            "h": h,
            "confidence": best.get("confidence"),
            "source": best.get("source", "pi"),
        })

    @flask_app.route("/api/annotate/bbox/<cat>/<filename>", methods=["POST"])
    def api_annotate_bbox_save(cat, filename):
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400
        img_path = _ann_image_path(cat, filename)
        if img_path is None or not img_path.is_file():
            return jsonify({"status": "error", "message": "Image not found"}), 404
        try:
            cx, cy, w, h = (float(data[k]) for k in ("cx", "cy", "w", "h"))
        except (KeyError, TypeError, ValueError):
            return jsonify({"status": "error", "message": "cx, cy, w, h required"}), 400
        # Center format -> [xmin, ymin, w, h], clamped to the image
        x1 = min(max(cx - w / 2, 0.0), 1.0)
        y1 = min(max(cy - h / 2, 0.0), 1.0)
        x2 = min(max(cx + w / 2, 0.0), 1.0)
        y2 = min(max(cy + h / 2, 0.0), 1.0)
        if x2 <= x1 or y2 <= y1:
            return jsonify({"status": "error", "message": "Empty bbox"}), 400
        source = "auto" if data.get("source") == "auto" else "manual"
        confidence = float(data.get("confidence", 1.0)) if source == "auto" else 1.0

        meta = _ann_read_sidecar(img_path)
        if meta is None:
            meta = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "session_id": "",
                "image_size": _ann_image_size(img_path),
                "created_by": "annotate",
            }
        elif "original_detections" not in meta and meta.get("created_by") != "annotate":
            # First edit of a Pi sidecar: keep the Pi's detections restorable
            meta["original_detections"] = meta.get("detections", [])
        meta["detections"] = [{
            "label": "cat",
            "confidence": round(confidence, 4),
            "bbox_norm": [round(v, 6) for v in (x1, y1, x2 - x1, y2 - y1)],
            "source": source,
        }]
        _ann_write_sidecar(img_path, meta)
        return jsonify({"status": "ok"})

    @flask_app.route("/api/annotate/bbox/<cat>/<filename>", methods=["DELETE"])
    def api_annotate_bbox_delete(cat, filename):
        img_path = _ann_image_path(cat, filename)
        if img_path is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400
        meta = _ann_read_sidecar(img_path)
        if meta is None:
            return jsonify({"status": "ok", "result": "none"})
        if "original_detections" in meta:
            meta["detections"] = meta.pop("original_detections")
            _ann_write_sidecar(img_path, meta)
            return jsonify({"status": "ok", "result": "restored"})
        if meta.get("created_by") == "annotate":
            img_path.with_suffix(".json").unlink()
            return jsonify({"status": "ok", "result": "deleted"})
        # Untouched Pi detection: never destroy capture data
        return jsonify({"status": "ok", "result": "kept_pi"})

    @flask_app.route("/api/annotate/auto-detect", methods=["POST"])
    def api_annotate_auto_detect():
        data = request.get_json()
        img_path = _ann_image_path(data.get("cat", ""), data.get("filename", ""))
        if img_path is None or not img_path.is_file():
            return jsonify({"status": "error", "message": "Image not found"}), 404

        try:
            from ultralytics import YOLO
            # Cache model in module-level variable
            if not hasattr(api_annotate_auto_detect, "_coco_model"):
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                api_annotate_auto_detect._coco_model = YOLO("yolov8n.pt").to(device)
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
    stats = get_dataset_stats(step_mgr.get_version_dir(), cats, PROJECT_ROOT)
    outputs = get_output_files(PROJECT_ROOT, step_mgr.get_version_dir())
    print_info(f"Raw images: {stats['raw_total']}")
    if stats["dataset_ready"]:
        sc = stats["split_counts"]
        print_info(f"Dataset ready: train={sc['train']}, val={sc['val']}")
    else:
        print_warning("Dataset not prepared yet (run 'Prepare Dataset' step)")

    for name, info in outputs.items():
        if info.get("exists"):
            sz = info.get("size_mb")
            print_success(f"  {name}: {sz} MB" if sz is not None else f"  {name}: present")

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
    threading.Thread(target=step_mgr._precompute_version_sizes, daemon=True).start()
    flask_app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
