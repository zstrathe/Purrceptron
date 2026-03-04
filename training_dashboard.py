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
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

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

STEP_ORDER = ["pull", "prepare", "train", "export", "convert", "deploy"]

STEP_LABELS = {
    "pull": "Pull Data from Pi",
    "prepare": "Prepare Dataset",
    "train": "Train Model",
    "export": "Export ONNX",
    "convert": "Convert HEF",
    "deploy": "Deploy to Pi",
}


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
            self._active_version = name
            ver_dir = self.project_root / "versions" / name
            ver_dir.mkdir(parents=True, exist_ok=True)

    def get_versions(self) -> list[dict]:
        """List available training versions."""
        versions_dir = self.project_root / "versions"
        versions = []
        if versions_dir.is_dir():
            for d in sorted(versions_dir.iterdir()):
                if d.is_dir():
                    has_best = (d / "training" / "weights" / "best.pt").exists()
                    has_onnx = (d / "training" / "weights" / "best.onnx").exists()
                    versions.append({"name": d.name, "has_model": has_best, "has_onnx": has_onnx})
        return versions

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
            return ["bash", "train/convert_hef.sh", onnx_path, num_classes, calib_dir, hw_arch, model_size]
        elif step_name == "deploy":
            return ["bash", "scripts/transfer_model.sh"]
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def run_step(self, step_name: str) -> dict:
        """Launch a single training step."""
        if step_name not in self._steps:
            return {"status": "error", "message": f"Unknown step: {step_name}"}

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
            name = self._current_step
            self._current_step = None
            return {"status": "ok", "step": name}

    def _monitor_step(self, step: TrainingStep) -> None:
        """Read output and detect completion (runs in daemon thread)."""
        if step.process and step.process.stdout:
            try:
                for line in step.process.stdout:
                    step.output.append(line.rstrip())
            except (ValueError, OSError):
                pass

        # Process finished
        exit_code = step.process.wait() if step.process else -1
        step.exit_code = exit_code
        step.end_time = time.time()

        with self._lock:
            if exit_code == 0:
                step.state = "done"
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
        """Return the next step name after current, or None if done."""
        idx = STEP_ORDER.index(current)
        if idx + 1 < len(STEP_ORDER):
            return STEP_ORDER[idx + 1]
        return None

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
                            "patience", "val_split", "test_split"):
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
        "cat_detector_hef": project_root / "models" / "cat_detector.hef",
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
