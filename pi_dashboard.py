#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Unified collection dashboard — capture, label review, and inference control.

Single-page tabbed web UI for the full data collection workflow:
- Capture tab: live detection feed with bounding boxes, mode switching, manual capture
- Review tab: iterate images, assign labels, delete bad captures, thumbnail navigation

Usage:
    python pi_dashboard.py                   # reads configs/project.yaml
    python pi_dashboard.py --config alt.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

from datetime import datetime, timedelta, timezone

from flask import Flask, Response, jsonify, render_template, request, send_file

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import PROJECT_ROOT, add_common_args, load_config, resolve_path, get_cat_colors_hex
from utils.display import print_info, print_success, print_error, print_warning, print_header

from collect.label_review import LabelReviewApp
from remote_training import settings as training_settings
from remote_training.backends import BackendError
from remote_training.backup.destinations import BackupError
from remote_training.backup.manager import BackupManager
from remote_training.bundle import _low_priority_prefix
from remote_training.manager import TrainingJobManager
from utils import sidecars
from utils.secrets import SecretsStore
from utils.runtime_paths import CLASSIFIER_STATUS_JSON, PREVIEW_DETECTIONS_JSON, PREVIEW_JPG

import yaml


# ═══════════════════════════════════════════════════════════════════════════
# Classifier Events Cache
# ═══════════════════════════════════════════════════════════════════════════

_EVENTS_CACHE_TTL = 30  # seconds

_events_cache: dict = {}  # {cache_key: {"data": [...], "expires": float}, "_db_path": Path}


def _refresh_events_cache(db_path=None, start=None, end=None):
    """Refresh the in-memory classifier events cache from SQLite."""
    path = db_path or _events_cache.get("_db_path")
    if path:
        _events_cache["_db_path"] = path
    if not path or not Path(str(path)).is_file():
        return []
    # Evict oldest entry if cache grows beyond 7 keys
    cache_keys = [k for k in _events_cache if k != "_db_path"]
    if len(cache_keys) > 7:
        oldest = min(cache_keys, key=lambda k: _events_cache[k].get("expires", 0))
        del _events_cache[oldest]
    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        if start and end:
            rows = conn.execute(
                "SELECT * FROM classifier_events "
                "WHERE start_time >= ? AND start_time < ? ORDER BY id DESC",
                (start, end),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM classifier_events ORDER BY id DESC LIMIT 50",
            ).fetchall()
    data = [dict(r) for r in rows]
    if start:
        _events_cache[start] = {
            "data": data,
            "expires": time.monotonic() + _EVENTS_CACHE_TTL,
        }
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Auto-Capture State
# ═══════════════════════════════════════════════════════════════════════════

class AutoCaptureState:
    """Tracks auto-capture mode: enabled, target cat, and interval."""

    def __init__(self, cats: list[str], interval: float = 3.0):
        self.enabled = False
        self.target_cat = cats[0] if cats else ""
        self.interval = interval
        self.last_capture_time = 0.0
        self._lock = threading.Lock()

    def toggle(self) -> tuple[bool, str]:
        """Toggle enabled state. Returns (new_enabled, target_cat)."""
        with self._lock:
            self.enabled = not self.enabled
            if self.enabled:
                self.last_capture_time = time.time()
            return self.enabled, self.target_cat

    def set_target(self, cat: str) -> None:
        with self._lock:
            self.target_cat = cat

    def get_state(self) -> dict:
        with self._lock:
            return {
                "capture_mode": "auto" if self.enabled else "manual",
                "auto_target": self.target_cat,
            }


def _auto_capture_loop(state: AutoCaptureState, pipeline_mgr: 'DetectionPipelineManager') -> None:
    """Background thread that sends periodic capture commands when auto mode is on."""
    while True:
        time.sleep(0.25)
        with state._lock:
            if not state.enabled:
                continue
            now = time.time()
            if now - state.last_capture_time < state.interval:
                continue
            cat = state.target_cat
            state.last_capture_time = now

        if pipeline_mgr.is_running():
            pipeline_mgr.send_command({"cmd": "capture", "cat": cat})


# ═══════════════════════════════════════════════════════════════════════════
# Detection Pipeline Manager
# ═══════════════════════════════════════════════════════════════════════════

class DetectionPipelineManager:
    """Manages the always-running detect.py subprocess in default/COCO mode."""

    PREVIEW_PATH = PREVIEW_JPG          # written by detect.py, in RAM

    def __init__(self, project_root: Path, config_path: Path | None = None,
                 hailo_apps_dir: Path | None = None):
        self.project_root = project_root
        self.config_path = config_path
        self.hailo_apps_dir = hailo_apps_dir
        self.process: subprocess.Popen | None = None
        self._output_lines: collections.deque[str] = collections.deque(maxlen=100)
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.mode: str = "default"
        self._config: dict | None = None
        # Watchdog auto-restart state
        self._intentional_stop = False
        self._watchdog_thread: threading.Thread | None = None
        self._restart_info: dict | None = None
        # Log file for persisting full output to disk
        self._log_file = None
        self._log_dir = self.project_root / "logs" / "detection"
        # Model file watcher: auto-restart when classifier model changes
        self._model_watcher_thread: threading.Thread | None = None
        self._model_watcher_stop = threading.Event()

    def set_config(self, config: dict) -> None:
        self._config = config

    def is_running(self) -> bool:
        with self._lock:
            if self.process is None:
                return False
            return self.process.poll() is None

    def start(self) -> dict:
        """Start detect.py subprocess in default/COCO mode.

        Launches detect.py inside the hailo-apps environment by sourcing
        setup_env.sh first, which activates the hailo-apps venv and sets
        GStreamer plugin paths.
        """
        if self.is_running():
            return {"status": "error", "message": "Already running"}

        self._intentional_stop = False
        self._restart_info = None

        # Build the detect.py command with absolute paths.
        # IMPORTANT: cwd must stay at hailo-apps dir — hailo-apps resolves
        # post-processing resources relative to its install directory.
        detect_script = self.project_root / "deploy" / "detect.py"
        detect_flags = "--enable-preview --enable-stdin-commands"
        if self.config_path:
            config_abs = self.config_path if self.config_path.is_absolute() else self.project_root / self.config_path
            detect_flags += f" --config {config_abs}"

        # Launch inside hailo-apps environment (provides gi, hailo, hailo_apps)
        # Stay in hailo-apps dir so resource lookups work correctly.
        if self.hailo_apps_dir:
            shell_cmd = (
                f"cd {self.hailo_apps_dir} && source setup_env.sh && "
                f"python {detect_script} {detect_flags}"
            )
            cmd = ["bash", "-c", shell_cmd]
        else:
            # Fallback: try sys.executable directly (for dev/testing without Hailo)
            cmd = [sys.executable, "deploy/detect.py",
                   "--enable-preview", "--enable-stdin-commands"]
            if self.config_path:
                cmd += ["--config", str(self.config_path)]

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        self._output_lines.clear()
        # Open a log file for full output persistence
        self._close_log_file()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._prune_old_logs()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = open(self._log_dir / f"default_{ts}.log", "w")

        self._reader_thread = threading.Thread(
            target=self._read_output, daemon=True
        )
        self._reader_thread.start()

        # Start watchdog for auto-restart on crash
        self._watchdog_thread = threading.Thread(
            target=self._watchdog, daemon=True
        )
        self._watchdog_thread.start()

        # Start model file watcher for auto-restart on new model deployment
        self._start_model_watcher()

        return {"status": "ok", "pid": self.process.pid, "mode": "default"}

    def stop(self) -> dict:
        """Stop the detect.py subprocess."""
        self._intentional_stop = True
        self._model_watcher_stop.set()
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                self.process = None
                return {"status": "ok", "message": "Not running"}

            # Close stdin first to signal graceful shutdown
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass

            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

            self.process = None
            self._close_log_file()
            return {"status": "ok"}

    def _start_model_watcher(self) -> None:
        """Watch the classifier model file for changes and auto-restart."""
        if not self._config:
            return
        clf_cfg = self._config.get("classifier", {})
        if not clf_cfg.get("enabled", False):
            return
        model_path = Path(clf_cfg.get("model_path", "models/cat_classifier.pte"))
        if not model_path.is_absolute():
            model_path = self.project_root / model_path

        self._model_watcher_stop.clear()
        self._model_watcher_thread = threading.Thread(
            target=self._watch_model_file,
            args=(model_path,),
            daemon=True,
        )
        self._model_watcher_thread.start()

    def _watch_model_file(self, model_path: Path) -> None:
        """Poll model file mtime and trigger restart on change."""
        try:
            last_mtime = model_path.stat().st_mtime if model_path.exists() else None
        except OSError:
            last_mtime = None

        while not self._model_watcher_stop.wait(timeout=5):
            try:
                if not model_path.exists():
                    continue
                mtime = model_path.stat().st_mtime
                if last_mtime is not None and mtime != last_mtime:
                    print(f"[model-watcher] Model file changed, restarting pipeline...",
                          flush=True)
                    self.restart()
                    return  # restart() starts a new watcher
                last_mtime = mtime
            except OSError:
                continue

    def restart(self) -> dict:
        """Stop and restart the pipeline with a delay for device release."""
        self.stop()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=3)
        time.sleep(3)  # kernel needs time to release /dev/video0 and Hailo NPU
        return self.start()

    def _close_log_file(self) -> None:
        if self._log_file and not self._log_file.closed:
            try:
                self._log_file.close()
            except Exception:
                pass
        self._log_file = None

    def _prune_old_logs(self) -> None:
        """Delete detection logs older than logging.keep_days (default 90;
        0 keeps them all). A new log starts with every pipeline start, so
        without this they would pile up forever."""
        keep_days = ((self._config or {}).get("logging") or {}).get("keep_days", 90)
        if not keep_days or keep_days <= 0:
            return
        cutoff = time.time() - keep_days * 86400
        for f in self._log_dir.glob("*.log"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

    def get_log_files(self) -> list[dict]:
        """List available log files, newest first."""
        if not self._log_dir.is_dir():
            return []
        logs = []
        for f in sorted(self._log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
            st = f.stat()
            logs.append({
                "name": f.name,
                "size_kb": round(st.st_size / 1024, 1),
                "modified": st.st_mtime,
            })
        return logs

    def send_command(self, cmd: dict) -> bool:
        """Send a JSON command to detect.py via stdin pipe."""
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                return False
            if self.process.stdin is None:
                return False
            try:
                self.process.stdin.write(json.dumps(cmd) + "\n")
                self.process.stdin.flush()
                return True
            except (BrokenPipeError, OSError):
                return False

    def get_output(self) -> list[str]:
        with self._lock:
            return list(self._output_lines)

    def get_status(self) -> dict:
        status = {
            "running": self.is_running(),
            "mode": self.mode,
            "pid": self.process.pid if self.process and self.is_running() else None,
            "output": self.get_output(),
        }
        if self._restart_info:
            status["restart_info"] = self._restart_info
        return status

    def _watchdog(self) -> None:
        """Monitor subprocess and auto-restart on unexpected exit."""
        backoff = 3
        max_restarts = 5
        restarts = 0
        while restarts < max_restarts:
            if self.process is None:
                return
            ret = self.process.wait()
            if self._intentional_stop:
                return
            restarts += 1
            self._restart_info = {
                "count": restarts,
                "max": max_restarts,
                "last_exit_code": ret,
                "waiting_seconds": backoff,
            }
            time.sleep(backoff)
            if self._intentional_stop:
                return
            self.process = None
            self._restart_info = None
            result = self.start()
            if result["status"] != "ok":
                self._restart_info = {"exhausted": True, "count": restarts}
            return  # start() created a new watchdog; exit this one
        self._restart_info = {"exhausted": True, "count": restarts}

    def _read_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        try:
            for line in self.process.stdout:
                with self._lock:
                    self._output_lines.append(line.rstrip())
                if self._log_file and not self._log_file.closed:
                    try:
                        self._log_file.write(line)
                        self._log_file.flush()
                    except Exception:
                        pass
        except (ValueError, OSError):
            pass
        finally:
            self._close_log_file()


# ═══════════════════════════════════════════════════════════════════════════
# Config Manager — read/write project.yaml for inference settings
# ═══════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """Read/write inference-related settings.

    Reads from project.yaml (shared defaults) merged with runtime.yaml
    (Pi-local overrides).  Writes only to runtime.yaml so that project.yaml
    can be synced freely without overwriting Pi-specific state like ROI,
    enabled flags, etc.
    """

    # Keys that get saved to runtime.yaml when changed via the dashboard UI.
    _RUNTIME_KEYS: dict[str, tuple[str, ...]] = {
        "deployment": ("hef_path", "labels_json", "track_all", "tracking_objects"),
        "detection_capture": ("enabled", "unlabeled_classes", "min_interval_seconds",
                              "max_captures_per_session", "save_labeled_only"),
        "feeding": ("enabled", "cooldown_seconds", "min_detection_frames"),
        "classifier": ("enabled", "confidence_threshold", "stability_frames",
                        "classify_interval_seconds", "bowl_roi",
                        "roi_overlap_threshold"),
    }

    def __init__(self, config_path: Path, runtime_path: Path | None = None):
        self.config_path = config_path
        self.runtime_path = runtime_path or config_path.parent / "runtime.yaml"
        self._lock = threading.Lock()

    def _read_runtime(self) -> dict:
        """Read runtime overrides (empty dict if file doesn't exist)."""
        if self.runtime_path.exists():
            with open(self.runtime_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def read(self) -> dict:
        """Return project config with runtime overrides merged in."""
        with self._lock:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)
            runtime = self._read_runtime()
            # Deep-merge runtime overrides into config
            for section, overrides in runtime.items():
                if isinstance(overrides, dict) and section in cfg and isinstance(cfg[section], dict):
                    cfg[section].update(overrides)
                else:
                    cfg[section] = overrides
            return cfg

    def get_inference_settings(self) -> dict:
        """Return the subset of config relevant to inference UI controls."""
        cfg = self.read()
        return {
            "deployment": cfg.get("deployment", {}),
            "detection_capture": cfg.get("detection_capture", {}),
            "feeding": cfg.get("feeding", {}),
            "classifier": cfg.get("classifier", {}),
        }

    def update_inference_settings(self, settings: dict) -> None:
        """Update inference-related fields in runtime.yaml only."""
        with self._lock:
            runtime = self._read_runtime()

            for section, keys in self._RUNTIME_KEYS.items():
                if section in settings:
                    runtime.setdefault(section, {})
                    for key in keys:
                        if key in settings[section]:
                            runtime[section][key] = settings[section][key]

            self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.runtime_path, "w") as f:
                yaml.dump(runtime, f, default_flow_style=False, sort_keys=False)

    def update_training_backend(self, backend: str, settings: dict,
                                min_images_per_cat: int | None = None) -> None:
        """Save the selected training backend and its (non-secret) settings
        to runtime.yaml. Secrets go to SecretsStore instead."""
        # Start from the merged view: runtime.yaml replaces whole sections'
        # keys, so the backend block must be written complete.
        current = (self.read().get("training_backend") or {}).get(backend) or {}
        with self._lock:
            runtime = self._read_runtime()
            tb = runtime.setdefault("training_backend", {})
            tb["backend"] = backend
            if min_images_per_cat is not None:
                tb["min_images_per_cat"] = int(min_images_per_cat)
            tb[backend] = {**current, **settings}
            self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.runtime_path, "w") as f:
                yaml.dump(runtime, f, default_flow_style=False, sort_keys=False)

    def update_backup(self, block: dict) -> None:
        """Save the complete backup settings block to runtime.yaml."""
        with self._lock:
            runtime = self._read_runtime()
            runtime["backup"] = block
            self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.runtime_path, "w") as f:
                yaml.dump(runtime, f, default_flow_style=False, sort_keys=False)

    def update_training_settings(self, values: dict) -> None:
        """Save classifier training settings (already validated by
        remote_training.settings.parse) to runtime.yaml. classifier_training
        is written complete, because runtime.yaml replaces a section's
        top-level keys (the augmentation dict as a whole)."""
        tc, crop_padding = training_settings.apply(self.read(), values)
        with self._lock:
            runtime = self._read_runtime()
            runtime["classifier_training"] = tc
            if crop_padding is not None:
                runtime.setdefault("classifier", {})["crop_padding"] = crop_padding
            self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.runtime_path, "w") as f:
                yaml.dump(runtime, f, default_flow_style=False, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════════
# Flask application factory
# ═══════════════════════════════════════════════════════════════════════════

# Endpoints the dashboard page polls continuously (preview every 67 ms, the
# rest every 0.2-30 s). Logging every successful poll would add a line to
# logs/pi_dashboard.out many times a second; errors and all other requests
# are still logged.
_QUIET_POLL_RE = re.compile(
    r'"(?:\x1b\[[0-9;]*m)*GET /api/(?:preview|detections|status|classifier/current|'
    r'classifier/events|training/status|backup/status)(?:\?\S*)? HTTP/[0-9.]+(?:\x1b\[[0-9;]*m)*" '
    r'(?:200|204|304) ')


class _QuietPollingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _QUIET_POLL_RE.search(record.getMessage())


def create_app(
    reviewer: LabelReviewApp,
    pipeline_mgr: DetectionPipelineManager,
    config_mgr: ConfigManager,
    auto_state: AutoCaptureState,
    cats: list[str],
    min_images: int,
    training_mgr: TrainingJobManager,
    secrets: SecretsStore,
    backup_mgr: BackupManager,
) -> Flask:
    template_dir = Path(__file__).resolve().parent / "templates"
    flask_app = Flask(__name__, template_folder=str(template_dir))
    logging.getLogger("werkzeug").addFilter(_QuietPollingFilter())

    # ── Page ──────────────────────────────────────────────────────
    @flask_app.route("/")
    def index():
        cats_indexed = list(enumerate(cats, start=1))
        cat_colors = get_cat_colors_hex(cats)
        return render_template(
            "dashboard.html",
            cats=cats_indexed,
            cat_names=cats,
            cat_colors=cat_colors,
            min_images=min_images,
        )

    # ── Preview Frame (latest JPEG from detect.py) ───────────────
    @flask_app.route("/api/preview")
    def api_preview():
        """Return the latest preview frame as a single JPEG image.

        The client polls this endpoint to display live video.
        This avoids MJPEG streaming buffering issues that cause
        multi-second delay on the Raspberry Pi.
        """
        preview_path = DetectionPipelineManager.PREVIEW_PATH
        try:
            with open(preview_path, "rb") as f:
                jpeg_bytes = f.read()
            if not jpeg_bytes:
                return '', 204
            return Response(jpeg_bytes, mimetype="image/jpeg",
                            headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})
        except (FileNotFoundError, OSError):
            return '', 204

    @flask_app.route("/api/detections")
    def api_detections():
        """Return current detection labels from the sidecar JSON."""
        det_path = PREVIEW_DETECTIONS_JSON
        try:
            with open(det_path) as f:
                return Response(f.read(), mimetype="application/json",
                                headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})
        except (FileNotFoundError, OSError):
            return Response("[]", mimetype="application/json")

    # ── Capture API (sends command to detect.py via stdin) ───────
    @flask_app.route("/api/capture/<cat_name>", methods=["POST"])
    def api_capture(cat_name):
        if cat_name not in cats and cat_name != "unlabeled":
            return jsonify({"status": "error", "message": f"Unknown cat: {cat_name}"}), 400
        if not pipeline_mgr.is_running():
            return jsonify({"status": "error", "message": "Detection pipeline not running"}), 503

        sent = pipeline_mgr.send_command({"cmd": "capture", "cat": cat_name})
        if sent:
            return jsonify({"status": "ok", "cat": cat_name})
        else:
            return jsonify({"status": "error", "message": "Failed to send capture command"}), 500

    # ── Auto-Capture API ─────────────────────────────────────────
    @flask_app.route("/api/mode", methods=["POST"])
    def api_mode():
        enabled, target = auto_state.toggle()
        return jsonify({
            "status": "ok",
            "mode": "auto" if enabled else "manual",
            "auto_target": target,
        })

    @flask_app.route("/api/auto_target", methods=["POST"])
    def api_auto_target():
        data = request.get_json()
        cat = data.get("cat", "").strip() if data else ""
        if cat not in cats:
            return jsonify({"status": "error", "message": f"Unknown cat: {cat}"}), 400
        auto_state.set_target(cat)
        return jsonify({"status": "ok", "auto_target": cat})

    # ── Status API ───────────────────────────────────────────────
    @flask_app.route("/api/status")
    def api_status():
        result = {}
        result["pipeline"] = pipeline_mgr.get_status()
        result.update(auto_state.get_state())

        # Per-cat image counts (filesystem)
        data_dir = reviewer.data_dir
        counts = {}
        for cat in cats:
            cat_dir = data_dir / cat
            counts[cat] = len(list(cat_dir.glob("*.jpg"))) if cat_dir.is_dir() else 0
        unlabeled_dir = data_dir / "unlabeled"
        counts["unlabeled"] = len(list(unlabeled_dir.glob("*.jpg"))) if unlabeled_dir.is_dir() else 0
        result["image_counts"] = counts

        # Classifier state
        cfg = config_mgr.read()
        classifier_cfg = cfg.get("classifier", {})
        model_path = resolve_path(
            classifier_cfg.get("model_path", "models/cat_classifier.pte"),
            PROJECT_ROOT,
        )
        result["classifier"] = {
            "enabled": classifier_cfg.get("enabled", False),
            "model_exists": model_path.is_file(),
        }

        # Review state
        result["review"] = {
            "unreviewed_count": reviewer.unreviewed_count(),
            "total": reviewer.total(),
        }

        return jsonify(result)

    # ── Review API ────────────────────────────────────────────────
    @flask_app.route("/api/review/current")
    def api_review_current():
        current = reviewer.get_current()
        if current is None:
            return jsonify({"done": True, "stats": reviewer.get_stats()})
        return jsonify({"done": False, **current})

    @flask_app.route("/api/review/image")
    def api_review_image():
        current = reviewer.get_current()
        if current is None:
            return "No image", 404
        return send_file(current["path"], mimetype="image/jpeg")

    @flask_app.route("/api/review/label", methods=["POST"])
    def api_review_label():
        data = request.get_json()
        label = data.get("label", "").strip()
        if not label:
            return jsonify({"status": "error", "message": "No label"}), 400
        result = reviewer.assign_label(label)
        result["stats"] = reviewer.get_stats()
        result["current"] = reviewer.get_current()
        return jsonify(result)

    @flask_app.route("/api/review/delete", methods=["POST"])
    def api_review_delete():
        result = reviewer.delete_current()
        result["stats"] = reviewer.get_stats()
        result["current"] = reviewer.get_current()
        return jsonify(result)

    @flask_app.route("/api/review/skip", methods=["POST"])
    def api_review_skip():
        result = reviewer.skip()
        result["stats"] = reviewer.get_stats()
        result["current"] = reviewer.get_current()
        return jsonify(result)

    @flask_app.route("/api/review/prev", methods=["POST"])
    def api_review_prev():
        result = reviewer.prev()
        result["stats"] = reviewer.get_stats()
        result["current"] = reviewer.get_current()
        return jsonify(result)

    @flask_app.route("/api/review/rescan", methods=["POST"])
    def api_review_rescan():
        reviewer._scan_images()
        reviewer.index = 0
        return jsonify({"status": "ok", "total": reviewer.total()})

    # ── Review Thumbnails API ─────────────────────────────────────
    @flask_app.route("/api/review/thumbnails")
    def api_review_thumbnails():
        all_images = reviewer.get_all_images()
        return jsonify({
            "images": [
                {
                    "index": i,
                    "filename": img["filename"],
                    "label": img["label"],
                    "relative_path": img["relative_path"],
                    "reviewed": img["reviewed"],
                }
                for i, img in enumerate(all_images)
            ],
            "current_index": reviewer.index,
        })

    @flask_app.route("/api/review/thumbnail/<path:relative_path>")
    def api_review_thumbnail(relative_path):
        thumb_path = reviewer.get_or_create_thumbnail(relative_path)
        if thumb_path is None:
            return "Not found", 404
        return send_file(str(thumb_path), mimetype="image/jpeg")

    @flask_app.route("/api/review/goto", methods=["POST"])
    def api_review_goto():
        data = request.get_json()
        index = data.get("index", 0)
        result = reviewer.goto(index)
        result["stats"] = reviewer.get_stats()
        result["current"] = reviewer.get_current()
        return jsonify(result)

    @flask_app.route("/api/review/batch-label", methods=["POST"])
    def api_review_batch_label():
        data = request.get_json()
        indices = data.get("indices", [])
        label = data.get("label", "").strip()
        if not label or not indices:
            return jsonify({"status": "error", "message": "Missing indices or label"}), 400
        result = reviewer.batch_label(indices, label)
        result["stats"] = reviewer.get_stats()
        result["current"] = reviewer.get_current()
        return jsonify(result)

    @flask_app.route("/api/review/batch-delete", methods=["POST"])
    def api_review_batch_delete():
        data = request.get_json()
        indices = data.get("indices", [])
        if not indices:
            return jsonify({"status": "error", "message": "No indices provided"}), 400
        result = reviewer.batch_delete(indices)
        result["stats"] = reviewer.get_stats()
        result["current"] = reviewer.get_current()
        return jsonify(result)

    # ── Detection Pipeline API ─────────────────────────────────────
    @flask_app.route("/api/inference/status")
    def api_inference_status():
        return jsonify(pipeline_mgr.get_status())

    @flask_app.route("/api/pipeline/restart", methods=["POST"])
    def api_pipeline_restart():
        return jsonify(pipeline_mgr.restart())

    @flask_app.route("/api/inference/settings", methods=["GET"])
    def api_inference_settings_get():
        return jsonify(config_mgr.get_inference_settings())

    @flask_app.route("/api/inference/settings", methods=["POST"])
    def api_inference_settings_post():
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400
        config_mgr.update_inference_settings(data)
        # Push detection capture changes to running detect.py process
        if "detection_capture" in data:
            dc = data["detection_capture"]
            pipeline_mgr.send_command({
                "cmd": "set_detection_capture",
                "enabled": dc.get("enabled", False),
                "min_interval_seconds": dc.get("min_interval_seconds", 15),
                "save_labeled_only": dc.get("save_labeled_only", False),
            })
        # Push bowl_roi changes live to classifier gate
        if "classifier" in data and "bowl_roi" in data["classifier"]:
            pipeline_mgr.send_command({
                "cmd": "set_bowl_roi",
                "bowl_roi": data["classifier"]["bowl_roi"],
            })
        # Push ROI overlap threshold live
        if "classifier" in data and "roi_overlap_threshold" in data["classifier"]:
            pipeline_mgr.send_command({
                "cmd": "set_roi_overlap_threshold",
                "roi_overlap_threshold": data["classifier"]["roi_overlap_threshold"],
            })
        return jsonify({"status": "ok"})

    # ── Classifier ─────────────────────────────────────────────────
    @flask_app.route("/api/classifier/current")
    def api_classifier_current():
        """Return live classifier status from the status file."""
        status_path = CLASSIFIER_STATUS_JSON
        if not status_path.is_file():
            return jsonify({"prediction": None})
        try:
            with open(status_path) as f:
                return jsonify(json.load(f))
        except (json.JSONDecodeError, OSError):
            return jsonify({"prediction": None})

    @flask_app.route("/api/classifier/events")
    def api_classifier_events():
        start = request.args.get("start")  # UTC ISO string
        end = request.args.get("end")      # UTC ISO string
        if start and start in _events_cache:
            entry = _events_cache[start]
            if time.monotonic() < entry["expires"]:
                return jsonify({"events": entry["data"]})
        events = _refresh_events_cache(start=start, end=end)
        return jsonify({"events": events})

    # ── Logs ───────────────────────────────────────────────────────
    @flask_app.route("/api/logs")
    def api_logs():
        return jsonify({"logs": pipeline_mgr.get_log_files()})

    @flask_app.route("/api/logs/<filename>")
    def api_log_file(filename):
        log_path = pipeline_mgr._log_dir / filename
        if not log_path.exists() or not log_path.is_file():
            return "Not found", 404
        # Prevent path traversal
        try:
            log_path.resolve().relative_to(pipeline_mgr._log_dir.resolve())
        except ValueError:
            return "Forbidden", 403
        return send_file(str(log_path), mimetype="text/plain")

    # ── Backups ────────────────────────────────────────────────────
    @flask_app.route("/api/backup/config", methods=["GET"])
    def api_backup_config_get():
        return jsonify(backup_mgr.describe())

    @flask_app.route("/api/backup/config", methods=["POST"])
    def api_backup_config_post():
        try:
            return jsonify({"status": "ok", **backup_mgr.save(request.get_json() or {})})
        except BackupError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

    @flask_app.route("/api/backup/test", methods=["POST"])
    def api_backup_test():
        """Test the settings currently in the form (saved or not)."""
        try:
            return jsonify(backup_mgr.test(request.get_json() or {}))
        except BackupError as exc:
            return jsonify({"ok": False, "message": str(exc)})
        except Exception as exc:
            return jsonify({"ok": False, "message": f"{type(exc).__name__}: {exc}"})

    @flask_app.route("/api/backup/run", methods=["POST"])
    def api_backup_run():
        res = backup_mgr.run_now()
        return jsonify(res), (200 if res.get("status") == "ok" else 409)

    @flask_app.route("/api/backup/status")
    def api_backup_status():
        return jsonify(backup_mgr.status())

    @flask_app.route("/api/backup/snapshots")
    def api_backup_snapshots():
        try:
            return jsonify({"status": "ok", "snapshots": backup_mgr.snapshots()})
        except BackupError as exc:
            return jsonify({"status": "error", "message": str(exc)})
        except Exception as exc:
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"})

    # ── Training (remote backends) ─────────────────────────────────
    @flask_app.route("/api/training/config", methods=["GET"])
    def api_training_config_get():
        return jsonify(training_mgr.describe())

    @flask_app.route("/api/training/config", methods=["POST"])
    def api_training_config_post():
        data = request.get_json() or {}
        backend = data.get("backend", "")
        try:
            settings, new_secrets = training_mgr.parse_form(backend, data.get("values") or {})
            min_images_val = data.get("min_images_per_cat")
            min_images_val = int(min_images_val) if min_images_val not in (None, "") else None
        except (BackendError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        config_mgr.update_training_backend(backend, settings, min_images_val)
        if new_secrets:
            secrets.update_section(backend, new_secrets)
        return jsonify({"status": "ok", **training_mgr.describe()})

    @flask_app.route("/api/training/test", methods=["POST"])
    def api_training_test():
        """Test the settings currently in the form (saved or not)."""
        data = request.get_json() or {}
        backend = data.get("backend", "")
        try:
            settings, form_secrets = training_mgr.parse_form(backend, data.get("values") or {})
            result = training_mgr.make_backend(backend, settings, form_secrets).test_connection()
        except BackendError as exc:
            return jsonify({"ok": False, "message": str(exc)})
        except Exception as exc:
            return jsonify({"ok": False, "message": f"{type(exc).__name__}: {exc}"})
        return jsonify(result.to_dict())

    @flask_app.route("/api/training/images")
    def api_training_images():
        return jsonify({
            "counts": training_mgr.image_counts(),
            "min_images_per_cat": training_mgr.settings()["min_images_per_cat"],
        })

    @flask_app.route("/api/training/status")
    def api_training_status():
        return jsonify(training_mgr.status())

    @flask_app.route("/api/training/jobs")
    def api_training_jobs():
        return jsonify({"jobs": training_mgr.list_jobs()})

    @flask_app.route("/api/training/start", methods=["POST"])
    def api_training_start():
        data = request.get_json(silent=True) or {}
        folds = data.get("cv_folds")
        try:
            folds = int(folds) if folds not in (None, "") else None
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Folds must be a number"}), 400
        res = training_mgr.start(kind=data.get("kind", "train"), cv_folds=folds)
        return jsonify(res), (200 if res["status"] == "ok" else 409)

    @flask_app.route("/api/training/settings", methods=["GET"])
    def api_training_settings_get():
        return jsonify({"fields": training_settings.SETTINGS,
                        "values": training_settings.current_values(config_mgr.read())})

    @flask_app.route("/api/training/settings", methods=["POST"])
    def api_training_settings_post():
        try:
            values = training_settings.parse((request.get_json() or {}).get("values") or {})
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        config_mgr.update_training_settings(values)
        return jsonify({"status": "ok",
                        "values": training_settings.current_values(config_mgr.read())})

    @flask_app.route("/api/training/metrics/<job_id>")
    def api_training_metrics(job_id):
        return jsonify(training_mgr.job_metrics(job_id))

    @flask_app.route("/api/training/job/<job_id>")
    def api_training_job(job_id):
        job = training_mgr.job_state(job_id)
        if job is None:
            return jsonify({"status": "error", "message": "Unknown job"}), 404
        return jsonify(job)

    # ── Annotate (bounding boxes in data/raw sidecars) ─────────────
    raw_dir = training_mgr.raw_dir()
    ann_folders = [*cats, "unlabeled"]

    def _ann_path(folder: str, filename: str) -> Path | None:
        if folder not in ann_folders:
            return None
        if not filename or Path(filename).name != filename \
                or not filename.lower().endswith((".jpg", ".jpeg")):
            return None
        return raw_dir / folder / filename

    def _bbox_json(meta: dict | None) -> dict | None:
        best = sidecars.best_detection(meta)
        if best is None:
            return None
        return {**sidecars.to_center(best["bbox_norm"]),
                "confidence": best.get("confidence"),
                "source": best.get("source", "pi")}

    @flask_app.route("/api/annotate/images")
    def api_annotate_images():
        images = []
        for folder in ann_folders:
            d = raw_dir / folder
            if not d.is_dir():
                continue
            for img in sorted(d.glob("*.jpg")):
                bbox = _bbox_json(sidecars.read_sidecar(img))
                images.append({"cat": folder, "filename": img.name,
                               "has_bbox": bbox is not None, "bbox": bbox})
        backend = training_mgr.settings()["backend"]
        return jsonify({
            "images": images,
            "instant_detect": training_mgr.instant_detect_available(),
            "backend": backend,
            "job_active": training_mgr.is_active(),
        })

    @flask_app.route("/api/annotate/thumb/<folder>/<filename>")
    def api_annotate_thumb(folder, filename):
        if _ann_path(folder, filename) is None:
            return "Not found", 404
        thumb = reviewer.get_or_create_thumbnail(f"{folder}/{filename}")
        if thumb is None:
            return "Not found", 404
        return send_file(str(thumb), mimetype="image/jpeg", max_age=86400)

    @flask_app.route("/api/annotate/image/<folder>/<filename>")
    def api_annotate_image(folder, filename):
        img = _ann_path(folder, filename)
        if img is None or not img.is_file():
            return "Not found", 404
        return send_file(str(img), mimetype="image/jpeg")

    @flask_app.route("/api/annotate/bbox/<folder>/<filename>", methods=["GET"])
    def api_annotate_bbox_get(folder, filename):
        img = _ann_path(folder, filename)
        bbox = _bbox_json(sidecars.read_sidecar(img)) if img else None
        return jsonify({"exists": bbox is not None, **(bbox or {})})

    @flask_app.route("/api/annotate/bbox/<folder>/<filename>", methods=["POST"])
    def api_annotate_bbox_save(folder, filename):
        img = _ann_path(folder, filename)
        if img is None or not img.is_file():
            return jsonify({"status": "error", "message": "Image not found"}), 404
        data = request.get_json() or {}
        try:
            cx, cy, w, h = (float(data[k]) for k in ("cx", "cy", "w", "h"))
        except (KeyError, TypeError, ValueError):
            return jsonify({"status": "error", "message": "cx, cy, w, h required"}), 400
        bbox = sidecars.from_center(cx, cy, w, h)
        if bbox is None:
            return jsonify({"status": "error", "message": "Empty bounding box"}), 400
        auto = data.get("source") == "auto-detect"
        sidecars.set_bbox(img, bbox, source="auto-detect" if auto else "manual",
                          confidence=float(data.get("confidence", 1.0)) if auto else 1.0)
        return jsonify({"status": "ok"})

    @flask_app.route("/api/annotate/bbox/<folder>/<filename>", methods=["DELETE"])
    def api_annotate_bbox_delete(folder, filename):
        img = _ann_path(folder, filename)
        if img is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400
        return jsonify({"status": "ok", "result": sidecars.clear_bbox(img)})

    @flask_app.route("/api/annotate/auto-detect", methods=["POST"])
    def api_annotate_auto_detect():
        data = request.get_json() or {}
        img = _ann_path(data.get("cat", ""), data.get("filename", ""))
        if img is None or not img.is_file():
            return jsonify({"status": "error", "message": "Image not found"}), 404
        try:
            bbox = training_mgr.instant_detect(img)
        except BackendError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 409
        if bbox:
            return jsonify({"status": "ok", "detected": True, **bbox})
        return jsonify({"status": "ok", "detected": False})

    @flask_app.route("/api/training/cancel", methods=["POST"])
    def api_training_cancel():
        return jsonify(training_mgr.cancel())

    @flask_app.route("/api/training/deploy", methods=["POST"])
    def api_training_deploy():
        job_id = (request.get_json() or {}).get("job_id", "")
        res = training_mgr.deploy(job_id)
        return jsonify(res), (200 if res["status"] == "ok" else 409)

    @flask_app.route("/api/training/rollback", methods=["POST"])
    def api_training_rollback():
        res = training_mgr.rollback()
        return jsonify(res), (200 if res["status"] == "ok" else 409)

    @flask_app.route("/api/training/log/<job_id>")
    def api_training_log(job_id):
        log_path = training_mgr.log_dir / f"{job_id}.log"
        try:
            log_path.resolve().relative_to(training_mgr.log_dir.resolve())
        except ValueError:
            return "Forbidden", 403
        if not log_path.is_file():
            return "Not found", 404
        return send_file(str(log_path), mimetype="text/plain")

    return flask_app


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cat collection dashboard — capture, review, and inference"
    )
    add_common_args(parser)
    parser.add_argument("--host", default=None,
                        help="Web UI host (default: from config)")
    parser.add_argument("--port", type=int, default=None,
                        help="Web UI port (default: from config)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Skip detection pipeline (review-only mode)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load base config from project.yaml, then merge runtime.yaml overrides
    yaml_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "project.yaml"
    config_mgr = ConfigManager(yaml_path)
    config = config_mgr.read()

    cats: list[str] = config["cats"]
    web_cfg = config["web_ui"]
    collection_cfg = config["collection"]
    review_cfg = config.get("label_review", {})

    host = args.host or web_cfg["host"]
    port = args.port or web_cfg["port"]
    data_dir = resolve_path(collection_cfg["output_dir"], PROJECT_ROOT)
    trash_dir = resolve_path(review_cfg.get("trash_dir", "data/trash"), PROJECT_ROOT)
    min_images = collection_cfg.get("min_images_per_cat", 300)

    print_header("Cat Collection Dashboard")
    print_info(f"Cats: {', '.join(cats)}")

    # Label reviewer (with review state tracking and thumbnail cache)
    reviewer = LabelReviewApp(
        data_dir=data_dir,
        trash_dir=trash_dir,
        cats=cats,
        scope="all",
    )

    # Detection pipeline manager (replaces InferenceManager + CameraThread)
    config_path = args.config if args.config else None
    deploy_cfg = config.get("deployment", {})

    # Resolve hailo-apps directory (expand ~ and resolve relative to project root)
    hailo_apps_dir_str = deploy_cfg.get("hailo_apps_dir", "./hailo-apps")
    hailo_apps_dir = Path(os.path.expanduser(hailo_apps_dir_str))
    if not hailo_apps_dir.is_absolute():
        hailo_apps_dir = PROJECT_ROOT / hailo_apps_dir
    if not (hailo_apps_dir / "setup_env.sh").exists():
        print_warning(f"hailo-apps setup_env.sh not found at {hailo_apps_dir}/setup_env.sh")
        print_warning("Detection pipeline may fail. Install hailo-apps or set deployment.hailo_apps_dir in project.yaml")
        hailo_apps_dir = None  # Will fall back to sys.executable

    pipeline_mgr = DetectionPipelineManager(PROJECT_ROOT, config_path, hailo_apps_dir)
    pipeline_mgr.set_config(config)

    # Remote training jobs (local PC / cloud backends); resumes a running job
    secrets = SecretsStore(yaml_path.parent / "secrets.yaml")

    def _on_model_installed() -> str | None:
        # The pipeline's model watcher restarts detection when the model file
        # changes, but it only runs if the classifier was enabled at startup.
        # Without it, restart here so the new model is loaded.
        watcher = pipeline_mgr._model_watcher_thread
        if watcher and watcher.is_alive():
            return None
        if not pipeline_mgr.is_running():
            return "Model installed. Detection is not running; it loads on the next start."
        threading.Thread(target=pipeline_mgr.restart, daemon=True).start()
        return "Restarting detection with the new model."

    training_mgr = TrainingJobManager(PROJECT_ROOT, yaml_path, config_mgr.read, secrets,
                                      on_model_installed=_on_model_installed)
    backup_mgr = BackupManager(PROJECT_ROOT, yaml_path, config_mgr.read,
                               config_mgr.update_backup, secrets, sys.executable,
                               command_prefix=_low_priority_prefix)
    backup_mgr.start()

    # Auto-capture state + timer
    auto_interval = collection_cfg.get("auto_capture_interval_sec", 3.0)
    auto_state = AutoCaptureState(cats, interval=auto_interval)
    auto_thread = threading.Thread(
        target=_auto_capture_loop, args=(auto_state, pipeline_mgr), daemon=True
    )
    auto_thread.start()

    # Start detection pipeline (unless --no-camera)
    if not args.no_camera:
        result = pipeline_mgr.start()
        if result["status"] == "ok":
            print_success(f"Detection pipeline started (COCO mode), PID: {result['pid']}")
        else:
            print_warning(f"Failed to start detection pipeline: {result.get('message', '?')}")
            print_warning("Running in review-only mode")

    # Graceful shutdown
    def shutdown(*_args):
        print_info("\nShutting down...")
        pipeline_mgr.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print_info(f"Review images: {reviewer.total()} ({reviewer.unreviewed_count()} unreviewed)")
    print_success(f"Dashboard: http://{host}:{port}")

    # Warm classifier events cache so first dashboard load is instant
    today_local = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_local.astimezone(timezone.utc).isoformat()
    tomorrow_start_utc = (today_local + timedelta(days=1)).astimezone(timezone.utc).isoformat()
    _refresh_events_cache(
        resolve_path(
            config.get("classifier", {}).get("db_path", "data/classifier_log.db"),
            PROJECT_ROOT,
        ),
        start=today_start_utc,
        end=tomorrow_start_utc,
    )

    flask_app = create_app(
        reviewer=reviewer,
        pipeline_mgr=pipeline_mgr,
        config_mgr=config_mgr,
        auto_state=auto_state,
        cats=cats,
        min_images=min_images,
        training_mgr=training_mgr,
        secrets=secrets,
        backup_mgr=backup_mgr,
    )
    flask_app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
