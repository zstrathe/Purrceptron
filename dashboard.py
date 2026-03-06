#!/usr/bin/env python3
"""Unified collection dashboard — capture, label review, and inference control.

Single-page tabbed web UI for the full data collection workflow:
- Capture tab: live detection feed with bounding boxes, mode switching, manual capture
- Review tab: iterate images, assign labels, delete bad captures, thumbnail navigation

Usage:
    python dashboard.py                   # reads configs/project.yaml
    python dashboard.py --config alt.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, load_config, resolve_path
from utils.display import print_info, print_success, print_error, print_warning, print_header

from collect.label_review import LabelReviewApp

import yaml


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
    """Manages the always-running detect.py subprocess.

    Two modes:
    - "default": COCO yolov8m HEF, filter_classes=["cat"], labels=coco_labels.json
    - "inference": fine-tuned cat_detector.hef, filter_classes=[], labels=labels.json
    """

    PREVIEW_PATH = "/tmp/cat_detection_preview.jpg"

    def __init__(self, project_root: Path, config_path: Path | None = None,
                 hailo_apps_dir: Path | None = None):
        self.project_root = project_root
        self.config_path = config_path
        self.hailo_apps_dir = hailo_apps_dir
        self.process: subprocess.Popen | None = None
        self._output_lines: collections.deque[str] = collections.deque(maxlen=100)
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.mode: str = "default"  # "default" or "inference"
        self._config: dict | None = None
        # Original inference paths — preserved before mode switching can overwrite them
        self._inference_hef = "models/cat_detector.hef"
        self._inference_labels = "deploy/labels.json"

    def set_config(self, config: dict) -> None:
        self._config = config
        # Preserve original inference paths before mode switching can contaminate them
        deploy = config.get("deployment", {})
        self._inference_hef = deploy.get("hef_path") or "models/cat_detector.hef"
        self._inference_labels = deploy.get("labels_json") or "deploy/labels.json"

    def scan_models(self) -> list[str]:
        """Scan models/ for .hef files, return relative paths."""
        models_dir = self.project_root / "models"
        if not models_dir.is_dir():
            return []
        return sorted(
            str(p.relative_to(self.project_root))
            for p in models_dir.glob("*.hef")
        )

    def is_running(self) -> bool:
        with self._lock:
            if self.process is None:
                return False
            return self.process.poll() is None

    def _write_mode_config(self, mode: str) -> None:
        """Update project.yaml with model paths for the given mode."""
        if not self._config:
            return
        yaml_path = self.config_path or (self.project_root / "configs" / "project.yaml")
        with open(yaml_path) as f:
            yaml_cfg = yaml.safe_load(f)
        if mode == "default":
            yaml_cfg["deployment"]["hef_path"] = ""
            yaml_cfg["deployment"]["labels_json"] = ""
            yaml_cfg["deployment"]["filter_classes"] = []
        else:  # "inference"
            # hef_path and labels_json already set by saveInferenceSettings
            yaml_cfg["deployment"]["filter_classes"] = []
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)

    def start(self, mode: str = "default") -> dict:
        """Start detect.py subprocess in the given mode.

        Launches detect.py inside the hailo-apps environment by sourcing
        setup_env.sh first, which activates the hailo-apps venv and sets
        GStreamer plugin paths.
        """
        if self.is_running():
            return {"status": "error", "message": "Already running"}

        self.mode = mode
        self._write_mode_config(mode)

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
        self._reader_thread = threading.Thread(
            target=self._read_output, daemon=True
        )
        self._reader_thread.start()

        return {"status": "ok", "pid": self.process.pid, "mode": mode}

    def stop(self) -> dict:
        """Stop the detect.py subprocess."""
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
            return {"status": "ok"}

    def switch_mode(self, new_mode: str) -> dict:
        """Stop detect.py, update config for new mode, restart."""
        self.stop()
        # Brief pause for camera release
        time.sleep(1)
        return self.start(mode=new_mode)

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
        return {
            "running": self.is_running(),
            "mode": self.mode,
            "pid": self.process.pid if self.process and self.is_running() else None,
            "output": self.get_output(),
            "models": self.scan_models(),
        }

    def _read_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        try:
            for line in self.process.stdout:
                with self._lock:
                    self._output_lines.append(line.rstrip())
        except (ValueError, OSError):
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Config Manager — read/write project.yaml for inference settings
# ═══════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """Read/write inference-related settings in project.yaml."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._lock = threading.Lock()

    def read(self) -> dict:
        with self._lock:
            with open(self.config_path) as f:
                return yaml.safe_load(f)

    def get_inference_settings(self) -> dict:
        """Return the subset of config relevant to inference UI controls."""
        cfg = self.read()
        return {
            "deployment": cfg.get("deployment", {}),
            "detection_capture": cfg.get("detection_capture", {}),
            "feeding": cfg.get("feeding", {}),
        }

    def update_inference_settings(self, settings: dict) -> None:
        """Update inference-related fields in project.yaml."""
        with self._lock:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)

            # Update deployment section
            if "deployment" in settings:
                for key in ("hef_path", "labels_json", "track_all", "filter_classes"):
                    if key in settings["deployment"]:
                        cfg["deployment"][key] = settings["deployment"][key]

            # Update detection_capture section
            if "detection_capture" in settings:
                for key in ("enabled", "unlabeled_classes", "min_interval_seconds",
                            "max_captures_per_session", "filter_labels"):
                    if key in settings["detection_capture"]:
                        cfg["detection_capture"][key] = settings["detection_capture"][key]

            # Update feeding section
            if "feeding" in settings:
                for key in ("enabled", "cooldown_seconds", "min_detection_frames"):
                    if key in settings["feeding"]:
                        cfg["feeding"][key] = settings["feeding"][key]

            with open(self.config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════════
# Flask application factory
# ═══════════════════════════════════════════════════════════════════════════

def create_app(
    reviewer: LabelReviewApp,
    pipeline_mgr: DetectionPipelineManager,
    config_mgr: ConfigManager,
    auto_state: AutoCaptureState,
    cats: list[str],
    min_images: int,
) -> Flask:
    template_dir = Path(__file__).resolve().parent / "templates"
    flask_app = Flask(__name__, template_folder=str(template_dir))

    # ── Page ──────────────────────────────────────────────────────
    @flask_app.route("/")
    def index():
        cats_indexed = list(enumerate(cats, start=1))
        return render_template(
            "dashboard.html",
            cats=cats_indexed,
            cat_names=cats,
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
        det_path = DetectionPipelineManager.PREVIEW_PATH.replace(".jpg", "_detections.json")
        try:
            with open(det_path) as f:
                return Response(f.read(), mimetype="application/json",
                                headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})
        except (FileNotFoundError, OSError):
            return Response("[]", mimetype="application/json")

    @flask_app.route("/api/ignore_labels", methods=["POST"])
    def api_ignore_labels():
        """Send ignore labels list to detect.py to hide from video overlay."""
        labels = request.json.get("labels", [])
        pipeline_mgr.send_command({"cmd": "set_ignore_labels", "labels": labels})
        return jsonify({"status": "ok"})

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

    # ── Inference API (mode switching) ────────────────────────────
    @flask_app.route("/api/inference/status")
    def api_inference_status():
        return jsonify(pipeline_mgr.get_status())

    @flask_app.route("/api/inference/models")
    def api_inference_models():
        return jsonify({"models": pipeline_mgr.scan_models()})

    @flask_app.route("/api/inference/settings", methods=["GET"])
    def api_inference_settings_get():
        return jsonify(config_mgr.get_inference_settings())

    @flask_app.route("/api/inference/settings", methods=["POST"])
    def api_inference_settings_post():
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400
        config_mgr.update_inference_settings(data)
        return jsonify({"status": "ok"})

    @flask_app.route("/api/inference/labels")
    def api_inference_labels():
        """Return the labels list from the currently configured labels JSON."""
        cfg = config_mgr.read()
        labels_path = cfg.get("deployment", {}).get("labels_json", "")
        full_path = project_root / labels_path
        if full_path.exists():
            with open(full_path) as f:
                data = json.load(f)
            return jsonify({"labels": data.get("labels", [])})
        return jsonify({"labels": []})

    @flask_app.route("/api/inference/start", methods=["POST"])
    def api_inference_start():
        """Switch to inference mode (fine-tuned model)."""
        result = pipeline_mgr.switch_mode("inference")
        return jsonify(result)

    @flask_app.route("/api/inference/stop", methods=["POST"])
    def api_inference_stop():
        """Switch back to default mode (COCO model)."""
        result = pipeline_mgr.switch_mode("default")
        return jsonify(result)

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
    config = get_config_with_args(args)

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

    # Resolve hailo-apps directory (expand ~ to home dir)
    hailo_apps_dir_str = deploy_cfg.get("hailo_apps_dir", "~/hailo-apps")
    hailo_apps_dir = Path(os.path.expanduser(hailo_apps_dir_str))
    if not (hailo_apps_dir / "setup_env.sh").exists():
        print_warning(f"hailo-apps setup_env.sh not found at {hailo_apps_dir}/setup_env.sh")
        print_warning("Detection pipeline may fail. Install hailo-apps or set deployment.hailo_apps_dir in project.yaml")
        hailo_apps_dir = None  # Will fall back to sys.executable

    pipeline_mgr = DetectionPipelineManager(PROJECT_ROOT, config_path, hailo_apps_dir)
    pipeline_mgr.set_config(config)

    # Auto-capture state + timer
    auto_interval = collection_cfg.get("auto_capture_interval_sec", 3.0)
    auto_state = AutoCaptureState(cats, interval=auto_interval)
    auto_thread = threading.Thread(
        target=_auto_capture_loop, args=(auto_state, pipeline_mgr), daemon=True
    )
    auto_thread.start()

    # Config manager
    yaml_path = args.config if args.config else PROJECT_ROOT / "configs" / "project.yaml"
    config_mgr = ConfigManager(yaml_path)

    # Start detection pipeline in default mode (unless --no-camera)
    if not args.no_camera:
        result = pipeline_mgr.switch_mode("default")
        if result["status"] == "ok":
            print_success(f"Detection pipeline started (default/COCO mode), PID: {result['pid']}")
        else:
            print_warning(f"Failed to start detection pipeline: {result.get('message', '?')}")
            print_warning("Running in review-only mode")

    # Graceful shutdown
    def shutdown(*_args):
        print_info("\nShutting down...")
        pipeline_mgr.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    models = pipeline_mgr.scan_models()
    print_info(f"Models found: {len(models)}" + (f" ({', '.join(models)})" if models else ""))
    print_info(f"Review images: {reviewer.total()} ({reviewer.unreviewed_count()} unreviewed)")
    print_success(f"Dashboard: http://{host}:{port}")

    flask_app = create_app(
        reviewer=reviewer,
        pipeline_mgr=pipeline_mgr,
        config_mgr=config_mgr,
        auto_state=auto_state,
        cats=cats,
        min_images=min_images,
    )
    flask_app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
