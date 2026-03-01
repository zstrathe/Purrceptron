#!/usr/bin/env python3
"""Unified collection dashboard — capture, label review, and inference control.

Single-page tabbed web UI for the full data collection workflow:
- Capture tab: live camera feed + manual capture, inference controls
- Review tab: iterate images, assign labels, delete bad captures

Usage:
    python dashboard.py                   # reads configs/project.yaml
    python dashboard.py --config alt.yaml
"""
from __future__ import annotations

import argparse
import collections
import glob as globmod
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, load_config, resolve_path
from utils.display import print_info, print_success, print_error, print_warning, print_header

from collect.capture import CameraThread, SaveWorker, CaptureApp
from collect.label_review import LabelReviewApp

import yaml


# ═══════════════════════════════════════════════════════════════════════════
# Inference Manager
# ═══════════════════════════════════════════════════════════════════════════

class InferenceManager:
    """Start/stop detect.py as a managed subprocess."""

    def __init__(self, project_root: Path, config_path: Path | None = None):
        self.project_root = project_root
        self.config_path = config_path
        self.process: subprocess.Popen | None = None
        self._output_lines: collections.deque[str] = collections.deque(maxlen=100)
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()

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

    def start(self) -> dict:
        """Start detect.py subprocess."""
        if self.is_running():
            return {"status": "error", "message": "Already running"}

        cmd = [sys.executable, "deploy/detect.py"]
        if self.config_path:
            cmd += ["--config", str(self.config_path)]

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
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

        return {"status": "ok", "pid": self.process.pid}

    def stop(self) -> dict:
        """Stop the detect.py subprocess."""
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                self.process = None
                return {"status": "ok", "message": "Not running"}

            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

            self.process = None
            return {"status": "ok"}

    def get_output(self) -> list[str]:
        with self._lock:
            return list(self._output_lines)

    def get_status(self) -> dict:
        return {
            "running": self.is_running(),
            "pid": self.process.pid if self.process and self.is_running() else None,
            "output": self.get_output(),
            "models": self.scan_models(),
        }

    def _read_output(self) -> None:
        """Background thread: read subprocess stdout line by line."""
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
                            "max_captures_per_session"):
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
# Auto-capture ticker
# ═══════════════════════════════════════════════════════════════════════════

def _auto_capture_loop(capture_app: CaptureApp) -> None:
    """Run in a daemon thread: periodically trigger auto-captures."""
    while True:
        try:
            capture_app.auto_capture_tick()
        except Exception:
            pass
        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Flask application factory
# ═══════════════════════════════════════════════════════════════════════════

def create_app(
    capture_app: CaptureApp | None,
    reviewer: LabelReviewApp,
    inference_mgr: InferenceManager,
    config_mgr: ConfigManager,
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
            camera_available=capture_app is not None,
        )

    # ── Capture API ───────────────────────────────────────────────
    @flask_app.route("/video_feed")
    def video_feed():
        if capture_app is None or capture_app.camera is None:
            return "No camera", 503

        def generate():
            while True:
                frame = capture_app.camera.get_display_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue
                ret, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if not ret:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + jpeg.tobytes() + b"\r\n")

        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @flask_app.route("/api/capture/<cat_name>", methods=["POST"])
    def api_capture(cat_name):
        if capture_app is None:
            return jsonify({"status": "error", "message": "No camera"}), 503
        if inference_mgr.is_running():
            return jsonify({"status": "error", "message": "Camera in use by inference"}), 409
        return jsonify(capture_app.capture(cat_name))

    @flask_app.route("/api/mode", methods=["POST"])
    def api_mode():
        if capture_app is None:
            return jsonify({"status": "error", "message": "No camera"}), 503
        return jsonify(capture_app.toggle_mode())

    @flask_app.route("/api/auto_target", methods=["POST"])
    def api_auto_target():
        if capture_app is None:
            return jsonify({"status": "error", "message": "No camera"}), 503
        data = request.get_json(silent=True) or {}
        return jsonify(capture_app.set_auto_target(data.get("cat", "")))

    @flask_app.route("/api/status")
    def api_status():
        result = {}
        if capture_app:
            result.update(capture_app.get_status())
        result["inference"] = inference_mgr.get_status()

        # Per-cat image counts (filesystem)
        data_dir = reviewer.data_dir
        counts = {}
        for cat in cats:
            cat_dir = data_dir / cat
            counts[cat] = len(list(cat_dir.glob("*.jpg"))) if cat_dir.is_dir() else 0
        unlabeled_dir = data_dir / "unlabeled"
        counts["unlabeled"] = len(list(unlabeled_dir.glob("*.jpg"))) if unlabeled_dir.is_dir() else 0
        result["image_counts"] = counts

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

    # ── Inference API ─────────────────────────────────────────────
    @flask_app.route("/api/inference/status")
    def api_inference_status():
        return jsonify(inference_mgr.get_status())

    @flask_app.route("/api/inference/models")
    def api_inference_models():
        return jsonify({"models": inference_mgr.scan_models()})

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

    @flask_app.route("/api/inference/start", methods=["POST"])
    def api_inference_start():
        # Stop camera if running (inference needs exclusive access)
        if capture_app and capture_app.camera:
            capture_app.camera.stop()

        result = inference_mgr.start()
        return jsonify(result)

    @flask_app.route("/api/inference/stop", methods=["POST"])
    def api_inference_stop():
        result = inference_mgr.stop()

        # Restart camera for manual capture
        if capture_app and capture_app.camera:
            capture_app.camera.start()

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
                        help="Skip camera initialization (review-only mode)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config_with_args(args)

    cats: list[str] = config["cats"]
    cam_cfg = config["camera"]
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

    # Camera (optional)
    capture_app = None
    if not args.no_camera:
        try:
            camera = CameraThread(
                device=cam_cfg["device"],
                width=cam_cfg["width"],
                height=cam_cfg["height"],
                fps=cam_cfg["fps"],
                fourcc=cam_cfg["format"],
                preview_w=web_cfg["preview_width"],
                preview_h=web_cfg["preview_height"],
            )
            camera.start()
            time.sleep(1)
            if camera.get_frame() is not None:
                saver = SaveWorker()
                saver.start()
                capture_app = CaptureApp(config, camera, saver)
                print_success(f"Camera: {cam_cfg['device']}")
            else:
                camera.stop()
                print_warning("Camera not available — capture disabled, review-only mode")
        except Exception as exc:
            print_warning(f"Camera init failed: {exc} — review-only mode")

    # Label reviewer
    reviewer = LabelReviewApp(
        data_dir=data_dir,
        trash_dir=trash_dir,
        cats=cats,
        scope="all",
    )

    # Inference manager
    config_path = args.config if args.config else None
    inference_mgr = InferenceManager(PROJECT_ROOT, config_path)

    # Config manager
    yaml_path = args.config if args.config else PROJECT_ROOT / "configs" / "project.yaml"
    config_mgr = ConfigManager(yaml_path)

    # Auto-capture ticker (only if camera available)
    if capture_app:
        ticker = threading.Thread(
            target=_auto_capture_loop, args=(capture_app,), daemon=True
        )
        ticker.start()

    # Graceful shutdown
    def shutdown(*_args):
        print_info("\nShutting down...")
        inference_mgr.stop()
        if capture_app:
            capture_app.camera.stop()
            capture_app.saver.drain()
            capture_app.print_summary()
            capture_app.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    models = inference_mgr.scan_models()
    print_info(f"Models found: {len(models)}" + (f" ({', '.join(models)})" if models else ""))
    print_info(f"Review images: {reviewer.total()}")
    print_success(f"Dashboard: http://{host}:{port}")

    flask_app = create_app(
        capture_app=capture_app,
        reviewer=reviewer,
        inference_mgr=inference_mgr,
        config_mgr=config_mgr,
        cats=cats,
        min_images=min_images,
    )
    flask_app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
