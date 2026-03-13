#!/usr/bin/env python3
"""Cat detection inference on Raspberry Pi 5 + Hailo 10H.

Wraps the Hailo GStreamer detection pipeline with custom class filtering
and tracker configuration for individual cat identification.

Supports dashboard integration via:
- Preview frame output: writes annotated frames to a shared file for the dashboard
- Stdin commands: accepts JSON commands from the dashboard for manual capture

All settings are read from project.yaml. No custom CLI flags required.

Usage:
    python deploy/detect.py                        # reads configs/project.yaml
    python deploy/detect.py --config alt.yaml      # use alternate config
    python deploy/detect.py --input /dev/video2    # override a Hailo arg
    python deploy/detect.py --enable-preview       # write preview frames for dashboard
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import select
import sys
import threading
import time
import json as _json
from datetime import datetime, timezone
from pathlib import Path
from queue import Full as QueueFull

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import PROJECT_ROOT, get_config_with_args, load_config, resolve_path, add_common_args
from utils.display import print_info, print_error, print_success, print_warning

# Hailo imports — only available on the Raspberry Pi
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import hailo
    from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
    from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
    HAILO_AVAILABLE = True
except (ImportError, ValueError) as e:
    HAILO_AVAILABLE = False
    HAILO_IMPORT_ERROR = str(e)


# ═══════════════════════════════════════════════════════════════════════════
# Preview Frame Writer
# ═══════════════════════════════════════════════════════════════════════════

class PreviewFrameWriter:
    """Writes annotated preview frames to a shared file for the dashboard.

    Frame processing (resize, bbox drawing, JPEG encode, file write) runs
    in a background thread so the GStreamer callback is not blocked.
    Uses atomic write (write to .tmp, then os.replace) to prevent
    the dashboard from reading a partially-written file.
    """

    def __init__(self, preview_path: str, preview_fps: int,
                 preview_width: int, preview_height: int,
                 jpeg_quality: int):
        self.preview_path = preview_path
        self.tmp_path = preview_path + ".tmp"
        self.detections_path = preview_path.replace(".jpg", "_detections.json")
        self.target_interval = 1.0 / preview_fps
        self.preview_size = (preview_width, preview_height)
        self.jpeg_quality = jpeg_quality
        self._last_write_time: float = 0.0
        self._queue: queue.Queue[tuple[np.ndarray, list[dict]] | None] = queue.Queue(
            maxsize=4
        )
        self._thread = threading.Thread(target=self._write_loop, daemon=True)

    def start(self) -> None:
        """Start the background writer thread."""
        self._thread.start()

    def should_write(self) -> bool:
        return (time.time() - self._last_write_time) >= self.target_interval

    def enqueue_frame(self, frame: np.ndarray, detections: list[dict] | None = None) -> None:
        """Enqueue a frame for background processing.

        Args:
            frame: BGR numpy array (must be a copy, not a buffer view).
            detections: Optional list of dicts with 'bbox_norm', 'label',
                        'confidence', 'track_id' keys. Bounding boxes are
                        drawn AFTER resizing to preview dimensions.
        """
        self._last_write_time = time.time()
        try:
            self._queue.put_nowait((frame, detections or []))
        except queue.Full:
            pass  # Drop frame if writer can't keep up

    def _write_loop(self) -> None:
        """Background thread: dequeue frames, resize, draw bboxes, encode, write."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame, detections = item
            try:
                resized = cv2.resize(frame, self.preview_size)
                if detections:
                    pw, ph = self.preview_size  # (width, height)
                    pixel_dets = []
                    for det in detections:
                        nx, ny, nw, nh = det["bbox_norm"]
                        pixel_dets.append({
                            "label": det["label"],
                            "confidence": det["confidence"],
                            "bbox": (int(nx * pw), int(ny * ph),
                                     int((nx + nw) * pw), int((ny + nh) * ph)),
                            "track_id": det.get("track_id"),
                        })
                    draw_bounding_boxes(resized, pixel_dets)
                ret, jpeg = cv2.imencode(
                    ".jpg", resized,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if ret:
                    with open(self.tmp_path, "wb") as f:
                        f.write(jpeg.tobytes())
                    os.replace(self.tmp_path, self.preview_path)
                # Write detection metadata sidecar
                det_tmp = self.detections_path + ".tmp"
                with open(det_tmp, "w") as f:
                    json.dump([
                        {"label": d["label"], "confidence": d["confidence"],
                         "track_id": d.get("track_id")}
                        for d in detections
                    ], f)
                os.replace(det_tmp, self.detections_path)
            except Exception as exc:
                print_warning(f"PreviewFrameWriter error: {exc}")
            self._queue.task_done()

    def shutdown(self) -> None:
        """Drain the write queue, stop the thread, and remove preview files."""
        self._queue.join()
        self._queue.put(None)
        self._thread.join(timeout=5)
        for p in (self.preview_path, self.tmp_path,
                  self.detections_path, self.detections_path + ".tmp"):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Stdin Command Listener
# ═══════════════════════════════════════════════════════════════════════════

class StdinCommandListener:
    """Reads JSON commands from stdin in a background thread.

    Commands are placed in a thread-safe queue for the main callback to process.
    Protocol: one JSON object per line (newline-delimited JSON).
    """

    def __init__(self):
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=16)
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running:
            try:
                if select.select([sys.stdin], [], [], 1.0)[0]:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        try:
                            cmd = json.loads(line)
                            self._queue.put_nowait(cmd)
                        except json.JSONDecodeError:
                            print_warning(f"Invalid command: {line}")
            except (OSError, ValueError):
                break

    def get_command(self) -> dict | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


# ═══════════════════════════════════════════════════════════════════════════
# Classifier Gate
# ═══════════════════════════════════════════════════════════════════════════

class ClassifierGate:
    """Manages Stage 2 classifier interaction from the GStreamer callback.

    Filters detections to the bowl ROI, applies a stability gate (N consecutive
    frames), throttles classification to classify_interval seconds, produces
    224×224 crops, and optionally saves them for labeling.
    """

    def __init__(
        self,
        queue,
        bowl_roi: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        stability_frames: int,
        classify_interval: float,
        crop_padding: float,
        save_crops: bool,
        labeling_dir: Path | None,
        roi_overlap_threshold: float = 0.01,
        trigger_labels: set[str] | None = None,
    ):
        self._queue = queue
        self._bowl_roi = bowl_roi          # (x_min, y_min, x_max, y_max) in pixels
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._stability_frames = stability_frames
        self._classify_interval = classify_interval
        self._crop_padding = crop_padding
        self._save_crops = save_crops
        self._labeling_dir = labeling_dir
        self._roi_overlap_threshold = roi_overlap_threshold
        self._trigger_labels = trigger_labels or {"cat"}
        self._roi_consecutive_frames: int = 0
        self._last_classify_time: float = 0.0
        self._was_tracking: bool = False
        self.pending_bbox: tuple | None = None  # normalized (xmin, ymin, w, h)

    _last_log_time: float = 0.0

    def update(self, frame_detections: list[dict]) -> bool:
        """Check detections against bowl ROI.

        Returns True when a crop should be produced this frame.
        """
        W, H = self._frame_w, self._frame_h
        x0, y0, x1, y1 = self._bowl_roi
        now = time.monotonic()

        roi_hit = False
        roi_area = (x1 - x0) * (y1 - y0)
        best_bbox = None
        for det in frame_detections:
            if det["label"] not in self._trigger_labels:
                continue
            xmin, ymin, w, h = det["bbox_norm"]
            bx1, by1 = xmin * W, ymin * H
            bx2, by2 = (xmin + w) * W, (ymin + h) * H
            # Compute intersection area
            ix1, iy1 = max(x0, bx1), max(y0, by1)
            ix2, iy2 = min(x1, bx2), min(y1, by2)
            if ix2 > ix1 and iy2 > iy1 and roi_area > 0:
                overlap = (ix2 - ix1) * (iy2 - iy1) / roi_area
                if overlap >= self._roi_overlap_threshold:
                    roi_hit = True
                    best_bbox = det["bbox_norm"]
                    break
                elif now - self._last_log_time >= 5.0:
                    print(f"[classifier-gate] Overlap {overlap:.0%} < "
                          f"threshold {self._roi_overlap_threshold:.0%}",
                          flush=True)
                    self._last_log_time = now
            elif now - self._last_log_time >= 5.0:
                print(f"[classifier-gate] Detection outside ROI: "
                      f"bbox=({bx1:.0f},{by1:.0f})-({bx2:.0f},{by2:.0f}) "
                      f"ROI=({x0},{y0})-({x1},{y1})",
                      flush=True)
                self._last_log_time = now

        if not roi_hit:
            if self._was_tracking:
                print("[classifier-gate] Cat left ROI — sending departure", flush=True)
                self.send_reset()
            self._roi_consecutive_frames = 0
            self._was_tracking = False
            return False

        self._roi_consecutive_frames += 1
        self._was_tracking = True

        if self._roi_consecutive_frames < self._stability_frames:
            if now - self._last_log_time >= 5.0:
                print(f"[classifier-gate] Stability: {self._roi_consecutive_frames}"
                      f"/{self._stability_frames} frames", flush=True)
                self._last_log_time = now
            return False

        if (now - self._last_classify_time) >= self._classify_interval:
            self.pending_bbox = best_bbox
            self._last_classify_time = now
            print(f"[classifier-gate] Crop sent to classifier", flush=True)
            return True

        return False

    def put_crop(self, frame: np.ndarray, bbox_norm: tuple) -> None:
        """Crop, resize to 224×224, enqueue for classifier, and optionally save full frame + sidecar."""
        H, W = frame.shape[:2]
        xmin, ymin, bw, bh = bbox_norm
        pad_x = bw * self._crop_padding
        pad_y = bh * self._crop_padding
        x1 = max(0, int((xmin - pad_x) * W))
        y1 = max(0, int((ymin - pad_y) * H))
        x2 = min(W, int((xmin + bw + pad_x) * W))
        y2 = min(H, int((ymin + bh + pad_y) * H))

        if x2 <= x1 or y2 <= y1:
            return

        crop = cv2.resize(frame[y1:y2, x1:x2], (224, 224))

        try:
            self._queue.put_nowait(crop)
        except QueueFull:
            pass

        if self._save_crops and self._labeling_dir:
            stem = str(int(time.time() * 1000))
            try:
                # Save full frame (not crop) so padding can be adjusted on the dashboard
                cv2.imwrite(str(self._labeling_dir / f"{stem}.jpg"), frame)
                sidecar = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": "",
                    "image_size": [W, H],
                    "detections": [{
                        "label": "cat",
                        "confidence": 1.0,
                        "bbox_norm": [round(float(v), 6) for v in bbox_norm],
                    }],
                }
                with open(self._labeling_dir / f"{stem}.json", "w") as f:
                    _json.dump(sidecar, f)
            except Exception:
                pass

    def send_reset(self) -> None:
        """Send departure sentinel to classifier worker."""
        try:
            self._queue.put_nowait(None)
        except QueueFull:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Bounding Box Drawing
# ═══════════════════════════════════════════════════════════════════════════

# Color palette for different labels (BGR)
_BBOX_COLORS = {
    "cat":     (0, 165, 255),   # orange
    "Cat3":  (0, 255, 0),     # green
    "Cat1":  (255, 100, 100), # blue
    "Cat2":   (0, 200, 255),   # yellow
}
_BBOX_DEFAULT_COLOR = (200, 200, 200)


def draw_bounding_boxes(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw labeled bounding boxes on a frame.

    Args:
        frame: BGR numpy array
        detections: list of dicts with keys: label, confidence, bbox (xmin, ymin, xmax, ymax), track_id

    Returns:
        Annotated frame (modified in-place).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        xmin, ymin, xmax, ymax = det["bbox"]
        track_id = det.get("track_id")

        color = _BBOX_COLORS.get(label, _BBOX_DEFAULT_COLOR)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)

        text = f"{label} {conf:.0%}"
        if track_id is not None:
            text += f" #{track_id}"

        (tw, th), baseline = cv2.getTextSize(text, font, 0.7, 2)
        cv2.rectangle(frame, (xmin, ymin - th - baseline - 4),
                      (xmin + tw + 4, ymin), color, -1)
        cv2.putText(frame, text, (xmin + 2, ymin - baseline - 2),
                    font, 0.7, (0, 0, 0), 2)

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Hailo Detection Callback
# ═══════════════════════════════════════════════════════════════════════════

if HAILO_AVAILABLE:
    class CatDetectionCallback(app_callback_class):
        """Tracks detection counts and optionally filters by class name."""

        def __init__(self):
            super().__init__()
            self.filter_labels: set[str] | None = None  # None = show all
            self.detection_log: dict[str, int] = {}      # label -> count
            self.feeding_tracker = None  # Optional FeedingTracker instance
            self.detection_capturer = None  # Optional DetectionCapturer instance
            self.capture_mode = None  # "eating" or "detections"
            self.capture_filter_labels: set[str] | None = None  # labels to auto-capture
            self.unlabeled_classes: set[str] = set()  # COCO classes to save as "unlabeled"
            self.ignore_labels: set[str] = set()  # labels hidden from video/pills
            self.preview_writer: PreviewFrameWriter | None = None
            self.command_listener: StdinCommandListener | None = None
            self.manual_capture_dir: Path | None = None
            self.classifier_gate: ClassifierGate | None = None
            self.classifier_process: multiprocessing.Process | None = None
            self.classifier_queue = None  # multiprocessing.Queue

        def set_filter(self, labels: list[str]) -> None:
            self.filter_labels = set(labels)


    def app_callback(element, buffer, user_data: CatDetectionCallback):
        """Process each detection frame.

        Called by hailo-apps _internal_callback_wrapper via the identity
        element's 'handoff' signal.  The wrapper has already incremented
        the frame counter.

        Args:
            element: GStreamer identity element
            buffer: Gst.Buffer containing frame data and Hailo metadata
            user_data: CatDetectionCallback instance
        """
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        # Diagnostic: log raw detection count before filtering
        if user_data.get_count() % 30 == 0:
            print_info(f"Frame {user_data.get_count()}: {len(detections)} raw detections (before filter)")

        # Collect detected cat names and tracker IDs this frame
        detected_cats: set[str] = set()
        detected_cats_with_ids: dict[str, set[int | None]] = {}
        frame_detections: list[dict] = []

        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()

            # Filter if configured
            if user_data.filter_labels and label not in user_data.filter_labels:
                roi.remove_object(detection)
                continue

            # Hide ignored labels from video overlay and pills
            if user_data.ignore_labels and label in user_data.ignore_labels:
                roi.remove_object(detection)
                continue

            # Hide labels not in capture filter from overlay and preview
            if user_data.capture_filter_labels and label not in user_data.capture_filter_labels:
                roi.remove_object(detection)
                continue

            # Track unique IDs if tracker is active
            track_objs = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = track_objs[0].get_id() if track_objs else None

            # Log detection
            user_data.detection_log[label] = user_data.detection_log.get(label, 0) + 1
            detected_cats.add(label)
            detected_cats_with_ids.setdefault(label, set()).add(track_id)

            # Collect bbox for preview drawing (normalized coords)
            bbox = detection.get_bbox()
            frame_detections.append({
                "label": label,
                "confidence": confidence,
                "bbox_norm": (bbox.xmin(), bbox.ymin(), bbox.width(), bbox.height()),
                "track_id": track_id,
            })

            if user_data.get_count() % 30 == 0:  # Print every 30 frames to avoid spam
                print_info(f"Frame {user_data.get_count()}: {label} ({confidence:.2f}) track={track_id}")

        # Periodic summary — confirms model is producing detections
        if user_data.get_count() % 150 == 0:
            print_info(f"Frame {user_data.get_count()}: {len(frame_detections)} detections this frame")

        # Process stdin commands (cheap — just a queue check)
        if user_data.command_listener:
            cmd = user_data.command_listener.get_command()
            if cmd:
                if cmd.get("cmd") == "capture":
                    _handle_capture_command(element, buffer, user_data, cmd, frame_detections)
                elif cmd.get("cmd") == "set_ignore_labels":
                    user_data.ignore_labels = set(cmd.get("labels", []))
                elif cmd.get("cmd") == "set_detection_capture":
                    _handle_set_detection_capture(user_data, cmd)
                elif cmd.get("cmd") == "set_bowl_roi":
                    if user_data.classifier_gate:
                        roi = cmd.get("bowl_roi", [0, 0, 1920, 1080])
                        user_data.classifier_gate._bowl_roi = tuple(roi)
                        print(f"[classifier-gate] ROI updated: {roi}", flush=True)
                elif cmd.get("cmd") == "set_roi_overlap_threshold":
                    if user_data.classifier_gate:
                        t = cmd.get("roi_overlap_threshold", 0.01)
                        user_data.classifier_gate._roi_overlap_threshold = t
                        print(f"[classifier-gate] Overlap threshold: {t:.0%}", flush=True)

        # Feed into session tracker (with tracker IDs)
        if user_data.feeding_tracker:
            user_data.feeding_tracker.update(detected_cats_with_ids)

        # ── Classifier subprocess health check ─────────────────────────
        if (user_data.classifier_process is not None
                and user_data.frame_count % 150 == 0  # ~every 5s at 30fps
                and not user_data.classifier_process.is_alive()):
            print_warning(f"Classifier subprocess died "
                          f"(exit code: {user_data.classifier_process.exitcode})")
            user_data.classifier_process = None

        # ── Classifier gate (Stage 2) ──────────────────────────────────
        needs_classify = False
        classify_bbox = None
        if user_data.classifier_gate and frame_detections:
            needs_classify = user_data.classifier_gate.update(frame_detections)
            if needs_classify:
                classify_bbox = user_data.classifier_gate.pending_bbox

        # ── Decide what needs a frame BEFORE extracting (extraction is expensive) ──
        from deploy.detection_capturer import FrameExtractor

        needs_preview = (user_data.preview_writer
                         and user_data.preview_writer.should_write())

        needs_capture = False
        if user_data.detection_capturer and detected_cats:
            if user_data.capture_mode == "detections":
                # Filter to only configured labels (if set)
                capture_cats = detected_cats
                if user_data.capture_filter_labels:
                    capture_cats = detected_cats & user_data.capture_filter_labels
                for cat in capture_cats:
                    save_label = ("unlabeled" if user_data.unlabeled_classes
                                  and cat in user_data.unlabeled_classes else cat)
                    if user_data.detection_capturer.should_capture(save_label):
                        needs_capture = True
                        break
            elif user_data.capture_mode == "eating" and user_data.feeding_tracker:
                from deploy.feeding_tracker import CatState
                for cat in detected_cats:
                    if user_data.feeding_tracker.get_cat_state(cat) == CatState.EATING:
                        needs_capture = True
                        break

        # Extract frame AT MOST ONCE, shared between preview, capture, and classify
        frame = None
        if needs_preview or needs_capture or needs_classify:
            frame = FrameExtractor.extract_frame(buffer, element)

        # Write preview frame with bounding boxes (async — non-blocking)
        if frame is not None and needs_preview:
            user_data.preview_writer.enqueue_frame(frame, frame_detections)

        # Classifier crop (using shared frame)
        if frame is not None and needs_classify and classify_bbox is not None:
            user_data.classifier_gate.put_crop(frame, classify_bbox)

        # Detection-triggered capture (using shared frame)
        if frame is not None and needs_capture:
            if user_data.capture_mode == "eating" and user_data.feeding_tracker:
                _maybe_capture_eating_cats(user_data, detected_cats, frame, frame_detections)
            elif user_data.capture_mode == "detections":
                _maybe_capture_detections(user_data, detected_cats, frame, frame_detections)


    def _handle_capture_command(element, buffer, user_data: CatDetectionCallback, cmd: dict,
                               frame_detections: list[dict] | None = None):
        """Handle a manual capture command from the dashboard via stdin.

        Always saves directly to manual_capture_dir, bypassing DetectionCapturer's
        automatic throttle which is only appropriate for detection-triggered captures.
        Writes a JSON sidecar with current frame detections for crop-on-dashboard.
        """
        from deploy.detection_capturer import FrameExtractor

        cat_name = cmd.get("cat", "unlabeled")
        frame = FrameExtractor.extract_frame(buffer, element)
        if frame is None:
            print(json.dumps({"response": "capture_error", "message": "frame extraction failed"}),
                  flush=True)
            return

        if not user_data.manual_capture_dir:
            print(json.dumps({"response": "capture_error", "message": "no capture dir configured"}),
                  flush=True)
            return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = f"{cat_name}_{ts}.jpg"
        filepath = user_data.manual_capture_dir / cat_name / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            if frame_detections:
                H, W = frame.shape[:2]
                sidecar = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": "",
                    "image_size": [W, H],
                    "detections": [
                        {
                            "label": d["label"],
                            "confidence": round(float(d["confidence"]), 4),
                            "bbox_norm": [round(float(v), 6) for v in d["bbox_norm"]],
                        }
                        for d in frame_detections
                    ],
                }
                try:
                    with open(filepath.with_suffix(".json"), "w") as f:
                        _json.dump(sidecar, f)
                except Exception:
                    pass
            print(json.dumps({"response": "capture_ok", "cat": cat_name, "filename": filename}),
                  flush=True)
        else:
            print(json.dumps({"response": "capture_error", "message": "imwrite failed"}),
                  flush=True)


    def _maybe_capture_detections(user_data, detected_cats, frame, frame_detections=None):
        """Capture a frame for each detected class (no feeding tracker required).

        Args:
            user_data: CatDetectionCallback with detection_capturer
            detected_cats: set of detected label names
            frame: Pre-extracted BGR numpy array (shared with preview writer)
            frame_detections: list of detection dicts (label, confidence, bbox_norm)
        """
        # Filter by configured labels
        capture_cats = detected_cats
        if user_data.capture_filter_labels:
            capture_cats = detected_cats & user_data.capture_filter_labels

        for cat in capture_cats:
            save_label = cat
            if user_data.unlabeled_classes and cat in user_data.unlabeled_classes:
                save_label = "unlabeled"
            # Build per-cat detection list for sidecar (only detections matching this cat)
            det_data = None
            if frame_detections:
                det_data = [
                    {"label": d["label"], "confidence": d["confidence"],
                     "bbox_norm": list(d["bbox_norm"])}
                    for d in frame_detections if d["label"] == cat
                ] or None
            user_data.detection_capturer.maybe_capture(save_label, frame, detections=det_data)


    def _maybe_capture_eating_cats(user_data, detected_cats, frame, frame_detections=None):
        """Capture a frame for cats confirmed EATING.

        Args:
            user_data: CatDetectionCallback with feeding_tracker + detection_capturer
            detected_cats: set of detected label names
            frame: Pre-extracted BGR numpy array (shared with preview writer)
            frame_detections: list of detection dicts (label, confidence, bbox_norm)
        """
        from deploy.feeding_tracker import CatState

        for cat in detected_cats:
            state = user_data.feeding_tracker.get_cat_state(cat)
            if state != CatState.EATING:
                continue
            session_id = user_data.feeding_tracker.get_session_id(cat)
            det_data = None
            if frame_detections:
                det_data = [
                    {"label": d["label"], "confidence": d["confidence"],
                     "bbox_norm": list(d["bbox_norm"])}
                    for d in frame_detections if d["label"] == cat
                ] or None
            user_data.detection_capturer.maybe_capture(cat, frame, session_id,
                                                       detections=det_data)


    def _handle_set_detection_capture(user_data: CatDetectionCallback, cmd: dict):
        """Handle live toggle of detection capture from the dashboard."""
        enabled = cmd.get("enabled", False)
        filter_labels = cmd.get("filter_labels", [])

        if enabled:
            if not user_data.detection_capturer:
                from deploy.detection_capturer import DetectionCapturer

                cfg = load_config()
                capture_cfg = cfg.get("detection_capture", {})
                collection_cfg = cfg.get("collection", {})

                capture_dir = resolve_path(
                    capture_cfg.get("output_dir", "data/raw"),
                    PROJECT_ROOT,
                )
                interval = collection_cfg.get(
                    "auto_capture_interval_sec",
                    capture_cfg.get("min_interval_seconds", 15.0),
                )
                user_data.detection_capturer = DetectionCapturer(
                    output_dir=capture_dir,
                    cats=cfg.get("cats", []),
                    min_interval_seconds=interval,
                    max_captures_per_session=capture_cfg.get("max_captures_per_session", 0),
                    jpeg_quality=capture_cfg.get("jpeg_quality", 95),
                    csv_logging=capture_cfg.get("csv_logging", True),
                )
                user_data.detection_capturer.start()

                unlabeled = capture_cfg.get("unlabeled_classes", [])
                if unlabeled:
                    user_data.unlabeled_classes = set(unlabeled)

            user_data.capture_mode = "detections"
            if filter_labels:
                user_data.capture_filter_labels = set(filter_labels)
            else:
                user_data.capture_filter_labels = None
            print_info(f"Detection capture enabled (filter: {filter_labels or 'all'})")
        else:
            if user_data.detection_capturer:
                user_data.detection_capturer.shutdown()
                user_data.detection_capturer = None
            user_data.capture_mode = None
            print_info("Detection capture disabled")


def main():
    if not HAILO_AVAILABLE:
        print_error("Hailo libraries not available.")
        print_error(f"Import error: {HAILO_IMPORT_ERROR}")
        print_info("This script must run on the Raspberry Pi inside the hailo-apps venv:")
        print_info("  cd <project_root>/hailo-apps && source setup_env.sh")
        sys.exit(1)

    # Only --config is ours; everything else passes through to Hailo
    parser = argparse.ArgumentParser(
        description="Cat detection with Hailo 10H",
        add_help=False,
    )
    add_common_args(parser)
    parser.add_argument("--enable-preview", action="store_true",
                        help="Write preview frames to shared file for dashboard")
    parser.add_argument("--enable-stdin-commands", action="store_true",
                        help="Listen for JSON commands on stdin")
    args, remaining_argv = parser.parse_known_args()

    # Load config
    cfg = get_config_with_args(args)
    deploy_cfg = cfg.get("deployment", {})
    camera_cfg = cfg.get("camera", {})
    capture_cfg = cfg.get("detection_capture", {})
    feeding_cfg = cfg.get("feeding", {})
    web_cfg = cfg.get("web_ui", {})

    # Build Hailo sys.argv from config; CLI args in remaining_argv override.
    # If model/labels are empty or files don't exist, omit the flags so
    # hailo-apps uses its built-in defaults (e.g. the standard COCO yolov8 model).
    raw_hef = deploy_cfg.get("hef_path", "")
    raw_labels = deploy_cfg.get("labels_json", "")
    camera_device = camera_cfg.get("device", "/dev/video0")

    sys.argv = [sys.argv[0], "--input", camera_device]

    if raw_hef:
        hef_path = str(resolve_path(raw_hef, PROJECT_ROOT))
        if Path(hef_path).is_file():
            sys.argv += ["--hef-path", hef_path]
            print_info(f"Model: {hef_path}")
            # Only pass labels when using a custom model
            if raw_labels:
                labels_path = str(resolve_path(raw_labels, PROJECT_ROOT))
                if Path(labels_path).is_file():
                    sys.argv += ["--labels-json", labels_path]
                    print_info(f"Labels: {labels_path}")
                else:
                    print_warning(f"Labels not found: {labels_path}")
        else:
            print_warning(f"Model not found: {hef_path} — using hailo-apps defaults")
    else:
        print_info("No custom model configured — using hailo-apps defaults")

    sys.argv += remaining_argv
    print_info(f"sys.argv for hailo-apps: {sys.argv}")
    print_info(f"Working directory: {os.getcwd()}")

    # Setup callback
    user_data = CatDetectionCallback()

    filter_classes = deploy_cfg.get("filter_classes", [])
    if filter_classes:
        user_data.set_filter(filter_classes)
        print_info(f"Filtering for: {filter_classes}")

    track_all = deploy_cfg.get("track_all", True)
    tracker_class_id = -1 if track_all else 0
    print_info(f"Tracker class-id: {tracker_class_id} (-1 = all classes)")

    # Setup preview writer
    if args.enable_preview:
        preview_path = deploy_cfg.get("preview_path", "/tmp/cat_detection_preview.jpg")
        preview_fps = deploy_cfg.get("preview_fps", 10)
        user_data.preview_writer = PreviewFrameWriter(
            preview_path=preview_path,
            preview_fps=preview_fps,
            preview_width=web_cfg.get("preview_width", 960),
            preview_height=web_cfg.get("preview_height", 540),
            jpeg_quality=deploy_cfg.get("preview_jpeg_quality", 70),
        )
        user_data.preview_writer.start()
        print_info(f"Preview writer: {preview_path} @ {preview_fps} fps")

    # Setup stdin command listener
    if args.enable_stdin_commands:
        user_data.command_listener = StdinCommandListener()
        user_data.command_listener.start()
        print_info("Stdin command listener started")

    # Store manual capture dir for fallback (when no DetectionCapturer)
    user_data.manual_capture_dir = resolve_path(
        capture_cfg.get("output_dir", cfg.get("collection", {}).get("output_dir", "data/raw")),
        PROJECT_ROOT,
    )

    # ── Classifier setup (BEFORE GStreamer init — fork safety) ───────────
    classifier_cfg = cfg.get("classifier", {})
    if classifier_cfg.get("enabled", False) or classifier_cfg.get("save_crops_for_labeling", False):
        bowl_roi = tuple(classifier_cfg.get("bowl_roi", [0, 0, 1920, 1080]))
        classifier_q = multiprocessing.Queue(maxsize=5)

        save_crops = classifier_cfg.get("save_crops_for_labeling", False)
        labeling_dir = None
        if save_crops:
            labeling_dir = resolve_path(
                classifier_cfg.get("labeling_output_dir", "data/raw/unlabeled"),
                PROJECT_ROOT,
            )
            labeling_dir.mkdir(parents=True, exist_ok=True)

        # Clear stale classifier status from previous runs
        status_file = Path("/tmp/cat_classifier_status.json")
        if status_file.exists():
            status_file.unlink(missing_ok=True)

        classifier_proc = None
        if classifier_cfg.get("enabled", False):
            from deploy.cat_classifier import classifier_worker
            classifier_cfg_resolved = {**classifier_cfg, "cats": cfg["cats"]}
            if "model_path" in classifier_cfg_resolved:
                classifier_cfg_resolved["model_path"] = str(resolve_path(classifier_cfg_resolved["model_path"], PROJECT_ROOT))
            if "db_path" in classifier_cfg_resolved:
                classifier_cfg_resolved["db_path"] = str(resolve_path(classifier_cfg_resolved["db_path"], PROJECT_ROOT))
            classifier_proc = multiprocessing.Process(
                target=classifier_worker,
                args=(classifier_q, classifier_cfg_resolved),
                daemon=True,
                name="cat_classifier",
            )
            classifier_proc.start()   # fork happens here, BEFORE GStreamer init
            print_info(f"Classifier process started (pid={classifier_proc.pid})")

        user_data.classifier_queue = classifier_q
        user_data.classifier_process = classifier_proc
        user_data.classifier_gate = ClassifierGate(
            queue=classifier_q,
            bowl_roi=bowl_roi,
            frame_w=camera_cfg.get("width", 1920),
            frame_h=camera_cfg.get("height", 1080),
            stability_frames=classifier_cfg.get("stability_frames", 5),
            classify_interval=classifier_cfg.get("classify_interval_seconds", 1.0),
            crop_padding=classifier_cfg.get("crop_padding", 0.20),
            save_crops=save_crops,
            labeling_dir=labeling_dir,
            roi_overlap_threshold=classifier_cfg.get("roi_overlap_threshold", 0.01),
            trigger_labels=set(classifier_cfg.get("trigger_labels", ["cat"])),
        )
        print_info(f"Classifier gate active — ROI={bowl_roi}, save_crops={save_crops}")

    # Create the detection app (Hailo parses sys.argv for --hef-path, --input, etc.)
    app = GStreamerDetectionApp(app_callback, user_data)

    # Diagnostic: log what model hailo-apps resolved to
    try:
        print_info(f"Hailo resolved HEF: {getattr(app, 'hef_path', 'unknown')}")
    except Exception:
        pass
    try:
        print_info(f"Hailo options: {vars(app.options_menu)}")
    except Exception:
        pass

    # Configure only the queue feeding the user callback to drop old frames.
    # IMPORTANT: Do NOT modify all queues — that breaks inference pipeline
    # buffering and causes 0 detections.
    try:
        identity = app.pipeline.get_by_name("identity_callback")
        if identity:
            # Walk upstream from the identity element to find its queue
            pad = identity.get_static_pad("sink")
            if pad:
                peer = pad.get_peer()
                if peer:
                    upstream = peer.get_parent_element()
                    factory = upstream.get_factory() if upstream else None
                    if factory and factory.get_name() == "queue":
                        upstream.set_property("leaky", 2)  # drop newest when full
                        upstream.set_property("max-size-buffers", 3)
                        print_info(f"Queue '{upstream.get_name()}': leaky=downstream, max-buffers=3")
                    else:
                        print_info(f"Upstream of callback is '{factory.get_name() if factory else 'unknown'}', not a queue — skipping")
        else:
            print_warning("identity_callback not found in pipeline")
    except Exception as exc:
        print_warning(f"Could not configure callback queue: {exc}")

    # Configure tracker class-id (-1 = track all classes)
    try:
        tracker = app.pipeline.get_by_name("hailo_tracker")
        if tracker:
            tracker.set_property("class-id", tracker_class_id)
            print_info(f"Tracker '{tracker.get_name()}': class-id={tracker_class_id}")
        else:
            # Try iterating to find by factory name
            it = app.pipeline.iterate_elements()
            found = False
            while True:
                ok, elem = it.next()
                if ok != 1:  # Gst.IteratorResult.OK
                    break
                factory = elem.get_factory()
                if factory and factory.get_name() == "hailotracker":
                    elem.set_property("class-id", tracker_class_id)
                    print_info(f"Tracker '{elem.get_name()}': class-id={tracker_class_id}")
                    found = True
                    break
            if not found:
                print_warning("No hailotracker element found in pipeline")
    except Exception as exc:
        print_warning(f"Could not configure tracker: {exc}")

    # Setup feeding tracker if enabled
    if feeding_cfg.get("enabled", False):
        from deploy.feeding_tracker import FeedingTracker

        db_path = resolve_path(
            feeding_cfg.get("db_path", "data/feeding_log.db"),
            PROJECT_ROOT,
        )
        user_data.feeding_tracker = FeedingTracker(
            db_path=db_path,
            cats=cfg["cats"],
            cooldown_seconds=feeding_cfg.get("cooldown_seconds", 30),
            min_detection_frames=feeding_cfg.get("min_detection_frames", 3),
            log_interval_seconds=feeding_cfg.get("log_interval_seconds", 60),
        )
        print_info(f"Feeding tracker enabled, logging to {db_path}")

    # Setup detection capture if enabled
    if capture_cfg.get("enabled", False):
        from deploy.detection_capturer import DetectionCapturer

        capture_dir = resolve_path(
            capture_cfg.get("output_dir", "data/raw"),
            PROJECT_ROOT,
        )
        # Use auto-capture interval as the throttle (fallback to detection_capture's own)
        collection_cfg = cfg.get("collection", {})
        interval = collection_cfg.get(
            "auto_capture_interval_sec",
            capture_cfg.get("min_interval_seconds", 15.0),
        )
        user_data.detection_capturer = DetectionCapturer(
            output_dir=capture_dir,
            cats=cfg.get("cats", []),
            min_interval_seconds=interval,
            max_captures_per_session=capture_cfg.get("max_captures_per_session", 0),
            jpeg_quality=capture_cfg.get("jpeg_quality", 95),
            csv_logging=capture_cfg.get("csv_logging", True),
        )

        # Set label filter if configured
        filter_labels = capture_cfg.get("filter_labels", [])
        if filter_labels:
            user_data.capture_filter_labels = set(filter_labels)
            print_info(f"Capture filter: {filter_labels}")

        # Capture mode: "eating" if feeding tracker active, else "detections"
        if user_data.feeding_tracker:
            user_data.capture_mode = "eating"
            print_info(f"Detection capture (eating mode): saving to {capture_dir}")
        else:
            user_data.capture_mode = "detections"
            unlabeled = capture_cfg.get("unlabeled_classes", [])
            if unlabeled:
                user_data.unlabeled_classes = set(unlabeled)
                print_info(f"Classes {unlabeled} will be saved as 'unlabeled'")
            print_info(f"Detection capture (any detection): saving to {capture_dir}")

        user_data.detection_capturer.start()
        print_info(
            f"  Interval: {user_data.detection_capturer.min_interval}s, "
            f"Max/session: {user_data.detection_capturer.max_per_session or 'unlimited'}"
        )

    print_success("Starting detection pipeline...")
    try:
        app.run()
    finally:
        # Classifier shutdown (graceful: send stop sentinel, then terminate if needed)
        if user_data.classifier_process and user_data.classifier_process.is_alive():
            try:
                user_data.classifier_queue.put(None, timeout=1)  # flush final event
            except Exception:
                pass
            user_data.classifier_process.join(timeout=5)
            if user_data.classifier_process.is_alive():
                user_data.classifier_process.terminate()

        # Cleanup preview on shutdown
        if user_data.preview_writer:
            user_data.preview_writer.shutdown()
        if user_data.command_listener:
            user_data.command_listener.stop()

        # Finalize any active sessions on shutdown
        if user_data.feeding_tracker:
            user_data.feeding_tracker.shutdown()

        # Shutdown capturer after feeding tracker (so final sessions are logged first)
        if user_data.detection_capturer:
            user_data.detection_capturer.print_stats()
            user_data.detection_capturer.shutdown()


if __name__ == "__main__":
    main()
