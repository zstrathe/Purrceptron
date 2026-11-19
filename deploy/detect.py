#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

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
from collections import deque
from queue import Full as QueueFull

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import PROJECT_ROOT, get_config_with_args, load_config, resolve_path, add_common_args, get_cat_colors_bgr
from utils.display import print_info, print_error, print_success, print_warning
from utils.runtime_paths import PREVIEW_DETECTIONS_JSON, PREVIEW_JPG, runtime_dir
from deploy.visits import (DetectionRecorder, VisitTracker, visit_gap_from,  # noqa: E402
                           visit_hold_from)

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

    def __init__(self, preview_path: str, detections_path: str, preview_fps: int,
                 preview_width: int, preview_height: int,
                 jpeg_quality: int):
        self.preview_path = preview_path
        self.tmp_path = preview_path + ".tmp"
        self.detections_path = detections_path
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
                            "name": det.get("name"),
                            "duplicate": det.get("duplicate", False),
                            "held": det.get("held", False),
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
                         "track_id": d.get("track_id"), "in_roi": d.get("in_roi", True),
                         "visit_id": d.get("visit_id"), "stable": d.get("stable", False),
                         "duplicate": d.get("duplicate", False), "held": d.get("held", False),
                         "name": d.get("name"), "named_by": d.get("named_by")}
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

class VisitFeeder:
    """Feeds the Stage 2 classifier from the GStreamer callback.

    Every stable visit from deploy/visits.py's VisitTracker gets a 224×224
    crop of its animal every classify_interval seconds; the worker keeps a
    vote, a name and a log row per visit. Messages (see cat_classifier.py):
    ("crop", visit_id, seen_at, crop), ("departed", visit_id, last_seen).
    A departure that finds the queue full is kept and
    sent again on the next frame, so a visit is never left open.
    """

    def __init__(self, queue, classify_interval: float, crop_padding: float):
        self._queue = queue
        self._classify_interval = classify_interval
        self._crop_padding = crop_padding
        self._pending: deque = deque()          # departures waiting for room
        self._fed: set[float] = set()           # visits the worker has heard of
        self._visits: list = []                 # this frame's visits

    def update(self, visits: list, departed: list) -> list:
        """Called every frame with the tracker's visits. Returns the visits
        due a crop in this frame."""
        self._visits = visits
        for v in departed:
            if v.visit_id in self._fed:
                self._fed.discard(v.visit_id)
                self._pending.append(("departed", v.visit_id, v.last_seen))
        while self._pending:
            try:
                self._queue.put_nowait(self._pending[0])
            except QueueFull:
                break
            self._pending.popleft()
        now = time.monotonic()
        return [v for v in visits if v.seen and v.stable
                and now - v.last_classify >= self._classify_interval]

    def put_crop(self, frame: np.ndarray, visit) -> None:
        """Crop the visit's animal, resize to 224×224 and enqueue it."""
        visit.last_classify = time.monotonic()
        H, W = frame.shape[:2]
        xmin, ymin, bw, bh = visit.bbox_norm
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
            self._queue.put_nowait(("crop", visit.visit_id, visit.last_seen, crop))
            self._fed.add(visit.visit_id)
        except QueueFull:
            pass

    def _primary(self):
        return next((v for v in self._visits if v.stable), None)

    @property
    def visit_start(self) -> float | None:
        """The oldest stable visit's id (its first frame's wall time), or None."""
        v = self._primary()
        return v.visit_id if v else None

    @property
    def last_hit(self) -> float | None:
        v = self._primary()
        return v.last_seen if v else None


# ═══════════════════════════════════════════════════════════════════════════
# Bounding Box Drawing
# ═══════════════════════════════════════════════════════════════════════════

# Color palette for different labels (BGR).
# "cat" is a fixed COCO label; per-cat colors are added at startup from config.
_BBOX_COLORS: dict[str, tuple[int, int, int]] = {
    "cat": (0, 165, 255),   # orange
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
        if det.get("duplicate"):
            continue            # a second bounding box on an animal already drawn
        label = det["label"]
        conf = det["confidence"]
        xmin, ymin, xmax, ymax = det["bbox"]
        track_id = det.get("track_id")
        name = det.get("name")

        if name:
            # An animal in the region: who it is ("Cat1", "Cat1?" while the
            # vote is open, "?" when unknown), in that animal's colour.
            color = _BBOX_COLORS.get(name.rstrip("?"), _BBOX_DEFAULT_COLOR)
            text = name
        else:
            color = _BBOX_COLORS.get(label, _BBOX_DEFAULT_COLOR)
            text = f"{label} {conf:.0%}"
            if track_id is not None:
                text += f" #{track_id}"

        # Thin while held: the detector lost the animal for a moment.
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1 if det.get("held") else 3)

        (tw, th), baseline = cv2.getTextSize(text, font, 0.7, 2)
        tag_h = th + baseline + 4
        # Above the bounding box, or just inside it when it touches the top.
        top = ymin - tag_h if ymin - tag_h >= 0 else ymin
        cv2.rectangle(frame, (xmin, top), (xmin + tw + 4, top + tag_h), color, -1)
        cv2.putText(frame, text, (xmin + 2, top + tag_h - baseline - 2),
                    font, 0.7, (0, 0, 0), 2)

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Capture labeling from classifier status
# ═══════════════════════════════════════════════════════════════════════════

from deploy.cat_classifier import (  # noqa: E402  (numpy-only import)
    STATUS_PATH as CLASSIFIER_STATUS_PATH,
    confirmed_visits,
    read_status as read_classifier_status,
    visit_record,
)

# Detector labels the classifier names (detection_capture.unlabeled_classes) share
# one capture throttle: it is the same animal at the bowl, whichever name its
# images end up under.
CLASSIFIED_THROTTLE_KEY = "classified"


def classifier_capture_label(status: dict | None,
                             visit_id: float | None) -> tuple[str | None, bool]:
    """Decide how a detection capture of one visit's animal, taken right now,
    should be labeled.

    Returns (label, backlabel_eligible):
      - label is the name the visit has now (the same one its log row has),
        or None to save the image as unlabeled.
      - backlabel_eligible is True when the image is unlabeled only because the
        vote has not decided yet; it is renamed if this visit is named later.
    Never named, and never eligible: images taken while the classifier is
    rejecting the visit's crops (energy above threshold), while its name is
    held by another animal in view, or while its vote majority has moved to
    a different name than the visit's.
    A visit that has ended is looked up in the list of confirmed visits.
    """
    if visit_id is None:
        return None, False
    if not status:
        return None, True
    rec = visit_record(status, visit_id)
    if rec is not None:
        if rec.get("last_crop_rejected") or rec.get("name_taken"):
            return None, False
        visit_cat = rec.get("visit_cat")
        majority = rec.get("confirmed_cat")
    else:
        visit_cat = confirmed_visits(status).get(visit_id)
        majority = None
    if visit_cat:
        if majority and majority != visit_cat:
            return None, False
        return visit_cat, False
    return None, True


SAVE_WHICH = ("all", "identified", "unidentified")


def save_which_from(cfg: dict) -> str:
    """detection_capture.save_which, or the older save_labeled_only flag."""
    which = cfg.get("save_which")
    if which in SAVE_WHICH:
        return which
    return "identified" if cfg.get("save_labeled_only") else "all"


def capture_plan(status: dict | None, visit_start: float | None, classifier_on: bool,
                 save_which: str) -> tuple[str | None, bool, bool]:
    """(label, backlabel_eligible, skip) for an unlabeled_classes detection.

    save_which applies only while the classifier runs (without it nothing is
    identified): "identified" skips images that can never be named,
    "unidentified" skips images already named (the ones worth labeling by
    hand are the ones the model can't name)."""
    if not classifier_on:
        return None, False, False
    label, eligible = classifier_capture_label(status, visit_start)
    if save_which == "identified":
        skip = label is None and not eligible
    elif save_which == "unidentified":
        skip = label is not None
    else:
        skip = False
    return label, eligible, skip


def visit_display(visit, status: dict | None, human_names: dict,
                  identifying: bool) -> tuple[str, str | None]:
    """(text, named_by) for an animal's bounding box on the live view:
    a name a person gave ("you"), the model's name ("model"), the model's
    guess with "?" while its vote is open, or "?"."""
    name = human_names.get(visit.visit_id)
    if name:
        return name, "you"
    if not identifying:
        return "?", None
    rec = visit_record(status, visit.visit_id)
    if rec is None:
        return "?", None
    if rec.get("visit_cat"):
        return rec["visit_cat"], "model"
    if rec.get("prediction") and not rec.get("unknown"):
        return f"{rec['prediction']}?", None
    return "?", None


def held_detections(visits: list) -> list[dict]:
    """Stand-in detections for stable visits not seen in this frame, at their
    last position. Only for the live view: never cropped, recorded or counted."""
    return [{"label": v.label, "confidence": v.confidence, "bbox_norm": v.bbox_norm,
             "track_id": v.track_id, "visit_id": v.visit_id, "in_roi": True,
             "duplicate": False, "held": True}
            for v in visits if v.stable and not v.seen]


def label_visits(detections: list[dict], visits: list, status: dict | None,
                 human_names: dict, identifying: bool) -> None:
    """Add name, named_by and stable to the detections that are stable visits."""
    by_id = {v.visit_id: v for v in visits}
    for d in detections:
        v = by_id.get(d.get("visit_id"))
        d["stable"] = bool(v and v.stable)
        if v is not None and v.stable:
            d["name"], d["named_by"] = visit_display(v, status, human_names, identifying)


# ═══════════════════════════════════════════════════════════════════════════
# Hailo Detection Callback
# ═══════════════════════════════════════════════════════════════════════════

if HAILO_AVAILABLE:
    # The callback structure follows Hailo's hailo-apps detection example
    # (MIT License; see THIRD_PARTY_NOTICES.md).
    class CatDetectionCallback(app_callback_class):
        """Tracks detection counts and optionally filters by class name."""

        def __init__(self):
            super().__init__()
            self.filter_labels: set[str] | None = None  # None = show all
            self.detection_log: dict[str, int] = {}      # label -> count
            self.feeding_tracker = None  # Optional FeedingTracker instance
            self.detection_capturer = None  # Optional DetectionCapturer instance
            self.capture_mode = None  # "eating" or "detections"
            self.unlabeled_classes: set[str] = set()  # COCO classes to save as "unlabeled"
            self.preview_writer: PreviewFrameWriter | None = None
            self.command_listener: StdinCommandListener | None = None
            self.manual_capture_dir: Path | None = None
            self.classifier_gate: VisitFeeder | None = None
            self.classifier_process: multiprocessing.Process | None = None
            self.classifier_queue = None  # multiprocessing.Queue
            self.reload_config = None     # () -> current config (project + runtime)
            self.save_which: str = "all"      # all / identified / unidentified
            self.bowl_roi: tuple = (0, 0, 1920, 1080)
            self.roi_overlap_threshold: float = 0.01
            self.stability_frames: int = 5
            self.frame_w: int = 1920
            self.frame_h: int = 1080
            self.visits: VisitTracker | None = None   # each animal in the region, by position
            self.human_names: dict[float, str] = {}   # visit_id -> name given on the dashboard
            self.recorder: DetectionRecorder | None = None
            from deploy.detection_capturer import VisitBackLabeler
            self.backlabeler = VisitBackLabeler()  # names a visit's early captures once confirmed
            # Classifier status as of the last read (2 Hz poll or a capture). The
            # pre-check uses it only to decide whether a frame is worth extracting;
            # each capture re-reads the file before choosing a label.
            self._status: dict | None = None
            self._status_last_poll: float = 0.0

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

        # Process stdin commands (cheap — just a queue check). A capture runs
        # after the visit tracker has seen this frame, so it knows who is in it.
        capture_cmd = None
        if user_data.command_listener:
            cmd = user_data.command_listener.get_command()
            if cmd:
                if cmd.get("cmd") == "capture":
                    capture_cmd = cmd
                elif cmd.get("cmd") == "set_classifier":
                    _switch_classifier(user_data, restart=False)
                elif cmd.get("cmd") == "reload_classifier":
                    _switch_classifier(user_data, restart=True)
                elif cmd.get("cmd") == "set_detection_capture":
                    _handle_set_detection_capture(user_data, cmd)
                elif cmd.get("cmd") == "set_bowl_roi":
                    roi = cmd.get("bowl_roi", [0, 0, 1920, 1080])
                    user_data.bowl_roi = tuple(roi)
                    user_data.visits.set_roi(roi)
                    print(f"[visits] ROI updated: {roi}", flush=True)
                elif cmd.get("cmd") == "set_roi_overlap_threshold":
                    t = cmd.get("roi_overlap_threshold", 0.01)
                    user_data.roi_overlap_threshold = t
                    user_data.visits.set_overlap_threshold(t)
                    print(f"[visits] Overlap threshold: {t:.0%}", flush=True)
                elif cmd.get("cmd") == "set_visit_names":
                    names = {}
                    for k, v in (cmd.get("names") or {}).items():
                        try:
                            names[float(k)] = str(v)
                        except (TypeError, ValueError):
                            pass
                    user_data.human_names = names
                elif cmd.get("cmd") == "record_detections":
                    secs = min(900.0, max(1.0, float(cmd.get("seconds", 600))))
                    if user_data.recorder:
                        user_data.recorder.close()
                    user_data.recorder = DetectionRecorder(
                        PROJECT_ROOT / "logs" / "detection", secs,
                        frame_size=(user_data.frame_w, user_data.frame_h),
                        frame_every=float(cmd.get("frame_every", 0) or 0),
                        idle_every=float(cmd.get("idle_every", 0) or 0))
                    print(f"[visits] Recording detections for {secs:.0f} s to "
                          f"{user_data.recorder.path}", flush=True)

        # ── Visits: each animal in the region, followed by position (every frame) ──
        visits, departed = user_data.visits.update(frame_detections)
        for v in departed:
            print(f"[visits] Visit {v.visit_id:.3f} ended ({v.frames} frames)", flush=True)
            user_data.human_names.pop(v.visit_id, None)
            if user_data.detection_capturer:
                user_data.detection_capturer.forget(f"visit:{v.visit_id}")
        if user_data.recorder and not user_data.recorder.add(frame_detections):
            print(f"[visits] Recording saved: {user_data.recorder.path}", flush=True)
            user_data.recorder = None
        if capture_cmd:
            _handle_capture_command(element, buffer, user_data, capture_cmd, frame_detections)

        # Feed into session tracker (with tracker IDs)
        if user_data.feeding_tracker:
            user_data.feeding_tracker.update(detected_cats_with_ids)

        # ── Periodic pipeline heartbeat ──────────────────────────────
        if user_data.frame_count % 300 == 0:
            n_det = len(frame_detections)
            print(f"[pipeline] Frame {user_data.frame_count}: {n_det} detections, "
                  f"{len(visits)} visit(s)", flush=True)

        # ── Classifier subprocess health check ─────────────────────────
        if (user_data.classifier_process is not None
                and user_data.frame_count % 150 == 0  # ~every 5s at 30fps
                and not user_data.classifier_process.is_alive()):
            print_warning(f"Classifier subprocess died "
                          f"(exit code: {user_data.classifier_process.exitcode})")
            user_data.classifier_process = None

        # ── Classifier feeder (Stage 2) ────────────────────────────────
        # One read per frame: the classifier can be switched off (feeder set
        # to None) from another thread at any moment.
        clf_gate = user_data.classifier_gate
        to_classify = clf_gate.update(visits, departed) if clf_gate else []
        needs_classify = bool(to_classify)

        # ── Decide what needs a frame BEFORE extracting (extraction is expensive) ──
        from deploy.detection_capturer import FrameExtractor

        needs_preview = (user_data.preview_writer
                         and user_data.preview_writer.should_write())

        needs_capture = False
        if user_data.detection_capturer and detected_cats:
            if user_data.capture_mode == "detections":
                # Some stable visit in the region is due an image (cheap check
                # with the last status read; the capture reads it again).
                needs_capture = bool(_capture_targets(user_data, visits, user_data._status))
            elif user_data.capture_mode == "eating" and user_data.feeding_tracker:
                from deploy.feeding_tracker import CatState
                for cat in detected_cats:
                    if user_data.feeding_tracker.get_cat_state(cat) == CatState.EATING:
                        needs_capture = True
                        break

        # Extract frame AT MOST ONCE, shared between preview, capture, and classify
        record_frame_kind = (user_data.recorder.frame_due(bool(visits))
                             if user_data.recorder else None)
        needs_record_frame = record_frame_kind is not None

        frame = None
        if needs_preview or needs_capture or needs_classify or needs_record_frame:
            frame = FrameExtractor.extract_frame(buffer, element)

        # A frame for the detection recording, picked by the clock (before the
        # preview writer draws on it)
        if frame is not None and needs_record_frame:
            cv2.imwrite(str(user_data.recorder.frame_path(record_frame_kind)), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Write preview frame with bounding boxes (async — non-blocking)
        if frame is not None and needs_preview:
            # The dashboard reads in_roi, visit_id and name (set by the visit
            # tracker and label_visits) to follow each animal.
            # A stable visit the detector lost for a moment keeps its bounding
            # box on the live view (drawn thin), so it doesn't vanish and return.
            shown = frame_detections + held_detections(visits)
            label_visits(shown, visits, user_data._status, user_data.human_names,
                         identifying=clf_gate is not None)
            user_data.preview_writer.enqueue_frame(frame, shown)

        # Classifier crops (using shared frame), one per visit due
        if frame is not None:
            for v in to_classify:
                clf_gate.put_crop(frame, v)

        # Detection-triggered capture (using shared frame)
        if frame is not None and needs_capture:
            if user_data.capture_mode == "eating" and user_data.feeding_tracker:
                _maybe_capture_eating_cats(user_data, detected_cats, frame, frame_detections)
            elif user_data.capture_mode == "detections":
                _maybe_capture_detections(user_data, detected_cats, frame, frame_detections, visits)

        # Classifier status (names on the live view, naming captures): read at
        # most twice a second, and only while a visit is open or captures are
        # waiting to be named.
        if clf_gate:
            now = time.time()
            if (now - user_data._status_last_poll >= 0.5
                    and (visits or user_data.backlabeler)):
                user_data._status_last_poll = now
                user_data._status = read_classifier_status()
                if user_data.detection_capturer and user_data.capture_mode == "detections":
                    _apply_backlabels(user_data, now)
        else:
            user_data._status = None


    def _handle_capture_command(element, buffer, user_data: CatDetectionCallback, cmd: dict,
                               frame_detections: list[dict] | None = None):
        """Handle a manual capture command from the dashboard via stdin.

        Always saves directly to manual_capture_dir, bypassing DetectionCapturer's
        automatic throttle which is only appropriate for detection-triggered captures.
        Writes a JSON sidecar with current frame detections for crop-on-dashboard.
        With "require_cat" (auto-capture), saves only when a tracked animal is
        in the bowl area, so empty frames aren't saved under the cat's name.
        """
        from deploy.detection_capturer import FrameExtractor

        # The animal the image is of: a named visit's own (saved only while that
        # visit is in the region), else the only visit in progress, if one.
        dets = frame_detections or []
        visit_id = cmd.get("visit_id")
        subject = None
        if visit_id is not None:
            v = user_data.visits.get(visit_id) if user_data.visits else None
            if v is None or not v.seen:
                return
            subject = v.detection
        else:
            if cmd.get("require_cat") and not any(d.get("in_roi") for d in dets):
                return
            in_progress = user_data.visits.visits() if user_data.visits else []
            if len(in_progress) == 1 and in_progress[0].seen:
                subject, visit_id = in_progress[0].detection, in_progress[0].visit_id
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
            if dets:
                H, W = frame.shape[:2]
                rounded = lambda ds: [
                    {"label": d["label"], "confidence": round(float(d["confidence"]), 4),
                     "bbox_norm": [round(float(x), 6) for x in d["bbox_norm"]]} for d in ds]
                sidecar = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": "",
                    "image_size": [W, H],
                }
                if subject is not None:
                    sidecar["detections"] = rounded([subject])
                    sidecar["visit_id"] = visit_id
                    sidecar["other_detections"] = rounded([d for d in dets if d is not subject])
                else:
                    sidecar["detections"] = rounded(dets)
                try:
                    with open(filepath.with_suffix(".json"), "w") as f:
                        _json.dump(sidecar, f)
                except Exception:
                    pass
            print(json.dumps({"response": "capture_ok", "cat": cat_name, "filename": filename,
                              "visit_id": visit_id}), flush=True)
        else:
            print(json.dumps({"response": "capture_error", "message": "imwrite failed"}),
                  flush=True)


    def _capture_targets(user_data, visits, status) -> list[tuple]:
        """The visits to save an image of in this frame: (visit, label,
        backlabel_eligible, throttle_key). One image per animal: each stable
        visit seen in this frame, when its throttle allows and save_which
        doesn't skip it."""
        capturer = user_data.detection_capturer
        gate = user_data.classifier_gate
        out, keys = [], set()
        for v in visits or []:
            if not (v.seen and v.stable):
                continue
            classified = _is_classified(user_data, v.label)
            key = _throttle_key(user_data, v.label, v.visit_id)
            if key in keys or not capturer.should_capture(key):
                continue
            label, eligible = v.label, False
            if classified:
                confirmed, eligible, skip = capture_plan(
                    status, v.visit_id, gate is not None, user_data.save_which)
                if skip:
                    continue
                label = confirmed or "unlabeled"
            keys.add(key)
            out.append((v, label, eligible and classified and label == "unlabeled", key))
        return out


    def _sidecar_dets(dets) -> list[dict]:
        return [{"label": d["label"], "confidence": d["confidence"],
                 "bbox_norm": list(d["bbox_norm"])} for d in dets]


    def _maybe_capture_detections(user_data, detected_cats, frame, frame_detections=None,
                                  visits=None):
        """Save one image per animal in the region (no feeding tracker required).

        Each stable visit gets its own image, named for its animal: for
        unlabeled_classes labels the name comes from capture_plan() (the name
        the classifier gave that visit, or unlabeled/; undecided ones are
        registered for back-labeling), other labels keep the detector's label.
        The sidecar's "detections" holds only that animal's bounding box (the
        one training crops); the other bounding boxes in the region go under
        "other_detections". save_which "identified" skips unlabeled images
        that can never be named; "unidentified" skips named ones.
        """
        if user_data.classifier_gate:
            user_data._status = read_classifier_status()
        in_region = [d for d in (frame_detections or []) if d.get("in_roi")]
        for v, label, eligible, key in _capture_targets(user_data, visits, user_data._status):
            subject = v.detection
            if subject is None:
                continue
            others = [d for d in in_region if d is not subject]
            saved = user_data.detection_capturer.maybe_capture(
                label, frame, detections=_sidecar_dets([subject]), throttle_key=key,
                extra={"visit_id": v.visit_id, "other_detections": _sidecar_dets(others)})
            if saved:
                print(f"[capture] Saved as '{label}' (visit {v.visit_id:.3f})", flush=True)
                if eligible:
                    user_data.backlabeler.add(v.visit_id, saved)


    def _is_classified(user_data, cat: str) -> bool:
        """True for detector labels whose captures are named by the classifier."""
        return bool(user_data.unlabeled_classes) and cat in user_data.unlabeled_classes


    def _throttle_key(user_data, cat: str, visit_id: float | None = None) -> str:
        """What the minimum time between images applies to: each animal (visit)
        for labels the classifier names, whatever name its images get; the
        label for the others."""
        if not _is_classified(user_data, cat):
            return cat
        return f"visit:{visit_id}" if visit_id is not None else CLASSIFIED_THROTTLE_KEY


    def _apply_backlabels(user_data, now: float) -> None:
        """Name the captures of visits the classifier has confirmed; with
        save_which "identified", delete those of visits that ended unconfirmed
        ("unidentified" deletes the confirmed ones instead)."""
        if not user_data.backlabeler:
            return
        capturer = user_data.detection_capturer
        for paths, cat in user_data.backlabeler.poll(confirmed_visits(user_data._status)):
            if user_data.save_which == "unidentified":
                capturer.delete(paths)      # it was identified after all
                continue
            capturer.relabel(paths, cat)
            print(f"[capture] Visit confirmed as '{cat}': labeling "
                  f"{len(paths)} earlier capture(s)", flush=True)
        if user_data.classifier_gate is None:      # switched off meanwhile
            return
        current = {v.visit_id: v.last_seen for v in user_data.visits.visits()}
        expired = user_data.backlabeler.expire(current, now)
        if expired and user_data.save_which == "identified":
            capturer.delete(expired)


    def _flush_backlabels(user_data) -> None:
        """Before the capturer stops: name what is already confirmed, then drop the
        rest (deleted under save_which "identified", otherwise left in unlabeled/)."""
        capturer = user_data.detection_capturer
        if not capturer or not user_data.backlabeler:
            return
        if user_data.classifier_gate:
            user_data._status = read_classifier_status()
            for paths, cat in user_data.backlabeler.poll(confirmed_visits(user_data._status)):
                if user_data.save_which == "unidentified":
                    capturer.delete(paths)
                else:
                    capturer.relabel(paths, cat)
        leftover = user_data.backlabeler.drain()
        if leftover and user_data.save_which == "identified":
            capturer.delete(leftover)


    def _maybe_capture_eating_cats(user_data, detected_cats, frame, frame_detections=None):
        """Capture a frame for cats confirmed EATING.

        Args:
            user_data: CatDetectionCallback with feeding_tracker + detection_capturer
            detected_cats: set of detected label names
            frame: Pre-extracted BGR numpy array (shared with preview writer)
            frame_detections: list of detection dicts (label, confidence, bbox_norm)
        """
        from deploy.feeding_tracker import CatState

        gate = user_data.classifier_gate
        confirmed = (classifier_capture_label(read_classifier_status(), gate.visit_start)[0]
                     if gate else None)
        for cat in detected_cats:
            state = user_data.feeding_tracker.get_cat_state(cat)
            if state != CatState.EATING:
                continue
            session_id = user_data.feeding_tracker.get_session_id(cat)
            save_label = confirmed if confirmed else cat
            det_data = None
            if frame_detections:
                det_data = [
                    {"label": d["label"], "confidence": d["confidence"],
                     "bbox_norm": list(d["bbox_norm"])}
                    for d in frame_detections if d["label"] == cat
                ] or None
            user_data.detection_capturer.maybe_capture(save_label, frame, session_id,
                                                       detections=det_data, throttle_key=cat)


    def _handle_set_detection_capture(user_data: CatDetectionCallback, cmd: dict):
        """Handle live toggle of detection capture from the dashboard."""
        enabled = cmd.get("enabled", False)
        interval = cmd.get("min_interval_seconds", 15.0)
        save_which = save_which_from(cmd)

        if enabled:
            if not user_data.detection_capturer:
                from deploy.detection_capturer import DetectionCapturer

                cfg = load_config()
                capture_cfg = cfg.get("detection_capture", {})

                capture_dir = resolve_path(
                    capture_cfg.get("output_dir", "data/raw"),
                    PROJECT_ROOT,
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
            else:
                user_data.detection_capturer.min_interval = interval

            user_data.capture_mode = "detections"
            user_data.save_which = save_which
            print_info(f"Detection capture enabled (interval={interval}s, saving {save_which})")
        else:
            if user_data.detection_capturer:
                _flush_backlabels(user_data)
                user_data.detection_capturer.shutdown()
                user_data.detection_capturer = None
            user_data.capture_mode = None
            print_info("Detection capture disabled")


def _start_classifier(user_data, cfg: dict, ctx) -> None:
    """Start the classifier subprocess and the gate that feeds it crops.

    ctx is the multiprocessing context: "fork" at startup, before GStreamer
    starts its threads; "spawn" afterwards, because forking a process that
    runs threads is unsafe (spawn starts a fresh interpreter instead).
    """
    classifier_cfg = cfg.get("classifier", {})
    CLASSIFIER_STATUS_PATH.unlink(missing_ok=True)      # stale status of an earlier run

    from deploy.cat_classifier import classifier_worker
    resolved = {**classifier_cfg, "cats": cfg["cats"]}
    for key in ("model_path", "db_path"):
        if key in resolved:
            resolved[key] = str(resolve_path(resolved[key], PROJECT_ROOT))
    q = ctx.Queue(maxsize=16)
    proc = ctx.Process(target=classifier_worker, args=(q, resolved), daemon=True,
                       name="cat_classifier")
    proc.start()
    print_info(f"Classifier process started (pid={proc.pid})")
    gate = VisitFeeder(
        queue=q,
        classify_interval=classifier_cfg.get("classify_interval_seconds", 1.0),
        crop_padding=classifier_cfg.get("crop_padding", 0.20),
    )
    # Publish the gate last: the frame callback starts feeding it at once.
    user_data.classifier_queue = q
    user_data.classifier_process = proc
    user_data.classifier_gate = gate
    print_info("Classifier feeder active")


def _stop_classifier(user_data) -> None:
    """Stop feeding the classifier, let it close the current visit, end it."""
    proc, q = user_data.classifier_process, user_data.classifier_queue
    if proc is None:
        return
    user_data.classifier_gate = None          # the frame callback stops at once
    user_data.classifier_process = None
    user_data.classifier_queue = None
    if proc and proc.is_alive():
        try:
            q.put(None, timeout=1)            # sentinel: drop an unfinished visit, exit
        except Exception:
            pass
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
    print_info("Classifier stopped")


_classifier_switch_lock = threading.Lock()


def _switch_classifier(user_data, restart: bool) -> None:
    """Apply the classifier settings in the (re-read) config without touching
    the camera: on, off, or restarted with a new model or certainty. Runs in
    a thread so the frame callback never waits for it."""
    def work():
        with _classifier_switch_lock:
            cfg = user_data.reload_config()
            enabled = bool(cfg.get("classifier", {}).get("enabled"))
            running = user_data.classifier_process is not None
            if running and (restart or not enabled):
                _stop_classifier(user_data)
                running = False
            if enabled and not running:
                _start_classifier(user_data, cfg, multiprocessing.get_context("spawn"))
            elif not enabled:
                CLASSIFIER_STATUS_PATH.unlink(missing_ok=True)
    threading.Thread(target=work, daemon=True, name="classifier-switch").start()


def _disable_video_window() -> None:
    """Make hailo-apps' detection pipeline end in a fakesink instead of its
    display branch (overlay, videoconvert and a video window). Must run
    before GStreamerDetectionApp is created, which builds the pipeline."""
    import hailo_apps.python.pipeline_apps.detection.detection_pipeline as det_pipeline
    # Not named hailo_display: hailo-apps hooks an FPS signal onto that name.
    det_pipeline.DISPLAY_PIPELINE = (
        lambda *args, **kwargs: "fakesink name=no_display sync=false async=false")
    print_info("No screen: video window off")


def _set_camera_mirror(mirror: bool) -> None:
    """hailo-apps mirrors camera input by default, like a selfie camera; a
    feeder camera should show the room the way it is. Sets the source's
    mirror_image to camera.mirror instead. Must run before
    GStreamerDetectionApp is created, which builds the pipeline."""
    import hailo_apps.python.pipeline_apps.detection.detection_pipeline as det_pipeline
    source_pipeline = det_pipeline.SOURCE_PIPELINE
    det_pipeline.SOURCE_PIPELINE = (
        lambda *args, **kwargs: source_pipeline(*args, **{**kwargs, "mirror_image": mirror}))
    print_info(f"Camera picture: {'mirrored' if mirror else 'not mirrored'}")


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

    # Populate per-cat bbox colors from config
    _BBOX_COLORS.update(get_cat_colors_bgr(cfg.get("cats", [])))

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

    tracking_objects = deploy_cfg.get("tracking_objects", ["cat"])
    user_data.set_filter(tracking_objects)
    print_info(f"Filtering for: {tracking_objects}")

    track_all = deploy_cfg.get("track_all", True)
    tracker_class_id = -1 if track_all else 0
    print_info(f"Tracker class-id: {tracker_class_id} (-1 = all classes)")

    # Setup preview writer
    if args.enable_preview:
        # In RAM (utils/runtime_paths.py): rewritten preview_fps times a second
        runtime_dir()
        preview_path = str(PREVIEW_JPG)
        preview_fps = deploy_cfg.get("preview_fps", 10)
        user_data.preview_writer = PreviewFrameWriter(
            preview_path=preview_path,
            detections_path=str(PREVIEW_DETECTIONS_JSON),
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

    # ── ROI config (always loaded — used by detection capture even without classifier) ──
    classifier_cfg = cfg.get("classifier", {})
    user_data.bowl_roi = tuple(classifier_cfg.get("bowl_roi", [0, 0, 1920, 1080]))
    user_data.roi_overlap_threshold = classifier_cfg.get("roi_overlap_threshold", 0.01)
    user_data.stability_frames = classifier_cfg.get("stability_frames", 5)
    user_data.frame_w = camera_cfg.get("width", 1920)
    user_data.frame_h = camera_cfg.get("height", 1080)
    # Visits: each animal in the region, followed by position (track IDs
    # change constantly). Runs with identification on or off.
    user_data.visits = VisitTracker(
        labels=set(tracking_objects),
        roi=user_data.bowl_roi,
        frame_w=user_data.frame_w,
        frame_h=user_data.frame_h,
        overlap_threshold=user_data.roi_overlap_threshold,
        stability_frames=user_data.stability_frames,
        gap_seconds=visit_gap_from(classifier_cfg, camera_cfg),
        hold_seconds=visit_hold_from(classifier_cfg),
    )
    print_info(f"Visits: up to {user_data.visits.max_visits} at a time, end after "
               f"{user_data.visits.gap_seconds:.1f} s out of the region, or "
               f"{user_data.visits.hold_seconds:.1f} s for an animal back in the same place")

    # ── Classifier setup (BEFORE GStreamer init — fork safety) ───────────
    # Later switches (dashboard: identification on/off, a new model, a new
    # certainty) restart only the classifier, never the camera; see
    # _switch_classifier.
    user_data.reload_config = lambda: get_config_with_args(args)
    if classifier_cfg.get("enabled", False):
        _start_classifier(user_data, cfg, multiprocessing.get_context("fork"))

    # Without a screen (the systemd service has none), hailo-apps' video window
    # was still drawn, with OpenGL on the CPU: a whole core for a picture nobody
    # sees, and it held the callback (and so the visits) to about 8 frames a
    # second. Send the frames to a sink that drops them instead.
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        _disable_video_window()
    _set_camera_mirror(bool(camera_cfg.get("mirror", False)))

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
        interval = capture_cfg.get("min_interval_seconds", 15.0)
        user_data.save_which = save_which_from(capture_cfg)
        user_data.detection_capturer = DetectionCapturer(
            output_dir=capture_dir,
            cats=cfg.get("cats", []),
            min_interval_seconds=interval,
            max_captures_per_session=capture_cfg.get("max_captures_per_session", 0),
            jpeg_quality=capture_cfg.get("jpeg_quality", 95),
            csv_logging=capture_cfg.get("csv_logging", True),
        )

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
        _stop_classifier(user_data)

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
            _flush_backlabels(user_data)
            user_data.detection_capturer.print_stats()
            user_data.detection_capturer.shutdown()


if __name__ == "__main__":
    main()
