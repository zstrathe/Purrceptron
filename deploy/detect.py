#!/usr/bin/env python3
"""Cat detection inference on Raspberry Pi 5 + Hailo 10H.

Wraps the Hailo GStreamer detection pipeline with custom class filtering
and tracker configuration for individual cat identification.

All settings are read from project.yaml. No custom CLI flags required.

Usage:
    python deploy/detect.py                        # reads configs/project.yaml
    python deploy/detect.py --config alt.yaml      # use alternate config
    python deploy/detect.py --input /dev/video2    # override a Hailo arg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import PROJECT_ROOT, get_config_with_args, resolve_path, add_common_args
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
            self.unlabeled_classes: set[str] = set()  # COCO classes to save as "unlabeled"

        def set_filter(self, labels: list[str]) -> None:
            self.filter_labels = set(labels)


    def app_callback(pad, info, user_data: CatDetectionCallback):
        """Process each detection frame.

        Args:
            pad: GStreamer pad
            info: GStreamer probe info (access buffer via info.get_buffer())
            user_data: CatDetectionCallback instance
        """
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        user_data.increment()  # increment frame count
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        # Collect detected cat names this frame
        detected_cats: set[str] = set()

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

            if user_data.get_count() % 30 == 0:  # Print every 30 frames to avoid spam
                print_info(f"Frame {user_data.get_count()}: {label} ({confidence:.2f}) track={track_id}")

        # Feed into session tracker
        if user_data.feeding_tracker:
            user_data.feeding_tracker.update(detected_cats)

        # Detection-triggered capture
        if user_data.detection_capturer and detected_cats:
            if user_data.capture_mode == "eating" and user_data.feeding_tracker:
                _maybe_capture_eating_cats(pad, buffer, user_data, detected_cats)
            elif user_data.capture_mode == "detections":
                _maybe_capture_detections(pad, buffer, user_data, detected_cats)

        return Gst.PadProbeReturn.OK


    def _maybe_capture_detections(pad, buffer, user_data, detected_cats):
        """Capture a frame for each detected class (no feeding tracker required)."""
        from deploy.detection_capturer import FrameExtractor

        frame = None  # Lazy extraction

        for cat in detected_cats:
            save_label = cat
            if user_data.unlabeled_classes and cat in user_data.unlabeled_classes:
                save_label = "unlabeled"

            if frame is None:
                frame = FrameExtractor.extract_frame(buffer, pad)
                if frame is None:
                    return

            user_data.detection_capturer.maybe_capture(save_label, frame)


    def _maybe_capture_eating_cats(pad, buffer, user_data, detected_cats):
        """Check if any detected cats are EATING and capture a frame if throttle allows."""
        from deploy.feeding_tracker import CatState
        from deploy.detection_capturer import FrameExtractor

        frame = None  # Lazy extraction: only extract once if needed

        for cat in detected_cats:
            state = user_data.feeding_tracker.get_cat_state(cat)
            if state != CatState.EATING:
                continue

            if frame is None:
                frame = FrameExtractor.extract_frame(buffer, pad)
                if frame is None:
                    return

            session_id = user_data.feeding_tracker.get_session_id(cat)
            user_data.detection_capturer.maybe_capture(cat, frame, session_id)


def main():
    if not HAILO_AVAILABLE:
        print_error("Hailo libraries not available.")
        print_error(f"Import error: {HAILO_IMPORT_ERROR}")
        print_info("This script must run on the Raspberry Pi inside the hailo-apps venv:")
        print_info("  source ~/hailo-apps/venv_hailo_apps/bin/activate")
        sys.exit(1)

    # Only --config is ours; everything else passes through to Hailo
    parser = argparse.ArgumentParser(
        description="Cat detection with Hailo 10H",
        add_help=False,
    )
    add_common_args(parser)
    args, remaining_argv = parser.parse_known_args()

    # Load config
    cfg = get_config_with_args(args)
    deploy_cfg = cfg.get("deployment", {})
    camera_cfg = cfg.get("camera", {})
    capture_cfg = cfg.get("detection_capture", {})
    feeding_cfg = cfg.get("feeding", {})

    # Build Hailo sys.argv from config; CLI args in remaining_argv override
    hef_path = str(resolve_path(deploy_cfg["hef_path"], PROJECT_ROOT))
    labels_path = str(resolve_path(deploy_cfg["labels_json"], PROJECT_ROOT))
    camera_device = camera_cfg.get("device", "/dev/video0")

    sys.argv = [
        sys.argv[0],
        "--hef-path", hef_path,
        "--input", camera_device,
        "--labels-json", labels_path,
    ] + remaining_argv

    # Setup callback
    user_data = CatDetectionCallback()

    filter_classes = deploy_cfg.get("filter_classes", [])
    if filter_classes:
        user_data.set_filter(filter_classes)
        print_info(f"Filtering for: {filter_classes}")

    track_all = deploy_cfg.get("track_all", True)
    tracker_class_id = -1 if track_all else 0
    print_info(f"Tracker class-id: {tracker_class_id} (-1 = all classes)")

    # Create the detection app (Hailo parses sys.argv for --hef-path, --input, etc.)
    app = GStreamerDetectionApp(app_callback, user_data)

    # Modify tracker AFTER pipeline construction but BEFORE app.run()
    tracker = app.pipeline.get_by_name("hailo_tracker")
    if tracker:
        tracker.set_property("class-id", tracker_class_id)
        print_info(f"Tracker configured: class-id={tracker_class_id}")
    else:
        print_warning("hailo_tracker element not found in pipeline")

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
        user_data.detection_capturer = DetectionCapturer(
            output_dir=capture_dir,
            cats=cfg.get("cats", []),
            min_interval_seconds=capture_cfg.get("min_interval_seconds", 15.0),
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
    app.run()

    # Finalize any active sessions on shutdown
    if user_data.feeding_tracker:
        user_data.feeding_tracker.shutdown()

    # Shutdown capturer after feeding tracker (so final sessions are logged first)
    if user_data.detection_capturer:
        user_data.detection_capturer.print_stats()
        user_data.detection_capturer.shutdown()


if __name__ == "__main__":
    main()
