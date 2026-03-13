#!/usr/bin/env python3
"""Camera capture classes for cat data collection.

Provides CameraThread, SaveWorker, and CaptureApp classes for use by the
unified dashboard (pi_dashboard.py).
"""
from __future__ import annotations

import collections
import csv
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, resolve_path
from utils.display import print_info, print_success, print_error, print_warning, print_header


# ═══════════════════════════════════════════════════════════════════════════
# Camera Thread
# ═══════════════════════════════════════════════════════════════════════════

class CameraThread:
    """Continuously reads frames from a camera in a background daemon thread."""

    def __init__(self, device: str, width: int, height: int, fps: int,
                 fourcc: str, preview_w: int, preview_h: int):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.preview_size = (preview_w, preview_h)

        self._frame: np.ndarray | None = None
        self._display_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # FPS tracking
        self._timestamps: collections.deque[float] = collections.deque(maxlen=30)
        self.fps_value: float = 0.0

        # Flash effect (set externally, decremented each frame)
        self.flash_frames_remaining: int = 0

        # Overlay text set by CaptureApp
        self.overlay_mode: str = "MANUAL"
        self.overlay_counts: str = ""

    # ------------------------------------------------------------------
    def _open_camera(self) -> cv2.VideoCapture | None:
        """Try to open the camera device, retrying up to 5 times."""
        for attempt in range(5):
            # Try as path first, then as integer index
            dev: str | int = self.device
            try:
                dev = int(self.device)
            except (ValueError, TypeError):
                pass

            cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                if self.fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc(*self.fourcc))
                return cap

            print_warning(f"Camera open attempt {attempt + 1}/5 failed, retrying in 1s...")
            cap.release()
            time.sleep(1)

        return None

    # ------------------------------------------------------------------
    def _run(self) -> None:
        cap = self._open_camera()
        if cap is None:
            print_error(f"Could not open camera device: {self.device}")
            self._running = False
            return

        print_success(f"Camera opened: {self.device} "
                      f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
                      f"{int(cap.get(cv2.CAP_PROP_FPS))} fps)")

        while self._running:
            ok, frame = cap.read()
            if not ok:
                print_warning("Frame read failed, retrying...")
                time.sleep(0.05)
                continue

            now = time.time()
            self._timestamps.append(now)
            if len(self._timestamps) > 1:
                self.fps_value = (len(self._timestamps) - 1) / (
                    self._timestamps[-1] - self._timestamps[0]
                )

            # Build display frame
            display = cv2.resize(frame, self.preview_size)
            self._draw_overlay(display)

            with self._lock:
                self._frame = frame
                self._display_frame = display

        cap.release()

    # ------------------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray) -> None:
        """Draw semi-transparent status overlay on the preview frame."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)

        # Mode text (top-left)
        mode_color = (0, 255, 0) if "MANUAL" in self.overlay_mode else (0, 180, 255)
        cv2.putText(frame, self.overlay_mode, (10, 28),
                    font, 0.6, mode_color, 2)

        # FPS (top-right)
        fps_text = f"FPS: {self.fps_value:.1f}"
        cv2.putText(frame, fps_text, (w - 130, 28), font, 0.6, (200, 200, 200), 2)

        # Counts (bottom bar)
        if self.overlay_counts:
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - 34), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.4, frame, 0.6, 0, dst=frame)
            cv2.putText(frame, self.overlay_counts, (10, h - 10),
                        font, 0.5, (200, 200, 200), 1)

        # Flash border effect
        if self.flash_frames_remaining > 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 8)
            self.flash_frames_remaining -= 1

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_display_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._display_frame.copy() if self._display_frame is not None else None


# ═══════════════════════════════════════════════════════════════════════════
# Save Worker
# ═══════════════════════════════════════════════════════════════════════════

class SaveWorker:
    """Background thread that writes captured frames to disk."""

    def __init__(self, maxsize: int = 64):
        self._queue: queue.Queue[tuple[np.ndarray, Path, int] | None] = queue.Queue(
            maxsize=maxsize
        )
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame, filepath, quality = item
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(filepath), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
            except Exception as exc:
                print_error(f"Save failed {filepath.name}: {exc}")
            self._queue.task_done()

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, frame: np.ndarray, filepath: Path, quality: int) -> None:
        self._queue.put((frame, filepath, quality))

    def drain(self) -> None:
        """Wait for all pending saves to finish, then signal the thread to exit."""
        self._queue.join()
        self._queue.put(None)
        self._thread.join(timeout=5)


# ═══════════════════════════════════════════════════════════════════════════
# Capture Application State
# ═══════════════════════════════════════════════════════════════════════════

class CaptureApp:
    """Holds all mutable application state for the capture session."""

    def __init__(self, config: dict, camera: CameraThread, saver: SaveWorker):
        self.config = config
        self.camera = camera
        self.saver = saver

        self.cats: list[str] = config["cats"]
        self.output_dir: Path = resolve_path(
            config["collection"]["output_dir"], PROJECT_ROOT
        )
        self.jpeg_quality: int = config["collection"]["jpeg_quality"]
        self.auto_interval: float = config["collection"]["auto_capture_interval_sec"]
        self.min_images: int = config["collection"]["min_images_per_cat"]

        self.mode: str = "manual"
        self.auto_target_cat: str = self.cats[0]
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time: float = time.time()

        # Counts
        self.counts: dict[str, int] = {c: 0 for c in self.cats}
        self.session_counts: dict[str, int] = {c: 0 for c in self.cats}
        self._count_existing_images()

        # CSV log
        self.csv_path = self.output_dir / "capture_log.csv"
        self._csv_file = None
        self._csv_writer = None
        self._open_csv()

        # Auto-capture timing
        self.last_auto_capture_time: float = 0.0

        self._update_overlay()

    # ------------------------------------------------------------------
    def _count_existing_images(self) -> None:
        for cat in self.cats:
            cat_dir = self.output_dir / cat
            if cat_dir.is_dir():
                self.counts[cat] = len(list(cat_dir.glob("*.jpg")))

    def _open_csv(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.exists()
        self._csv_file = open(self.csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow(["filename", "cat", "timestamp", "session_id"])

    def _update_overlay(self) -> None:
        """Push current state to the camera overlay."""
        if self.mode == "auto":
            self.camera.overlay_mode = f"AUTO: {self.auto_target_cat}"
        else:
            self.camera.overlay_mode = "MANUAL"
        parts = [f"{c}: {self.counts[c]}" for c in self.cats]
        self.camera.overlay_counts = "  ".join(parts)

    # ------------------------------------------------------------------
    def capture(self, cat_name: str) -> dict:
        if cat_name not in self.cats:
            return {"status": "error", "message": f"Unknown cat: {cat_name}"}

        frame = self.camera.get_frame()
        if frame is None:
            return {"status": "error", "message": "No frame available"}

        ts = datetime.now()
        filename = f"{cat_name}_{ts.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        filepath = self.output_dir / cat_name / filename

        self.saver.enqueue(frame, filepath, self.jpeg_quality)

        # Log to CSV
        if self._csv_writer:
            self._csv_writer.writerow([
                filename, cat_name, ts.isoformat(), self.session_id
            ])
            self._csv_file.flush()

        self.counts[cat_name] += 1
        self.session_counts[cat_name] += 1

        # Trigger flash
        self.camera.flash_frames_remaining = 15

        self._update_overlay()

        return {
            "status": "ok",
            "cat": cat_name,
            "filename": filename,
            "count": self.counts[cat_name],
        }

    def toggle_mode(self) -> dict:
        self.mode = "auto" if self.mode == "manual" else "manual"
        self.last_auto_capture_time = time.time()
        self._update_overlay()
        return {"status": "ok", "mode": self.mode, "auto_target": self.auto_target_cat}

    def set_auto_target(self, cat_name: str) -> dict:
        if cat_name not in self.cats:
            return {"status": "error", "message": f"Unknown cat: {cat_name}"}
        self.auto_target_cat = cat_name
        self._update_overlay()
        return {"status": "ok", "auto_target": self.auto_target_cat}

    def get_status(self) -> dict:
        return {
            "mode": self.mode,
            "auto_target": self.auto_target_cat,
            "counts": dict(self.counts),
            "session_counts": dict(self.session_counts),
            "session_id": self.session_id,
            "fps": self.camera.fps_value,
        }

    def auto_capture_tick(self) -> None:
        if self.mode != "auto":
            return
        now = time.time()
        if now - self.last_auto_capture_time >= self.auto_interval:
            self.capture(self.auto_target_cat)
            self.last_auto_capture_time = now

    def close(self) -> None:
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    def print_summary(self) -> None:
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        print_header("Session Summary")
        print(f"  Session:  {self.session_id}")
        print(f"  Duration: {mins}m {secs:02d}s\n")

        print("  This session:")
        for cat in self.cats:
            total = self.counts[cat]
            sess = self.session_counts[cat]
            print(f"    {cat:8s} +{sess:<4d} (total: {total})")

        print(f"\n  Progress toward {self.min_images} minimum:")
        for cat in self.cats:
            total = self.counts[cat]
            pct = min(100, total * 100 // self.min_images)
            filled = pct // 10
            bar = "\u2588" * filled + "\u2591" * (10 - filled)
            print(f"    {cat:8s} {bar} {total}/{self.min_images} ({pct}%)")
        print()


