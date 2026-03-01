#!/usr/bin/env python3
"""Detection-triggered image capture — saves frames when cats are confirmed eating."""
from __future__ import annotations

import csv
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.display import print_error, print_info


class FrameExtractor:
    """Extract numpy frames from GStreamer buffers."""

    @staticmethod
    def extract_frame(buffer, pad) -> np.ndarray | None:
        """Convert a GStreamer buffer to a numpy BGR array.

        Args:
            buffer: Gst.Buffer from info.get_buffer()
            pad: Gst.Pad from the probe callback (for video caps)

        Returns:
            numpy array in BGR format (OpenCV-compatible), or None on failure.
        """
        from gi.repository import Gst

        caps = pad.get_current_caps()
        if caps is None:
            return None

        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        fmt = structure.get_value("format")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            if fmt == "RGB":
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data,
                )
                frame = frame[:, :, ::-1].copy()
            elif fmt == "BGR":
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data,
                ).copy()
            elif fmt == "NV12":
                yuv = np.ndarray(
                    shape=(int(height * 1.5), width),
                    dtype=np.uint8,
                    buffer=map_info.data,
                )
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                return None
            return frame
        finally:
            buffer.unmap(map_info)


class DetectionCapturer:
    """Capture and save frames when cats are confirmed eating.

    Throttles captures per-cat to avoid near-identical frames.
    Saves are performed asynchronously via a background thread.
    """

    def __init__(
        self,
        output_dir: Path,
        cats: list[str],
        min_interval_seconds: float = 15.0,
        max_captures_per_session: int = 0,
        jpeg_quality: int = 95,
        csv_logging: bool = True,
        queue_size: int = 32,
    ):
        self.output_dir = output_dir
        self.cats = cats
        self.min_interval = min_interval_seconds
        self.max_per_session = max_captures_per_session
        self.jpeg_quality = jpeg_quality

        # Per-cat throttle state
        self._last_capture_time: dict[str, float] = {cat: 0.0 for cat in cats}
        self._session_capture_count: dict[str, int] = {cat: 0 for cat in cats}
        self._active_session_ids: dict[str, str | None] = {cat: None for cat in cats}

        # Async save queue
        self._queue: queue.Queue[tuple[np.ndarray, Path, int] | None] = queue.Queue(
            maxsize=queue_size
        )
        self._thread = threading.Thread(target=self._save_loop, daemon=True)
        self._lock = threading.Lock()

        # CSV log
        self._csv_file = None
        self._csv_writer = None
        if csv_logging:
            self._open_csv()

        # Stats
        self.total_captured: int = 0
        self.total_skipped_throttle: int = 0
        self.total_skipped_limit: int = 0

    def start(self) -> None:
        """Start the background save thread."""
        self._thread.start()

    def maybe_capture(
        self,
        cat_name: str,
        frame: np.ndarray,
        session_id: str | None = None,
    ) -> bool:
        """Attempt to capture a frame for the given cat.

        Returns True if the frame was enqueued for saving, False if throttled/limited.
        """
        now = time.time()
        with self._lock:
            # Check: is this a new feeding session for this cat?
            if session_id != self._active_session_ids.get(cat_name):
                self._active_session_ids[cat_name] = session_id
                self._session_capture_count[cat_name] = 0

            # Check: max captures per session
            if (
                self.max_per_session > 0
                and self._session_capture_count.get(cat_name, 0) >= self.max_per_session
            ):
                self.total_skipped_limit += 1
                return False

            # Check: throttle interval
            elapsed = now - self._last_capture_time.get(cat_name, 0.0)
            if elapsed < self.min_interval:
                self.total_skipped_throttle += 1
                return False

            # Build filename and path
            ts = datetime.now()
            filename = f"{cat_name}_{ts.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
            filepath = self.output_dir / cat_name / filename

            # Enqueue (non-blocking: drop frame if queue full)
            try:
                self._queue.put_nowait((frame, filepath, self.jpeg_quality))
            except queue.Full:
                return False

            # Update state
            self._last_capture_time[cat_name] = now
            self._session_capture_count[cat_name] = (
                self._session_capture_count.get(cat_name, 0) + 1
            )
            self.total_captured += 1

            # CSV log
            if self._csv_writer:
                self._csv_writer.writerow([
                    filename, cat_name, ts.isoformat(), session_id or ""
                ])
                self._csv_file.flush()

            return True

    def reset_session(self, cat_name: str) -> None:
        """Reset capture count for a cat when its feeding session ends."""
        with self._lock:
            self._session_capture_count[cat_name] = 0
            self._active_session_ids[cat_name] = None

    def _save_loop(self) -> None:
        """Background thread: dequeue frames and write JPEGs."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame, filepath, quality = item
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            except Exception as exc:
                print_error(f"Detection capture save failed {filepath.name}: {exc}")
            self._queue.task_done()

    def _open_csv(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "capture_log.csv"
        file_exists = csv_path.exists()
        self._csv_file = open(csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow(["filename", "cat", "timestamp", "session_id"])

    def shutdown(self) -> None:
        """Drain the save queue and close resources."""
        self._queue.join()
        self._queue.put(None)
        self._thread.join(timeout=5)
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    def print_stats(self) -> None:
        print_info(
            f"Detection captures: {self.total_captured} saved, "
            f"{self.total_skipped_throttle} throttled, "
            f"{self.total_skipped_limit} session-limited"
        )
