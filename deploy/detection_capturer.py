#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Detection-triggered image capture, plus back-labeling of a visit's early captures."""
from __future__ import annotations

import csv
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.display import print_error, print_info


class FrameExtractor:
    """Extract numpy frames from GStreamer buffers."""

    @staticmethod
    def extract_frame(buffer, pad_or_element) -> np.ndarray | None:
        """Convert a GStreamer buffer to a numpy BGR array.

        Args:
            buffer: Gst.Buffer containing frame data
            pad_or_element: Gst.Pad or Gst.Element (identity element from
                            hailo-apps callback). Caps are read from the pad;
                            if an element is passed, its sink pad is used.

        Returns:
            numpy array in BGR format (OpenCV-compatible), or None on failure.
        """
        from gi.repository import Gst

        # Get caps — works with both a Pad and an Element
        caps = None
        if hasattr(pad_or_element, 'get_current_caps'):
            caps = pad_or_element.get_current_caps()
        if caps is None and hasattr(pad_or_element, 'get_static_pad'):
            sink_pad = pad_or_element.get_static_pad("sink")
            if sink_pad:
                caps = sink_pad.get_current_caps()
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

        # Async save queue: (frame, filepath, quality, detections, ts_iso, session_id)
        # or a small control item ("relabel" | "delete", paths, label). The queue is
        # unbounded so control items never block the GStreamer callback; frames are
        # bounded by hand in maybe_capture() (dropped when queue_size are waiting).
        self._queue: queue.Queue[tuple | None] = queue.Queue()
        self._max_queued_frames = queue_size
        # Paths handed out but possibly not written yet (name-collision check)
        self._recent_paths: deque[Path] = deque(maxlen=queue_size * 2)
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

    def should_capture(self, throttle_key: str) -> bool:
        """Check whether the throttle interval for throttle_key has passed.

        Use this to avoid expensive frame extraction when capture would be rejected.
        A capture refused by the per-session cap also restarts the interval, so
        this stays accurate without knowing the label the image would be saved as.
        """
        now = time.time()
        with self._lock:
            elapsed = now - self._last_capture_time.get(throttle_key, 0.0)
            return elapsed >= self.min_interval

    def maybe_capture(
        self,
        cat_name: str,
        frame: np.ndarray,
        session_id: str | None = None,
        detections: list[dict] | None = None,
        throttle_key: str | None = None,
        extra: dict | None = None,
    ) -> Path | None:
        """Attempt to capture a frame for the given cat.

        Returns the path the frame will be saved to, or None if throttled/limited.
        The file is written asynchronously; relabel()/delete() are queued behind it.
        If detections is provided, a JSON sidecar is written alongside the JPEG with
        normalized bounding box coordinates of the triggering detections.
        throttle_key (default: cat_name) is what the minimum interval applies to, so
        one animal at the bowl is throttled once whatever name its images are saved
        under. The per-session cap applies per saved label (cat_name).
        extra is merged into the sidecar (visit_id, other_detections).
        """
        key = throttle_key or cat_name
        now = time.time()
        with self._lock:
            # Check: throttle interval
            elapsed = now - self._last_capture_time.get(key, 0.0)
            if elapsed < self.min_interval:
                self.total_skipped_throttle += 1
                return None

            # Check: is this a new feeding session for this label?
            if session_id != self._active_session_ids.get(cat_name):
                self._active_session_ids[cat_name] = session_id
                self._session_capture_count[cat_name] = 0

            # Check: max captures per session (restart the interval so the caller's
            # should_capture() pre-check does not pass on every frame)
            if (
                self.max_per_session > 0
                and self._session_capture_count.get(cat_name, 0) >= self.max_per_session
            ):
                self._last_capture_time[key] = now
                self.total_skipped_limit += 1
                return None

            # Build filename and path (suffix if the name is taken, e.g. two
            # detector labels saved under the same label in the same millisecond)
            ts = datetime.now(timezone.utc)
            ts_iso = ts.isoformat()
            stem = f"{cat_name}_{ts.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            filepath = self.output_dir / cat_name / f"{stem}.jpg"
            counter = 1
            while filepath in self._recent_paths or filepath.exists():
                filepath = self.output_dir / cat_name / f"{stem}_{counter}.jpg"
                counter += 1
            filename = filepath.name

            # Enqueue (non-blocking: drop the frame if the writer is behind)
            if self._queue.qsize() >= self._max_queued_frames:
                return None
            self._queue.put_nowait((frame, filepath, self.jpeg_quality,
                                    detections, ts_iso, session_id or "", extra))

            # Update state
            self._last_capture_time[key] = now
            self._session_capture_count[cat_name] = (
                self._session_capture_count.get(cat_name, 0) + 1
            )
            self._recent_paths.append(filepath)
            self.total_captured += 1

            # CSV log
            if self._csv_writer:
                self._csv_writer.writerow([
                    filename, cat_name, ts.isoformat(), session_id or ""
                ])
                self._csv_file.flush()

            return filepath

    def relabel(self, paths: list[Path], new_label: str) -> None:
        """Move saved captures (and their sidecars) into new_label's folder.

        Queued behind pending writes, so it is safe to call right after
        maybe_capture() returned the paths.
        """
        if paths:
            self._queue.put_nowait(("relabel", list(paths), new_label))

    def delete(self, paths: list[Path]) -> None:
        """Delete saved captures and their sidecars (queued behind pending writes)."""
        if paths:
            self._queue.put_nowait(("delete", list(paths), None))

    def forget(self, throttle_key: str) -> None:
        """Drop a throttle key that won't be used again (a visit that ended)."""
        with self._lock:
            self._last_capture_time.pop(throttle_key, None)

    def reset_session(self, cat_name: str) -> None:
        """Reset capture count for a cat when its feeding session ends."""
        with self._lock:
            self._session_capture_count[cat_name] = 0
            self._active_session_ids[cat_name] = None

    def _save_loop(self) -> None:
        """Background thread: dequeue frames, write JPEGs and optional JSON sidecars."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            # Control items are ("relabel" | "delete", paths, label); capture
            # items start with an ndarray, so test the type rather than compare.
            if isinstance(item[0], str):
                action, paths, new_label = item
                try:
                    if action == "relabel":
                        self._relabel_now(paths, new_label)
                    elif action == "delete":
                        self._delete_now(paths)
                except Exception as exc:
                    print_error(f"Detection capture {action} failed: {exc}")
                self._queue.task_done()
                continue
            frame, filepath, quality, detections, ts_iso, session_id, extra = item
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                print_info(f"Detection capture saved: {filepath.name}")
                if detections is not None:
                    sidecar = {
                        "timestamp": ts_iso,
                        "session_id": session_id,
                        "image_size": [frame.shape[1], frame.shape[0]],
                        "detections": [
                            {
                                "label": d["label"],
                                "confidence": round(float(d["confidence"]), 4),
                                "bbox_norm": [round(float(v), 6) for v in d["bbox_norm"]],
                            }
                            for d in detections
                        ],
                        **(extra or {}),
                    }
                    with open(filepath.with_suffix(".json"), "w") as f:
                        json.dump(sidecar, f)
            except Exception as exc:
                print_error(f"Detection capture save failed {filepath.name}: {exc}")
            self._queue.task_done()

    def _relabel_now(self, paths: list[Path], new_label: str) -> None:
        dest_dir = self.output_dir / new_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            try:
                self._relabel_one(src, dest_dir, new_label)
            except OSError as exc:  # keep going with the rest of the batch
                print_error(f"Detection capture relabel failed for {src.name}: {exc}")

    def _relabel_one(self, src: Path, dest_dir: Path, new_label: str) -> None:
        if not src.exists():
            return
        old_label = src.parent.name
        stem = src.stem
        if stem.startswith(old_label + "_"):
            stem = stem[len(old_label) + 1:]
        dest = dest_dir / f"{new_label}_{stem}{src.suffix}"
        counter = 1
        while dest.exists() or dest.with_suffix(".json").exists():
            dest = dest_dir / f"{new_label}_{stem}_{counter}{src.suffix}"
            counter += 1
        src_json = src.with_suffix(".json")
        captured_at, session_id = "", ""
        if src_json.exists():
            try:
                meta = json.loads(src_json.read_text())
                captured_at = meta.get("timestamp", "")
                session_id = meta.get("session_id", "")
            except (OSError, ValueError):
                pass
        src.rename(dest)
        if src_json.exists():
            src_json.rename(dest.with_suffix(".json"))
        print_info(f"Detection capture relabeled: {src.name} -> {new_label}/{dest.name}")
        with self._lock:
            if self._csv_writer:
                self._csv_writer.writerow([dest.name, new_label, captured_at, session_id])
                self._csv_file.flush()

    @staticmethod
    def _delete_now(paths: list[Path]) -> None:
        for p in paths:
            try:
                p.unlink(missing_ok=True)
                p.with_suffix(".json").unlink(missing_ok=True)
                print_info(f"Detection capture removed (visit never confirmed): {p.name}")
            except OSError as exc:
                print_error(f"Detection capture delete failed for {p.name}: {exc}")

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


class VisitBackLabeler:
    """Track unlabeled captures per visit and name them once the visit is confirmed.

    A capture is added only when it was taken while the classifier had not yet
    confirmed the cat and was not rejecting crops. Visits are keyed by the
    detector's visit start time, which the classifier echoes in its status file,
    so a confirmation is never applied to another visit's images.
    """

    def __init__(self, expire_after_seconds: float = 30.0):
        self._expire_after = expire_after_seconds
        # visit_start -> {"paths": [...], "last_seen": last in-ROI frame time}
        # Both keys come from the visit tracker, the same visit boundaries the
        # classifier's event log uses for arrival and departure.
        self._pending: dict[float, dict] = {}

    def __bool__(self) -> bool:
        return bool(self._pending)

    def drain(self) -> list[Path]:
        """Forget every pending visit and return their paths."""
        paths = [p for entry in self._pending.values() for p in entry["paths"]]
        self._pending.clear()
        return paths

    def add(self, visit_start: float | None, path: Path, now: float | None = None) -> None:
        if visit_start is None:
            return
        entry = self._pending.setdefault(visit_start, {"paths": [], "last_seen": 0.0})
        entry["paths"].append(path)
        entry["last_seen"] = time.time() if now is None else now

    def poll(self, confirmed: dict[float, str]) -> list[tuple[list[Path], str]]:
        """Return (paths, cat) for each pending visit that has been confirmed.

        confirmed maps visit_start -> cat (see cat_classifier.confirmed_visits).
        """
        ready = []
        for visit in [v for v in self._pending if v in confirmed]:
            ready.append((self._pending.pop(visit)["paths"], confirmed[visit]))
        return ready

    def expire(self, current: dict[float, float] | None = None,
               now: float | None = None) -> list[Path]:
        """Forget visits that ended without confirmation; return their paths.

        current maps each visit in progress to when its animal was last seen.
        A visit no longer in progress expires expire_after_seconds after its
        last frame in the region.
        """
        now = time.time() if now is None else now
        current = current or {}
        expired: list[Path] = []
        for visit in list(self._pending):
            if visit in current:
                self._pending[visit]["last_seen"] = current[visit]
                continue
            if now - self._pending[visit]["last_seen"] >= self._expire_after:
                expired.extend(self._pending.pop(visit)["paths"])
        return expired
