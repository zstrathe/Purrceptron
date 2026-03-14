"""Stage 2: MobileNetV2-INT8 cat identity classifier worker.

Runs as a separate process pinned to a single CPU core.  Receives
pre-cropped 224×224 BGR numpy arrays from Stage 1 (detect.py) via a
bounded multiprocessing.Queue.

A None sentinel on the queue signals cat departure and resets all vote state.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty

import numpy as np


# ── ImageNet normalisation constants ────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess(frame_bgr: np.ndarray):
    """Convert a 224×224 BGR uint8 frame to a normalised CHW float tensor."""
    import torch
    rgb = frame_bgr[:, :, ::-1].copy()          # BGR → RGB
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1))   # HWC → CHW
    return tensor.unsqueeze(0)                           # add batch dim


# ═══════════════════════════════════════════════════════════════════════════
# Vote Window
# ═══════════════════════════════════════════════════════════════════════════

class VoteWindow:
    """Rolling majority-vote buffer.

    A prediction is "confirmed" when one class holds a strict majority
    (> window_size / 2 votes) AND its average confidence ≥ threshold.
    """

    def __init__(self, window_size: int = 10, confidence_threshold: float = 0.85):
        self._window_size = window_size
        self._threshold = confidence_threshold
        self._votes: deque[tuple[int, float]] = deque(maxlen=window_size)

    def add(self, class_idx: int, confidence: float) -> tuple[int, float] | None:
        """Add a vote.  Returns (class_idx, avg_conf) if confirmed, else None."""
        self._votes.append((class_idx, confidence))

        # Count votes per class
        counts: dict[int, list[float]] = {}
        for idx, conf in self._votes:
            counts.setdefault(idx, []).append(conf)

        majority = self._window_size / 2
        for idx, confs in counts.items():
            if len(confs) > majority:
                avg_conf = float(np.mean(confs))
                if avg_conf >= self._threshold:
                    return (idx, avg_conf)
        return None

    def get_counts(self) -> dict[int, int]:
        """Return vote counts per class index."""
        counts: dict[int, int] = {}
        for idx, _ in self._votes:
            counts[idx] = counts.get(idx, 0) + 1
        return counts

    def reset(self) -> None:
        self._votes.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Feeding Event Logger
# ═══════════════════════════════════════════════════════════════════════════

class FeedingEventLogger:
    """Persists classifier feeding events to SQLite (WAL mode)."""

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS classifier_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cat_name TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT,
        duration_seconds REAL,
        confidence_avg REAL NOT NULL,
        frame_count INTEGER NOT NULL DEFAULT 0,
        confirmed INTEGER NOT NULL DEFAULT 1,
        in_progress INTEGER NOT NULL DEFAULT 0
    )
    """

    def __init__(self, db_path: Path, cats: list[str]):
        self._db_path = db_path
        self._cats = cats
        self._active_row_id: int | None = None
        self._start_time: datetime | None = None
        self._confidence_sum: float = 0.0
        self._frame_count: int = 0
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(self._CREATE_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def on_confirmed(self, cat_name: str, confidence: float) -> None:
        """Insert a new in-progress event row."""
        if self._active_row_id is not None:
            return  # Already tracking; ignore duplicate confirmation
        self._start_time = datetime.now()
        self._confidence_sum = confidence
        self._frame_count = 1
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO classifier_events "
                "(cat_name, start_time, confidence_avg, frame_count, in_progress) "
                "VALUES (?, ?, ?, 1, 1)",
                (cat_name, self._start_time.isoformat(), confidence),
            )
            self._active_row_id = cur.lastrowid

    def on_vote(self, confidence: float) -> None:
        """Accumulate confidence for the active event."""
        if self._active_row_id is None:
            return
        self._confidence_sum += confidence
        self._frame_count += 1

    def on_departure(self) -> None:
        """Finalise the active event row and reset state."""
        if self._active_row_id is None:
            return
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds() if self._start_time else None
        avg_conf = self._confidence_sum / max(self._frame_count, 1)
        with self._connect() as conn:
            conn.execute(
                "UPDATE classifier_events "
                "SET end_time=?, duration_seconds=?, confidence_avg=?, "
                "frame_count=?, in_progress=0 "
                "WHERE id=?",
                (end_time.isoformat(), duration, avg_conf,
                 self._frame_count, self._active_row_id),
            )
        self._active_row_id = None
        self._start_time = None
        self._confidence_sum = 0.0
        self._frame_count = 0


# ═══════════════════════════════════════════════════════════════════════════
# Classifier Worker (multiprocessing target)
# ═══════════════════════════════════════════════════════════════════════════

def _write_status(status: dict) -> None:
    """Atomically write classifier status JSON for the dashboard."""
    try:
        tmp = Path("/tmp/cat_classifier_status.json.tmp")
        tmp.write_text(json.dumps(status))
        os.replace(str(tmp), "/tmp/cat_classifier_status.json")
    except OSError:
        pass


def classifier_worker(q, cfg: dict) -> None:
    """Entry point for the classifier subprocess.

    Args:
        q:   multiprocessing.Queue — receives 224×224 BGR uint8 arrays
             or None sentinels (cat departure).
        cfg: classifier config dict (from project.yaml["classifier"] merged
             with {"cats": [...]}).
    """
    try:
        _classifier_worker_inner(q, cfg)
    except Exception as e:
        print(f"[classifier] FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        _write_status({"error": str(e), "prediction": None})


def _classifier_worker_inner(q, cfg: dict) -> None:
    """Actual classifier loop — called from classifier_worker with error handling."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    # Pin to dedicated CPU core to avoid competing with GStreamer threads
    try:
        os.sched_setaffinity(0, {cfg.get("cpu_core", 3)})
    except (AttributeError, OSError):
        pass

    torch.backends.quantized.engine = "qnnpack"

    model_path = Path(cfg["model_path"])
    if not model_path.is_file():
        msg = f"Model not found: {model_path}"
        print(f"[classifier] {msg}", flush=True)
        _write_status({"error": msg, "prediction": None})
        return

    extra_files = {"classes.json": "", "energy_threshold": ""}
    model = torch.jit.load(str(model_path), map_location="cpu",
                           _extra_files=extra_files)
    model.eval()

    # Use class order from training (embedded in model) to avoid index mismatch
    if extra_files["classes.json"]:
        cats: list[str] = json.loads(extra_files["classes.json"])
        print(f"[classifier] Loaded class order from model: {cats}", flush=True)
    else:
        cats: list[str] = cfg.get("cats", [])
        print(f"[classifier] WARNING: no class order in model, using config: {cats}", flush=True)

    # Energy-based OOD rejection
    energy_ood_enabled = cfg.get("energy_ood_enabled", True)
    energy_threshold: float | None = None
    if energy_ood_enabled:
        # Config override takes priority over model-embedded threshold
        cfg_threshold = cfg.get("energy_ood_threshold")
        if cfg_threshold is not None:
            energy_threshold = float(cfg_threshold)
            print(f"[classifier] Energy OOD threshold from config: {energy_threshold:.4f}", flush=True)
        elif extra_files["energy_threshold"]:
            et_data = json.loads(extra_files["energy_threshold"])
            # Prefer explicit threshold key, fall back to p99/p95 for older models
            if "threshold" in et_data:
                energy_threshold = float(et_data["threshold"])
                pct = et_data.get("percentile", "?")
                print(f"[classifier] Energy OOD threshold from model (p{pct}): {energy_threshold:.4f}", flush=True)
            elif "p99" in et_data:
                energy_threshold = float(et_data["p99"])
                print(f"[classifier] Energy OOD threshold from model (p99): {energy_threshold:.4f}", flush=True)
            else:
                energy_threshold = float(et_data["p95"])
                print(f"[classifier] Energy OOD threshold from model (p95): {energy_threshold:.4f}", flush=True)
        else:
            energy_ood_enabled = False
            print("[classifier] Energy OOD: no threshold available, disabled", flush=True)
    else:
        print("[classifier] Energy OOD: disabled by config", flush=True)

    vote_window = VoteWindow(
        window_size=cfg.get("vote_window_size", 10),
        confidence_threshold=cfg.get("confidence_threshold", 0.85),
    )
    logger = FeedingEventLogger(
        db_path=Path(cfg["db_path"]),
        cats=cats,
    )

    ood_consecutive_min = max(1, vote_window._window_size // 2)
    print(f"[classifier] Ready -- {len(cats)} classes: {cats}", flush=True)

    consecutive_ood = 0

    while True:
        try:
            item = q.get(timeout=1.0)
        except Empty:
            continue

        if item is None:
            # Departure sentinel -- finalise active event, reset votes
            vote_window.reset()
            logger.on_departure()
            consecutive_ood = 0
            _write_status({"prediction": None})
            continue

        tensor = _preprocess(item)
        with torch.no_grad():
            logits = model(tensor)

        # Energy-based OOD rejection (before voting)
        energy = float(-torch.logsumexp(logits, dim=1).item())
        if energy_ood_enabled and energy_threshold is not None and energy > energy_threshold:
            consecutive_ood += 1
            # Only show "Unknown" after sustained rejection, not isolated noisy frames
            if consecutive_ood >= ood_consecutive_min:
                vote_counts = vote_window.get_counts()
                _write_status({
                    "prediction": None,
                    "confidence": 0.0,
                    "confirmed": logger._active_row_id is not None,
                    "confirmed_cat": None,
                    "votes": {cats[k] if k < len(cats) else "unknown": v for k, v in vote_counts.items()},
                    "vote_window": vote_window._window_size,
                    "energy": round(energy, 3),
                    "ood_rejected": True,
                })
            continue

        consecutive_ood = 0
        probs = F.softmax(logits, dim=1)[0]
        class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())

        result = vote_window.add(class_idx, confidence)
        confirmed_cat = None
        if result is not None:
            confirmed_cat = cats[result[0]] if result[0] < len(cats) else "unknown"
            logger.on_confirmed(confirmed_cat, result[1])
            print(f"[classifier] Confirmed: {confirmed_cat} ({result[1]:.1%})", flush=True)
        elif logger._active_row_id is not None:
            logger.on_vote(confidence)

        # Write live status for dashboard
        vote_counts = vote_window.get_counts()
        _write_status({
            "prediction": cats[class_idx] if class_idx < len(cats) else "unknown",
            "confidence": round(confidence, 3),
            "confirmed": confirmed_cat is not None or logger._active_row_id is not None,
            "confirmed_cat": confirmed_cat,
            "votes": {cats[k] if k < len(cats) else "unknown": v for k, v in vote_counts.items()},
            "vote_window": vote_window._window_size,
            "energy": round(energy, 3),
            "ood_rejected": False,
        })
