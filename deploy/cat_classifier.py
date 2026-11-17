# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Stage 2: MobileNetV2-INT8 cat identity classifier worker.

Runs the ExecuTorch model (.pte, XNNPACK backend) exported by
train/train_classifier.py.

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
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty

import numpy as np

from utils.runtime_paths import CLASSIFIER_STATUS_JSON


# ── ImageNet normalisation constants ────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess(frame_bgr: np.ndarray):
    """Convert a 224×224 BGR uint8 frame to a normalised NCHW float32 tensor.

    The result must be C-contiguous: ExecuTorch reads the tensor's memory
    directly, and a transposed (non-contiguous) view gives wrong logits
    without raising any error.
    """
    import torch
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0   # BGR → RGB
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))        # HWC → CHW
    return torch.from_numpy(chw).unsqueeze(0)                 # add batch dim


# Flatbuffer file identifier of an ExecuTorch program, at bytes 4..8.
_PTE_MAGIC = b"ET12"


def _is_pte_file(path: Path) -> bool:
    """True if the file starts like an ExecuTorch program. Checked before
    loading: the loader aborts the whole process on a garbage header."""
    try:
        with open(path, "rb") as f:
            return f.read(8)[4:8] == _PTE_MAGIC
    except OSError:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Vote Window
# ═══════════════════════════════════════════════════════════════════════════

class VoteWindow:
    """Rolling majority-vote buffer.

    A prediction is "confirmed" when one class holds a strict majority
    (> window_size / 2 votes) AND its average confidence ≥ threshold.
    Each vote keeps the time its frame was seen, so a change of name can be
    dated (first_seen_of).
    """

    def __init__(self, window_size: int = 10, confidence_threshold: float = 0.85):
        self._window_size = window_size
        self._threshold = confidence_threshold
        self._votes: deque[tuple[int, float, float]] = deque(maxlen=window_size)

    def add(self, class_idx: int, confidence: float,
            seen_at: float = 0.0) -> tuple[int, float] | None:
        """Add a vote.  Returns (class_idx, avg_conf) if confirmed, else None."""
        self._votes.append((class_idx, confidence, seen_at))
        return self.result()

    def result(self) -> tuple[int, float] | None:
        counts: dict[int, list[float]] = {}
        for idx, conf, _ in self._votes:
            counts.setdefault(idx, []).append(conf)
        majority = self._window_size / 2
        for idx, confs in counts.items():
            if len(confs) > majority:
                avg_conf = float(np.mean(confs))
                if avg_conf >= self._threshold:
                    return (idx, avg_conf)
        return None

    def first_seen_of(self, class_idx: int) -> float | None:
        """When class_idx took over: the oldest of its votes from which on it
        holds at least two thirds of the votes (so a stray early vote is ignored)."""
        ours = total = 0
        when = None
        for idx, _, seen_at in reversed(self._votes):
            total += 1
            if idx == class_idx:
                ours += 1
                if ours * 3 >= total * 2:
                    when = seen_at
        return when

    def get_counts(self) -> dict[int, int]:
        """Return vote counts per class index."""
        counts: dict[int, int] = {}
        for idx, _, _ in self._votes:
            counts[idx] = counts.get(idx, 0) + 1
        return counts

    def reset(self) -> None:
        self._votes.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Feeding Event Logger
# ═══════════════════════════════════════════════════════════════════════════

class FeedingEventLogger:
    """Persists classifier visits to SQLite (WAL mode), several at a time.

    A row is opened when a visit is named, gets the confidence of each crop
    that agrees with its name, and is closed when the visit ends. A visit of
    the same animal that starts within a few seconds of its previous one
    ending continues that row (reopen_recent), so a detector dropout does not
    split one meal into two.
    """

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

    def __init__(self, db_path: Path, cats: list[str] | None = None):
        self._db_path = db_path
        # row id -> {"start": datetime, "sum": float, "count": int, "reopened": bool}
        self._rows: dict[int, dict] = {}
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(self._CREATE_SQL)
            # Left in progress by a crash, a power cut or a restart mid-visit:
            # a continued row keeps its earlier end; a new one never got an
            # end and is dropped, as a shutdown mid-visit does.
            conn.execute("UPDATE classifier_events SET in_progress=0 "
                         "WHERE in_progress=1 AND end_time IS NOT NULL")
            cur = conn.execute("DELETE FROM classifier_events WHERE in_progress=1")
            if cur.rowcount:
                print(f"[classifier] Dropped {cur.rowcount} unfinished visit(s) of an earlier run",
                      flush=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @staticmethod
    def _to_utc(ts: float | None) -> datetime:
        """Convert a time.time() timestamp to an aware UTC datetime (now if None)."""
        if ts is None:
            return datetime.now(timezone.utc)
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def open(self, cat_name: str, confidence: float, start_time: float | None = None) -> int:
        """Insert an in-progress row; start_time is when the animal arrived."""
        start = self._to_utc(start_time)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO classifier_events "
                "(cat_name, start_time, confidence_avg, frame_count, in_progress) "
                "VALUES (?, ?, ?, 1, 1)",
                (cat_name, start.isoformat(), confidence),
            )
            row = int(cur.lastrowid)
        self._rows[row] = {"start": start, "sum": confidence, "count": 1, "reopened": False}
        return row

    def reopen_recent(self, cat_name: str, not_before: float) -> int | None:
        """Continue this animal's latest finished row if it ended at or after
        not_before (time.time()). Its end stays in place until it is closed
        again, so a crash meanwhile leaves the earlier visit as it was."""
        held = tuple(self._rows) or (-1,)
        with self._connect() as conn:
            found = conn.execute(
                "SELECT id, start_time, end_time, confidence_avg, frame_count "
                "FROM classifier_events WHERE cat_name=? AND in_progress=0 "
                f"AND end_time IS NOT NULL AND id NOT IN ({','.join('?' * len(held))}) "
                "ORDER BY end_time DESC LIMIT 1",
                (cat_name, *held),
            ).fetchone()
            if not found:
                return None
            row, start, end, avg, count = found
            if datetime.fromisoformat(end) < self._to_utc(not_before):
                return None
            conn.execute("UPDATE classifier_events SET in_progress=1 WHERE id=?", (row,))
        count = int(count or 0)
        self._rows[row] = {"start": datetime.fromisoformat(start), "sum": float(avg) * count,
                           "count": count, "reopened": True}
        return row

    def vote(self, row: int | None, confidence: float) -> None:
        """Count one crop that agrees with the row's name."""
        r = self._rows.get(row)
        if r is not None:
            r["sum"] += confidence
            r["count"] += 1

    def close(self, row: int | None, end_time: float | None = None) -> None:
        """Finish the row at end_time (when the animal was last seen)."""
        r = self._rows.pop(row, None)
        if r is None:
            return
        end = self._to_utc(end_time)
        duration = max(0.0, (end - r["start"]).total_seconds())
        with self._connect() as conn:
            conn.execute(
                "UPDATE classifier_events "
                "SET end_time=?, duration_seconds=?, confidence_avg=?, "
                "frame_count=?, in_progress=0 WHERE id=?",
                (end.isoformat(), duration, r["sum"] / max(r["count"], 1), r["count"], row),
            )

    def drop(self, row: int | None) -> None:
        """Forget the part of the row in progress: a new row is deleted, a
        continued one goes back to how it was."""
        r = self._rows.pop(row, None)
        if r is None:
            return
        with self._connect() as conn:
            if r["reopened"]:
                conn.execute("UPDATE classifier_events SET in_progress=0 WHERE id=?", (row,))
            else:
                conn.execute("DELETE FROM classifier_events WHERE id=? AND in_progress=1", (row,))

    def abandon_all(self) -> None:
        """Identification stopped mid-visit: no end was seen, so nothing in
        progress is logged."""
        if self._rows:
            print("[classifier] Stopped during a visit: not logging it", flush=True)
        for row in list(self._rows):
            self.drop(row)

    @property
    def open_rows(self) -> list[int]:
        return list(self._rows)


# ═══════════════════════════════════════════════════════════════════════════
# Identification per visit
# ═══════════════════════════════════════════════════════════════════════════

class _VisitState:
    def __init__(self, visit_id: float, window: VoteWindow):
        self.visit_id = visit_id
        self.window = window
        self.crops = 0
        self.consecutive_ood = 0
        self.cat: str | None = None          # the visit's name (= its log row)
        self.row: int | None = None
        self.unknown = False                 # the model gave up (see VisitIdentifier)
        self.name_taken: str | None = None   # its vote's name is held by another animal
        self.last_seen = visit_id            # wall time of its latest crop's frame
        self.since = visit_id                # since when it has had no name (for a new row)
        self.last_crop_mono = 0.0
        self.updated = 0.0                   # wall time this record last changed
        self.record: dict = {"prediction": None}


class VisitIdentifier:
    """Names each visit from its crops' votes, and logs it. No model inside:
    the worker passes each crop's result, so this is tested on its own.

    - A visit is named once its vote window confirms a name.
    - Unknown: the model has given up on a visit that has no name (its crops
      are rejected, or a full window passed with no winner). It sticks until
      a name is confirmed, so the dashboard's "Who is this?" does not flicker.
    - One name per animal at a time: a name held by a visit seen at the same
      time is not given to a second one (name_taken; it stays unidentified).
    - A visit of an animal whose previous visit ended within rejoin_seconds
      continues that log row.
    - A log row never holds two animals: when a named visit's vote confirms
      another name, its row is closed when the other name took over, and the
      other name's row starts then (or the two visits swap rows, when two
      animals swapped places).
    """

    def __init__(self, cats: list[str], logger: FeedingEventLogger, *,
                 window_size: int = 10, confidence_threshold: float = 0.85,
                 ood_min: int | None = None, rejoin_seconds: float = 5.0,
                 stale_after: float = 60.0, recent: int = 32):
        self.cats = cats
        self.logger = logger
        self.window_size = window_size
        self.threshold = confidence_threshold
        self.ood_min = ood_min if ood_min is not None else max(1, window_size // 2)
        self.rejoin_seconds = rejoin_seconds
        self.stale_after = stale_after
        self._visits: dict[float, _VisitState] = {}
        # Recent [visit_id, name] pairs, newest last; detect.py names a visit's
        # early captures from it, also after the visit has ended.
        self.confirmed_visits: deque = deque(maxlen=recent)

    def _name(self, idx: int) -> str:
        return self.cats[idx] if 0 <= idx < len(self.cats) else "unknown"

    def _state(self, visit_id: float) -> _VisitState:
        st = self._visits.get(visit_id)
        if st is None:
            st = self._visits[visit_id] = _VisitState(
                visit_id, VoteWindow(self.window_size, self.threshold))
        return st

    def _holder(self, name: str, other_than: _VisitState) -> _VisitState | None:
        return next((v for v in self._visits.values() if v is not other_than and v.cat == name), None)

    # ── messages ──
    def on_crop(self, visit_id: float, seen_at: float, class_idx: int, confidence: float,
                energy: float | None = None, rejected: bool = False,
                mono: float | None = None) -> None:
        st = self._state(visit_id)
        st.crops += 1
        st.last_seen = max(st.last_seen, seen_at)
        st.last_crop_mono = time.monotonic() if mono is None else mono
        st.updated = time.time()
        energy_r = round(energy, 3) if energy is not None else None

        if rejected:
            st.consecutive_ood += 1
            if st.consecutive_ood >= self.ood_min:
                if st.cat is None:
                    st.unknown = True
                st.record = {"prediction": None, "confidence": 0.0, "confirmed_cat": None,
                             "energy": energy_r, "ood_rejected": True}
            st.record["last_crop_rejected"] = True
            return

        st.consecutive_ood = 0
        result = st.window.add(class_idx, confidence, seen_at)
        majority = self._name(result[0]) if result else None
        st.record = {"prediction": self._name(class_idx), "confidence": round(confidence, 3),
                     "confirmed_cat": majority, "energy": energy_r,
                     "ood_rejected": False, "last_crop_rejected": False}

        if st.cat is not None:
            if majority is not None and majority != st.cat:
                self._change_name(st, majority, result[1])
            elif self._name(class_idx) == st.cat:
                self.logger.vote(st.row, confidence)
            return
        if result is None:
            if st.crops >= self.window_size:
                st.unknown = True           # a full window and still no winner
            return
        self._give_name(st, majority, result[1], start=st.since)

    def on_departed(self, visit_id: float, last_seen: float | None = None) -> None:
        st = self._visits.pop(visit_id, None)
        if st is not None and st.row is not None:
            self.logger.close(st.row, last_seen if last_seen is not None else st.last_seen)

    def expire_stale(self, mono: float | None = None) -> None:
        """End visits with no crop for stale_after seconds (a lost departure)."""
        mono = time.monotonic() if mono is None else mono
        for vid, st in list(self._visits.items()):
            if mono - st.last_crop_mono > self.stale_after:
                print(f"[classifier] No crops for {self.stale_after:.0f} s -- ending visit {vid:.3f}",
                      flush=True)
                self.on_departed(vid, st.last_seen)

    def abandon_all(self) -> None:
        self.logger.abandon_all()
        self._visits.clear()

    # ── naming ──
    def _give_name(self, st: _VisitState, name: str, confidence: float, start: float) -> None:
        holder = self._holder(name, st)
        if holder is not None:
            if holder.last_seen >= st.visit_id:
                # Two bounding boxes showed this animal at the same time.
                st.name_taken = name
                st.unknown = True
                return
            # The holder is this same animal, lost by the tracker and found
            # again before its departure arrived: continue its row.
            del self._visits[holder.visit_id]
            st.row = holder.row
        else:
            st.row = None
            if self.rejoin_seconds > 0:
                st.row = self.logger.reopen_recent(name, not_before=start - self.rejoin_seconds)
                if st.row is not None:
                    print(f"[classifier] {name} is back within {self.rejoin_seconds:.0f} s: "
                          "continuing its visit", flush=True)
            if st.row is None:
                st.row = self.logger.open(name, confidence, start_time=start)
        st.cat = name
        st.unknown = False
        st.name_taken = None
        self.confirmed_visits.append([st.visit_id, name])
        print(f"[classifier] Confirmed: {name} ({confidence:.1%})", flush=True)

    def _change_name(self, st: _VisitState, new: str, confidence: float) -> None:
        """A named visit's vote now confirms another animal."""
        switch_at = st.window.first_seen_of(self.cats.index(new)) if new in self.cats else None
        switch_at = switch_at or st.last_seen
        old = st.cat
        holder = self._holder(new, st)
        if holder is not None and holder.record.get("confirmed_cat") == old:
            # The two animals swapped places: swap the rows, nothing ends.
            st.row, holder.row = holder.row, st.row
            st.cat, holder.cat = new, old
            self.confirmed_visits.append([st.visit_id, new])
            self.confirmed_visits.append([holder.visit_id, old])
            print(f"[classifier] {old} and {new} swapped places", flush=True)
            return
        print(f"[classifier] Visit of {old} is now {new}: ending {old}'s visit", flush=True)
        self.logger.close(st.row, end_time=switch_at)
        st.row, st.cat = None, None
        st.since = switch_at
        if holder is not None:
            st.name_taken = new
            st.unknown = True
            return
        self._give_name(st, new, confidence, start=switch_at)

    # ── status ──
    def visit_record(self, st: _VisitState) -> dict:
        counts = st.window.get_counts()
        rec = {
            "visit_id": st.visit_id,
            "prediction": None, "confidence": 0.0, "confirmed_cat": None, "energy": None,
            "ood_rejected": False, "last_crop_rejected": False,
            **st.record,
            "visit_cat": st.cat,
            "confirmed": st.cat is not None,
            "votes": {self._name(k): v for k, v in counts.items()},
            "vote_window": self.window_size,
            "unknown": st.unknown,
            "name_taken": st.name_taken,
            "crops": st.crops,
            "updated": round(st.updated, 3),
        }
        return rec

    def status(self) -> dict:
        visits = [self.visit_record(st) for st in sorted(self._visits.values(),
                                                          key=lambda v: v.visit_id)]
        return {"visits": visits, "confirmed_visits": list(self.confirmed_visits)}


# ═══════════════════════════════════════════════════════════════════════════
# Classifier Worker (multiprocessing target)
# ═══════════════════════════════════════════════════════════════════════════

# In RAM (utils/runtime_paths.py): rewritten for every classified crop.
STATUS_PATH = CLASSIFIER_STATUS_JSON


def _write_status(status: dict) -> None:
    """Atomically write classifier status JSON for the dashboard and detect.py."""
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_name(STATUS_PATH.name + ".tmp")
        tmp.write_text(json.dumps(status))
        os.replace(str(tmp), str(STATUS_PATH))
    except OSError:
        pass


def read_status() -> dict | None:
    """Read the status file (None if missing, unreadable or mid-replace)."""
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def confirmed_visits(status: dict | None) -> dict[float, str]:
    """{visit_start: cat} for the recent visits the classifier confirmed.

    The cat is the visit's first confirmation, the same one the event log records.
    """
    if not status:
        return {}
    out: dict[float, str] = {}
    for pair in status.get("confirmed_visits") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[1]:
            out[pair[0]] = pair[1]
    return out


def visit_record(status: dict | None, visit_id: float | None) -> dict | None:
    """The status record of one visit in progress (None if it has none)."""
    if not status or visit_id is None:
        return None
    return next((r for r in status.get("visits") or []
                 if isinstance(r, dict) and r.get("visit_id") == visit_id), None)


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

    model_path = Path(cfg["model_path"])
    if not model_path.is_file():
        msg = f"Model not found: {model_path}"
        print(f"[classifier] {msg}", flush=True)
        _write_status({"error": msg, "prediction": None})
        return
    if not _is_pte_file(model_path):
        msg = (f"{model_path.name} is not an ExecuTorch model (.pte). Train and deploy a "
               "model from the Training tab; classifier.model_path must point to it.")
        print(f"[classifier] {msg}", flush=True)
        _write_status({"error": msg, "prediction": None})
        return

    from executorch.runtime import Runtime
    try:
        # One inference thread: this process is pinned to a single core.
        from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool
        _unsafe_reset_threadpool(1)
    except Exception:
        pass
    torch.set_num_threads(1)

    # `program` must stay referenced for as long as `forward` is used.
    program = Runtime.get().load_program(str(model_path))
    forward = program.load_method("forward")

    def _model_constant(name: str) -> str:
        """A string the exporter embedded as a constant method ('' if absent)."""
        if name not in program.method_names:
            return ""
        method = program.load_method(name)
        return method.execute([])[0]

    extra_files = {"classes.json": _model_constant("classes_json"),
                   "energy_threshold": _model_constant("energy_threshold_json")}

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

    logger = FeedingEventLogger(db_path=Path(cfg["db_path"]))
    window = int(cfg.get("vote_window_size", 10))
    ident = VisitIdentifier(
        cats, logger,
        window_size=window,
        confidence_threshold=cfg.get("confidence_threshold", 0.85),
        rejoin_seconds=float(cfg.get("visit_rejoin_seconds", 5.0)),
    )
    print(f"[classifier] Ready -- {len(cats)} classes: {cats}", flush=True)

    # Queue messages from detect.py (deploy/detect.py, VisitFeeder):
    #   ("crop", visit_id, seen_at, crop)    a 224x224 crop of one visit's animal
    #   ("departed", visit_id, last_seen)    the visit ended
    #   None                                 shutdown: drop visits in progress, exit
    last_expire = time.monotonic()
    while True:
        try:
            item = q.get(timeout=1.0)
        except Empty:
            item = ()
        if time.monotonic() - last_expire >= 1.0:
            last_expire = time.monotonic()
            ident.expire_stale()
        if item == ():
            continue
        if item is None:
            # Shutdown (identification off, a new model, detect.py stopping)
            # in the middle of visits: drop them rather than log visits whose
            # end was never seen, then exit.
            ident.abandon_all()
            _write_status(ident.status())
            return
        kind = item[0] if isinstance(item, tuple) and item and isinstance(item[0], str) else None
        if kind == "departed":
            ident.on_departed(item[1], item[2])
        elif kind == "crop":
            _, visit_id, seen_at, crop = item
            logits = forward.execute([_preprocess(crop)])[0]
            # Energy-based OOD rejection (before voting)
            energy = float(-torch.logsumexp(logits, dim=1).item())
            rejected = bool(energy_ood_enabled and energy_threshold is not None
                            and energy > energy_threshold)
            probs = F.softmax(logits, dim=1)[0]
            class_idx = int(torch.argmax(probs).item())
            ident.on_crop(visit_id, seen_at, class_idx, float(probs[class_idx].item()),
                          energy=energy, rejected=rejected)
        else:
            continue
        _write_status(ident.status())
