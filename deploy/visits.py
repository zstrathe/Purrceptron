# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Visits: each animal's stay in the region of interest, followed by position.

The detector's tracker gives every bounding box a track ID, but the IDs change
constantly (in the visit logs only a third of visits kept one ID). So a visit
follows where its animal is instead: a bounding box continues the visit it
overlaps or is nearest to, and the track ID is only a hint. A visit the
detector lost for longer than the gap is held for hold_seconds, continued only
by a bounding box back in the same place.

Pure Python (no numpy, OpenCV or Hailo), so it is unit-tested anywhere.

    tracker = VisitTracker(labels={"cat"}, roi=(760, 390, 1160, 790),
                           frame_w=1920, frame_h=1080, overlap_threshold=0.01)
    visits, departed = tracker.update(frame_detections, time.time())

update() runs on every frame, including frames with no detections, so visits
end on time. It also writes three keys into each detection dict: ``in_roi``,
``visit_id`` (or None) and ``duplicate``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

# Two bounding boxes are one animal when they overlap this much (IoU), or when
# this much of the smaller one lies inside the larger one. In the saved images,
# 94 of 102 frames with two bounding boxes were a second box on the same cat.
DUPLICATE_IOU = 0.5
DUPLICATE_INSIDE = 0.8
# A bounding box can continue a visit when it overlaps the visit's last one by
# more than this (IoU), or its centre is within the larger side of the two.
MATCH_IOU = 0.1
# Added to the position score when the track ID is the same as last frame's.
TRACK_ID_BONUS = 0.1
# A visit unseen for longer than the gap is held for a while (hold_seconds):
# only a bounding box overlapping its last one by at least this much (IoU) can
# continue it. The detector often loses an eating cat for a second or two, and
# it comes back in the same place (IoU 0.8 to 0.99 in recordings); a cat that
# leaves is seen walking off, and one that arrives appears somewhere else first.
HOLD_IOU = 0.5


@dataclass
class Visit:
    visit_id: float          # wall time of the first frame, rounded to ms, unique
    label: str               # detector label, e.g. "cat"
    bbox_norm: tuple         # latest (xmin, ymin, w, h), normalized 0-1
    confidence: float        # latest detector confidence
    track_id: int | None     # latest track ID (a tie-break only)
    last_seen: float         # wall time of the latest matched frame
    last_seen_mono: float    # the same, on the monotonic clock (for the gap)
    frames: int = 1          # frames matched so far
    seen: bool = True        # matched in the current frame
    stable: bool = False     # frames >= the tracker's stability_frames
    detection: dict | None = None   # this frame's detection dict, when seen
    resumes: int = 0         # times it continued after a hold
    last_classify: float = 0.0      # used by detect.py: monotonic time of the last crop


# ── Geometry (pixels: normalized x and y have different scales) ─────────────

def _px(bbox_norm, frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox_norm
    return x * frame_w, y * frame_h, (x + w) * frame_w, (y + h) * frame_h


def _intersection(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return ix * iy if ix > 0 and iy > 0 else 0.0


def _area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def iou(a, b) -> float:
    """Intersection over union of two pixel rectangles (x1, y1, x2, y2)."""
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def inside_fraction(a, b) -> float:
    """How much of the smaller rectangle lies inside the other (0 to 1)."""
    small = min(_area(a), _area(b))
    return _intersection(a, b) / small if small > 0 else 0.0


def _centre_distance(a, b) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _reach(a, b) -> float:
    """How far apart two centres may be for one animal: the larger side of the two."""
    return max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)


def in_roi(bbox_norm, roi, frame_w: int, frame_h: int, threshold: float) -> bool:
    """True when the bounding box covers at least ``threshold`` of the region
    of interest (intersection area / region area). A region with no area
    accepts everything."""
    x0, y0, x1, y1 = roi
    roi_area = (x1 - x0) * (y1 - y0)
    if roi_area <= 0:
        return True
    inter = _intersection(_px(bbox_norm, frame_w, frame_h), (x0, y0, x1, y1))
    return inter > 0 and inter / roi_area >= threshold


def visit_gap_from(classifier_cfg: dict, camera_cfg: dict) -> float:
    """classifier.visit_gap_seconds, or the older miss_tolerance in frames
    (15 at 30 fps = 0.5 s)."""
    gap = classifier_cfg.get("visit_gap_seconds")
    if gap is not None:
        return max(0.0, float(gap))
    fps = float(camera_cfg.get("fps") or 30) or 30.0
    return max(0.0, float(classifier_cfg.get("miss_tolerance", 15)) / fps)


def visit_hold_from(classifier_cfg: dict) -> float:
    """classifier.visit_hold_seconds (default 2.5; 0 turns the hold off)."""
    return max(0.0, float(classifier_cfg.get("visit_hold_seconds", 2.5)))


# ── The tracker ─────────────────────────────────────────────────────────────

class VisitTracker:
    """Follows each animal in the region of interest as a visit."""

    MAX_VISITS = 8           # animals followed at once; more are ignored

    def __init__(self, *, labels, roi, frame_w: int, frame_h: int,
                 overlap_threshold: float = 0.01, stability_frames: int = 5,
                 gap_seconds: float = 0.5, hold_seconds: float = 0.0,
                 max_visits: int = MAX_VISITS):
        self.labels = set(labels)
        self.roi = tuple(roi)
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.overlap_threshold = overlap_threshold
        self.stability_frames = max(1, int(stability_frames))
        self.gap_seconds = gap_seconds
        # A visit ends after the longer of the two; past the gap, only a
        # bounding box in the same place continues it (see HOLD_IOU).
        self.hold_seconds = hold_seconds
        self.max_visits = max(1, int(max_visits))
        self._visits: list[Visit] = []      # oldest first
        self._last_id = 0.0

    # ── settings (applied live) ──
    def set_roi(self, roi) -> None:
        self.roi = tuple(roi)

    def set_overlap_threshold(self, t: float) -> None:
        self.overlap_threshold = t

    # ── queries ──
    def visits(self) -> list[Visit]:
        return list(self._visits)

    def get(self, visit_id: float) -> Visit | None:
        return next((v for v in self._visits if v.visit_id == visit_id), None)

    def oldest_stable(self) -> Visit | None:
        return next((v for v in self._visits if v.stable), None)

    # ── the per-frame step ──
    def update(self, detections: list[dict], now: float | None = None,
               mono: float | None = None) -> tuple[list[Visit], list[Visit]]:
        """Match this frame's detections to visits. Returns (visits in
        progress, oldest first; visits that ended this frame)."""
        now = time.time() if now is None else now
        mono = time.monotonic() if mono is None else mono
        W, H = self.frame_w, self.frame_h

        # Departures first: a visit not seen for longer than the gap (or the
        # hold) has ended, even if no frames came in meanwhile (so it can't be
        # continued now).
        departed = []
        ends_after = max(self.gap_seconds, self.hold_seconds)
        for v in list(self._visits):
            if mono - v.last_seen_mono > ends_after:
                self._visits.remove(v)
                departed.append(v)
        for v in self._visits:
            v.seen = False
            v.detection = None

        # 1. Animals in the region. A new visit needs the overlap threshold;
        # one in progress continues while its animal touches the region at all
        # (a cat sitting back from the bowl is often a smaller bounding box,
        # mostly outside the region, while it is still there).
        cands = []
        keep_only: set[int] = set()      # ids of detections that can't start a visit
        for d in detections:
            d["visit_id"] = None
            d["duplicate"] = False
            d["in_roi"] = in_roi(d["bbox_norm"], self.roi, W, H, self.overlap_threshold)
            if d.get("label") not in self.labels:
                continue
            if d["in_roi"]:
                cands.append(d)
            elif self._visits and in_roi(d["bbox_norm"], self.roi, W, H, 0.0):
                cands.append(d)
                keep_only.add(id(d))

        # 2. Duplicates: keep the most confident of overlapping bounding boxes,
        # preferring one in the region over one that only touches it.
        cands.sort(key=lambda d: (id(d) not in keep_only, d.get("confidence", 0.0)), reverse=True)
        kept: list[tuple[dict, tuple]] = []
        for d in cands:
            r = _px(d["bbox_norm"], W, H)
            if any(iou(r, kr) > DUPLICATE_IOU or inside_fraction(r, kr) >= DUPLICATE_INSIDE
                   for _, kr in kept):
                d["duplicate"] = True
                continue
            kept.append((d, r))

        # 3. Match to visits in progress.
        free_v = list(self._visits)
        free_c = list(kept)
        pairs: list[tuple[Visit, dict]] = []

        def held(v) -> bool:
            return mono - v.last_seen_mono > self.gap_seconds

        def plausible(v, vr, cr) -> bool:
            if held(v):         # lost for a while: only the same place continues it
                return iou(vr, cr) >= HOLD_IOU
            return iou(vr, cr) > MATCH_IOU or _centre_distance(vr, cr) <= _reach(vr, cr)

        def score(v, vr, c) -> float:
            # Position decides; the same track ID only breaks near-ties, so a
            # tracker that swaps IDs between two neighbours can't swap visits.
            same_id = v.track_id is not None and c[0].get("track_id") == v.track_id
            return (iou(vr, c[1]) + 0.5 * (1 - min(1.0, _centre_distance(vr, c[1]) / _reach(vr, c[1])))
                    + (TRACK_ID_BONUS if same_id else 0.0))

        options = []
        for v in free_v:
            vr = _px(v.bbox_norm, W, H)
            for c in free_c:
                if plausible(v, vr, c[1]):
                    options.append((score(v, vr, c), v, c))
        options.sort(key=lambda o: o[0], reverse=True)
        for _, v, c in options:
            if v in free_v and c in free_c:
                pairs.append((v, c[0]))
                free_v.remove(v)
                free_c.remove(c)

        # 4. Update matched visits.
        for v, d in pairs:
            if held(v):
                v.resumes += 1
            v.bbox_norm = tuple(d["bbox_norm"])
            v.confidence = float(d.get("confidence", 0.0))
            v.track_id = d.get("track_id")
            v.last_seen = now
            v.last_seen_mono = mono
            v.frames += 1
            v.seen = True
            v.stable = v.frames >= self.stability_frames
            v.detection = d
            d["visit_id"] = v.visit_id
            d["in_roi"] = True

        # 5. New visits, while there is room.
        for d, _ in free_c:
            if id(d) in keep_only:
                continue
            if len(self._visits) >= self.max_visits:
                break
            v = Visit(visit_id=self._new_id(now), label=d.get("label", ""),
                      bbox_norm=tuple(d["bbox_norm"]), confidence=float(d.get("confidence", 0.0)),
                      track_id=d.get("track_id"), last_seen=now, last_seen_mono=mono)
            v.stable = v.frames >= self.stability_frames
            v.detection = d
            d["visit_id"] = v.visit_id
            self._visits.append(v)
            # An animal appearing where a held visit's animal was lost ends that
            # visit now: it is another animal, or the same one looking different,
            # and either way one animal never has two visits at once.
            nr = _px(v.bbox_norm, W, H)
            for hv in [h for h in self._visits if h is not v and not h.seen and held(h)]:
                hr = _px(hv.bbox_norm, W, H)
                if iou(hr, nr) > MATCH_IOU or _centre_distance(hr, nr) <= _reach(hr, nr):
                    self._visits.remove(hv)
                    departed.append(hv)

        return list(self._visits), departed

    def _new_id(self, now: float) -> float:
        vid = round(now, 3)
        taken = {v.visit_id for v in self._visits}
        while vid <= self._last_id or vid in taken:
            vid = round(vid + 0.001, 3)
        self._last_id = vid
        return vid


# ── Recording detections, to replay real visits through the tracker ────────

class DetectionRecorder:
    """Writes each frame's detections to logs/detection/frames_<time>.jsonl for
    a while (bounding boxes only: no images, no names), so real visits can be
    replayed with scripts/replay_visits.py. About 16 MB an hour.

    Line format: {"t": wall time, "dets": [[label, confidence, xmin, ymin, w, h,
    track_id], ...]}; the first line is {"frame_size": [w, h]}.

    With frame_every > 0, the caller also saves the camera frame every that
    many seconds while a visit is in progress, whether or not the frame had a
    detection, to frames_<time>/<wall time>_visit.jpg; with idle_every > 0,
    also every that many seconds with no visit, to <wall time>_idle.jpg (see
    frame_due and frame_path). Frames picked by the clock, not by the
    detector, are a fair test set for comparing detectors: visit frames for
    missed animals, idle frames for false alarms. About 0.3 MB a frame.
    """

    FLUSH_EVERY = 5.0

    def __init__(self, folder, seconds: float, frame_size=(1920, 1080),
                 frame_every: float = 0.0, idle_every: float = 0.0):
        from datetime import datetime
        from pathlib import Path
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._until = time.monotonic() + seconds
        self._lines: list[str] = [json.dumps({"frame_size": list(frame_size)})]
        self._last_flush = time.monotonic()
        self.frame_every = max(0.0, float(frame_every))
        self.idle_every = max(0.0, float(idle_every))
        self.frames_dir = folder / self.path.stem
        self._next_frame = {"visit": 0.0, "idle": 0.0}

    def frame_due(self, visit_open: bool) -> str | None:
        """"visit" or "idle" when a frame should be saved now (then not
        again for that kind's interval), else None."""
        kind = "visit" if visit_open else "idle"
        every = self.frame_every if visit_open else self.idle_every
        if not every:
            return None
        mono = time.monotonic()
        if mono < self._next_frame[kind]:
            return None
        self._next_frame[kind] = mono + every
        return kind

    def frame_path(self, kind: str = "visit", now: float | None = None):
        """Where to save the frame taken now; creates the folder."""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        return self.frames_dir / f"{time.time() if now is None else now:.3f}_{kind}.jpg"

    def add(self, detections: list[dict], now: float | None = None) -> bool:
        """Record one frame. Returns False once the time is up (file closed)."""
        mono = time.monotonic()
        dets = [[d.get("label"), round(float(d.get("confidence", 0.0)), 3),
                 *(round(float(x), 5) for x in d["bbox_norm"]), d.get("track_id")]
                for d in detections]
        self._lines.append(json.dumps({"t": round(time.time() if now is None else now, 3),
                                       "dets": dets}))
        if mono >= self._until:
            self.close()
            return False
        if mono - self._last_flush >= self.FLUSH_EVERY:
            self._flush()
        return True

    def _flush(self) -> None:
        self._last_flush = time.monotonic()
        if not self._lines:
            return
        try:
            with open(self.path, "a") as f:
                f.write("\n".join(self._lines) + "\n")
        except OSError:
            pass
        self._lines = []

    def close(self) -> None:
        self._flush()


def read_recording(path) -> tuple[tuple[int, int], list[tuple[float, list[dict]]]]:
    """(frame_size, [(wall time, detection dicts), ...]) from a recording."""
    size = (1920, 1080)
    frames = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "frame_size" in rec:
                size = tuple(rec["frame_size"])
                continue
            frames.append((rec["t"], [
                {"label": d[0], "confidence": d[1], "bbox_norm": tuple(d[2:6]), "track_id": d[6]}
                for d in rec["dets"]]))
    return size, frames
