#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Replay a detections recording through the visit tracker and list the visits.

Record on the Pi (bounding boxes only, at most 15 minutes):
    curl -X POST http://cats.local/api/debug/record-detections -H 'Content-Type: application/json' -d '{"seconds": 600}'
The file appears in logs/detection/frames_<time>.jsonl. Then, anywhere:
    python scripts/replay_visits.py logs/detection/frames_<time>.jsonl [--gap 0.5] [--hold 2.5]

The region of interest and settings come from --config (merged with
configs/runtime.yaml when it exists), as on the Pi.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploy.visits import VisitTracker, read_recording, visit_gap_from, visit_hold_from  # noqa: E402


def _load_config(path: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(path.read_text()) or {}
    runtime = path.parent / "runtime.yaml"
    if runtime.is_file():
        for section, values in (yaml.safe_load(runtime.read_text()) or {}).items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parent.parent / "configs" / "project.yaml")
    ap.add_argument("--gap", type=float, help="seconds out of the region that end a visit")
    ap.add_argument("--hold", type=float,
                    help="seconds a visit waits for its animal back in the same place (0 = off)")
    args = ap.parse_args()

    cfg = _load_config(args.config) if args.config.is_file() else {}
    clf, cam = cfg.get("classifier") or {}, cfg.get("camera") or {}
    (fw, fh), frames = read_recording(args.recording)
    tracker = VisitTracker(
        labels=set((cfg.get("deployment") or {}).get("tracking_objects") or ["cat"]),
        roi=clf.get("bowl_roi", [0, 0, fw, fh]), frame_w=fw, frame_h=fh,
        overlap_threshold=clf.get("roi_overlap_threshold", 0.01),
        stability_frames=clf.get("stability_frames", 5),
        gap_seconds=args.gap if args.gap is not None else visit_gap_from(clf, cam),
        hold_seconds=args.hold if args.hold is not None else visit_hold_from(clf))

    ids: dict[float, set] = {}
    ended = []
    for t, dets in frames:
        visits, departed = tracker.update(dets, now=t, mono=t)
        for v in visits:
            if v.seen and v.track_id is not None:
                ids.setdefault(v.visit_id, set()).add(v.track_id)
        ended += departed
    if frames:
        _, departed = tracker.update([], now=frames[-1][0] + 3600, mono=frames[-1][0] + 3600)
        ended += departed

    fmt = lambda t: datetime.fromtimestamp(t).strftime("%H:%M:%S")
    stable = [v for v in ended if v.stable]
    print(f"{len(frames)} frames, {len(ended)} visits ({len(stable)} stable), "
          f"gap {tracker.gap_seconds:.2f} s, hold {tracker.hold_seconds:.2f} s, "
          f"at most {tracker.max_visits} at once")
    print(f"{'start':>8} {'end':>8} {'secs':>6} {'frames':>6} {'holds':>5}  track IDs")
    for v in sorted(stable, key=lambda v: v.visit_id):
        tids = sorted(ids.get(v.visit_id, ()))
        print(f"{fmt(v.visit_id):>8} {fmt(v.last_seen):>8} {v.last_seen - v.visit_id:6.1f} "
              f"{v.frames:6d} {v.resumes:5d}  {len(tids)}: {tids[:12]}{' ...' if len(tids) > 12 else ''}")


if __name__ == "__main__":
    main()
