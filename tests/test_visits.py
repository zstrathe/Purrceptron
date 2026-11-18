# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""VisitTracker: visits follow positions, not the tracker's IDs."""
import pytest

from deploy.visits import VisitTracker, in_roi, visit_gap_from

W, H = 1000, 1000
FPS = 30
DT = 1 / FPS


def det(x, y, w=0.2, h=0.2, tid=None, conf=0.9, label="cat"):
    return {"label": label, "confidence": conf, "bbox_norm": (x, y, w, h), "track_id": tid}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def tick(self, dt=DT):
        self.t += dt
        return self.t


def make(max_visits=VisitTracker.MAX_VISITS, gap=0.5, roi=(0, 0, W, H), stability=5):
    return VisitTracker(labels={"cat"}, roi=roi, frame_w=W, frame_h=H, overlap_threshold=0.0001,
                        stability_frames=stability, gap_seconds=gap, max_visits=max_visits)


def run(tracker, clock, frames):
    """frames: list of detection lists. Returns all departures."""
    departed = []
    for dets in frames:
        t = clock.tick()
        _, gone = tracker.update(dets, now=t, mono=t)
        departed += gone
    return departed


def empty(seconds):
    return [[] for _ in range(round(seconds * FPS))]


# ── one animal ────────────────────────────────────────────────────────────

def test_changing_track_ids_stay_one_visit():
    tr, c = make(), Clock()
    run(tr, c, [[det(0.4, 0.4, tid=i // 3)] for i in range(60)])
    (v,) = tr.visits()
    assert v.frames == 60 and v.stable


def test_nested_duplicate_bounding_box_is_one_visit():
    tr, c = make(max_visits=8), Clock()
    dets = [det(0.4, 0.4, 0.3, 0.3, conf=0.9), det(0.45, 0.45, 0.2, 0.2, conf=0.6)]
    run(tr, c, [dets])
    assert len(tr.visits()) == 1
    assert dets[1]["duplicate"] and not dets[0]["duplicate"]
    assert dets[0]["visit_id"] == tr.visits()[0].visit_id and dets[1]["visit_id"] is None


def test_overlapping_duplicates_keep_the_more_confident():
    tr, c = make(max_visits=8), Clock()
    a, b = det(0.40, 0.40, conf=0.5), det(0.43, 0.42, conf=0.8)    # IoU well above 0.5
    run(tr, c, [[a, b]])
    assert len(tr.visits()) == 1 and a["duplicate"] and not b["duplicate"]


def test_short_dropout_continues():
    tr, c = make(), Clock()
    run(tr, c, [[det(0.4, 0.4)]] * 10 + empty(0.3) + [[det(0.42, 0.4)]] * 5)
    (v,) = tr.visits()
    assert v.frames == 15


def test_long_dropout_ends_visit_once_then_new_visit():
    tr, c = make(), Clock()
    first = run(tr, c, [[det(0.4, 0.4)]] * 10)
    vid = tr.visits()[0].visit_id
    gone = run(tr, c, empty(1.0))
    assert [v.visit_id for v in gone] == [vid] and not first
    assert not run(tr, c, empty(1.0))                          # reported once
    run(tr, c, [[det(0.4, 0.4)]])
    assert tr.visits()[0].visit_id != vid


def test_departure_keeps_last_seen_time():
    tr, c = make(), Clock()
    run(tr, c, [[det(0.4, 0.4)]] * 10)
    last = c.t
    (gone,) = run(tr, c, empty(1.0))
    assert gone.last_seen == pytest.approx(last)


def test_no_track_ids():
    tr, c = make(), Clock()
    run(tr, c, [[det(0.4 + i * 0.002, 0.4, tid=None)] for i in range(30)])
    assert len(tr.visits()) == 1 and tr.visits()[0].frames == 30


def test_outside_region_is_not_a_visit():
    tr, c = make(roi=(0, 0, 300, 300)), Clock()
    d = det(0.7, 0.7)
    run(tr, c, [[d]])
    assert not tr.visits() and d["in_roi"] is False


def test_region_without_area_accepts_everything():
    assert in_roi((0.9, 0.9, 0.05, 0.05), (0, 0, 0, 0), W, H, 0.5)


def test_other_labels_are_ignored():
    tr, c = make(), Clock()
    d = det(0.4, 0.4, label="person")
    run(tr, c, [[d]])
    assert not tr.visits() and d["in_roi"] is True and d["visit_id"] is None


def test_one_frame_false_detection_never_becomes_stable():
    tr, c = make(), Clock()
    run(tr, c, [[det(0.4, 0.4)]])
    assert not tr.visits()[0].stable
    (gone,) = run(tr, c, empty(1.0))
    assert not gone.stable


def test_stability_counts_frames_per_visit():
    tr, c = make(stability=5), Clock()
    run(tr, c, [[det(0.4, 0.4, tid=i)] for i in range(4)])     # a new track ID every frame
    assert not tr.visits()[0].stable
    run(tr, c, [[det(0.4, 0.4, tid=9)]])
    assert tr.visits()[0].stable


# ── several at once ───────────────────────────────────────────────────────

def test_second_animal_away_from_the_first_is_its_own_visit():
    tr, c = make(), Clock()
    run(tr, c, [[det(0.1, 0.1)]] * 5)
    gone = run(tr, c, [[det(0.7, 0.7)]] * 30)       # the first left as the second came
    assert len(gone) == 1 and tr.visits()[0].bbox_norm[0] == pytest.approx(0.7)


def test_several_two_animals_arriving_apart():
    tr, c = make(max_visits=8), Clock()
    run(tr, c, [[det(0.1, 0.1)]] * 60)
    run(tr, c, [[det(0.1, 0.1), det(0.7, 0.7)]] * 3)
    a, b = tr.visits()
    assert a.stable and not b.stable and a.visit_id < b.visit_id
    run(tr, c, [[det(0.1, 0.1), det(0.7, 0.7)]] * 3)
    assert all(v.stable for v in tr.visits())


def test_several_leave_together():
    tr, c = make(max_visits=8), Clock()
    run(tr, c, [[det(0.1, 0.1), det(0.7, 0.7)]] * 10)
    assert len(run(tr, c, empty(1.0))) == 2 and not tr.visits()


def test_several_one_drops_briefly_other_untouched():
    tr, c = make(max_visits=8), Clock()
    run(tr, c, [[det(0.1, 0.1), det(0.7, 0.7)]] * 10)
    ids = [v.visit_id for v in tr.visits()]
    gone = run(tr, c, [[det(0.7, 0.7)]] * 9 + [[det(0.1, 0.1), det(0.7, 0.7)]] * 3)
    assert not gone and [v.visit_id for v in tr.visits()] == ids


def test_several_swapped_track_ids_do_not_swap_visits():
    tr, c = make(max_visits=8), Clock()
    run(tr, c, [[det(0.30, 0.4, tid=1), det(0.52, 0.4, tid=2)]] * 10)
    left, right = tr.visits()
    run(tr, c, [[det(0.30, 0.4, tid=2), det(0.52, 0.4, tid=1)]] * 10)
    assert tr.get(left.visit_id).bbox_norm[0] == pytest.approx(0.30)
    assert tr.get(right.visit_id).bbox_norm[0] == pytest.approx(0.52)


def test_several_moving_animal_keeps_its_visit():
    tr, c = make(max_visits=8), Clock()
    run(tr, c, [[det(0.1 + i * 0.005, 0.1), det(0.7, 0.7)] for i in range(60)])
    assert len(tr.visits()) == 2 and all(v.frames == 60 for v in tr.visits())


def test_several_is_capped():
    tr, c = make(), Clock()
    run(tr, c, [[det(0.1 * i, 0.05 * i, 0.05, 0.05) for i in range(10)]])
    assert len(tr.visits()) == 8


def test_ids_unique_in_the_same_millisecond():
    tr, c = make(max_visits=8), Clock()
    tr.update([det(0.1, 0.1), det(0.7, 0.7)], now=5.0, mono=5.0)
    a, b = tr.visits()
    assert a.visit_id != b.visit_id


# ── settings ──────────────────────────────────────────────────────────────

def test_visit_gap_from_config():
    assert visit_gap_from({"visit_gap_seconds": 1.5}, {}) == 1.5
    assert visit_gap_from({"miss_tolerance": 15}, {"fps": 30}) == pytest.approx(0.5)
    assert visit_gap_from({}, {}) == pytest.approx(0.5)


# ── recording and replay ──────────────────────────────────────────────────

def test_recording_round_trip(tmp_path):
    from deploy.visits import DetectionRecorder, read_recording
    rec = DetectionRecorder(tmp_path, 60, frame_size=(1920, 1080))
    for i in range(3):
        assert rec.add([det(0.4, 0.4, tid=i)], now=100.0 + i)
    rec.close()
    size, frames = read_recording(rec.path)
    assert size == (1920, 1080) and len(frames) == 3
    assert frames[2][0] == 102.0 and frames[2][1][0]["track_id"] == 2
    assert frames[0][1][0]["bbox_norm"] == (0.4, 0.4, 0.2, 0.2)


def test_recording_frames_by_the_clock(tmp_path, monkeypatch):
    from deploy import visits as visits_mod
    clock = [500.0]
    monkeypatch.setattr(visits_mod.time, "monotonic", lambda: clock[0])
    off = visits_mod.DetectionRecorder(tmp_path, 60)
    assert off.frame_due(True) is None and off.frame_due(False) is None  # off by default
    rec = visits_mod.DetectionRecorder(tmp_path, 60, frame_every=1.0)
    assert rec.frame_due(False) is None                  # no idle frames unless asked
    assert rec.frame_due(True) == "visit"
    clock[0] += 0.5
    assert rec.frame_due(True) is None                   # not again within 1 s
    clock[0] += 0.6
    assert rec.frame_due(True) == "visit"
    p = rec.frame_path("visit", now=1234.5)
    assert p.parent.is_dir() and p.parent.name == rec.path.stem and p.name == "1234.500_visit.jpg"
    both = visits_mod.DetectionRecorder(tmp_path, 60, frame_every=1.0, idle_every=30.0)
    assert both.frame_due(False) == "idle" and both.frame_due(True) == "visit"
    clock[0] += 10
    assert both.frame_due(False) is None                 # idle interval is its own
    clock[0] += 21
    assert both.frame_due(False) == "idle"


def test_replay_script(tmp_path):
    import subprocess, sys as _sys
    from deploy.visits import DetectionRecorder
    rec = DetectionRecorder(tmp_path, 60, frame_size=(W, H))
    t = 1000.0
    for i in range(40):                      # a visit with track IDs changing
        rec.add([det(0.4, 0.4, tid=i // 10)], now=t); t += DT
    t += 2.0
    for i in range(40):                      # a second visit after 2 s
        rec.add([det(0.4, 0.4, tid=7)], now=t); t += DT
    rec.close()
    cfg = tmp_path / "project.yaml"
    cfg.write_text("classifier:\n  bowl_roi: [0, 0, 1000, 1000]\n  roi_overlap_threshold: 0.0001\n")
    out = subprocess.run([_sys.executable, "scripts/replay_visits.py", str(rec.path), "--config", str(cfg)],
                         capture_output=True, text=True, check=True).stdout
    assert "2 visits (2 stable)" in out and "4: [0, 1, 2, 3]" in out
