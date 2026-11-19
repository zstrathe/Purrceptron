# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Automatic retraining: the test set, confirmed images, the deploy decision
and the nightly check."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from collect.label_review import ReviewStateTracker
from remote_training.auto_retrain import AutoRetrainer, auto_settings, decide, parse_form
from remote_training.bundle import count_trainable_images, set_aside_test
from remote_training.test_set import TestSet, image_key

CATS = ["Cat1", "Cat2"]
T0 = datetime(2026, 11, 1, 8, 0, 0)


def name(cat, t, ms=0):
    return f"{cat}_{t:%Y%m%d_%H%M%S}_{ms:03d}.jpg"


def add_image(raw, cat, t, ms=0, bbox=True):
    d = raw / cat
    d.mkdir(parents=True, exist_ok=True)
    p = d / name(cat, t, ms)
    p.write_bytes(b"jpg")
    dets = [{"label": "cat", "confidence": 0.9, "bbox_norm": [0.1, 0.1, 0.5, 0.5]}] if bbox else []
    p.with_suffix(".json").write_text(json.dumps({"detections": dets}))
    return p


def add_sessions(raw, cat, n, per=3, start=T0):
    """n sessions an hour apart, `per` images 10 s apart in each."""
    out = []
    for s in range(n):
        base = start + timedelta(hours=s)
        out.append([add_image(raw, cat, base + timedelta(seconds=10 * i)) for i in range(per)])
    return out


def confirm(raw, paths, value=True):
    state_path = raw / ".review_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for p in paths:
        state[f"{p.parent.name}/{p.name}"] = value
    state_path.write_text(json.dumps(state))


# ── Test set ────────────────────────────────────────────────────

def test_image_key_ignores_the_cat_prefix():
    assert image_key("Cat1_20261101_080000_123.jpg") == "20261101_080000_123"
    assert image_key("Some_Cat_20261101_080000_123.jpg") == "20261101_080000_123"
    assert image_key("1741608123456.jpg") == "1741608123456"


def test_sessions_are_split_whole_and_about_the_right_share(tmp_path):
    sessions = add_sessions(tmp_path, "Cat1", 200)
    ts = TestSet(tmp_path)
    ts.assign(["Cat1"], 5, 0.15)
    sides = [{ts.is_test(p) for p in s} for s in sessions]
    assert all(len(side) == 1 for side in sides)            # never split
    share = sum(side == {True} for side in sides) / len(sides)
    assert 0.08 < share < 0.22


def test_assignment_is_permanent(tmp_path):
    sessions = add_sessions(tmp_path, "Cat1", 40)
    ts = TestSet(tmp_path)
    ts.assign(["Cat1"], 5, 0.15)
    ts.save()
    before = {p.name: ts.is_test(p) for s in sessions for p in s}
    # More sessions, and a different share: old ones stay where they were.
    add_sessions(tmp_path, "Cat1", 40, start=T0 + timedelta(days=5))
    again = TestSet(tmp_path)
    again.assign(["Cat1"], 5, 0.5)
    assert {p.name: again.is_test(p) for s in sessions for p in s} == before


def test_a_relabeled_image_keeps_its_side(tmp_path):
    sessions = add_sessions(tmp_path, "Cat1", 40)
    ts = TestSet(tmp_path)
    ts.assign(CATS, 5, 0.15)
    ts.save()
    test_img = next(p for s in sessions for p in s if ts.is_test(p))
    moved = tmp_path / "Cat2" / test_img.name.replace("Cat1_", "Cat2_")
    moved.parent.mkdir()
    test_img.rename(moved)
    again = TestSet(tmp_path)
    again.assign(CATS, 5, 0.15)
    assert again.is_test(moved)


def test_new_images_join_the_side_of_their_session(tmp_path):
    sessions = add_sessions(tmp_path, "Cat1", 40)
    ts = TestSet(tmp_path)
    ts.assign(["Cat1"], 5, 0.15)
    for s in sessions:
        late = add_image(tmp_path, "Cat1", T0 + timedelta(hours=sessions.index(s), seconds=60))
        ts.assign(["Cat1"], 5, 0.15)
        assert ts.is_test(late) == ts.is_test(s[0])


# ── Confirmed images and the bundle ─────────────────────────────

def test_skip_is_reviewed_but_not_confirmed(tmp_path):
    img = add_image(tmp_path, "Cat1", T0)
    rs = ReviewStateTracker(tmp_path)
    rs.mark_skipped(img)
    assert rs.is_reviewed(img) and not rs.is_confirmed(img)
    rs.mark_reviewed(img)
    rs.mark_skipped(img)                     # never downgrades a confirmation
    assert rs.is_confirmed(img)


def test_counts_and_set_aside(tmp_path):
    raw = tmp_path / "raw"
    config = {"classifier_training": {"test_frac": 0.3}}
    sessions = add_sessions(raw, "Cat1", 30)
    every = [p for s in sessions for p in s]
    confirm(raw, every[::2])                              # half confirmed
    confirm(raw, every[1:2], value="skipped")
    ts = TestSet(raw)
    ts.assign(["Cat1"], 5, 0.3)
    confirmed = set(every[::2])
    want_test = sum(1 for p in confirmed if ts.is_test(p))
    counts = count_trainable_images(raw, ["Cat1"], config)["Cat1"]
    assert counts["with_bbox"] == len(every)
    assert counts["test"] == want_test
    assert counts["confirmed"] == len(confirmed) - want_test

    # Crops are named like their images.
    crops, test_dir = tmp_path / "crops", tmp_path / "test"
    (crops / "Cat1").mkdir(parents=True)
    for p in every:
        (crops / "Cat1" / p.name).write_bytes(b"crop")
    aside = set_aside_test(crops, test_dir, raw, ["Cat1"], config, confirmed_only=True)
    left = {p.name for p in (crops / "Cat1").iterdir()}
    assert left == {p.name for p in confirmed if not ts.is_test(p)}
    assert {p.name for p in (test_dir / "Cat1").iterdir()} == \
        {p.name for p in confirmed if ts.is_test(p)}
    assert aside["test"]["Cat1"] == want_test
    assert (raw / ".test_set.json").is_file()


def test_manual_jobs_keep_unconfirmed_images_but_not_test_ones(tmp_path):
    raw = tmp_path / "raw"
    config = {"classifier_training": {"test_frac": 0.3}}
    every = [p for s in add_sessions(raw, "Cat1", 30) for p in s]
    crops = tmp_path / "crops"
    (crops / "Cat1").mkdir(parents=True)
    for p in every:
        (crops / "Cat1" / p.name).write_bytes(b"crop")
    set_aside_test(crops, tmp_path / "test", raw, ["Cat1"], config)
    ts = TestSet(raw)
    assert {p.name for p in (crops / "Cat1").iterdir()} == \
        {p.name for p in every if not ts.is_test(p)}
    assert not (tmp_path / "test").exists()               # nothing confirmed to score


# ── The deploy decision ─────────────────────────────────────────

def ev(**per_cat):
    pc = {c: {"correct": r, "total": t} for c, (r, t) in per_cat.items()}
    return {"overall": {"correct": sum(r for r, _ in per_cat.values()),
                        "total": sum(t for _, t in per_cat.values())}, "per_cat": pc}


def test_decide():
    old = ev(Cat1=(28, 30), Cat2=(29, 30))
    ok, _ = decide(ev(Cat1=(28, 30), Cat2=(29, 30)), old, model_in_use=True,
                   comparable=True, min_test_per_cat=20)
    assert ok                                            # a tie deploys
    ok, why = decide(ev(Cat1=(30, 30), Cat2=(28, 30)), old, model_in_use=True,
                     comparable=True, min_test_per_cat=20)
    assert not ok and "Cat2" in why                      # better overall, one cat worse
    ok, _ = decide(ev(Cat1=(27, 30), Cat2=(29, 30)), old, model_in_use=True,
                   comparable=True, min_test_per_cat=20)
    assert not ok
    ok, why = decide(ev(Cat1=(30, 30), Cat2=(30, 30)), None, model_in_use=True,
                     comparable=False, min_test_per_cat=20)
    assert not ok and "by hand" in why                   # model in use saw the test set
    ok, why = decide(ev(Cat1=(10, 10), Cat2=(30, 30)), old, model_in_use=True,
                     comparable=True, min_test_per_cat=20)
    assert not ok and "Too few" in why
    ok, _ = decide(ev(Cat1=(25, 30), Cat2=(25, 30)), None, model_in_use=False,
                   comparable=False, min_test_per_cat=20)
    assert ok                                            # nothing in use yet


def test_settings_from_hand_edited_yaml():
    s = auto_settings({"auto_retrain": {"enabled": "yes", "time": 120, "min_new_images": "x"}})
    assert s["enabled"] is False and s["time"] == "02:00" and s["min_new_images"] == 25
    assert parse_form({"enabled": True, "time": "23:30", "min_new_images": "5",
                       "min_test_per_cat": "10"})["time"] == "23:30"


# ── The nightly check ───────────────────────────────────────────

class FakeManager:
    def __init__(self, counts, last=None, active=False, reachable=True, missing=()):
        self.counts, self.last, self.active = counts, last, active
        self.backend = SimpleNamespace(
            label="Local PC", missing_settings=lambda: list(missing),
            test_connection=lambda: SimpleNamespace(ok=reachable, message="refused"))
        self.started = []
        self.on_train_ready = None
        self.deployed = []
        self.decisions = []

    def is_active(self): return self.active
    def image_counts(self): return self.counts
    def cached_image_counts(self): return self.counts
    def last_trained_counts(self): return self.last
    def settings(self): return {"min_images_per_cat": 30}
    def make_backend(self): return self.backend

    def start(self, kind, trigger):
        self.started.append((kind, trigger))
        return {"status": "ok", "job_id": "20261118-020000"}

    def deploy(self, job_id):
        self.deployed.append(job_id)
        return {"status": "ok", "message": "loaded"}

    def record_auto_decision(self, job_id, **kw): self.decisions.append((job_id, kw))


def retrainer(tmp_path, mgr, **cfg):
    config = {"auto_retrain": {"enabled": True, "time": "02:00", **cfg}}
    return AutoRetrainer(tmp_path, lambda: config, lambda block: None, mgr)


COUNTS = {"Cat1": {"confirmed": 80}, "Cat2": {"confirmed": 90}}


def test_skips_without_enough_new_confirmed_images(tmp_path):
    mgr = FakeManager(COUNTS, last={"Cat1": {"confirmed": 70}, "Cat2": {"confirmed": 90}})
    r = retrainer(tmp_path, mgr)
    assert r.run_once().startswith("Skipped: 10 new")
    assert not mgr.started and r._state["last_success"]      # done for tonight


def test_old_jobs_without_confirmed_counts_count_everything_as_new(tmp_path):
    mgr = FakeManager(COUNTS, last={"Cat1": {"with_bbox": 500}})
    r = retrainer(tmp_path, mgr)
    assert r.run_once().startswith("Started")
    assert mgr.started == [("train", "auto")]


def test_waits_for_a_running_job(tmp_path):
    mgr = FakeManager(COUNTS, active=True)
    r = retrainer(tmp_path, mgr)
    assert r.run_once().startswith("waiting")
    assert "last_attempt" not in r._state                    # checked again shortly


def test_unreachable_backend_is_retried_not_packaged(tmp_path):
    mgr = FakeManager(COUNTS, reachable=False)
    r = retrainer(tmp_path, mgr)
    assert "unreachable" in r.run_once()
    assert not mgr.started
    assert r._state.get("last_attempt") and not r._state.get("last_success")


def test_too_few_confirmed_images_for_a_cat(tmp_path):
    mgr = FakeManager({"Cat1": {"confirmed": 80}, "Cat2": {"confirmed": 5}})
    r = retrainer(tmp_path, mgr)
    assert "Cat2 (5)" in r.run_once()
    assert not mgr.started


def test_after_training_deploys_only_automatic_jobs_that_pass(tmp_path):
    mgr = FakeManager(COUNTS)
    r = retrainer(tmp_path, mgr)
    good = {"job_id": "j1", "trigger": "auto", "test_eval": {
        "model_in_use": True, "comparable": True,
        "candidate": ev(Cat1=(29, 30), Cat2=(30, 30)), "deployed": ev(Cat1=(28, 30), Cat2=(30, 30))}}
    r.on_train_ready(good)
    assert mgr.deployed == ["j1"]
    r.on_train_ready({**good, "job_id": "j2", "trigger": "manual"})
    assert mgr.deployed == ["j1"]                           # manual jobs: the user decides
    r.on_train_ready({**good, "job_id": "j3", "test_eval": {"error": "no confirmed test images yet"}})
    assert mgr.deployed == ["j1"] and mgr.decisions[-1][1]["deployed"] is False
