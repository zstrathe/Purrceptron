# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""The Capture tab's footer rows, and naming visits."""
import json
import os
from pathlib import Path

import pytest

from collect.visit_naming import VisitNamer, footer_rows

WORDS = {"animal": "cat", "an_animal": "a cat"}
ON = dict(enabled=True, model_exists=True, worker_error=None, running=True, words=WORDS)


def det(vid, stable=True):
    return {"visit_id": vid, "stable": stable, "in_roi": True}


def rec(vid, **kw):
    return {"visit_id": vid, "visit_cat": None, "prediction": None, "unknown": False,
            "name_taken": None, "ood_rejected": False, "votes": {}, "vote_window": 10, **kw}


def rows(dets, recs=(), named=None, **kw):
    return footer_rows(dets, {"visits": list(recs)}, named or {}, **{**ON, **kw})


# ── footer_rows ───────────────────────────────────────────────────────────

def test_model_named():
    (r,) = rows([det(1.0)], [rec(1.0, visit_cat="Cat1", prediction="Cat1", confidence=0.9,
                                 votes={"Cat1": 9})])["rows"]
    assert (r["label"], r["named_by"], r["prompt"]) == ("Cat1", "model", False)
    assert r["detail"] == "Cat1 9 of the last 10 frames" and r["conf"] == "90%"


def test_model_named_then_unclear():
    (r,) = rows([det(1.0)], [rec(1.0, visit_cat="Cat1", ood_rejected=True)])["rows"]
    assert r["label"] == "Cat1" and r["detail"] == "unclear right now" and not r["prompt"]


def test_voting_does_not_ask():
    (r,) = rows([det(1.0)], [rec(1.0, prediction="Cat2", confidence=0.8, votes={"Cat2": 3})])["rows"]
    assert r["label"] == "Cat2?" and not r["prompt"] and r["pending"]


def test_no_crop_yet():
    (r,) = rows([det(1.0)])["rows"]
    assert r["label"] == "Identifying..." and not r["prompt"]


@pytest.mark.parametrize("kw,label", [
    (dict(ood_rejected=True), "Unknown cat. Who is this?"),
    (dict(prediction="Cat2"), "Who is this? The model thinks Cat2, but is not sure."),
    (dict(), "Who is this?"),
])
def test_unknown_asks(kw, label):
    (r,) = rows([det(1.0)], [rec(1.0, unknown=True, **kw)])["rows"]
    assert r["label"] == label and r["prompt"] and r["tone"] == "warn"


def test_name_taken_asks():
    (r,) = rows([det(1.0)], [rec(1.0, unknown=True, name_taken="Cat2")])["rows"]
    assert r["label"] == "Looks like Cat2, but Cat2 is already here. Who is this?" and r["prompt"]


@pytest.mark.parametrize("kw", [dict(enabled=False), dict(model_exists=False), dict(worker_error="boom")])
def test_no_working_model_asks_at_once(kw):
    (r,) = rows([det(1.0)], **kw)["rows"]
    assert r["label"] == "Who is this?" and r["prompt"]


def test_named_by_you_wins_and_offers_correction():
    named = {1.0: {"cat": "Cat3", "saved": 2, "moved": 1, "in_view": True}}
    (r,) = rows([det(1.0)], [rec(1.0, visit_cat="Cat1")], named)["rows"]
    assert (r["label"], r["named_by"], r["prompt"], r["stop"]) == ("Cat3", "you", True, True)
    assert r["detail"].startswith("named by you · 3 images saved")


def test_named_visit_out_of_view_keeps_a_row():
    named = {1.0: {"cat": "Cat3", "saved": 0, "moved": 0, "in_view": False}}
    (r,) = rows([], named=named)["rows"]
    assert "stopping (no cat in view)" in r["detail"]


def test_two_rows_oldest_first_and_unstable_ignored():
    out = rows([det(2.0), det(1.0), det(3.0, stable=False)],
               [rec(1.0, visit_cat="Cat1"), rec(2.0, unknown=True)])
    assert [r["visit_id"] for r in out["rows"]] == [1.0, 2.0]
    assert [r["prompt"] for r in out["rows"]] == [False, True]


@pytest.mark.parametrize("kw,kind", [
    (dict(running=False), "not_running"), (dict(enabled=False), "ident_off"),
    (dict(model_exists=False), "no_model"), (dict(worker_error="Model not found: x"), "loading"),
    (dict(worker_error="boom"), "error"), (dict(), "waiting")])
def test_idle(kw, kind):
    out = rows([], **kw)
    assert out["rows"] == [] and out["idle"]["kind"] == kind


# ── VisitNamer ────────────────────────────────────────────────────────────

class Reviewer:
    def __init__(self, root: Path):
        self.data_dir = root
        self.reviewed = []

    def label_paths(self, paths, label):
        out = []
        for p in paths:
            dest = self.data_dir / label / p.name.replace(p.parent.name, label, 1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            p.rename(dest)
            if p.with_suffix(".json").exists():
                p.with_suffix(".json").rename(dest.with_suffix(".json"))
            out.append({"status": "ok", "filename": dest.name})
        return out

    def mark_reviewed_paths(self, paths):
        self.reviewed += list(paths)


@pytest.fixture
def namer(tmp_path):
    return VisitNamer(["Cat1", "Cat2", "Cat3"], 3.0, Reviewer(tmp_path))


def test_name_and_captures_every_interval(namer):
    namer.observe([1.0], 100)
    assert namer.name("Cat1", 1.0)["status"] == "ok"
    assert namer.due_captures(100) == [(1.0, "Cat1")]
    assert namer.due_captures(101) == []
    assert namer.due_captures(103.1) == [(1.0, "Cat1")]
    assert namer.names() == {"1.0": "Cat1"}


def test_name_follows_the_animal_through_a_dropout(namer):
    namer.observe([1.0], 100); namer.name("Cat1", 1.0)
    namer.observe([], 101)
    namer.observe([2.0], 105)                           # new visit, the only one in view
    assert namer.by_visit() == {2.0: {"cat": "Cat1", "saved": 0, "moved": 0, "in_view": True}}
    assert namer.due_captures(105) == [(2.0, "Cat1")]


def test_no_inheritance_with_two_in_view(namer):
    namer.observe([1.0], 100); namer.name("Cat1", 1.0)
    namer.observe([], 101)
    namer.observe([2.0, 3.0], 105)
    assert 2.0 not in namer.by_visit() and 3.0 not in namer.by_visit()


def test_two_named_visits(namer):
    namer.observe([1.0, 2.0], 100)
    namer.name("Cat1", 1.0); namer.name("Cat2", 2.0)
    assert sorted(namer.due_captures(100)) == [(1.0, "Cat1"), (2.0, "Cat2")]
    namer.on_output_line(json.dumps({"response": "capture_ok", "cat": "Cat2", "filename": "b.jpg", "visit_id": 2.0}))
    assert namer.by_visit()[2.0]["saved"] == 1 and namer.by_visit()[1.0]["saved"] == 0


def test_without_visit_id(namer):
    namer.observe([1.0, 2.0], 100)
    assert namer.name("Cat1")["status"] == "error"     # which one?
    namer.observe([1.0], 101)
    assert namer.name("Cat1")["status"] == "ok" and 1.0 in namer.by_visit()


def test_pending_name_given_to_the_next_visit(namer):
    namer.observe([], 100)
    namer.name("Cat2")
    namer.observe([5.0], 104)
    assert namer.by_visit()[5.0]["cat"] == "Cat2"


def test_finish_after_end_marks_reviewed(namer, tmp_path):
    namer.observe([1.0], 100); namer.name("Cat1", 1.0, now=100)
    namer.on_output_line(json.dumps({"response": "capture_ok", "cat": "Cat1", "filename": "a.jpg", "visit_id": 1.0}))
    (tmp_path / "Cat1").mkdir(); (tmp_path / "Cat1" / "a.jpg").write_text("x")
    namer.observe([], 105); namer.finish_if_ended(109)
    assert namer.by_visit()
    namer.finish_if_ended(111)
    assert not namer.by_visit() and namer.state()["last"]["saved"] == 1
    assert namer.reviewer.reviewed == [tmp_path / "Cat1" / "a.jpg"]


def test_correction_moves_saved_images(namer, tmp_path):
    namer.observe([1.0], 100); namer.name("Cat1", 1.0)
    (tmp_path / "Cat1").mkdir(); (tmp_path / "Cat1" / "Cat1_x.jpg").write_text("x")
    namer.on_output_line(json.dumps({"response": "capture_ok", "cat": "Cat1", "filename": "Cat1_x.jpg", "visit_id": 1.0}))
    namer.name("Cat2", 1.0)
    assert (tmp_path / "Cat2" / "Cat2_x.jpg").exists() and namer.by_visit()[1.0]["cat"] == "Cat2"


def test_claims_only_this_visits_unlabeled_images(namer, tmp_path):
    import time
    u = tmp_path / "unlabeled"; u.mkdir()
    for name, vid in [("unlabeled_a", 1.0), ("unlabeled_b", 2.0), ("unlabeled_c", None)]:
        (u / f"{name}.jpg").write_text("x")
        (u / f"{name}.json").write_text(json.dumps({"visit_id": vid}))
    now = time.time()
    namer.observe([1.0, 2.0], now)
    namer.name("Cat1", 1.0)
    assert sorted(p.name for p in u.glob("*.jpg")) == ["unlabeled_b.jpg", "unlabeled_c.jpg"]
    assert namer.by_visit()[1.0]["moved"] == 1


def test_stop_one_visit(namer):
    namer.observe([1.0, 2.0], 100)
    namer.name("Cat1", 1.0); namer.name("Cat2", 2.0)
    namer.stop(1.0)
    assert list(namer.by_visit()) == [2.0]
