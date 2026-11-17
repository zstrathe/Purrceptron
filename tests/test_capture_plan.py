# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""How detection captures are named, from each visit's own status record."""
from deploy.detect import capture_plan, classifier_capture_label


def status(*visits, confirmed=()):
    return {"visits": list(visits), "confirmed_visits": [list(p) for p in confirmed]}


def visit(vid, cat=None, majority=None, **kw):
    return {"visit_id": vid, "visit_cat": cat, "confirmed_cat": majority,
            "last_crop_rejected": False, "name_taken": None, **kw}


def test_each_visit_gets_its_own_name():
    st = status(visit(1.0, "Cat1", "Cat1"), visit(2.0, "Cat2", "Cat2"))
    assert classifier_capture_label(st, 1.0) == ("Cat1", False)
    assert classifier_capture_label(st, 2.0) == ("Cat2", False)


def test_undecided_visit_is_eligible_for_back_labelling():
    st = status(visit(1.0), visit(2.0, "Cat2", "Cat2"))
    assert classifier_capture_label(st, 1.0) == (None, True)


def test_rejected_name_taken_and_moved_majority_are_never_named():
    assert classifier_capture_label(status(visit(1.0, last_crop_rejected=True)), 1.0) == (None, False)
    assert classifier_capture_label(status(visit(1.0, name_taken="Cat2")), 1.0) == (None, False)
    assert classifier_capture_label(status(visit(1.0, "Cat1", "Cat2")), 1.0) == (None, False)


def test_ended_visit_found_in_confirmed_list_newest_wins():
    st = status(confirmed=[(1.0, "Cat1"), (1.0, "Cat2")])
    assert classifier_capture_label(st, 1.0) == ("Cat2", False)


def test_no_status_or_no_visit():
    assert classifier_capture_label(None, 1.0) == (None, True)
    assert classifier_capture_label(status(), None) == (None, False)


def test_save_which():
    st = status(visit(1.0, "Cat1", "Cat1"), visit(2.0), visit(3.0, last_crop_rejected=True))
    assert capture_plan(st, 1.0, True, "unidentified")[2] is True
    assert capture_plan(st, 2.0, True, "identified")[2] is False      # may still be named
    assert capture_plan(st, 3.0, True, "identified")[2] is True
    assert capture_plan(st, 1.0, False, "identified") == (None, False, False)


# ── saving per animal ─────────────────────────────────────────────────────

import json
import time

import numpy as np

from deploy.detection_capturer import DetectionCapturer, VisitBackLabeler
from utils.sidecars import best_detection


def test_capturer_writes_subject_and_extra(tmp_path):
    cap = DetectionCapturer(tmp_path, ["Cat1"], min_interval_seconds=0, csv_logging=False)
    cap.start()
    frame = np.zeros((10, 10, 3), np.uint8)
    subj = {"label": "cat", "confidence": 0.7, "bbox_norm": [0.1, 0.1, 0.2, 0.2]}
    other = {"label": "cat", "confidence": 0.9, "bbox_norm": [0.6, 0.6, 0.2, 0.2]}
    a = cap.maybe_capture("Cat1", frame, detections=[subj], throttle_key="visit:1",
                          extra={"visit_id": 1.0, "other_detections": [other]})
    b = cap.maybe_capture("Cat2", frame, detections=[other], throttle_key="visit:2",
                          extra={"visit_id": 2.0, "other_detections": [subj]})
    cap.shutdown()
    assert a and b and a != b
    meta = json.loads(a.with_suffix(".json").read_text())
    assert meta["visit_id"] == 1.0 and len(meta["detections"]) == 1
    # Training crops the subject, even though the other animal is more confident.
    assert best_detection(meta)["bbox_norm"] == [0.1, 0.1, 0.2, 0.2]


def test_capturer_forget_throttle():
    cap = DetectionCapturer.__new__(DetectionCapturer)
    import threading
    cap._lock = threading.Lock(); cap._last_capture_time = {"visit:1": time.time()}; cap.min_interval = 60
    assert not cap.should_capture("visit:1")
    cap.forget("visit:1")
    assert cap.should_capture("visit:1")


def test_backlabeler_with_several_visits():
    bl = VisitBackLabeler(expire_after_seconds=30)
    bl.add(1.0, "a.jpg", now=100); bl.add(2.0, "b.jpg", now=100)
    assert bl.poll({1.0: "Cat1"}) == [(["a.jpg"], "Cat1")]
    assert bl.expire({2.0: 125.0}, now=140) == []          # still in progress
    assert bl.expire({}, now=150) == []                    # ended at 125: not yet 30 s
    assert bl.expire({}, now=156) == ["b.jpg"]


def test_crop_images_uses_the_subject(tmp_path):
    import cv2
    from train.crop_images import process_cat
    src = tmp_path / "src"; src.mkdir()
    img = np.zeros((100, 100, 3), np.uint8); img[10:30, 10:30] = 255
    cv2.imwrite(str(src / "x.jpg"), img)
    (src / "x.json").write_text(json.dumps({"detections": [
        {"label": "cat", "confidence": 0.5, "bbox_norm": [0.1, 0.1, 0.2, 0.2]}],
        "other_detections": [{"label": "cat", "confidence": 0.99, "bbox_norm": [0.6, 0.6, 0.3, 0.3]}]}))
    assert process_cat(src, tmp_path / "out", 0.0, 20)[0] == 1
    assert cv2.imread(str(tmp_path / "out" / "x.jpg")).mean() > 200     # the white square
