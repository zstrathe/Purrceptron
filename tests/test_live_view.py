# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Names on each animal's bounding box in the live view."""
import json

import numpy as np

import deploy.detect as d
from deploy.visits import VisitTracker


def tracked(*boxes):
    tr = VisitTracker(labels={"cat"}, roi=(0, 0, 1920, 1080), frame_w=1920, frame_h=1080,
                      stability_frames=1, max_visits=8)
    dets = [{"label": "cat", "confidence": 0.9, "bbox_norm": b, "track_id": None} for b in boxes]
    visits, _ = tr.update(dets, now=10.0, mono=10.0)
    return dets, sorted(visits, key=lambda v: v.bbox_norm[0])


def status(*recs):
    return {"visits": list(recs), "confirmed_visits": []}


def test_display_texts():
    dets, (a, b) = tracked((0.1, 0.1, 0.2, 0.2), (0.6, 0.6, 0.2, 0.2))
    st = status({"visit_id": a.visit_id, "visit_cat": "Cat1"},
                {"visit_id": b.visit_id, "visit_cat": None, "prediction": "Cat2", "unknown": False})
    assert d.visit_display(a, st, {}, True) == ("Cat1", "model")
    assert d.visit_display(b, st, {}, True) == ("Cat2?", None)
    assert d.visit_display(b, st, {b.visit_id: "Cat3"}, True) == ("Cat3", "you")
    st["visits"][1]["unknown"] = True
    assert d.visit_display(b, st, {}, True) == ("?", None)
    assert d.visit_display(a, st, {}, False) == ("?", None)        # identification off
    assert d.visit_display(a, None, {}, True) == ("?", None)       # no record yet


def test_preview_json_and_drawing(tmp_path):
    dets, (a, b) = tracked((0.1, 0.1, 0.2, 0.2), (0.6, 0.6, 0.2, 0.2), (0.62, 0.62, 0.15, 0.15))
    d.label_visits(dets, [a, b], status({"visit_id": a.visit_id, "visit_cat": "Cat1"}), {}, True)
    d._BBOX_COLORS["Cat1"] = (0, 0, 255)
    w = d.PreviewFrameWriter(str(tmp_path / "p.jpg"), str(tmp_path / "p.json"), 15, 480, 270, 80)
    w.start()
    w.enqueue_frame(np.zeros((1080, 1920, 3), np.uint8), dets)
    w._queue.join()
    out = json.loads((tmp_path / "p.json").read_text())
    assert [(x["name"], x["named_by"], x["duplicate"]) for x in out] == [
        ("Cat1", "model", False), ("?", None, False), (None, None, True)]
    assert out[0]["visit_id"] == a.visit_id and out[0]["stable"]
    import cv2
    img = cv2.imread(str(tmp_path / "p.jpg"))
    x, y = int(0.1 * 480), int(0.1 * 270) + 10
    assert img[y, x - 1:x + 2, 2].max() > 150          # Cat1's red bounding box edge
    w.shutdown()


def test_held_visit_keeps_its_bounding_box_and_name():
    tr = VisitTracker(labels={"cat"}, roi=(0, 0, 1920, 1080), frame_w=1920, frame_h=1080,
                      stability_frames=1, hold_seconds=2.5)
    tr.update([{"label": "cat", "confidence": 0.9, "bbox_norm": (0.4, 0.4, 0.2, 0.2),
                "track_id": 3}], now=10.0, mono=10.0)
    visits, _ = tr.update([], now=11.5, mono=11.5)            # lost for 1.5 s: held
    (h,) = d.held_detections(visits)
    assert h["held"] and h["visit_id"] == visits[0].visit_id and h["bbox_norm"] == (0.4, 0.4, 0.2, 0.2)
    d.label_visits([h], visits, status({"visit_id": h["visit_id"], "visit_cat": "Cat1"}),
                   {}, identifying=True)
    assert h["name"] == "Cat1"
    thin, thick = np.zeros((60, 60, 3), np.uint8), np.zeros((60, 60, 3), np.uint8)
    box = {"label": "cat", "confidence": 0.9, "bbox": (5, 30, 55, 55), "name": None}
    d.draw_bounding_boxes(thin, [dict(box, held=True)])
    d.draw_bounding_boxes(thick, [box])
    assert 0 < (thin > 0).sum() < (thick > 0).sum()
    seen, _ = tr.update([{"label": "cat", "confidence": 0.9, "bbox_norm": (0.4, 0.4, 0.2, 0.2),
                          "track_id": 3}], now=11.6, mono=11.6)
    assert d.held_detections(seen) == []                     # back: drawn as a detection again
