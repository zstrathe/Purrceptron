# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Bounding-box sidecars: <image>.json next to each captured image.

Format (written by the Pi's detection capture, read by train/crop_images.py):

    {"timestamp": ..., "session_id": ..., "image_size": [w, h],
     "detections": [{"label": "cat", "confidence": 0.63,
                     "bbox_norm": [xmin, ymin, w, h]}],   # normalized 0-1
     "original_detections": [...],   # only after a bounding box was edited
     "created_by": "annotate"}       # only when the sidecar was made by hand

Cropping uses the highest-confidence detection. Editing a Pi sidecar keeps the
Pi's detections in ``original_detections`` so clearing the edit restores them;
an untouched Pi sidecar is never deleted.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def sidecar_path(img_path: Path) -> Path:
    return img_path.with_suffix(".json")


def read_sidecar(img_path: Path) -> dict | None:
    sidecar = sidecar_path(img_path)
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return None


def write_sidecar(img_path: Path, meta: dict) -> None:
    """Write atomically (temp file, then rename)."""
    sidecar = sidecar_path(img_path)
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta))
    tmp.replace(sidecar)


def best_detection(meta: dict | None) -> dict | None:
    """Highest-confidence detection with a valid bounding box (the same rule
    crop_images.py uses)."""
    if not meta:
        return None
    valid = [d for d in meta.get("detections", [])
             if isinstance(d.get("bbox_norm"), list) and len(d["bbox_norm"]) == 4]
    return max(valid, key=lambda d: d.get("confidence", 0)) if valid else None


def has_bbox(img_path: Path) -> bool:
    return best_detection(read_sidecar(img_path)) is not None


def to_center(bbox_norm: list[float]) -> dict:
    xmin, ymin, w, h = (float(v) for v in bbox_norm)
    return {"cx": xmin + w / 2, "cy": ymin + h / 2, "w": w, "h": h}


def from_center(cx: float, cy: float, w: float, h: float) -> list[float] | None:
    """Center format to [xmin, ymin, w, h], clamped to the image. None if empty."""
    x1 = min(max(cx - w / 2, 0.0), 1.0)
    y1 = min(max(cy - h / 2, 0.0), 1.0)
    x2 = min(max(cx + w / 2, 0.0), 1.0)
    y2 = min(max(cy + h / 2, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(v, 6) for v in (x1, y1, x2 - x1, y2 - y1)]


def _image_size(img_path: Path) -> list[int] | None:
    try:
        import cv2
        img = cv2.imread(str(img_path))
        return [int(img.shape[1]), int(img.shape[0])] if img is not None else None
    except Exception:
        return None


def set_bbox(img_path: Path, bbox_norm: list[float], *, source: str,
             confidence: float = 1.0) -> None:
    """Replace the image's detections with one cat bounding box.

    ``source`` is "manual" or "auto-detect". The first edit of a Pi sidecar
    keeps the Pi's detections in ``original_detections``.
    """
    meta = read_sidecar(img_path)
    if meta is None:
        meta = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "session_id": "",
            "image_size": _image_size(img_path),
            "created_by": "annotate",
        }
    elif "original_detections" not in meta and meta.get("created_by") != "annotate":
        meta["original_detections"] = meta.get("detections", [])
    meta["detections"] = [{
        "label": "cat",
        "confidence": round(float(confidence), 4),
        "bbox_norm": [round(float(v), 6) for v in bbox_norm],
        "source": source,
    }]
    write_sidecar(img_path, meta)


def clear_bbox(img_path: Path) -> str:
    """Undo an edit. Returns "restored" (Pi detections back), "deleted"
    (hand-made sidecar removed), "kept_pi" (untouched Pi sidecar left alone)
    or "none"."""
    meta = read_sidecar(img_path)
    if meta is None:
        return "none"
    if "original_detections" in meta:
        meta["detections"] = meta.pop("original_detections")
        write_sidecar(img_path, meta)
        return "restored"
    if meta.get("created_by") == "annotate":
        sidecar_path(img_path).unlink()
        return "deleted"
    return "kept_pi"


def move_with_sidecar(src: Path, dest: Path) -> None:
    """Move an image and its sidecar together, keeping the image's mtime."""
    mtime = src.stat().st_mtime
    shutil.move(str(src), str(dest))
    os.utime(str(dest), (mtime, mtime))
    side = sidecar_path(src)
    if side.is_file():
        shutil.move(str(side), str(sidecar_path(dest)))
