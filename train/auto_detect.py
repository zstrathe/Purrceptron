# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Find the cat bounding box in still images (annotation helper).

Uses torchvision's COCO Faster R-CNN ResNet50-FPN-v2 (BSD-3). It was chosen
over Ultralytics YOLOv8n (AGPL-3.0) after comparing both against the Pi's
Hailo bounding boxes: it finds every cat, and classifier accuracy on its crops
matches. It is too heavy for the Pi's CPU, so it runs on the training machine:
one image at a time through the worker (train_worker.py), or as a batch job
on any training backend.

Usage (batch):
    python train/auto_detect.py --images-dir DIR --output detections.json

Output maps each image path relative to DIR to its best cat bounding box in
center format, or null when no cat was found:
    {"Cat2/x.jpg": {"cx": 0.5, "cy": 0.4, "w": 0.3, "h": 0.5, "confidence": 0.97}}
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

MIN_CONF = 0.15

_lock = threading.Lock()
_model = None  # (detector, preprocess, cat_label, device), loaded once


def _load():
    global _model
    if _model is None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=MIN_CONF,
        ).eval().to(device)
        _model = (detector, weights.transforms(),
                  weights.meta["categories"].index("cat"), device)
    return _model


def detect_best(image_path: Path) -> dict | None:
    """Best cat bounding box as {cx, cy, w, h, confidence} (normalized), or
    None. Thread-safe; the model loads on first use."""
    import torch
    from torchvision.io import ImageReadMode, read_image

    with _lock:  # load once; serialize GPU inference
        detector, preprocess, cat_label, device = _load()
        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        img_h, img_w = image.shape[1], image.shape[2]
        with torch.no_grad():
            output = detector([preprocess(image).to(device)])[0]

    best = None
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        conf = float(score)
        if int(label) == cat_label and conf >= MIN_CONF and (best is None or conf > best["confidence"]):
            x1, y1, x2, y2 = box.tolist()
            best = {
                "cx": ((x1 + x2) / 2) / img_w,
                "cy": ((y1 + y2) / 2) / img_h,
                "w": (x2 - x1) / img_w,
                "h": (y2 - y1) / img_h,
                "confidence": round(conf, 4),
            }
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-detect cat bounding boxes")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = sorted(p for p in args.images_dir.rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg"))
    print(f"Auto-detecting cats in {len(images)} images", flush=True)
    t0 = time.time()
    results: dict[str, dict | None] = {}
    found = 0
    for i, img in enumerate(images, 1):
        rel = img.relative_to(args.images_dir).as_posix()
        try:
            results[rel] = detect_best(img)
        except Exception as exc:
            print(f"  {rel}: failed ({exc})", flush=True)
            results[rel] = None
        found += results[rel] is not None
        if i % 25 == 0 or i == len(images):
            print(f"  {i}/{len(images)} done, cat found in {found}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    print(f"Found a cat in {found} of {len(images)} images ({time.time() - t0:.0f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
