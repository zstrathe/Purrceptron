# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Score classifiers on the test set's crops, the way the live view uses them.

Run as a subprocess by the job manager (like validate_model), so ExecuTorch
never loads into the dashboard. A crop counts as right only when the model
names the right cat and doesn't reject it as an unknown cat. Prints one JSON
object: {"<model path>": {"overall": {correct, total}, "per_cat": {cat:
{correct, total}}}, ...} and exits 0, or {"error": ...} and exits 1.

    python -m remote_training.evaluate_model --test-dir job/test \\
        --model new.pte --model deployed.pte [--config configs/project.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_IMG_EXT = {".jpg", ".jpeg"}


def evaluate(model_path: Path, test_dir: Path, cfg: dict) -> dict:
    import cv2
    from deploy.cat_classifier import CatModel
    model = CatModel(model_path, cfg, log=lambda m: None)
    per_cat: dict[str, dict[str, int]] = {}
    for cat_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        cat = cat_dir.name
        right = total = 0
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXT:
                continue
            crop = cv2.imread(str(img_path))
            if crop is None:
                continue
            if crop.shape[:2] != (224, 224):
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            idx, _conf, _energy, rejected = model.classify(crop)
            total += 1
            right += int(not rejected and 0 <= idx < len(model.cats) and model.cats[idx] == cat)
        per_cat[cat] = {"correct": right, "total": total}
    return {"overall": {"correct": sum(v["correct"] for v in per_cat.values()),
                        "total": sum(v["total"] for v in per_cat.values())},
            "per_cat": per_cat}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, action="append", required=True)
    ap.add_argument("--config", type=Path, default=None,
                    help="Project config, for the unknown-cat settings the live view uses")
    args = ap.parse_args()
    try:
        import torch
        torch.set_num_threads(1)
        cfg: dict = {}
        if args.config:
            from utils import load_config
            cfg = dict(load_config(args.config).get("classifier") or {})
        out = {str(m): evaluate(m, args.test_dir, cfg) for m in args.model}
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
