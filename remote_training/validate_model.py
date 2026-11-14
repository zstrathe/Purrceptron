# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Check that a returned classifier works with this machine's torch.

Run as a subprocess by the job manager so torch never loads into the
dashboard process. Prints one JSON object and exits 0 if the model is
usable, 1 otherwise.

    python -m remote_training.validate_model --model m.pt --cats Cat1,Cat2
"""
from __future__ import annotations

import argparse
import json
import sys


def validate(model_path: str, cats: list[str]) -> dict:
    result: dict = {"ok": False, "errors": []}
    try:
        import torch
    except ImportError as exc:
        result["errors"].append(f"torch not available: {exc}")
        return result
    result["torch_version"] = torch.__version__
    torch.set_num_threads(1)
    torch.backends.quantized.engine = "qnnpack"

    extra = {"classes.json": "", "energy_threshold": ""}
    try:
        model = torch.jit.load(model_path, map_location="cpu", _extra_files=extra)
        model.eval()
    except Exception as exc:
        result["errors"].append(f"Model does not load with torch {torch.__version__}: {exc}")
        return result

    classes = json.loads(extra["classes.json"]) if extra["classes.json"] else None
    result["classes"] = classes
    if classes is None:
        result["errors"].append("Model has no embedded class list")
    elif sorted(classes) != sorted(cats):
        result["errors"].append(f"Model classes {classes} do not match configured cats {cats}")

    if extra["energy_threshold"]:
        result["energy_threshold"] = json.loads(extra["energy_threshold"])
    else:
        result["errors"].append("Model has no embedded energy threshold")

    try:
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 224, 224))
        result["output_shape"] = list(out.shape)
        if tuple(out.shape) != (1, len(cats)):
            result["errors"].append(f"Output shape {tuple(out.shape)}, expected (1, {len(cats)})")
    except Exception as exc:
        result["errors"].append(f"Test inference failed: {exc}")

    result["ok"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cats", required=True, help="Comma-separated configured cat names")
    args = parser.parse_args()
    res = validate(args.model, [c for c in args.cats.split(",") if c])
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
