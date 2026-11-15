# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Check that a returned classifier runs with this machine's ExecuTorch runtime.

Run as a subprocess by the job manager, so neither torch nor ExecuTorch loads
into the dashboard process, and a loader crash cannot take the dashboard down.
Prints one JSON object and exits 0 if the model is usable, 1 otherwise.

    python -m remote_training.validate_model --model m.pte --cats Cat1,Cat2
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata

# Flatbuffer file identifier of an ExecuTorch program, at bytes 4..8.
PTE_MAGIC = b"ET12"


def is_pte_file(path) -> bool:
    """True if the file starts like an ExecuTorch program. Checked before a
    file is handed to the loader, which aborts the whole process (no Python
    exception) on a file whose header is garbage."""
    try:
        with open(path, "rb") as f:
            return f.read(8)[4:8] == PTE_MAGIC
    except OSError:
        return False


def validate(model_path: str, cats: list[str]) -> dict:
    result: dict = {"ok": False, "errors": []}
    if not is_pte_file(model_path):
        result["errors"].append("Not an ExecuTorch model file (.pte)")
        return result
    try:
        import torch
        from executorch.runtime import Runtime
    except ImportError as exc:
        result["errors"].append(f"ExecuTorch runtime not available on the Pi: {exc}")
        return result
    result["torch_version"] = torch.__version__
    try:
        result["executorch_version"] = metadata.version("executorch")
    except metadata.PackageNotFoundError:
        result["executorch_version"] = None
    torch.set_num_threads(1)

    try:
        program = Runtime.get().load_program(model_path)
        forward = program.load_method("forward")
    except Exception as exc:
        result["errors"].append(
            f"Model does not load with ExecuTorch {result['executorch_version']}: {exc}")
        return result

    def constant(name: str):
        if name not in program.method_names:
            return None
        method = program.load_method(name)
        return json.loads(method.execute([])[0])

    try:
        classes = constant("classes_json")
        energy = constant("energy_threshold_json")
    except Exception as exc:
        result["errors"].append(f"Model metadata is unreadable: {exc}")
        return result
    result["classes"] = classes
    if classes is None:
        result["errors"].append("Model has no embedded class list")
    elif sorted(classes) != sorted(cats):
        result["errors"].append(f"Model classes {classes} do not match configured cats {cats}")
    if energy is None:
        result["errors"].append("Model has no embedded energy threshold")
    else:
        result["energy_threshold"] = energy

    try:
        out = forward.execute([torch.zeros(1, 3, 224, 224)])[0]
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
