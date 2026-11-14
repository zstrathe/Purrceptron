# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Run one remote training job from an extracted job bundle.

Every training backend (local PC worker, Hugging Face Jobs, ...) runs this
same script, so a backend only has to move files and start a process.

Bundle layout (built on the Pi by remote_training/bundle.py):

    <work>/manifest.json        cats, classifier_training settings, Pi torch version
    <work>/dataset/train/<cat>/ 224x224 crops
    <work>/dataset/val/<cat>/
    <work>/code/train/          copy of the Pi's train/ package (this file included)
    <work>/code/utils/          copy of the Pi's utils/ package

The script writes <work>/code/configs/project.yaml from the manifest so the
bundled code finds its own project root, then runs train_classifier.py with
the bundled code. That keeps training from touching the host checkout (for
example the PC's own models/cat_classifier.pt).

Output (in --output-dir, and packed into --output-tar when given):

    cat_classifier.pt     INT8 TorchScript with classes and energy threshold
    final_metrics.json    val accuracy, per-class accuracy, threshold
    training/results.csv  per-epoch metrics
    status.json           exit code, duration, host torch version

Usage:
    python <work>/code/train/run_job.py --work <work> \\
        [--output-dir <dir>] [--output-tar <file.tar.gz>]
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

# Files from the output dir that are worth sending back. Checkpoints in
# training/*.pt are large and only needed for resuming, so they stay behind.
_RESULT_FILES = ("cat_classifier.pt", "final_metrics.json", "status.json",
                 "training/results.csv")


def _write_config(work: Path, manifest: dict) -> Path:
    cfg = {
        "cats": manifest["cats"],
        "classifier_training": manifest.get("classifier_training", {}),
        "classifier": {"crop_padding": manifest.get("crop_padding", 0.2)},
    }
    cfg_path = work / "code" / "configs" / "project.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def _torch_info() -> dict:
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - reported, not fatal
        return {"torch_error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bundled classifier training job")
    parser.add_argument("--work", type=Path, required=True,
                        help="Extracted bundle directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where training writes its results (default: <work>/output)")
    parser.add_argument("--output-tar", type=Path, default=None,
                        help="Also pack the result files into this .tar.gz")
    args = parser.parse_args()

    work = args.work.resolve()
    output_dir = (args.output_dir or work / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((work / "manifest.json").read_text())
    host = _torch_info()
    print(f"[run_job] job {manifest.get('job_id')} on {platform.node()}", flush=True)
    print(f"[run_job] host torch {host.get('torch_version')}, "
          f"Pi torch {manifest.get('pi_torch_version')}, "
          f"GPU: {host.get('gpu') or 'none (CPU)'}", flush=True)
    if (host.get("torch_version", "").split("+")[0]
            != str(manifest.get("pi_torch_version", "")).split("+")[0]):
        print("[run_job] WARNING: torch version differs from the Pi's; "
              "the Pi will check that the model loads before deploying it.", flush=True)

    cfg_path = _write_config(work, manifest)
    cmd = [
        sys.executable, "-u", str(work / "code" / "train" / "train_classifier.py"),
        "--config", str(cfg_path),
        "--dataset-dir", str(work / "dataset"),
        "--output-dir", str(output_dir),
    ]
    print("[run_job] stage=training", flush=True)
    start = time.time()
    rc = subprocess.call(cmd, cwd=str(work / "code"))
    duration = round(time.time() - start, 1)

    model_ok = (output_dir / "cat_classifier.pt").is_file()
    status = {
        "job_id": manifest.get("job_id"),
        "exit_code": rc,
        "ok": rc == 0 and model_ok,
        "duration_seconds": duration,
        "host": platform.node(),
        **host,
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(f"[run_job] training finished: exit {rc}, model "
          f"{'written' if model_ok else 'MISSING'}, {duration:.0f}s", flush=True)

    if args.output_tar:
        # Build locally, then copy: the destination may be a mounted cloud
        # bucket where renames are not supported.
        local_tar = work / "results.tar.gz"
        with tarfile.open(local_tar, "w:gz") as tar:
            for rel in _RESULT_FILES:
                p = output_dir / rel
                if p.is_file():
                    tar.add(p, arcname=rel)
        args.output_tar.parent.mkdir(parents=True, exist_ok=True)
        if local_tar.resolve() != args.output_tar.resolve():
            shutil.copyfile(local_tar, args.output_tar)
        print(f"[run_job] results packed: {args.output_tar}", flush=True)

    print("[run_job] stage=done" if status["ok"] else "[run_job] stage=failed", flush=True)
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
