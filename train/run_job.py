# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Run one remote job from an extracted job bundle.

Every training backend (local PC worker, Hugging Face Jobs, ...) runs this
same script, so a backend only has to move files and start a process. The
manifest's ``task`` says what to run:

    train     train_classifier.py on dataset/   -> model, metrics, results.csv
    crossval  cross_validate.py on cropped/     -> cv_summary.json
    detect    auto_detect.py on images/         -> detections.json

Bundle layout (built on the Pi by remote_training/bundle.py):

    <work>/manifest.json        task, cats, classifier_training settings, Pi torch version
    <work>/dataset/{train,val}/<cat>/   224x224 crops (train)
    <work>/cropped/<cat>/       224x224 crops, not split (crossval)
    <work>/images/<folder>/     full images without a bounding box (detect)
    <work>/code/train/          copy of the Pi's train/ package (this file included)
    <work>/code/utils/          copy of the Pi's utils/ package

The script writes <work>/code/configs/project.yaml from the manifest so the
bundled code finds its own project root, then runs the bundled code. That
keeps a job from touching the host checkout.

Output (in --output-dir, and packed into --output-tar when given):

    train:    cat_classifier.pte, final_metrics.json, training/results.csv
    crossval: cv_summary.json
    detect:   detections.json
    always:   status.json (exit code, duration, host torch and ExecuTorch versions)

Usage:
    python <work>/code/train/run_job.py --work <work> \\
        [--output-dir <dir>] [--output-tar <file.tar.gz>]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from importlib import metadata
from pathlib import Path

import yaml

# Per task: the file that proves success, and the files worth sending back.
# Checkpoints (training/*.pt, CV fold models) are large and stay behind.
_TASKS = {
    "train": ("cat_classifier.pte",
              ("cat_classifier.pte", "final_metrics.json", "training/results.csv")),
    "crossval": ("cv_summary.json", ("cv_summary.json",)),
    "detect": ("detections.json", ("detections.json",)),
}


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


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _torch_info() -> dict:
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "executorch_version": _package_version("executorch"),
            "torchao_version": _package_version("torchao"),
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
          f"ExecuTorch {host.get('executorch_version')} "
          f"(Pi: {manifest.get('pi_executorch_version') or 'unknown'}), "
          f"GPU: {host.get('gpu') or 'none (CPU)'}", flush=True)
    host_et = str(host.get("executorch_version") or "").split("+")[0]
    pi_et = str(manifest.get("pi_executorch_version") or "").split("+")[0]
    if host_et and pi_et and host_et != pi_et:
        print("[run_job] WARNING: ExecuTorch version differs from the Pi's; "
              "the Pi will check that the model loads before deploying it.", flush=True)

    task = manifest.get("task", "train")
    if task not in _TASKS:
        print(f"[run_job] unknown task {task!r}", flush=True)
        return 2
    proof, result_files = _TASKS[task]
    if task in ("train", "crossval") and not (host.get("executorch_version")
                                               and host.get("torchao_version")):
        print("[run_job] ERROR: this Python has no ExecuTorch/torchao, which quantization "
              "and the model export need. Install requirements-training.txt.", flush=True)
        return 3

    cfg_path = _write_config(work, manifest)
    scripts = work / "code" / "train"
    if task == "train":
        cmd = [sys.executable, "-u", str(scripts / "train_classifier.py"),
               "--config", str(cfg_path), "--dataset-dir", str(work / "dataset"),
               "--output-dir", str(output_dir)]
    elif task == "crossval":
        cmd = [sys.executable, "-u", str(scripts / "cross_validate.py"),
               "--config", str(cfg_path), "--source-dir", str(work / "cropped"),
               "--output-dir", str(output_dir),
               "--n-folds", str(manifest.get("cv_folds", 5))]
    else:
        cmd = [sys.executable, "-u", str(scripts / "auto_detect.py"),
               "--images-dir", str(work / "images"),
               "--output", str(output_dir / "detections.json")]
    print(f"[run_job] stage=training task={task}", flush=True)
    start = time.time()
    # Library deprecation notices repeat in every data-loader worker and
    # bury the progress lines in the dashboard's log; nothing to act on.
    # (Message prefixes only: PYTHONWARNINGS splits entries on commas.)
    quiet = ["ignore:`isinstance(treespec:FutureWarning",
             "ignore:torch.ao.quantization is deprecated:DeprecationWarning",
             "ignore:ShiftScaleRotate is a special case of Affine:UserWarning"]
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = ",".join(filter(None, [env.get("PYTHONWARNINGS", "")] + quiet))
    rc = subprocess.call(cmd, cwd=str(work / "code"), env=env)
    duration = round(time.time() - start, 1)

    proof_ok = (output_dir / proof).is_file()
    status = {
        "job_id": manifest.get("job_id"),
        "task": task,
        "exit_code": rc,
        "ok": rc == 0 and proof_ok,
        "duration_seconds": duration,
        "host": platform.node(),
        **host,
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(f"[run_job] {task} finished: exit {rc}, {proof} "
          f"{'written' if proof_ok else 'MISSING'}, {duration:.0f}s", flush=True)

    if args.output_tar:
        # Build locally, then copy: the destination may be a mounted cloud
        # bucket where renames are not supported.
        local_tar = work / "results.tar.gz"
        with tarfile.open(local_tar, "w:gz") as tar:
            for rel in (*result_files, "status.json"):
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
