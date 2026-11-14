# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Build a training job bundle on the Pi.

Runs the existing crop and train/val split scripts on the labeled images in
``collection.output_dir`` (data/raw/<cat>/), then packs the split crops, a
copy of train/ and utils/, and a manifest into one .tar.gz. See
train/run_job.py for the layout the remote side expects.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

from utils.config import resolve_path

_IMG_EXT = {".jpg", ".jpeg"}


class PackagingError(RuntimeError):
    pass


def pi_torch_version() -> str:
    """Torch version of this venv, read without importing torch."""
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return ""


def count_trainable_images(raw_dir: Path, cats: list[str]) -> dict[str, dict[str, int]]:
    """Per cat: labeled images, and how many have a bounding-box sidecar
    (only those can be cropped and used for training)."""
    counts = {}
    for cat in cats:
        d = raw_dir / cat
        imgs = [p for p in d.iterdir() if p.suffix.lower() in _IMG_EXT] if d.is_dir() else []
        counts[cat] = {
            "images": len(imgs),
            "with_bbox": sum(1 for p in imgs if p.with_suffix(".json").is_file()),
        }
    return counts


def _low_priority_prefix(config: dict) -> list[str]:
    """nice, and keep off the classifier's dedicated core, so packaging does
    not slow live detection."""
    prefix = []
    if shutil.which("nice"):
        prefix += ["nice", "-n", "10"]
    clf_core = config.get("classifier", {}).get("cpu_core")
    n = os.cpu_count() or 1
    if shutil.which("taskset") and clf_core is not None and n > 1:
        cpus = [str(c) for c in range(n) if c != clf_core]
        prefix += ["taskset", "-c", ",".join(cpus)]
    return prefix


def _run(label: str, cmd: list[str], cwd: Path, log: Callable[[str], None],
         should_stop: Callable[[], bool]) -> None:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
        if should_stop():
            proc.kill()
            proc.wait()
            raise PackagingError("Cancelled")
    rc = proc.wait()
    if rc != 0:
        raise PackagingError(f"{label} failed with exit code {rc}")


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=project_root,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _count_split(dataset_dir: Path, cats: list[str]) -> dict:
    out = {}
    for split in ("train", "val"):
        out[split] = {}
        for cat in cats:
            d = dataset_dir / split / cat
            out[split][cat] = sum(1 for p in d.iterdir()
                                  if p.suffix.lower() in _IMG_EXT) if d.is_dir() else 0
    return out


def build_bundle(
    job_id: str,
    job_dir: Path,
    *,
    project_root: Path,
    config_path: Path,
    config: dict,
    python_exe: str,
    log: Callable[[str], None],
    should_stop: Callable[[], bool] = lambda: False,
) -> tuple[Path, dict]:
    """Crop, split, and pack. Returns (bundle_tar, manifest)."""
    cats: list[str] = config["cats"]
    raw_dir = resolve_path(config.get("collection", {}).get("output_dir", "data/raw"),
                           project_root)
    crop_dir = job_dir / "cropped"
    dataset_dir = job_dir / "dataset"
    padding = config.get("classifier", {}).get("crop_padding", 0.20)
    prefix = _low_priority_prefix(config)

    log(f"Cropping labeled images from {raw_dir} (padding {padding})")
    _run("crop_images.py", prefix + [python_exe, "train/crop_images.py",
                   "--input-dir", str(raw_dir), "--output-dir", str(crop_dir),
                   "--padding", str(padding), "--config", str(config_path)],
         project_root, log, should_stop)

    log("Splitting into train/val by capture session")
    _run("prepare_classifier_dataset.py", prefix + [python_exe, "train/prepare_classifier_dataset.py",
                   "--source-dir", str(crop_dir), "--output-dir", str(dataset_dir),
                   "--config", str(config_path)],
         project_root, log, should_stop)

    split = _count_split(dataset_dir, cats)
    empty = [c for c in cats if not split["train"][c] or not split["val"][c]]
    if empty:
        raise PackagingError("No train or val images for: " + ", ".join(empty) +
                             ". Each cat needs images from at least two capture sessions.")

    manifest = {
        "job_id": job_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cats": cats,
        "classifier_training": config.get("classifier_training", {}),
        "crop_padding": padding,
        "pi_torch_version": pi_torch_version(),
        "git_commit": _git_commit(project_root),
        "split_counts": split,
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    bundle_tar = job_dir / "bundle.tar.gz"
    log("Packing bundle")
    t0 = time.time()

    def _skip_cache(ti: tarfile.TarInfo):
        return None if "__pycache__" in ti.name else ti

    # dereference: the split script symlinks crops instead of copying them
    with tarfile.open(bundle_tar, "w:gz", dereference=True) as tar:
        tar.add(job_dir / "manifest.json", arcname="manifest.json")
        tar.add(dataset_dir, arcname="dataset")
        for pkg in ("train", "utils"):
            tar.add(project_root / pkg, arcname=f"code/{pkg}", filter=_skip_cache)
    shutil.rmtree(crop_dir, ignore_errors=True)
    shutil.rmtree(dataset_dir, ignore_errors=True)

    size_mb = bundle_tar.stat().st_size / 1_048_576
    total = sum(sum(v.values()) for v in split.values())
    log(f"Bundle ready: {total} crops, {size_mb:.1f} MB ({time.time() - t0:.0f}s to pack)")
    manifest["bundle_mb"] = round(size_mb, 1)
    return bundle_tar, manifest


def safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract a tarball, refusing absolute paths, '..', and links that
    point outside dest."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        try:
            tar.extractall(dest, filter="data")
            return
        except TypeError:
            pass  # Python without extraction filters: check members by hand
        root = dest.resolve()
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if not (target == root or root in target.parents) or m.issym() or m.islnk() \
                    or m.isdev():
                raise PackagingError(f"Unsafe path in archive: {m.name}")
        tar.extractall(dest)
