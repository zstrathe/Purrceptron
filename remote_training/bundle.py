# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Build a job bundle on the Pi.

Packs the job's input, a copy of train/ and utils/, and a manifest into one
.tar.gz. The input depends on the task (see train/run_job.py for the layout
the remote side expects):

    train     crops of the labeled images in data/raw/<cat>/, split by session
    crossval  the same crops, not split
    detect    full images in data/raw/ that have no bounding box yet
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
from utils.sidecars import has_bbox

_IMG_EXT = {".jpg", ".jpeg"}

TASKS = ("train", "crossval", "detect")


class PackagingError(RuntimeError):
    pass


def pi_torch_version() -> str:
    """Torch version of this venv, read without importing torch."""
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return ""


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXT) if d.is_dir() else []


def count_trainable_images(raw_dir: Path, cats: list[str]) -> dict[str, dict[str, int]]:
    """Per cat: labeled images, and how many have a bounding box (only those
    can be cropped and used for training)."""
    counts = {}
    for cat in cats:
        imgs = _images(raw_dir / cat)
        counts[cat] = {"images": len(imgs), "with_bbox": sum(1 for p in imgs if has_bbox(p))}
    return counts


def images_missing_bbox(raw_dir: Path, cats: list[str]) -> list[Path]:
    """Images in the cat folders and unlabeled/ that have no bounding box."""
    return [p for folder in (*cats, "unlabeled") for p in _images(raw_dir / folder)
            if not has_bbox(p)]


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
    return {split: {cat: len(_images(dataset_dir / split / cat)) for cat in cats}
            for split in ("train", "val")}


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
    task: str = "train",
    cv_folds: int | None = None,
) -> tuple[Path, dict]:
    """Prepare the task's input and pack it. Returns (bundle_tar, manifest)."""
    if task not in TASKS:
        raise PackagingError(f"Unknown task {task!r}")
    cats: list[str] = config["cats"]
    tc = config.get("classifier_training", {})
    raw_dir = resolve_path(config.get("collection", {}).get("output_dir", "data/raw"),
                           project_root)
    crop_dir = job_dir / "cropped"
    dataset_dir = job_dir / "dataset"
    padding = config.get("classifier", {}).get("crop_padding", 0.20)
    prefix = _low_priority_prefix(config)
    manifest = {
        "job_id": job_id,
        "task": task,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cats": cats,
        "classifier_training": tc,
        "crop_padding": padding,
        "pi_torch_version": pi_torch_version(),
        "git_commit": _git_commit(project_root),
    }
    # (path inside the bundle, path on disk)
    payload: list[tuple[str, Path]] = []

    if task in ("train", "crossval"):
        log(f"Cropping labeled images from {raw_dir} (padding {padding})")
        _run("crop_images.py", prefix + [python_exe, "train/crop_images.py",
                       "--input-dir", str(raw_dir), "--output-dir", str(crop_dir),
                       "--padding", str(padding), "--config", str(config_path)],
             project_root, log, should_stop)

    if task == "train":
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
        manifest["split_counts"] = split
        total = sum(sum(v.values()) for v in split.values())
        payload.append(("dataset", dataset_dir))
    elif task == "crossval":
        folds = int(cv_folds or tc.get("cv_folds", 5))
        counts = {cat: len(_images(crop_dir / cat)) for cat in cats}
        short = [c for c, n in counts.items() if n < folds]
        if short:
            raise PackagingError(f"{folds}-fold cross-validation needs at least {folds} "
                                 "cropped images per cat; too few for: " + ", ".join(short))
        manifest["cv_folds"] = folds
        manifest["crop_counts"] = counts
        total = sum(counts.values())
        payload.append(("cropped", crop_dir))
    else:
        missing = images_missing_bbox(raw_dir, cats)
        if not missing:
            raise PackagingError("Every image already has a bounding box.")
        manifest["images"] = [p.relative_to(raw_dir).as_posix() for p in missing]
        total = len(missing)
        log(f"{total} images without a bounding box")
        payload += [(f"images/{p.relative_to(raw_dir).as_posix()}", p) for p in missing]

    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    bundle_tar = job_dir / "bundle.tar.gz"
    log("Packing bundle")
    t0 = time.time()

    def _skip_cache(ti: tarfile.TarInfo):
        return None if "__pycache__" in ti.name else ti

    # dereference: the split script symlinks crops instead of copying them
    with tarfile.open(bundle_tar, "w:gz", dereference=True) as tar:
        tar.add(job_dir / "manifest.json", arcname="manifest.json")
        for arcname, path in payload:
            if should_stop():
                raise PackagingError("Cancelled")
            tar.add(path, arcname=arcname)
        for pkg in ("train", "utils"):
            tar.add(project_root / pkg, arcname=f"code/{pkg}", filter=_skip_cache)
    shutil.rmtree(crop_dir, ignore_errors=True)
    shutil.rmtree(dataset_dir, ignore_errors=True)

    size_mb = bundle_tar.stat().st_size / 1_048_576
    noun = "images" if task == "detect" else "crops"
    log(f"Bundle ready: {total} {noun}, {size_mb:.1f} MB ({time.time() - t0:.0f}s to pack)")
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
