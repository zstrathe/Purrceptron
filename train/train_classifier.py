# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Four-phase MobileNetV2-INT8 classifier training for cat identity.

Phases:
  1. FP32 head-only warm-up  (~phase1_epochs epochs)
  2. FP32 partial fine-tune  (~phase2_epochs epochs, last 3 backbone blocks)
  3. Quantization-Aware Training (~phase3_epochs epochs)
  4. INT8 export → cat_classifier.pte (ExecuTorch program, XNNPACK backend)

Reads hyperparameters from configs/project.yaml [classifier_training:].
Writes results.csv with columns: epoch, phase, train_loss, val_loss, train_acc, val_acc, acc_<cat>...
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import json
import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# Albumentations — training augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

# Torchvision
import torchvision.datasets
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("pyyaml required: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

def _decode(path) -> np.ndarray:
    """One image file as an RGB uint8 array (H, W, 3)."""
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _decode_all(paths) -> list[np.ndarray]:
    """Decode many images in parallel (Pillow releases the GIL while decoding)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
        return list(pool.map(_decode, paths))


class _AlbumentationsWrapper(torchvision.datasets.ImageFolder):
    """ImageFolder with an Albumentations transform pipeline.

    With cache=True every image is decoded once up front and kept in memory,
    instead of being decoded again each time it is used (every epoch, and
    several times per epoch for oversampled cats). DataLoader workers started
    with fork share the decoded images with this process, copy-on-write.
    """

    def __init__(self, root, transform_alb, cache: bool = False):
        super().__init__(root, transform=None)
        self._transform_alb = transform_alb
        self._images = _decode_all([p for p, _ in self.samples]) if cache else None

    def decoded_images(self) -> list[np.ndarray]:
        """All images, decoded (from the cache when there is one)."""
        return self._images if self._images is not None else _decode_all(
            [p for p, _ in self.samples])

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self._images[idx].copy() if self._images is not None else _decode(path)
        augmented = self._transform_alb(image=img)
        return augmented["image"], target


def _make_train_transform(aug_cfg: dict | None = None):
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations required: pip install albumentations")
    if aug_cfg is None:
        aug_cfg = {}

    transforms = [A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))]

    if aug_cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_cfg.get("brightness_contrast", True):
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get("brightness_limit", 0.3),
            contrast_limit=aug_cfg.get("contrast_limit", 0.3), p=0.5))
    if aug_cfg.get("shift_scale_rotate", True):
        # Same transform as the former A.ShiftScaleRotate(shift_limit=0.05,
        # scale_limit=0.05, rotate_limit=r), which Albumentations 2 implements
        # as exactly this Affine and deprecates.
        r = aug_cfg.get("rotate_limit", 10)
        transforms.append(A.Affine(
            scale=(0.95, 1.05), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-r, r), shear=0, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5))
    if aug_cfg.get("gaussian_blur", True):
        transforms.append(A.GaussianBlur(p=0.1))
    if aug_cfg.get("coarse_dropout", True):
        transforms.append(A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.15))
    if aug_cfg.get("gaussian_noise", True):
        transforms.append(A.GaussNoise(p=0.15))
    if aug_cfg.get("hue_saturation", False):
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5))
    if aug_cfg.get("clahe", False):
        transforms.append(A.CLAHE(clip_limit=2.0, p=0.3))

    transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def _make_val_transform():
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations required: pip install albumentations")
    # Images are already 224×224 — just normalise
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _val_transform_lut() -> torch.Tensor:
    """_make_val_transform() as a (3, 256) table: the float value it gives
    each possible uint8 value of each channel. Albumentations normalizes
    uint8 images through the same kind of table, so looking values up here
    reproduces its output exactly. Checked against the real transform, so a
    future change to _make_val_transform that isn't per-pixel fails loudly.
    """
    transform = _make_val_transform()
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 3, axis=2)  # (256,1,3)
    lut = transform(image=ramp)["image"][:, :, 0].clone()                       # (3,256)
    probe = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    looked_up = lut[torch.arange(3), torch.from_numpy(probe).long()].permute(2, 0, 1)
    if not torch.equal(looked_up, transform(image=probe)["image"]):
        raise RuntimeError("_make_val_transform is no longer a per-pixel normalization; "
                           "update DeviceBatches")
    return lut


class DeviceBatches:
    """Batches of fixed, unaugmented 224x224 crops held on one device (the GPU
    when there is one), normalized exactly as _make_val_transform() would.

    Used instead of a DataLoader for the validation set and for QAT training,
    whose inputs are the same every epoch: they are decoded once and never
    touch the CPU again, so the GPU isn't left waiting for data. Yields
    (images float32 NCHW, labels int64) like the DataLoaders it replaces.
    """

    def __init__(self, images: list[np.ndarray], labels, batch_size: int, device,
                 sampler=None, shuffle: bool = False, drop_last: bool = False):
        shapes = {im.shape for im in images}
        if not images or shapes != {(224, 224, 3)}:
            raise ValueError(f"expected 224x224 RGB crops, got {len(images)} images "
                             f"with shapes {sorted(shapes)}")
        self.device = torch.device(device)          # where batches are used
        # Where the images are kept: the GPU when they fit (see _storage_for)
        self.storage, self.placement = _storage_for(len(images) * _CROP_BYTES, self.device)
        # Not pinned when kept in CPU memory: page-locked memory is limited
        # (notably under WSL2), and the training DataLoader needs it for its own
        # batches. Copying one 4.8 MB uint8 batch at a time is cheap anyway.
        self._x = torch.from_numpy(np.stack(images)).to(self.storage)   # (N,224,224,3) uint8
        self._y = torch.as_tensor(list(labels), dtype=torch.long).to(self.storage)
        self._lut = _val_transform_lut().to(self.device)
        self.batch_size = batch_size
        self.sampler = sampler
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(self._y)

    def on(self, device) -> "DeviceBatches":
        """The same images and labels on another device (in order, no sampler)."""
        other = copy.copy(self)
        other.device = other.storage = torch.device(device)
        other._x, other._y, other._lut = (t.to(other.device) for t in (self._x, self._y, self._lut))
        other.sampler, other.shuffle, other.drop_last = None, False, False
        return other

    def __len__(self) -> int:
        n = len(self.sampler) if self.sampler is not None else self.num_samples
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        if self.sampler is not None:
            order = torch.as_tensor(list(self.sampler), dtype=torch.long)
        elif self.shuffle:
            order = torch.randperm(self.num_samples)
        else:
            order = torch.arange(self.num_samples)
        order = order.to(self.storage)
        n = len(order)
        stop = n - n % self.batch_size if self.drop_last else n
        channels = torch.arange(3, device=self.device)
        for start in range(0, stop, self.batch_size):
            idx = order[start:start + self.batch_size]
            x, y = self._x[idx], self._y[idx]
            if self.storage != self.device:             # kept in CPU memory: copy uint8
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
            x = self._lut[channels, x.long()]                             # (B,224,224,3) float32
            yield x.permute(0, 3, 1, 2).contiguous(), y


# ═══════════════════════════════════════════════════════════════════════════
# Machine resources: decide what to cache where from what this machine has
# ═══════════════════════════════════════════════════════════════════════════

# Shares of each resource the data pipeline may take. The rest is left for the
# model, its activations and gradients (GPU), and for everything else (RAM).
GPU_CACHE_SHARE = 0.4     # of free GPU memory, for images kept on the GPU
RAM_CACHE_SHARE = 0.5     # of available RAM, for decoded images
SHM_SHARE = 0.75          # of free /dev/shm, for batches in flight from loader workers
_CROP_BYTES = 224 * 224 * 3


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _usable_cpus() -> int:
    """CPUs this process may use: its CPU affinity, capped by a container's
    CPU quota (cgroup v2 or v1), which os.cpu_count() doesn't see."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:          # not on Linux
        n = os.cpu_count() or 1
    quota = None
    try:
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        q, period = (_read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
                     _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        if q and q > 0 and period:
            quota = q / period
    if quota:
        n = min(n, max(1, int(quota)))
    return max(1, n)


def _available_ram() -> int | None:
    """Bytes of RAM available to this process: MemAvailable, capped by a
    container's memory limit (cgroup v2 or v1). None if unknown."""
    avail = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
                break
    except OSError:
        return None
    for limit_f, usage_f in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                             ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                              "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        limit, usage = _read_int(limit_f), _read_int(usage_f)
        if limit and usage is not None and limit < (1 << 60):    # "max" / no limit
            room = max(0, limit - usage)
            avail = room if avail is None else min(avail, room)
            break
    return avail


def _shm_free() -> int | None:
    try:
        return shutil.disk_usage("/dev/shm").free
    except OSError:
        return None


def _gpu_free(device: torch.device) -> int:
    """Free GPU memory, counting memory PyTorch has reserved but isn't using."""
    free, _total = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def _storage_for(nbytes: int, device) -> tuple[torch.device, str]:
    """Where to keep nbytes of fixed images used on device: on the GPU when
    they fit in its share of free memory, otherwise in CPU memory (copied to
    the GPU one batch at a time)."""
    device = torch.device(device)
    if device.type != "cuda":
        return device, f"in CPU memory ({_gb(nbytes)})"
    try:
        free = _gpu_free(device)
    except Exception as exc:          # driver query failed: play safe
        return torch.device("cpu"), f"in CPU memory ({_gb(nbytes)}; GPU memory unknown: {exc})"
    if nbytes <= free * GPU_CACHE_SHARE:
        return device, f"on the GPU ({_gb(nbytes)}; {_gb(free)} free)"
    return (torch.device("cpu"),
            f"in CPU memory, copied per batch ({_gb(nbytes)} is more than "
            f"{GPU_CACHE_SHARE:.0%} of the {_gb(free)} free on the GPU)")


def _ram_cache_ok(n_images: int) -> tuple[bool, str]:
    """Whether n decoded 224x224 crops fit in RAM's share, with the reason."""
    need = n_images * _CROP_BYTES
    avail = _available_ram()
    if avail is None:
        return False, "decoded from disk each time (available RAM unknown)"
    if need <= avail * RAM_CACHE_SHARE:
        return True, f"decoded once and kept in RAM ({_gb(need)}; {_gb(avail)} available)"
    return False, (f"decoded from disk each time ({_gb(need)} would be more than "
                   f"{RAM_CACHE_SHARE:.0%} of the {_gb(avail)} RAM available)")


def _loader_plan(batch_bytes: int) -> tuple[int, int, str]:
    """(workers, prefetch per worker, description) for the augmented training
    loader. Workers: the usable CPUs minus two for the training loop, at most
    8. Batches waiting in /dev/shm (workers x prefetch) are kept within its
    share of free space; with too little, data loads in the training process."""
    cpus = _usable_cpus()
    workers, prefetch = min(8, max(1, cpus - 2)), 4
    shm = _shm_free()
    if shm is not None:
        fits = int(shm * SHM_SHARE // max(batch_bytes, 1))
        while workers * prefetch > fits and prefetch > 1:
            prefetch -= 1
        while workers * prefetch > fits and workers > 1:
            workers -= 1
        if workers * prefetch > fits:
            workers, prefetch = 0, 0
    shm_note = f"/dev/shm {_gb(shm)} free" if shm is not None else "no /dev/shm"
    what = (f"{workers} loader worker{'s' if workers != 1 else ''}, "
            f"{prefetch} batch{'es' if prefetch != 1 else ''} ahead each"
            if workers else "loading in the training process")
    return workers, prefetch, f"{what} ({cpus} usable CPUs, {shm_note})"


# ═══════════════════════════════════════════════════════════════════════════
# Class balance
# ═══════════════════════════════════════════════════════════════════════════

BALANCE_METHODS = ("loss_weights", "oversample", "off")


def _balance_method(tc: dict) -> str:
    """classifier_training.balance_method (default oversample). Older configs
    without it use balance_classes: false means off, anything else oversample."""
    method = tc.get("balance_method")
    if method is None:
        return "off" if tc.get("balance_classes") is False else "oversample"
    if method not in BALANCE_METHODS:
        raise ValueError(f"balance_method must be one of {', '.join(BALANCE_METHODS)}, "
                         f"not {method!r}")
    return method


class BalancedEpochSampler(Sampler):
    """Each epoch: every real image once, plus repeats of the smaller classes'
    images until every class has as many samples as the largest one.

    With random augmentation each repeat is a new variant of the image, so a
    small class gets extra, different training examples every epoch. Repeats
    are spread evenly (whole extra passes first, then a random subset without
    replacement), and the epoch is shuffled. Works for any number of classes.
    """

    def __init__(self, labels, generator: torch.Generator | None = None):
        by_class: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            by_class.setdefault(int(label), []).append(i)
        self._by_class = {c: torch.tensor(idx) for c, idx in sorted(by_class.items())}
        self.per_class = max((len(v) for v in self._by_class.values()), default=0)
        self.generator = generator

    def __len__(self) -> int:
        return self.per_class * len(self._by_class)

    def __iter__(self):
        parts = []
        for idx in self._by_class.values():
            repeats, rest = divmod(self.per_class, len(idx))
            parts.append(idx.repeat(repeats))
            if rest:
                parts.append(idx[torch.randperm(len(idx), generator=self.generator)[:rest]])
        epoch = torch.cat(parts)
        return iter(epoch[torch.randperm(len(epoch), generator=self.generator)].tolist())


def _make_criterion(train_ds, method: str, device, label_smoothing: float = 0.0):
    """Loss for the chosen balance method, with a log line describing it."""
    from collections import Counter
    counts = Counter(label for _, label in train_ds.samples)
    names = train_ds.classes
    largest = max(counts.values())
    if method == "loss_weights":
        weights = torch.tensor([largest / counts.get(i, 1) for i in range(len(names))],
                               dtype=torch.float32, device=device)
        print(f"Class balance: loss weights  "
              f"{dict(zip(names, [f'{w:.2f}' for w in weights.tolist()]))}")
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    if method == "oversample":
        extra = [f"{names[i]} {counts[i]} real + {largest - counts[i]} repeats"
                 for i in range(len(names)) if counts.get(i, 0) < largest]
        print(f"Class balance: oversample, {largest} per cat per epoch "
              f"({', '.join(extra) if extra else 'already balanced'})")
    else:
        print("Class balance: OFF (uniform loss)")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _reseed_augmentation(transform, seed: int) -> None:
    """Give an Albumentations pipeline, and each transform in it, its own
    random stream derived from seed.

    Albumentations 2 creates its random generators once, when the pipeline is
    built. DataLoader workers forked from the training process all inherit
    copies of those generators in the same state, so without reseeding every
    worker draws the same augmentation sequence (and, with workers re-forked
    each epoch, the same sequence every epoch). Compose.set_random_seed isn't
    used because it gives every transform the same seed, which would tie their
    random decisions together (e.g. flip whenever brightness changes).
    """
    import random as _random
    children = list(getattr(transform, "transforms", []))
    seeds = np.random.SeedSequence(seed).generate_state(len(children) + 1)
    own = int(seeds[0])
    transform.seed = own
    transform.random_generator = np.random.default_rng(own)
    transform.py_random = _random.Random(own)
    for child, s in zip(children, seeds[1:]):
        if hasattr(child, "transforms"):
            _reseed_augmentation(child, int(s))
        else:
            child.set_random_seed(int(s))


def _worker_init(_worker_id: int) -> None:
    # One OpenCV thread per worker: Albumentations runs on OpenCV, which would
    # otherwise start a thread per core in every worker and oversubscribe the CPU.
    cv2.setNumThreads(1)
    # Independent augmentation per worker (see _reseed_augmentation). The
    # worker's seed differs per worker and per DataLoader, as PyTorch sets it.
    info = torch.utils.data.get_worker_info()
    transform = getattr(info.dataset, "_transform_alb", None) if info else None
    if transform is not None:
        _reseed_augmentation(transform, info.seed)


def _build_loaders(dataset_dir: Path, batch_size: int, aug_cfg: dict | None = None,
                   oversample: bool = False, device="cpu"):
    """(augmented train DataLoader, val DeviceBatches, class names).

    Images are decoded once into memory when DataLoader workers start with
    fork (Linux), which lets them share the decoded images; otherwise each
    worker decodes from disk as before.
    """
    import torch.multiprocessing as tmp
    n_train = len(torchvision.datasets.ImageFolder(str(dataset_dir / "train")).samples)
    if tmp.get_context().get_start_method() == "fork":
        cache, cache_note = _ram_cache_ok(n_train)
    else:
        cache, cache_note = False, "decoded from disk each time (workers don't start with fork)"
    train_ds = _AlbumentationsWrapper(str(dataset_dir / "train"), _make_train_transform(aug_cfg),
                                      cache=cache)
    print(f"Training images: {cache_note}")
    val_ds = torchvision.datasets.ImageFolder(str(dataset_dir / "val"))
    # Label val images by cat name with the train set's numbering, in case a
    # split lacks a cat folder (ImageFolder numbers the folders it finds).
    val_labels = [train_ds.class_to_idx[val_ds.classes[t]] for t in val_ds.targets]
    sampler = (BalancedEpochSampler([label for _, label in train_ds.samples])
               if oversample else None)
    workers, prefetch, plan = _loader_plan(batch_size * _CROP_BYTES * 4)   # float32 batches
    print(f"Data loading: {plan}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=workers,
                              pin_memory=torch.cuda.is_available(), drop_last=True,
                              persistent_workers=workers > 0,
                              prefetch_factor=prefetch if workers else None,
                              worker_init_fn=_worker_init if workers else None)
    val_batches = DeviceBatches(_decode_all([p for p, _ in val_ds.samples]), val_labels,
                                batch_size, device)
    print(f"Validation images: {val_batches.placement}")
    return train_loader, val_batches, train_ds.classes


def _qat_train_batches(train_loader: DataLoader, batch_size: int, device,
                       oversample: bool) -> DeviceBatches:
    """Phase 3 training data: the training images without augmentation (as
    _make_val_transform), balanced the same way as phases 1-2, on the device."""
    ds = train_loader.dataset
    sampler = BalancedEpochSampler(ds.targets) if oversample else None
    batches = DeviceBatches(ds.decoded_images(), ds.targets, batch_size, device,
                            sampler=sampler, shuffle=sampler is None, drop_last=True)
    print(f"QAT training images: {batches.placement}")
    return batches


# ═══════════════════════════════════════════════════════════════════════════
# BN freeze helper
# ═══════════════════════════════════════════════════════════════════════════

def _freeze_bn(module: nn.Module) -> None:
    """Put all BatchNorm2d layers into eval mode and freeze their params."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


# ═══════════════════════════════════════════════════════════════════════════
# Quantization (PT2E) and ExecuTorch export
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_flatc_on_path() -> None:
    """ExecuTorch's exporter runs the `flatc` program that pip installs next to
    the interpreter. Make it findable when the venv is used without activating
    it (the worker starts jobs with the interpreter's full path)."""
    if shutil.which("flatc") is None:
        bin_dir = str(Path(sys.executable).parent)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """FP32 MobileNetV2 -> exported graph prepared for quantization-aware training.

    The XNNPACK quantizer inserts fake-quantize modules and handles conv+BN
    folding itself, so no fuse_model() call is needed. The result supports
    .train()/.eval(), .to(device), parameters() and state_dict() like a module.
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from torchao.quantization.pt2e import allow_exported_model_train_eval
    from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e

    model = model.cpu().train()
    # Batch 2 in the example: torch.export specializes a dimension whose example
    # size is 1, which would fix the batch size.
    example = (torch.randn(2, 3, 224, 224),)
    batch = torch.export.Dim("batch", min=1, max=1024)
    exported = torch.export.export(model, example, dynamic_shapes={"x": {0: batch}}).module()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
    prepared = prepare_qat_pt2e(exported, quantizer)
    allow_exported_model_train_eval(prepared)
    return prepared


def freeze_observers(prepared: nn.Module) -> None:
    """Stop updating quantization ranges (fake-quantize stays on)."""
    from torchao.quantization.pt2e import disable_observer
    prepared.apply(disable_observer)


def convert_to_int8(prepared: nn.Module) -> nn.Module:
    """Prepared QAT graph (any device) -> converted INT8 graph on the CPU."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
    prepared = prepared.cpu()
    prepared.eval()
    return convert_pt2e(prepared)


def lower_to_pte(quantized: nn.Module, constants: dict[str, str] | None = None) -> bytes:
    """Converted INT8 graph -> ExecuTorch program bytes (XNNPACK delegate, batch 1).

    constants become zero-argument methods in the program (e.g. classes_json),
    the replacement for TorchScript's _extra_files.
    """
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    _ensure_flatc_on_path()
    exported = torch.export.export(quantized, (torch.zeros(1, 3, 224, 224),))
    program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()], constant_methods=constants or None,
    ).to_executorch()
    return bytes(program.buffer)


def load_pte(source):
    """Load a .pte (path or bytes). Returns (program, forward_method).

    Keep the returned program alive while the method is in use, and when
    loading from bytes keep the bytes object alive too."""
    from executorch.runtime import Runtime
    program = Runtime.get().load_program(source if isinstance(source, bytes) else str(source))
    return program, program.load_method("forward")


def read_pte_constant(program, name: str) -> str | None:
    """A constant method's string value, or None if the program has no such method."""
    if name not in program.method_names:
        return None
    method = program.load_method(name)
    return method.execute([])[0]


@torch.no_grad()
def eval_pte(forward, loader, num_classes: int):
    """Run the batch-1 program over a loader, one crop at a time.

    Returns (accuracy, per_class_accuracy, energies) where energies is a 1-D
    tensor of -logsumexp(logits) per crop. Inputs are made contiguous: the
    runtime reads tensor memory directly and misreads non-contiguous views.
    """
    correct = 0
    total = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    energies = []
    for images, labels in loader:
        logits = torch.cat([
            forward.execute([images[i:i + 1].contiguous()])[0].clone()
            for i in range(images.size(0))
        ])
        energies.append(-torch.logsumexp(logits, dim=1))
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        for c in range(num_classes):
            mask = labels == c
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()
    per_class_acc = [per_class_correct[c] / max(per_class_total[c], 1) for c in range(num_classes)]
    return correct / max(total, 1), per_class_acc, torch.cat(energies)


# ═══════════════════════════════════════════════════════════════════════════
# Train / eval loops
# ═══════════════════════════════════════════════════════════════════════════

def _train_epoch(model, loader, optimizer, criterion, device,
                 mixup_alpha: float = 0.0,
                 freeze_bn: bool = False,
                 grad_clip_norm: float = 0.0,
                 eval_mode: bool = False) -> tuple[float, float]:
    """One training epoch.

    eval_mode trains with the model in eval mode: BatchNorm running statistics
    and dropout are frozen while gradients still flow. Used for the last part
    of QAT, where the model is an exported graph and _freeze_bn cannot reach
    its BatchNorm nodes.
    """
    if eval_mode:
        model.eval()
    else:
        model.train()
        if freeze_bn:
            _freeze_bn(model)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(images.size(0), device=device)
            images = lam * images + (1 - lam) * images[idx]
            logits = model(images)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[idx])
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device,
                num_classes: int) -> tuple[float, float, list[float]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = [0] * num_classes
    per_class_total   = [0] * num_classes

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        for c in range(num_classes):
            mask = labels == c
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c]   += mask.sum().item()

    per_class_acc = [
        per_class_correct[c] / max(per_class_total[c], 1)
        for c in range(num_classes)
    ]
    return total_loss / total, correct / total, per_class_acc


# ═══════════════════════════════════════════════════════════════════════════
# CSV logger
# ═══════════════════════════════════════════════════════════════════════════

class _CSVLogger:
    def __init__(self, path: Path, cat_names: list[str]):
        self._path = path
        self._cat_names = cat_names
        self._cols = (
            ["epoch", "phase", "train_loss", "val_loss", "train_acc", "val_acc"]
            + [f"acc_{c}" for c in cat_names]
        )
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(self._cols)

    def log(self, epoch: int, phase: str, train_loss: float, val_loss: float,
            train_acc: float, val_acc: float, per_class: list[float]) -> None:
        row = [epoch, phase, f"{train_loss:.4f}", f"{val_loss:.4f}",
               f"{train_acc:.4f}", f"{val_acc:.4f}"]
        row += [f"{a:.4f}" for a in per_class]
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 cat classifier")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "project.yaml")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Override (default: data/classifier_dataset)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override version output dir (default: versions/classifier/)")
    parser.add_argument("--skip-phase", type=int, action="append", default=[],
                        help="Skip a phase number (1-3); repeatable")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tc  = cfg.get("classifier_training", {})
    cats: list[str] = cfg.get("cats", [])
    num_cats = len(cats)
    if num_cats < 2:
        print("ERROR: need at least 2 cats in config", file=sys.stderr)
        sys.exit(1)

    # Hyperparameters
    phase1_epochs   = tc.get("phase1_epochs", 20)
    phase2_epochs   = tc.get("phase2_epochs", 40)
    phase3_epochs   = tc.get("phase3_epochs", 10)
    phase1_lr       = tc.get("phase1_lr", 1e-3)
    phase2_bb_lr    = tc.get("phase2_backbone_lr", 1e-5)
    phase2_head_lr  = tc.get("phase2_head_lr", 1e-4)
    weight_decay    = tc.get("weight_decay", 0.005)
    batch_size      = tc.get("batch_size", 32)
    patience        = tc.get("patience", 15)
    mixup_alpha     = tc.get("mixup_alpha", 0.2)
    balance_method  = _balance_method(tc)
    energy_ood_pct  = tc.get("energy_ood_percentile", 0.99)
    label_smoothing = tc.get("label_smoothing", 0.1)
    grad_clip_norm  = tc.get("gradient_clip_norm", 1.0)
    warmup_epochs   = tc.get("warmup_epochs", 5)
    aug_cfg         = tc.get("augmentation", {})

    dataset_dir = args.dataset_dir or (PROJECT_ROOT / "data" / "classifier_dataset")
    if not (dataset_dir / "train").is_dir():
        print(f"ERROR: dataset not found at {dataset_dir}. "
              "Run prepare_classifier_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (PROJECT_ROOT / "versions" / "classifier")
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "training"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Cats ({num_cats}): {cats}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")

    # Print augmentation config
    enabled_augs = [k for k, v in aug_cfg.items() if v is True]
    disabled_augs = [k for k, v in aug_cfg.items() if v is False]
    print(f"Augmentation ON:  {', '.join(enabled_augs) if enabled_augs else '(none)'}")
    print(f"Augmentation OFF: {', '.join(disabled_augs) if disabled_augs else '(none)'}")
    strength_keys = {k: v for k, v in aug_cfg.items() if isinstance(v, (int, float))}
    if strength_keys:
        print(f"Augmentation params: {strength_keys}")

    train_loader, val_loader, class_names = _build_loaders(
        dataset_dir, batch_size, aug_cfg, oversample=balance_method == "oversample",
        device=device)
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}  "
          f"(loader workers: {train_loader.num_workers})")
    print(f"Classes (dataset order): {class_names}")

    csv_logger = _CSVLogger(ckpt_dir / "results.csv", class_names)

    # Imbalanced classes: weighted loss, or oversampling of the smaller classes
    criterion = _make_criterion(train_loader.dataset, balance_method, device, label_smoothing)
    if label_smoothing > 0:
        print(f"Label smoothing: {label_smoothing}")

    phase1_ckpt = ckpt_dir / "phase1_best.pt"
    phase2_ckpt = ckpt_dir / "phase2_best.pt"
    phase3_ckpt = ckpt_dir / "phase3_best.pt"

    # ── Phase 1: FP32 head-only ──────────────────────────────────────────
    if 1 not in args.skip_phase:
        print("\n" + "="*60)
        print("Phase 1: FP32 head-only warm-up")
        print("="*60)

        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in model.features.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(1280, num_cats)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=phase1_lr, weight_decay=weight_decay
        )
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(phase1_epochs - warmup_epochs, 1), eta_min=1e-7
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )

        best_val_acc = 0.0
        no_improve = 0
        for epoch in range(1, phase1_epochs + 1):
            tl, ta = _train_epoch(model, train_loader, optimizer, criterion, device,
                                  grad_clip_norm=grad_clip_norm)
            vl, va, per_cls = _eval_epoch(model, val_loader, criterion, device, num_cats)
            scheduler.step()
            csv_logger.log(epoch, "phase1", tl, vl, ta, va, per_cls)
            print(f"  [P1 {epoch:3d}/{phase1_epochs}] "
                  f"loss {tl:.3f}/{vl:.3f}  acc {ta:.1%}/{va:.1%}")
            if va > best_val_acc:
                best_val_acc = va
                no_improve = 0
                torch.save({"model_state_dict": model.state_dict(),
                            "val_acc": va, "epoch": epoch}, phase1_ckpt)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} (patience {patience})")
                    break

        print(f"Phase 1 done. Best val acc: {best_val_acc:.1%}  → {phase1_ckpt}")
    else:
        print("Phase 1 skipped.")

    # ── Phase 2: FP32 partial fine-tune ─────────────────────────────────
    if 2 not in args.skip_phase:
        print("\n" + "="*60)
        print("Phase 2: FP32 partial fine-tune (last 3 backbone blocks)")
        print("="*60)

        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, num_cats)
        if phase1_ckpt.is_file():
            ckpt = torch.load(phase1_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded Phase 1 checkpoint (val_acc={ckpt.get('val_acc', '?'):.1%})")
        else:
            print("  WARNING: Phase 1 checkpoint not found; starting from pretrained weights")

        # Freeze early backbone; unfreeze last 3 blocks + classifier
        for p in model.features[:-3].parameters():
            p.requires_grad = False
        for p in model.features[-3:].parameters():
            p.requires_grad = True
        _freeze_bn(model)
        model = model.to(device)

        optimizer = torch.optim.AdamW([
            {"params": model.features[-3:].parameters(),
             "lr": phase2_bb_lr, "weight_decay": weight_decay},
            {"params": model.classifier.parameters(),
             "lr": phase2_head_lr, "weight_decay": weight_decay},
        ])
        p2_warmup = min(3, phase2_epochs // 4)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=p2_warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(phase2_epochs - p2_warmup, 1), eta_min=1e-7
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[p2_warmup]
        )

        # SWA setup -- activate for last 25% of Phase 2
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
        swa_start_epoch = max(1, int(phase2_epochs * 0.75))
        swa_scheduler = SWALR(optimizer, swa_lr=[phase2_bb_lr * 0.5, phase2_head_lr * 0.5])
        swa_active = False

        best_val_acc = 0.0
        no_improve = 0
        for epoch in range(1, phase2_epochs + 1):
            tl, ta = _train_epoch(model, train_loader, optimizer, criterion, device,
                                  mixup_alpha=mixup_alpha, freeze_bn=True,
                                  grad_clip_norm=grad_clip_norm)
            vl, va, per_cls = _eval_epoch(model, val_loader, criterion, device, num_cats)

            if epoch >= swa_start_epoch:
                if not swa_active:
                    print(f"  SWA activated at epoch {epoch}")
                    swa_active = True
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            # epoch offset so CSV is continuous
            csv_logger.log(phase1_epochs + epoch, "phase2", tl, vl, ta, va, per_cls)
            print(f"  [P2 {epoch:3d}/{phase2_epochs}] "
                  f"loss {tl:.3f}/{vl:.3f}  acc {ta:.1%}/{va:.1%}"
                  f"{'  [SWA]' if swa_active else ''}")
            if va > best_val_acc:
                best_val_acc = va
                no_improve = 0
                torch.save({"model_state_dict": model.state_dict(),
                            "val_acc": va, "epoch": epoch}, phase2_ckpt)
            else:
                no_improve += 1
                if not swa_active and no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} (patience {patience})")
                    break

        # Finalize SWA: update BN stats and evaluate
        if swa_active:
            print("  Updating BN statistics for SWA model...")
            update_bn(train_loader, swa_model, device=device)
            swa_vl, swa_va, swa_cls = _eval_epoch(
                swa_model, val_loader, criterion, device, num_cats)
            print(f"  SWA val acc: {swa_va:.1%} (vs best single: {best_val_acc:.1%})")
            if swa_va >= best_val_acc:
                best_val_acc = swa_va
                # Save SWA model's inner module state dict
                torch.save({"model_state_dict": swa_model.module.state_dict(),
                            "val_acc": swa_va, "epoch": phase2_epochs, "swa": True},
                           phase2_ckpt)
                print(f"  SWA checkpoint saved (improved)")

        print(f"Phase 2 done. Best val acc: {best_val_acc:.1%}  -> {phase2_ckpt}")
    else:
        print("Phase 2 skipped.")

    # ── Phase 3: QAT ────────────────────────────────────────────────────
    if 3 not in args.skip_phase:
        print("\n" + "="*60)
        print("Phase 3: Quantization-Aware Training (QAT)")
        print("="*60)

        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        qat_float = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        qat_float.classifier[1] = nn.Linear(1280, num_cats)

        src_ckpt = phase2_ckpt if phase2_ckpt.is_file() else phase1_ckpt
        if src_ckpt.is_file():
            ckpt = torch.load(src_ckpt, map_location="cpu")
            qat_float.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded checkpoint: {src_ckpt.name} "
                  f"(val_acc={ckpt.get('val_acc', '?'):.1%})")
        else:
            print("  WARNING: no prior checkpoint found; QAT from pretrained weights")

        # Insert fake-quantize (PT2E, XNNPACK quantizer). Unlike the old eager
        # QNNPACK flow, this trains on the GPU when there is one.
        qat_model = prepare_qat_model(qat_float).to(device)

        qat_lr = 1e-6  # ~1% of training LR for stable QAT
        optimizer = torch.optim.AdamW(
            qat_model.parameters(), lr=qat_lr, weight_decay=weight_decay
        )

        # QAT trains on the training images without augmentation, balanced like
        # phases 1-2 (repeats are plain copies here); both sets stay on the device.
        qat_train_loader = _qat_train_batches(train_loader, batch_size, device,
                                              oversample=balance_method == "oversample")
        qat_val_loader = val_loader

        best_val_acc = -1.0
        best_state = None
        no_improve = 0
        freeze_epoch = max(1, int(math.ceil(phase3_epochs * 0.8)))
        observers_frozen = False
        for epoch in range(1, phase3_epochs + 1):
            # Freeze observers and BN at 80% to stabilize quantization grid
            if epoch >= freeze_epoch and not observers_frozen:
                freeze_observers(qat_model)
                observers_frozen = True
                print(f"  Froze observers and BN at epoch {epoch}")
            # eval_mode keeps BatchNorm statistics frozen for those last epochs
            tl, ta = _train_epoch(qat_model, qat_train_loader, optimizer, criterion,
                                  device, grad_clip_norm=grad_clip_norm,
                                  eval_mode=observers_frozen)
            vl, va, per_cls = _eval_epoch(qat_model, qat_val_loader, criterion,
                                          device, num_cats)
            csv_logger.log(phase1_epochs + phase2_epochs + epoch, "phase3",
                           tl, vl, ta, va, per_cls)
            print(f"  [P3 {epoch:3d}/{phase3_epochs}] "
                  f"loss {tl:.3f}/{vl:.3f}  acc {ta:.1%}/{va:.1%}")
            if va > best_val_acc:
                best_val_acc = va
                no_improve = 0
                best_state = copy.deepcopy(qat_model.state_dict())
                torch.save({"model_state_dict": best_state,
                            "val_acc": va, "epoch": epoch}, phase3_ckpt)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} (patience {patience})")
                    break

        if best_state is None:
            print("ERROR: Phase 3 ran no epochs; set phase3_epochs to at least 1.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Phase 3 done. Best val acc: {best_val_acc:.1%}  → {phase3_ckpt}")

        # ── Phase 4: INT8 Export ─────────────────────────────────────────
        print("\n" + "="*60)
        print("Phase 4: INT8 export (ExecuTorch)")
        print("="*60)

        # Convert the best QAT epoch to a real INT8 graph (on the CPU)
        qat_model.load_state_dict(best_state)
        quantized = convert_to_int8(qat_model)

        # ── Energy threshold calibration ───────────────────────────────
        # Lower once without metadata and score the val set with the ExecuTorch
        # runtime itself, so the threshold matches what the Pi computes.
        print("\n  Calibrating energy threshold on val set...")
        calib_bytes = lower_to_pte(quantized)
        calib_program, calib_forward = load_pte(calib_bytes)
        val_cpu = val_loader.on("cpu")      # the ExecuTorch runtime runs on the CPU
        _, _, all_energies = eval_pte(calib_forward, val_cpu, num_cats)
        energy_threshold = float(torch.quantile(all_energies, energy_ood_pct).item())
        energy_p95 = float(torch.quantile(all_energies, 0.95).item())
        energy_p99 = float(torch.quantile(all_energies, 0.99).item())
        energy_mean = float(all_energies.mean().item())
        energy_std = float(all_energies.std().item())
        print(f"  Energy stats (val set, n={len(all_energies)}):")
        print(f"    mean={energy_mean:.3f}  std={energy_std:.3f}  "
              f"p95={energy_p95:.3f}  p99={energy_p99:.3f}")
        print(f"  Using p{energy_ood_pct*100:.0f} threshold: {energy_threshold:.4f}")
        energy_threshold_data = {
            "threshold": round(energy_threshold, 4),
            "percentile": energy_ood_pct,
            "p95": round(energy_p95, 4),
            "p99": round(energy_p99, 4),
            "mean": round(energy_mean, 4),
            "std": round(energy_std, 4),
            "n_samples": len(all_energies),
        }

        # Final program: the same graph plus the class list and the threshold
        # as constant methods, so the model stays one self-describing file.
        pte_bytes = lower_to_pte(quantized, {
            "classes_json": json.dumps(class_names),
            "energy_threshold_json": json.dumps(energy_threshold_data),
        })
        model_out = output_dir / "cat_classifier.pte"
        model_out.write_bytes(pte_bytes)

        # Verify by loading the file back
        program, forward = load_pte(model_out)
        assert json.loads(read_pte_constant(program, "classes_json")) == class_names
        assert (json.loads(read_pte_constant(program, "energy_threshold_json"))
                == energy_threshold_data)
        out = forward.execute([torch.zeros(1, 3, 224, 224)])[0]
        assert tuple(out.shape) == (1, num_cats), f"Expected (1, {num_cats}), got {tuple(out.shape)}"
        size_mb = model_out.stat().st_size / 1_048_576
        print(f"  Exported: {model_out}  ({size_mb:.1f} MB)")
        print(f"  Output shape verified: {tuple(out.shape)}")

        # Convenience copy for command-line runs. A job from the Pi runs the
        # bundled code, whose PROJECT_ROOT is the job folder, so this never
        # touches the host checkout.
        local_copy = PROJECT_ROOT / "models" / "cat_classifier.pte"
        if local_copy.resolve() != model_out.resolve():
            local_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_out, local_copy)

        # Final per-class accuracy summary, measured with the exported file
        print("\n=== Final Val Accuracy (INT8 model) ===")
        va_final, per_cls_final, _ = eval_pte(forward, val_cpu, num_cats)
        print(f"  Overall: {va_final:.1%}")
        for cat, acc in zip(class_names, per_cls_final):
            flag = "" if acc >= 0.85 else "  ← below 85% threshold"
            print(f"  {cat}: {acc:.1%}{flag}")

        # Machine-readable summary for dashboards and remote training jobs
        final_metrics = {
            "val_acc": round(float(va_final), 4),
            "per_class_acc": {cat: round(float(acc), 4)
                              for cat, acc in zip(class_names, per_cls_final)},
            "classes": class_names,
            "energy_threshold": energy_threshold_data,
            "train_images": len(train_loader.dataset),
            "val_images": val_loader.num_samples,
            "torch_version": torch.__version__,
            "executorch_version": metadata.version("executorch"),
            "model_file": "cat_classifier.pte",
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        }
        with open(output_dir / "final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
    else:
        print("Phase 3 (QAT) and Phase 4 (export) skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
