"""Four-phase MobileNetV2-INT8 classifier training for cat identity.

Phases:
  1. FP32 head-only warm-up  (~phase1_epochs epochs)
  2. FP32 partial fine-tune  (~phase2_epochs epochs, last 3 backbone blocks)
  3. Quantization-Aware Training (~phase3_epochs epochs)
  4. INT8 export → models/cat_classifier.pt

Reads hyperparameters from configs/project.yaml [classifier_training:].
Writes results.csv with columns: epoch, phase, train_loss, val_loss, train_acc, val_acc, acc_<cat>...
"""
from __future__ import annotations

import argparse
import csv
import math
import json
import sys
from pathlib import Path

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
from torch.utils.data import DataLoader

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

class _AlbumentationsWrapper(torchvision.datasets.ImageFolder):
    """ImageFolder with an Albumentations transform pipeline."""

    def __init__(self, root, transform_alb):
        super().__init__(root, transform=None)
        self._transform_alb = transform_alb

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))
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
        transforms.append(A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05,
            rotate_limit=aug_cfg.get("rotate_limit", 10), p=0.5))
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


def _build_loaders(dataset_dir: Path, batch_size: int, aug_cfg: dict | None = None):
    train_ds = _AlbumentationsWrapper(str(dataset_dir / "train"), _make_train_transform(aug_cfg))
    val_ds   = _AlbumentationsWrapper(str(dataset_dir / "val"),   _make_val_transform())
    num_workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


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
# Train / eval loops
# ═══════════════════════════════════════════════════════════════════════════

def _train_epoch(model, loader, optimizer, criterion, device,
                 mixup_alpha: float = 0.0,
                 freeze_bn: bool = False,
                 grad_clip_norm: float = 0.0) -> tuple[float, float]:
    model.train()
    if freeze_bn:
        _freeze_bn(model)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

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
        images, labels = images.to(device), labels.to(device)
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
    balance_classes = tc.get("balance_classes", True)
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

    train_loader, val_loader, class_names = _build_loaders(dataset_dir, batch_size, aug_cfg)
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print(f"Classes (dataset order): {class_names}")

    csv_logger = _CSVLogger(ckpt_dir / "results.csv", class_names)

    # Class-weighted loss for imbalanced datasets
    if balance_classes:
        from collections import Counter
        class_counts = Counter(label for _, label in train_loader.dataset.samples)
        max_count = max(class_counts.values())
        weights = torch.tensor(
            [max_count / class_counts.get(i, 1) for i in range(num_cats)],
            dtype=torch.float32, device=device,
        )
        print(f"Class balance: ON  weights={dict(zip(class_names, [f'{w:.2f}' for w in weights.tolist()]))}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    else:
        print("Class balance: OFF (uniform loss)")
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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

        from torchvision.models.quantization import mobilenet_v2
        from torchvision.models import MobileNet_V2_Weights
        model = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1,
            quantize=False,
        )
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

        from torchvision.models.quantization import mobilenet_v2
        from torchvision.models import MobileNet_V2_Weights
        model = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1,
            quantize=False,
        )
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

        from torchvision.models.quantization import mobilenet_v2
        from torchvision.models import MobileNet_V2_Weights

        # Fresh model instance — fuse_model(is_qat=True) cannot be called twice
        qat_model = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1,
            quantize=False,
        )
        qat_model.classifier[1] = nn.Linear(1280, num_cats)

        src_ckpt = phase2_ckpt if phase2_ckpt.is_file() else phase1_ckpt
        if src_ckpt.is_file():
            ckpt = torch.load(src_ckpt, map_location="cpu")
            qat_model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded checkpoint: {src_ckpt.name} "
                  f"(val_acc={ckpt.get('val_acc', '?'):.1%})")
        else:
            print("  WARNING: no prior checkpoint found; QAT from pretrained weights")

        # QAT setup (CPU only for qnnpack)
        torch.backends.quantized.engine = "qnnpack"
        qat_model.train()
        qat_model.fuse_model(is_qat=True)
        qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
        torch.ao.quantization.prepare_qat(qat_model, inplace=True)
        qat_model = qat_model.cpu()

        qat_lr = 1e-6  # ~1% of training LR for stable QAT
        optimizer = torch.optim.AdamW(
            qat_model.parameters(), lr=qat_lr, weight_decay=weight_decay
        )

        # QAT uses val transforms for training (no augmentation)
        from torch.utils.data import DataLoader
        qat_train_ds = _AlbumentationsWrapper(str(dataset_dir / "train"), _make_val_transform())
        qat_train_loader = DataLoader(
            qat_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True  # num_workers=0 for stability with quantized ops
        )
        qat_val_loader = DataLoader(
            _AlbumentationsWrapper(str(dataset_dir / "val"), _make_val_transform()),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )

        best_val_acc = 0.0
        no_improve = 0
        phase3_device = torch.device("cpu")  # quantized models are CPU-only
        criterion = criterion.to(phase3_device)
        freeze_epoch = max(1, int(math.ceil(phase3_epochs * 0.8)))
        observers_frozen = False
        for epoch in range(1, phase3_epochs + 1):
            # Freeze observers and BN at 80% to stabilize quantization grid
            if epoch >= freeze_epoch and not observers_frozen:
                torch.ao.quantization.disable_observer(qat_model)
                for m in qat_model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                observers_frozen = True
                print(f"  Froze observers and BN at epoch {epoch}")
            tl, ta = _train_epoch(qat_model, qat_train_loader, optimizer, criterion,
                                  phase3_device, grad_clip_norm=grad_clip_norm)
            vl, va, per_cls = _eval_epoch(qat_model, qat_val_loader, criterion,
                                          phase3_device, num_cats)
            csv_logger.log(phase1_epochs + phase2_epochs + epoch, "phase3",
                           tl, vl, ta, va, per_cls)
            print(f"  [P3 {epoch:3d}/{phase3_epochs}] "
                  f"loss {tl:.3f}/{vl:.3f}  acc {ta:.1%}/{va:.1%}")
            if va > best_val_acc:
                best_val_acc = va
                no_improve = 0
                torch.save({"model_state_dict": qat_model.state_dict(),
                            "val_acc": va, "epoch": epoch}, phase3_ckpt)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} (patience {patience})")
                    break

        print(f"Phase 3 done. Best val acc: {best_val_acc:.1%}  → {phase3_ckpt}")

        # ── Phase 4: INT8 Export ─────────────────────────────────────────
        print("\n" + "="*60)
        print("Phase 4: INT8 export")
        print("="*60)

        # Reload best QAT checkpoint into a fresh model for conversion
        qat_model2 = mobilenet_v2(quantize=False)
        qat_model2.classifier[1] = nn.Linear(1280, num_cats)
        ckpt = torch.load(phase3_ckpt, map_location="cpu")
        torch.backends.quantized.engine = "qnnpack"
        qat_model2.train()
        qat_model2.fuse_model(is_qat=True)
        qat_model2.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
        torch.ao.quantization.prepare_qat(qat_model2, inplace=True)
        qat_model2.load_state_dict(ckpt["model_state_dict"])

        qat_model2 = qat_model2.cpu()
        qat_model2.eval()
        torch.ao.quantization.convert(qat_model2, inplace=True)

        example_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            traced = torch.jit.trace(qat_model2, example_input)
        traced = torch.jit.freeze(traced)

        # ── Energy threshold calibration ───────────────────────────────
        print("\n  Calibrating energy threshold on val set...")
        all_energies = []
        with torch.no_grad():
            for images, _ in qat_val_loader:
                logits = traced(images.cpu())
                energies = -torch.logsumexp(logits, dim=1)
                all_energies.append(energies)
        all_energies = torch.cat(all_energies)
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

        model_out = PROJECT_ROOT / "models" / "cat_classifier.pt"
        model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced, str(model_out),
                       _extra_files={
                           "classes.json": json.dumps(class_names),
                           "energy_threshold": json.dumps(energy_threshold_data),
                       })

        # Verify
        loaded = torch.jit.load(str(model_out), map_location="cpu")
        loaded.eval()
        with torch.no_grad():
            out = loaded(example_input)
        assert out.shape == (1, num_cats), f"Expected (1, {num_cats}), got {out.shape}"
        size_mb = model_out.stat().st_size / 1_048_576
        print(f"  Exported: {model_out}  ({size_mb:.1f} MB)")
        print(f"  Output shape verified: {tuple(out.shape)}")

        # Copy to version output dir too
        import shutil
        shutil.copy2(model_out, output_dir / "cat_classifier.pt")

        # Final per-class accuracy summary
        print("\n=== Final Val Accuracy (INT8 model) ===")
        with torch.no_grad():
            _, va_final, per_cls_final = _eval_epoch(
                loaded, qat_val_loader, criterion, torch.device("cpu"), num_cats
            )
        print(f"  Overall: {va_final:.1%}")
        for cat, acc in zip(class_names, per_cls_final):
            flag = "" if acc >= 0.85 else "  ← below 85% threshold"
            print(f"  {cat}: {acc:.1%}{flag}")
    else:
        print("Phase 3 (QAT) and Phase 4 (export) skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
