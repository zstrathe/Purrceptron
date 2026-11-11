# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""K-fold cross-validation for the MobileNetV2 cat classifier.

Runs StratifiedGroupKFold CV using session-aware grouping to prevent
data leakage.  Each fold trains the full 3-phase pipeline (head warm-up,
partial fine-tune, QAT) and reports per-fold + aggregate accuracy.

This is an *evaluation* tool — it does not produce a deployment model.
Use train_classifier.py for the actual training run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None

# Reuse building blocks from existing modules
from prepare_classifier_dataset import infer_sessions, link_or_copy
from train_classifier import (
    _AlbumentationsWrapper,
    _build_loaders,
    _eval_epoch,
    _freeze_bn,
    _make_train_transform,
    _make_val_transform,
    _train_epoch,
)


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("pyyaml required: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Fold directory creation
# ═══════════════════════════════════════════════════════════════════════════

def create_fold_dirs(
    fold_dir: Path,
    all_paths: list[Path],
    all_labels: list[int],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    cats: list[str],
) -> None:
    """Create {train,val}/{cat}/ symlink trees for one fold."""
    if fold_dir.exists():
        shutil.rmtree(fold_dir)

    for idx in train_indices:
        cat = cats[all_labels[idx]]
        link_or_copy(all_paths[idx], fold_dir / "train" / cat / all_paths[idx].name)

    for idx in val_indices:
        cat = cats[all_labels[idx]]
        link_or_copy(all_paths[idx], fold_dir / "val" / cat / all_paths[idx].name)


# ═══════════════════════════════════════════════════════════════════════════
# Single fold training (Phases 1-3, no export)
# ═══════════════════════════════════════════════════════════════════════════

def train_fold(
    fold_dir: Path,
    fold_idx: int,
    n_folds: int,
    num_cats: int,
    cats: list[str],
    tc: dict,
    device: torch.device,
) -> dict:
    """Train all 3 phases for a single fold. Returns fold metrics dict."""
    from torch.utils.data import DataLoader

    phase1_epochs = tc.get("phase1_epochs", 20)
    phase2_epochs = tc.get("phase2_epochs", 40)
    phase3_epochs = tc.get("phase3_epochs", 10)
    phase1_lr     = tc.get("phase1_lr", 1e-3)
    phase2_bb_lr  = tc.get("phase2_backbone_lr", 1e-5)
    phase2_head_lr = tc.get("phase2_head_lr", 1e-4)
    weight_decay  = tc.get("weight_decay", 0.005)
    batch_size    = tc.get("batch_size", 32)
    patience      = tc.get("patience", 15)
    mixup_alpha   = tc.get("mixup_alpha", 0.2)
    balance_classes = tc.get("balance_classes", True)
    aug_cfg       = tc.get("augmentation", {})

    prefix = f"[CV Fold {fold_idx + 1}/{n_folds}]"

    # Build loaders
    train_loader, val_loader, class_names = _build_loaders(fold_dir, batch_size, aug_cfg)
    print(f"{prefix} Train: {len(train_loader.dataset)} imgs, "
          f"Val: {len(val_loader.dataset)} imgs")

    # Class-weighted loss
    if balance_classes:
        class_counts = Counter(label for _, label in train_loader.dataset.samples)
        max_count = max(class_counts.values())
        weights = torch.tensor(
            [max_count / class_counts.get(i, 1) for i in range(num_cats)],
            dtype=torch.float32, device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_overall_acc = 0.0
    best_per_class = [0.0] * num_cats

    # ── Phase 1: head-only ─────────────────────────────────────────────
    from torchvision.models.quantization import mobilenet_v2
    from torchvision.models import MobileNet_V2_Weights

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(1280, num_cats)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=phase1_lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-7
    )

    best_p1_acc = 0.0
    best_p1_state = None
    no_improve = 0
    for epoch in range(1, phase1_epochs + 1):
        tl, ta = _train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va, per_cls = _eval_epoch(model, val_loader, criterion, device, num_cats)
        scheduler.step()
        print(f"  {prefix} [P1 {epoch:3d}/{phase1_epochs}] "
              f"loss {tl:.3f}/{vl:.3f}  acc {ta:.1%}/{va:.1%}")
        print(f"[CV PROGRESS] fold={fold_idx + 1}/{n_folds} phase=1 epoch={epoch}/{phase1_epochs}")
        if va > best_p1_acc:
            best_p1_acc = va
            best_p1_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  {prefix} P1 early stop at epoch {epoch}")
                break

    if best_p1_acc > best_overall_acc:
        best_overall_acc = best_p1_acc

    # ── Phase 2: partial fine-tune ─────────────────────────────────────
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False)
    model.classifier[1] = nn.Linear(1280, num_cats)
    if best_p1_state:
        model.load_state_dict(best_p1_state)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-7
    )

    best_p2_acc = 0.0
    best_p2_state = None
    no_improve = 0
    for epoch in range(1, phase2_epochs + 1):
        tl, ta = _train_epoch(model, train_loader, optimizer, criterion, device,
                              mixup_alpha=mixup_alpha, freeze_bn=True)
        vl, va, per_cls = _eval_epoch(model, val_loader, criterion, device, num_cats)
        scheduler.step()
        print(f"  {prefix} [P2 {epoch:3d}/{phase2_epochs}] "
              f"loss {tl:.3f}/{vl:.3f}  acc {ta:.1%}/{va:.1%}")
        print(f"[CV PROGRESS] fold={fold_idx + 1}/{n_folds} phase=2 epoch={epoch}/{phase2_epochs}")
        if va > best_p2_acc:
            best_p2_acc = va
            best_p2_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  {prefix} P2 early stop at epoch {epoch}")
                break

    if best_p2_acc > best_overall_acc:
        best_overall_acc = best_p2_acc

    # ── Phase 3: QAT ──────────────────────────────────────────────────
    qat_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False)
    qat_model.classifier[1] = nn.Linear(1280, num_cats)
    src_state = best_p2_state or best_p1_state
    if src_state:
        qat_model.load_state_dict(src_state)

    torch.backends.quantized.engine = "qnnpack"
    qat_model.train()
    qat_model.fuse_model(is_qat=True)
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    qat_model = qat_model.cpu()

    if balance_classes:
        qat_criterion = nn.CrossEntropyLoss(weight=criterion.weight.detach().cpu())
    else:
        qat_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-5, weight_decay=weight_decay)

    qat_train_ds = _AlbumentationsWrapper(str(fold_dir / "train"), _make_val_transform())
    qat_train_loader = DataLoader(qat_train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)
    qat_val_ds = _AlbumentationsWrapper(str(fold_dir / "val"), _make_val_transform())
    qat_val_loader = DataLoader(qat_val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0)

    cpu_device = torch.device("cpu")
    no_improve = 0
    for epoch in range(1, phase3_epochs + 1):
        tl, ta = _train_epoch(qat_model, qat_train_loader, optimizer, qat_criterion, cpu_device)
        vl, va, per_cls = _eval_epoch(qat_model, qat_val_loader, qat_criterion, cpu_device, num_cats)
        print(f"  {prefix} [P3 {epoch:3d}/{phase3_epochs}] "
              f"loss {tl:.3f}/{vl:.3f}  acc {ta:.1%}/{va:.1%}")
        print(f"[CV PROGRESS] fold={fold_idx + 1}/{n_folds} phase=3 epoch={epoch}/{phase3_epochs}")
        if va > best_overall_acc:
            best_overall_acc = va
            best_per_class = per_cls
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  {prefix} P3 early stop at epoch {epoch}")
                break

    # If P3 didn't improve, use P2's best per-class (re-eval)
    if best_per_class == [0.0] * num_cats and best_p2_state:
        # Re-evaluate P2 best model for per-class stats
        model = mobilenet_v2(quantize=False)
        model.classifier[1] = nn.Linear(1280, num_cats)
        model.load_state_dict(best_p2_state)
        model = model.to(device)
        _, _, best_per_class = _eval_epoch(model, val_loader, criterion, device, num_cats)
        del model

    print(f"  {prefix} Best val acc: {best_overall_acc:.1%}")

    # Clean up GPU memory
    del qat_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold_idx,
        "best_val_acc": round(best_overall_acc, 4),
        "per_class": {cat: round(acc, 4) for cat, acc in zip(cats, best_per_class)},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold cross-validation for cat classifier")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "project.yaml")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory with cropped images per cat")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for CV results (cv_summary.json)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds (2-10)")
    args = parser.parse_args()

    if not (2 <= args.n_folds <= 10):
        print("ERROR: n-folds must be between 2 and 10", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    tc = cfg.get("classifier_training", {})
    cats: list[str] = cfg.get("cats", [])
    num_cats = len(cats)
    if num_cats < 2:
        print("ERROR: need at least 2 cats in config", file=sys.stderr)
        sys.exit(1)

    session_gap = tc.get("session_gap_minutes", 5)
    n_folds = args.n_folds

    print(f"\n{'=' * 60}")
    print(f"  {n_folds}-Fold Cross-Validation")
    print(f"{'=' * 60}")
    print(f"Cats: {cats}")
    print(f"Source: {args.source_dir}")
    print(f"Folds: {n_folds}")

    # ── Collect images and sessions ───────────────────────────────────
    all_paths: list[Path] = []
    all_labels: list[int] = []
    all_groups: list[str] = []

    for cat_idx, cat in enumerate(cats):
        cat_dir = args.source_dir / cat
        if not cat_dir.is_dir():
            print(f"  [warn] No directory for '{cat}' at {cat_dir}")
            continue
        images = sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.jpeg"))
        if not images:
            print(f"  [warn] No images for '{cat}'")
            continue

        sessions = infer_sessions(images, gap_minutes=session_gap)
        print(f"  {cat}: {len(images)} images, {len(sessions)} sessions")

        # Validate enough sessions for k folds
        if len(sessions) < n_folds:
            print(f"ERROR: {cat} has only {len(sessions)} sessions but {n_folds} folds requested. "
                  f"Reduce folds or collect more data.", file=sys.stderr)
            sys.exit(1)

        for session_id, paths in sessions.items():
            for p in paths:
                all_paths.append(p)
                all_labels.append(cat_idx)
                all_groups.append(f"{cat}_{session_id}")

    if not all_paths:
        print("ERROR: no images found", file=sys.stderr)
        sys.exit(1)

    all_labels_arr = np.array(all_labels)
    all_groups_arr = np.array(all_groups)

    # ── Create folds ─────────────────────────────────────────────────
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(sgkf.split(all_paths, all_labels_arr, all_groups_arr))

    print(f"\nTotal: {len(all_paths)} images across {len(set(all_groups))} sessions")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_folds_dir = args.output_dir / "cv_folds"

    # ── Run folds ────────────────────────────────────────────────────
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'=' * 60}")

        fold_dir = cv_folds_dir / f"fold_{fold_idx}"
        create_fold_dirs(fold_dir, all_paths, all_labels, train_idx, val_idx, cats)

        result = train_fold(fold_dir, fold_idx, n_folds, num_cats, cats, tc, device)
        fold_results.append(result)

        # Clean up fold symlinks
        shutil.rmtree(fold_dir, ignore_errors=True)

    # Clean up cv_folds dir
    if cv_folds_dir.exists():
        shutil.rmtree(cv_folds_dir, ignore_errors=True)

    # ── Summary ──────────────────────────────────────────────────────
    accs = [r["best_val_acc"] for r in fold_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    per_class_mean = {}
    per_class_std = {}
    for cat in cats:
        cat_accs = [r["per_class"][cat] for r in fold_results]
        per_class_mean[cat] = round(float(np.mean(cat_accs)), 4)
        per_class_std[cat] = round(float(np.std(cat_accs)), 4)

    print(f"\n{'=' * 60}")
    print(f"  Cross-Validation Summary ({n_folds}-fold)")
    print(f"{'=' * 60}")
    print(f"  Overall: {mean_acc:.1%} \u00b1 {std_acc:.1%}")
    for cat in cats:
        print(f"  {cat}: {per_class_mean[cat]:.1%} \u00b1 {per_class_std[cat]:.1%}")
    print()

    # Per-fold table
    header = f"  {'Fold':<6} {'Overall':<10}"
    for cat in cats:
        header += f" {cat:<10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in fold_results:
        line = f"  {r['fold'] + 1:<6} {r['best_val_acc']:<10.1%}"
        for cat in cats:
            line += f" {r['per_class'][cat]:<10.1%}"
        print(line)
    print("  " + "-" * (len(header) - 2))
    line = f"  {'Mean':<6} {mean_acc:<10.1%}"
    for cat in cats:
        line += f" {per_class_mean[cat]:<10.1%}"
    print(line)
    line = "  Std    +/-{:<7.1%}".format(std_acc)
    for cat in cats:
        line += " +/-{:<7.1%}".format(per_class_std[cat])
    print(line)

    # Save summary JSON
    summary = {
        "n_folds": n_folds,
        "folds": fold_results,
        "mean_acc": round(mean_acc, 4),
        "std_acc": round(std_acc, 4),
        "per_class_mean": per_class_mean,
        "per_class_std": per_class_std,
    }
    summary_path = args.output_dir / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
