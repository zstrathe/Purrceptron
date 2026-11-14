# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Parse a training run's results.csv into chart-ready JSON (Training tab)."""
from __future__ import annotations

import csv
from pathlib import Path


def read_training_metrics(run_dir: Path) -> dict:
    """Parse results.csv from a classifier training run into chart-friendly JSON.

    Columns: epoch, phase, train_loss, val_loss, train_acc, val_acc, acc_<cat>...
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"available": False}

    epochs: list[int] = []
    phases: list[str] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    train_acc: list[float] = []
    val_acc: list[float] = []
    per_class: dict[str, list[float]] = {}

    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row.get("epoch", len(epochs) + 1)))
                phases.append(row.get("phase", ""))
                train_loss.append(float(row.get("train_loss", 0)))
                val_loss.append(float(row.get("val_loss", 0)))
                train_acc.append(float(row.get("train_acc", 0)))
                val_acc.append(float(row.get("val_acc", 0)))
                for key, val in row.items():
                    if key.startswith("acc_"):
                        cat_name = key[4:]
                        per_class.setdefault(cat_name, []).append(float(val or 0))
    except Exception:
        return {"available": False}

    if not epochs:
        return {"available": False}

    best_val_acc = max(val_acc) if val_acc else 0
    best_epoch = val_acc.index(best_val_acc) + 1 if val_acc else 0

    # Compute phase boundaries (start epoch of each phase)
    phase_boundaries: list[dict] = []
    prev_phase = ""
    for i, ph in enumerate(phases):
        if ph != prev_phase:
            phase_boundaries.append({"phase": ph, "epoch_idx": i})
            prev_phase = ph

    # Build per-phase breakdown
    phase_labels = {"phase1": "Head Warmup", "phase2": "Fine-tune", "phase3": "QAT"}
    phases_data: dict[str, dict] = {}
    for i, ph in enumerate(phases):
        if ph not in phases_data:
            phases_data[ph] = {
                "label": phase_labels.get(ph, ph),
                "local_epochs": [], "train_loss": [], "val_loss": [],
                "train_acc": [], "val_acc": [], "per_class": {},
                "best_val_acc": 0, "best_epoch_local": 0,
            }
        pd = phases_data[ph]
        local_ep = len(pd["local_epochs"]) + 1
        pd["local_epochs"].append(local_ep)
        pd["train_loss"].append(train_loss[i])
        pd["val_loss"].append(val_loss[i])
        pd["train_acc"].append(train_acc[i])
        pd["val_acc"].append(val_acc[i])
        for cat in per_class:
            pd["per_class"].setdefault(cat, []).append(per_class[cat][i])
        if val_acc[i] > pd["best_val_acc"]:
            pd["best_val_acc"] = round(val_acc[i], 4)
            pd["best_epoch_local"] = local_ep

    return {
        "available": True,
        "epochs": epochs,
        "phases": phases,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "per_class": per_class,
        "phase_boundaries": phase_boundaries,
        "phases_data": phases_data,
        "best_val_acc": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "total_epochs": len(epochs),
        "current_phase": phases[-1] if phases else "",
    }
