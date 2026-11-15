# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Classifier training settings the Pi dashboard can edit.

The Training tab builds its settings form from SETTINGS. Values are saved to
the Pi's runtime.yaml (``classifier_training`` and ``classifier.crop_padding``)
and travel to the backend in each job's manifest. Defaults match what
train/train_classifier.py uses when a key is missing.
"""
from __future__ import annotations

# (key, label, type, default, min, max, group, help)
# type is "int", "float", "bool" or "select" (choices in _OPTIONS).
# key "augmentation.x" lives in classifier_training.augmentation;
# key "crop_padding" lives in classifier.
_OPTIONS = {
    "balance_method": ["oversample", "loss_weights", "off"],
}

SETTINGS: list[dict] = [
    {"key": k, "label": lbl, "type": t, "default": d, "min": lo, "max": hi,
     "group": g, "help": h, "options": _OPTIONS.get(k)}
    for k, lbl, t, d, lo, hi, g, h in [
        ("phase1_epochs", "Head warmup epochs", "int", 20, 0, 500, "Phases",
         "Phase 1: only the new classifier head trains."),
        ("phase2_epochs", "Fine-tune epochs", "int", 40, 0, 500, "Phases",
         "Phase 2: the last backbone stages train too."),
        ("phase3_epochs", "QAT epochs", "int", 10, 1, 500, "Phases",
         "Phase 3: quantization-aware training for the INT8 model."),
        ("warmup_epochs", "LR warmup epochs", "int", 5, 0, 100, "Phases",
         "Linear warmup at the start of phase 1."),
        ("patience", "Early-stop patience", "int", 15, 1, 500, "Phases",
         "Stop a phase after this many epochs without improvement."),
        ("phase1_lr", "Head LR", "float", 1e-3, 1e-7, 1.0, "Optimizer", "Phase 1 learning rate."),
        ("phase2_backbone_lr", "Backbone LR", "float", 1e-5, 1e-8, 1.0, "Optimizer",
         "Phase 2 learning rate for the unfrozen backbone."),
        ("phase2_head_lr", "Head LR (fine-tune)", "float", 1e-4, 1e-8, 1.0, "Optimizer",
         "Phase 2 learning rate for the head."),
        ("weight_decay", "Weight decay", "float", 0.005, 0.0, 1.0, "Optimizer", ""),
        ("batch_size", "Batch size", "int", 32, 1, 512, "Optimizer", ""),
        ("label_smoothing", "Label smoothing", "float", 0.1, 0.0, 0.5, "Optimizer", ""),
        ("gradient_clip_norm", "Gradient clip norm", "float", 1.0, 0.0, 100.0, "Optimizer", ""),
        ("mixup_alpha", "Mixup alpha", "float", 0.2, 0.0, 2.0, "Data",
         "Phase 2 only; 0 turns mixup off."),
        ("balance_method", "Class balancing", "select", "oversample", None, None, "Data",
         "How cats with fewer images are kept from being outweighed. loss_weights: "
         "each of their images counts more in the loss. oversample: their images are "
         "repeated each epoch until every cat matches the largest, and each repeat "
         "gets new random distortions, so they act as extra synthetic examples. "
         "off: no balancing."),
        ("session_gap_minutes", "Session gap (min)", "float", 5, 0.0, 1440.0, "Data",
         "Captures closer than this belong to one session; a session is never "
         "split between train and val."),
        ("crop_padding", "Crop padding", "float", 0.20, 0.0, 1.0, "Data",
         "Extra margin around the bounding box when cropping (0.2 = 20%)."),
        ("energy_ood_percentile", "Unknown-cat percentile", "float", 0.99, 0.5, 1.0, "Data",
         "Energy threshold percentile on validation data; higher rejects fewer "
         "crops as 'not a known cat'."),
        ("cv_folds", "Cross-validation folds", "int", 5, 2, 10, "Data", ""),
        ("augmentation.horizontal_flip", "Horizontal flip", "bool", True, None, None,
         "Augmentation", ""),
        ("augmentation.brightness_contrast", "Brightness/contrast", "bool", True, None, None,
         "Augmentation", ""),
        ("augmentation.brightness_limit", "Brightness limit", "float", 0.3, 0.0, 1.0,
         "Augmentation", ""),
        ("augmentation.contrast_limit", "Contrast limit", "float", 0.3, 0.0, 1.0,
         "Augmentation", ""),
        ("augmentation.shift_scale_rotate", "Shift/scale/rotate", "bool", True, None, None,
         "Augmentation", ""),
        ("augmentation.rotate_limit", "Rotate limit (deg)", "int", 10, 0, 90,
         "Augmentation", ""),
        ("augmentation.gaussian_blur", "Gaussian blur", "bool", True, None, None,
         "Augmentation", ""),
        ("augmentation.gaussian_noise", "Gaussian noise", "bool", True, None, None,
         "Augmentation", ""),
        ("augmentation.coarse_dropout", "Coarse dropout", "bool", True, None, None,
         "Augmentation", ""),
        ("augmentation.hue_saturation", "Hue/saturation", "bool", False, None, None,
         "Augmentation", ""),
        ("augmentation.clahe", "CLAHE", "bool", False, None, None, "Augmentation", ""),
    ]
]
_BY_KEY = {s["key"]: s for s in SETTINGS}


def current_values(config: dict) -> dict:
    """Flat {key: value} from a (merged) config, defaults filled in."""
    tc = config.get("classifier_training") or {}
    aug = tc.get("augmentation") or {}
    out = {}
    for s in SETTINGS:
        k = s["key"]
        if k == "crop_padding":
            val = (config.get("classifier") or {}).get("crop_padding")
        elif k == "balance_method" and tc.get(k) is None and "balance_classes" in tc:
            # Older configs: balance_classes (same mapping as training)
            val = "off" if tc["balance_classes"] is False else "oversample"
        elif k.startswith("augmentation."):
            val = aug.get(k.split(".", 1)[1])
        else:
            val = tc.get(k)
        out[k] = s["default"] if val is None else val
    return out


def parse(values: dict) -> dict:
    """Validate submitted {key: value}; unknown keys are ignored. Raises
    ValueError with a message for the user."""
    out = {}
    for key, raw in values.items():
        s = _BY_KEY.get(key)
        if s is None:
            continue
        if s["type"] == "bool":
            if isinstance(raw, str):
                raw = raw.lower() in ("1", "true", "on", "yes")
            out[key] = bool(raw)
            continue
        if s["type"] == "select":
            if raw not in s["options"]:
                raise ValueError(f"{s['label']} must be one of {', '.join(s['options'])}")
            out[key] = raw
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{s['label']} must be a number") from None
        if s["type"] == "int":
            if not num.is_integer():
                raise ValueError(f"{s['label']} must be a whole number")
            num = int(num)
        if not s["min"] <= num <= s["max"]:
            raise ValueError(f"{s['label']} must be between {s['min']} and {s['max']}")
        out[key] = num
    return out


def apply(config: dict, values: dict) -> tuple[dict, float | None]:
    """Merge parsed values into the config's classifier_training section.
    Returns (complete classifier_training dict, crop_padding or None)."""
    tc = dict(config.get("classifier_training") or {})
    aug = dict(tc.get("augmentation") or {})
    crop_padding = None
    for key, val in values.items():
        if key == "crop_padding":
            crop_padding = val
        elif key.startswith("augmentation."):
            aug[key.split(".", 1)[1]] = val
        else:
            tc[key] = val
    if "balance_method" in values:
        tc.pop("balance_classes", None)   # replaced by balance_method
    tc["augmentation"] = aug
    return tc, crop_padding
