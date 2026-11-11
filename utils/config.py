# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Project configuration utilities: root detection, YAML loading, argparse helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def find_project_root(marker: str = "configs/project.yaml") -> Path:
    """Walk up from this file's directory until *marker* exists relative to a parent."""
    current = Path(__file__).resolve().parent
    while True:
        if (current / marker).exists():
            return current
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                f"Could not find project root (looked for '{marker}' in all parents "
                f"of {Path(__file__).resolve()})"
            )
        current = parent


PROJECT_ROOT: Path = find_project_root()


def resolve_path(path_str: str, root: Path) -> Path:
    """Resolve a potentially relative path string against *root*."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (root / p).resolve()


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load project YAML config, merging any runtime.yaml overrides."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "project.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Merge runtime overrides if present (Pi-only file written by dashboard)
    runtime_path = config_path.parent / "runtime.yaml"
    if runtime_path.exists():
        with open(runtime_path) as f:
            runtime = yaml.safe_load(f) or {}
        for section, overrides in runtime.items():
            if isinstance(overrides, dict) and section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(overrides)
            else:
                cfg[section] = overrides
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--config`` argument to an argument parser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to project config YAML (default: configs/project.yaml)",
    )


def get_config_with_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convenience wrapper: load config using the ``--config`` arg if provided."""
    return load_config(args.config)


# ── Cat color palette ─────────────────────────────────────────────────────
# Distinguishable colors assigned to cats by index.  Hex (for JS/HTML) and
# BGR (for OpenCV drawing) variants.

CAT_COLOR_PALETTE_HEX = ["#6b85a8", "#7ba87a", "#a87272", "#8a7a9e", "#b8896a", "#8e7a65"]
CAT_COLOR_PALETTE_BGR = [
    (168, 133, 107),  # #6b85a8  slate blue
    (122, 168, 123),  # #7ba87a  faded sage
    (114, 114, 168),  # #a87272  muted rose
    (158, 122, 138),  # #8a7a9e  dusty violet
    (106, 137, 184),  # #b8896a  warm copper
    (101, 122, 142),  # #8e7a65  driftwood
]


def get_cat_colors_hex(cats: list[str]) -> dict[str, str]:
    """Return {cat_name: hex_color} for each cat, cycling the palette."""
    return {cat: CAT_COLOR_PALETTE_HEX[i % len(CAT_COLOR_PALETTE_HEX)] for i, cat in enumerate(cats)}


def get_cat_colors_bgr(cats: list[str]) -> dict[str, tuple[int, int, int]]:
    """Return {cat_name: (B, G, R)} for each cat, cycling the palette."""
    return {cat: CAT_COLOR_PALETTE_BGR[i % len(CAT_COLOR_PALETTE_BGR)] for i, cat in enumerate(cats)}
