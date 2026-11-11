#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Package collected raw images into a tar.gz archive for transfer to the training machine."""
from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import tarfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, resolve_path
from utils.display import print_info, print_success, print_error, print_header


def count_sessions(csv_path: Path) -> int:
    """Count unique session IDs in the capture log CSV."""
    if not csv_path.exists():
        return 0
    sessions: set[str] = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("session_id", "")
            if sid:
                sessions.add(sid)
    return len(sessions)


def scan_images(output_dir: Path, cats: list[str]) -> dict[str, list[Path]]:
    """Return {cat_name: [image_paths]} for each cat directory."""
    result: dict[str, list[Path]] = {}
    for cat in cats:
        cat_dir = output_dir / cat
        if cat_dir.is_dir():
            result[cat] = sorted(cat_dir.glob("*.jpg"))
        else:
            result[cat] = []
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export collected images as a tar.gz archive"
    )
    add_common_args(parser)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for the archive (default: data/export/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config_with_args(args)

    cats: list[str] = config["cats"]
    output_dir = resolve_path(config["collection"]["output_dir"], PROJECT_ROOT)
    export_dir = args.output or resolve_path("data/export", PROJECT_ROOT)
    csv_path = output_dir / "capture_log.csv"

    # Scan images (cats + extra dirs like background/unlabeled)
    all_dirs = cats + ["background", "unlabeled"]
    cat_images = scan_images(output_dir, all_dirs)
    total = sum(len(imgs) for imgs in cat_images.values())

    if total == 0:
        print_info("No images found to export.")
        print_info(f"Looked in: {output_dir}")
        sys.exit(0)

    print_header("Export Dataset")
    print("  Exporting dataset:")
    for name in all_dirs:
        count = len(cat_images.get(name, []))
        if count > 0:
            print(f"    {name:12s} {count} images")
    print(f"    {'Total':12s} {total} images\n")

    # Prepare export directory
    export_dir.mkdir(parents=True, exist_ok=True)

    # Build manifest
    counts = {cat: len(imgs) for cat, imgs in cat_images.items()}
    num_sessions = count_sessions(csv_path)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cats": cats,
        "counts": counts,
        "total_images": total,
        "collection_sessions": num_sessions,
        "source_dir": str(output_dir),
    }

    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Create tar.gz archive
    archive_name = f"dataset_{datetime.now().strftime('%Y%m%d')}.tar.gz"
    archive_path = export_dir / archive_name

    print_info(f"Creating archive: {archive_path.name}")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(str(manifest_path), arcname="manifest.json")
        for name in all_dirs:
            for img_path in cat_images.get(name, []):
                tar.add(str(img_path), arcname=f"{name}/{img_path.name}")

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print_success(f"Archive created: {archive_path}")
    print_info(f"Size: {size_mb:.1f} MB")
    print()
    print(f"  Transfer to your training machine with:")
    print(f"    scp {archive_path} user@training-machine:~/")
    print()


if __name__ == "__main__":
    main()
