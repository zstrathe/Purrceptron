# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""
Crop raw captured images using stored bounding box sidecars.

Reads full-frame JPEGs + .json sidecars from raw_images/<cat>/,
applies padding around the detected bounding box, and writes
resized crops to cropped_images/<cat>/.

Always regenerates the output directory, so re-running with a different
--padding value produces a clean re-crop without stale images.

With --cache-dir, each crop is also kept in the cache and reused by later
runs while the image, its bounding box, the padding and the size are
unchanged. Output files are then hard links to the cached crops, so a run
that changes nothing writes no image data. Cached crops of images that are
gone, renamed or re-boxed are deleted at the end of the run.

Usage:
    python train/crop_images.py \\
        --input-dir versions/v1/raw_images \\
        --output-dir versions/v1/cropped_images \\
        [--padding 0.20] \\
        [--size 224] \\
        [--config configs/project.yaml] \\
        [--cache-dir data/crop_cache]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("pyyaml is required: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def crop_image(
    img: np.ndarray,
    bbox_norm: list[float],
    padding: float,
    size: int,
) -> np.ndarray | None:
    """Crop and resize one image using a normalized bbox + padding.

    bbox_norm: [xmin, ymin, width, height] in normalized coords (0-1).
    Returns the resized crop, or None if the bbox is degenerate.
    """
    H, W = img.shape[:2]
    xmin, ymin, bw, bh = bbox_norm
    pad_x = bw * padding
    pad_y = bh * padding
    x1 = max(0, int((xmin - pad_x) * W))
    y1 = max(0, int((ymin - pad_y) * H))
    x2 = min(W, int((xmin + bw + pad_x) * W))
    y2 = min(H, int((ymin + bh + pad_y) * H))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (size, size))


def _cache_key(jpg: Path, bbox: list, padding: float, size: int) -> str:
    """Everything a crop depends on: the image file (name, size, mtime), its
    bounding box, the padding and the output size."""
    st = jpg.stat()
    ident = json.dumps([jpg.name, st.st_size, st.st_mtime_ns, bbox, padding, size])
    return hashlib.sha1(ident.encode()).hexdigest()[:16]


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def process_cat(
    src_dir: Path,
    dst_dir: Path,
    padding: float,
    size: int,
    cache_dir: Path | None = None,
) -> tuple[int, int, int]:
    """Crop all images in src_dir into dst_dir.

    Returns (n_cropped, n_skipped, n_from_cache); n_cropped includes the
    crops taken from the cache.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    cropped = skipped = from_cache = 0
    jpgs = sorted(src_dir.glob("*.jpg"))

    for i, jpg in enumerate(jpgs, 1):
        if i % 250 == 0 and i < len(jpgs):
            print(f"  {src_dir.name}: {i} / {len(jpgs)} done", flush=True)
        sidecar = jpg.with_suffix(".json")
        if not sidecar.exists():
            print(f"  [SKIP] no sidecar: {jpg.name}")
            skipped += 1
            continue

        try:
            meta = json.loads(sidecar.read_text())
        except Exception as exc:
            print(f"  [SKIP] bad sidecar {jpg.name}: {exc}")
            skipped += 1
            continue

        dets = meta.get("detections", [])
        if not dets:
            print(f"  [SKIP] no detections in sidecar: {jpg.name}")
            skipped += 1
            continue

        # Use highest-confidence detection
        best = max(dets, key=lambda d: d.get("confidence", 0))
        bbox = best.get("bbox_norm")
        if not bbox or len(bbox) != 4:
            print(f"  [SKIP] invalid bbox in {jpg.name}")
            skipped += 1
            continue

        cached = None
        if cache_dir is not None:
            cached = cache_dir / f"{jpg.stem}.{_cache_key(jpg, bbox, padding, size)}.jpg"
            used.add(cached.name)
            if cached.is_file():
                _link_or_copy(cached, dst_dir / jpg.name)
                cropped += 1
                from_cache += 1
                continue

        img = cv2.imread(str(jpg))
        if img is None:
            print(f"  [SKIP] could not read image: {jpg.name}")
            skipped += 1
            continue

        crop = crop_image(img, bbox, padding, size)
        if crop is None:
            print(f"  [SKIP] degenerate bbox after padding: {jpg.name}")
            skipped += 1
            continue

        if cached is None:
            cv2.imwrite(str(dst_dir / jpg.name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            # Write the cache entry atomically (the .jpg suffix tells OpenCV
            # the format), then link it into the output.
            tmp = cached.with_name(cached.stem + ".tmp.jpg")
            if not cv2.imwrite(str(tmp), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"could not write {tmp}")
            os.replace(tmp, cached)
            _link_or_copy(cached, dst_dir / jpg.name)
        cropped += 1

    if cache_dir is not None:
        # Drop cached crops this run didn't use: images that were deleted,
        # renamed, moved to another cat, or given a new bounding box.
        for f in cache_dir.iterdir():
            if f.name not in used:
                f.unlink(missing_ok=True)

    return cropped, skipped, from_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop raw_images/ using bbox sidecars into cropped_images/"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Source raw_images directory (e.g. versions/v1/raw_images)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination cropped_images directory (e.g. versions/v1/cropped_images)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.20,
        help="Fractional padding around the bbox on each side (default: 0.20)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output image size in pixels (default: 224)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.yaml",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Keep crops here and reuse them while nothing they depend on changed",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cats: list[str] = cfg.get("cats", [])
    if not cats:
        print("ERROR: no cats defined in config", file=sys.stderr)
        sys.exit(1)

    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    cache_dir = None
    if args.cache_dir is not None:
        cache_dir = args.cache_dir if args.cache_dir.is_absolute() else PROJECT_ROOT / args.cache_dir

    # Wipe existing output so a re-run with different padding is always clean
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Padding: {args.padding:.2f}  Size: {args.size}×{args.size}")
    if cache_dir is not None:
        print(f"Cache:   {cache_dir}")
    print()

    total_cropped = total_skipped = total_cached = 0
    for cat in cats:
        src = input_dir / cat
        if not src.is_dir():
            print(f"  {cat}: no raw_images directory, skipping")
            continue
        n_cropped, n_skipped, n_cached = process_cat(
            src, output_dir / cat, args.padding, args.size,
            cache_dir / cat if cache_dir is not None else None)
        cached_note = f" ({n_cached} from cache)" if cache_dir is not None else ""
        print(f"  {cat}: {n_cropped} cropped{cached_note}, {n_skipped} skipped")
        total_cropped += n_cropped
        total_skipped += n_skipped
        total_cached += n_cached

    print()
    cached_note = f" ({total_cached} from cache)" if cache_dir is not None else ""
    print(f"Done: {total_cropped} images cropped{cached_note}, {total_skipped} skipped")
    if total_skipped > 0:
        print(
            "  Skipped images have no sidecar JSON — only images captured after this "
            "update include bounding box metadata."
        )


if __name__ == "__main__":
    main()
