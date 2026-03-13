"""
Crop raw captured images using stored bounding box sidecars.

Reads full-frame JPEGs + .json sidecars from raw_images/<cat>/,
applies padding around the detected bounding box, and writes
resized crops to cropped_images/<cat>/.

Always regenerates the output directory, so re-running with a different
--padding value produces a clean re-crop without stale images.

Usage:
    python train/crop_images.py \\
        --input-dir versions/v1/raw_images \\
        --output-dir versions/v1/cropped_images \\
        [--padding 0.20] \\
        [--size 224] \\
        [--config configs/project.yaml]
"""

import argparse
import json
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


def process_cat(
    src_dir: Path,
    dst_dir: Path,
    padding: float,
    size: int,
) -> tuple[int, int]:
    """Crop all images in src_dir into dst_dir.

    Returns (n_cropped, n_skipped).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    cropped = skipped = 0

    for jpg in sorted(src_dir.glob("*.jpg")):
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

        cv2.imwrite(str(dst_dir / jpg.name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cropped += 1

    return cropped, skipped


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    cats: list[str] = cfg.get("cats", [])
    if not cats:
        print("ERROR: no cats defined in config", file=sys.stderr)
        sys.exit(1)

    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    # Wipe existing output so a re-run with different padding is always clean
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Padding: {args.padding:.2f}  Size: {args.size}×{args.size}")
    print()

    total_cropped = total_skipped = 0
    for cat in cats:
        src = input_dir / cat
        if not src.is_dir():
            print(f"  {cat}: no raw_images directory, skipping")
            continue
        n_cropped, n_skipped = process_cat(src, output_dir / cat, args.padding, args.size)
        print(f"  {cat}: {n_cropped} cropped, {n_skipped} skipped")
        total_cropped += n_cropped
        total_skipped += n_skipped

    print()
    print(f"Done: {total_cropped} images cropped, {total_skipped} skipped")
    if total_skipped > 0:
        print(
            "  Skipped images have no sidecar JSON — only images captured after this "
            "update include bounding box metadata."
        )


if __name__ == "__main__":
    main()
