#!/usr/bin/env python3
"""Convert raw collected images into YOLO training format with auto-annotation."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, resolve_path
from utils.display import (
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset from raw collected images with auto-annotation"
    )
    add_common_args(parser)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to raw image directory (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for YOLO dataset output (default: data/dataset)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
        help="Minimum confidence for COCO cat detection (default: 0.15)",
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip auto-annotation; assume .txt label files already exist",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for split (default: from config)",
    )
    return parser.parse_args()


def auto_annotate(
    cats: list[str], input_dir: Path, min_confidence: float
) -> dict[str, int]:
    """Run YOLOv8n COCO model to auto-annotate cat images.

    Returns annotation statistics dict.
    """
    from ultralytics import YOLO

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    coco_model = YOLO("yolov8n.pt").to(device)

    stats = {"annotated": 0, "pre_annotated": 0, "no_detection": 0, "multi_detection": 0}

    for cat_idx, cat_name in enumerate(cats):
        cat_dir = input_dir / cat_name
        images = sorted(cat_dir.glob("*.jpg"))

        print_info(f"Annotating {cat_name}: {len(images)} images...")

        for img_path in images:
            # Skip images that already have a label file (manually annotated)
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                stats["pre_annotated"] += 1
                stats["annotated"] += 1
                continue

            results = coco_model.predict(
                str(img_path), verbose=False, conf=min_confidence
            )

            cat_boxes: list[str] = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 15:  # COCO class 15 = "cat"
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        img_h, img_w = r.orig_shape
                        # Convert to YOLO normalized format
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h
                        cat_boxes.append(
                            f"{cat_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                        )

            if len(cat_boxes) == 0:
                stats["no_detection"] += 1
                print_warning(f"  No cat detected: {img_path.name}")
            elif len(cat_boxes) > 1:
                stats["multi_detection"] += 1

            if cat_boxes:
                label_path = img_path.with_suffix(".txt")
                label_path.write_text("\n".join(cat_boxes) + "\n")
                stats["annotated"] += 1

    return stats


def print_annotation_report(stats: dict[str, int], total_images: int) -> None:
    """Print annotation summary."""
    print_header("Annotation Results")
    print_info(f"Total images: {total_images}")
    if total_images > 0:
        pct_ann = stats["annotated"] / total_images * 100
        pct_no = stats["no_detection"] / total_images * 100
        pct_multi = stats["multi_detection"] / total_images * 100
        pre = stats.get("pre_annotated", 0)
        print_info(f"Successfully annotated: {stats['annotated']} ({pct_ann:.1f}%)")
        if pre > 0:
            print_info(f"  Pre-annotated (manual): {pre}")
            print_info(f"  Auto-annotated: {stats['annotated'] - pre}")
        print_info(
            f"No cat detected: {stats['no_detection']} ({pct_no:.1f}%)"
            "  <- may need manual annotation"
        )
        print_info(f"Multiple cats in frame: {stats['multi_detection']} ({pct_multi:.1f}%)")


def split_dataset(
    cats: list[str],
    input_dir: Path,
    val_split: float,
    test_split: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    """Collect annotated samples and split into train/val/test."""
    all_samples: list[tuple[Path, Path]] = []
    for cat_name in cats:
        cat_dir = input_dir / cat_name
        for img in sorted(cat_dir.glob("*.jpg")):
            label = img.with_suffix(".txt")
            if label.exists():
                all_samples.append((img, label))

    random.seed(seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    val_count = int(n * val_split)
    test_count = int(n * test_split)
    train_count = n - val_count - test_count

    return {
        "train": all_samples[:train_count],
        "val": all_samples[train_count : train_count + val_count],
        "test": all_samples[train_count + val_count :],
    }


def create_yolo_structure(
    splits: dict[str, list[tuple[Path, Path]]], output_dir: Path
) -> None:
    """Copy images and labels into YOLO directory structure."""
    for split_name, samples in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in samples:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(label_path, lbl_dir / label_path.name)


def add_background_images(
    input_dir: Path, output_dir: Path, val_ratio: float = 0.2, seed: int = 42
) -> int:
    """Add background (negative) images with empty labels to the dataset.

    Background images teach the model what "no cat" looks like, reducing
    false positives. Expects images in input_dir/background/*.jpg.
    """
    bg_dir = input_dir / "background"
    if not bg_dir.is_dir():
        return 0

    bg_images = sorted(bg_dir.glob("*.jpg"))
    if not bg_images:
        return 0

    random.seed(seed)
    random.shuffle(bg_images)

    val_count = max(1, int(len(bg_images) * val_ratio))
    val_imgs = bg_images[:val_count]
    train_imgs = bg_images[val_count:]

    for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        for img_path in imgs:
            shutil.copy2(img_path, img_dir / img_path.name)
            # Empty label file = no objects in this image
            (lbl_dir / img_path.with_suffix(".txt").name).write_text("")

    return len(bg_images)


def create_calibration_dir(
    splits: dict[str, list[tuple[Path, Path]]], output_dir: Path, target: int = 500
) -> int:
    """Copy images for INT8 quantization calibration."""
    calib_dir = output_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Start with val images
    sources = [img for img, _ in splits["val"]]
    # Supplement with train images if needed
    if len(sources) < target:
        extra = [img for img, _ in splits["train"]]
        sources.extend(extra[: target - len(sources)])

    for img_path in sources:
        shutil.copy2(img_path, calib_dir / img_path.name)

    return len(sources)


def generate_dataset_yaml(
    output_dir: Path, cats: list[str]
) -> Path:
    """Write dataset.yaml for YOLO training."""
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(cats),
        "names": cats,
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    return yaml_path


def generate_labels_json(cats: list[str], detection_threshold: float) -> Path:
    """Write deploy/labels.json for Hailo inference."""
    labels_data = {
        "detection_threshold": detection_threshold,
        "max_boxes": 200,
        "labels": cats,
    }
    labels_path = PROJECT_ROOT / "deploy" / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)
    return labels_path


def print_dataset_stats(
    splits: dict[str, list[tuple[Path, Path]]],
    cats: list[str],
    input_dir: Path,
    calib_count: int,
    yaml_path: Path,
    labels_path: Path,
) -> None:
    """Print dataset statistics table with class balance check."""
    print_header("Dataset Statistics")

    # Count per-cat images in each split
    cat_counts: dict[str, dict[str, int]] = {
        split: {cat: 0 for cat in cats} for split in splits
    }
    for split_name, samples in splits.items():
        for img_path, _ in samples:
            # Cat name is the parent directory name
            cat_name = img_path.parent.name
            if cat_name in cat_counts[split_name]:
                cat_counts[split_name][cat_name] += 1

    # Header
    cat_headers = "  ".join(f"{c:>6s}" for c in cats)
    print(f"  {'Split':<8s}  {'Images':>6s}  {cat_headers}")
    print(f"  {'':─<{8 + 8 + (8 * len(cats))}}")

    totals_per_cat = {cat: 0 for cat in cats}
    for split_name in ("train", "val", "test"):
        n = len(splits[split_name])
        per_cat = "  ".join(
            f"{cat_counts[split_name][c]:>6d}" for c in cats
        )
        print(f"  {split_name.capitalize():<8s}  {n:>6d}  {per_cat}")
        for c in cats:
            totals_per_cat[c] += cat_counts[split_name][c]

    total = sum(len(s) for s in splits.values())
    total_cat = "  ".join(f"{totals_per_cat[c]:>6d}" for c in cats)
    print(f"  {'':─<{8 + 8 + (8 * len(cats))}}")
    print(f"  {'Total':<8s}  {total:>6d}  {total_cat}")
    print()
    print_info(f"Calibration images: {calib_count}")
    print()
    print_info(f"Dataset YAML: {yaml_path}")
    print_info(f"Labels JSON:  {labels_path}")

    # Class imbalance check
    if totals_per_cat:
        max_count = max(totals_per_cat.values())
        max_cat = max(totals_per_cat, key=totals_per_cat.get)
        for cat, count in totals_per_cat.items():
            if max_count > 0 and count / max_count < 0.65:
                print()
                print_warning("Class imbalance detected:")
                print_warning(
                    f"  {cat} has {count} images vs {max_cat}'s {max_count} "
                    f"(ratio: {count / max_count:.2f})"
                )
                print_warning(
                    f"  Consider collecting more {cat} images for better accuracy."
                )


def main() -> None:
    args = parse_args()
    cfg = get_config_with_args(args)

    cats: list[str] = cfg["cats"]
    input_dir = args.input or resolve_path(cfg["collection"]["output_dir"], PROJECT_ROOT)
    output_dir = args.output or resolve_path("data/dataset", PROJECT_ROOT)
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    val_split = cfg["training"]["val_split"]
    test_split = cfg["training"]["test_split"]

    # Verify input directory
    if not input_dir.exists():
        print_error(f"Input directory not found: {input_dir}")
        print_info("Run capture.py first to collect images.")
        sys.exit(1)

    missing = [c for c in cats if not (input_dir / c).is_dir()]
    if missing:
        print_error(f"Missing cat directories: {', '.join(missing)}")
        print_info(f"Expected subdirectories in {input_dir} for: {', '.join(cats)}")
        sys.exit(1)

    total_images = sum(
        len(list((input_dir / c).glob("*.jpg"))) for c in cats
    )

    if total_images == 0:
        print_error("No .jpg images found in input directories.")
        sys.exit(1)

    print_header("Prepare Dataset")
    print_info(f"Input: {input_dir}")
    print_info(f"Output: {output_dir}")
    print_info(f"Cats: {', '.join(cats)}")
    print_info(f"Total raw images: {total_images}")
    print()

    # Auto-annotation
    if not args.skip_annotation:
        stats = auto_annotate(cats, input_dir, args.min_confidence)
        print_annotation_report(stats, total_images)
    else:
        print_info("Skipping auto-annotation (--skip-annotation)")

    # Split dataset
    splits = split_dataset(cats, input_dir, val_split, test_split, seed)
    annotated_total = sum(len(s) for s in splits.values())

    if annotated_total == 0:
        print_error("No annotated images found. Cannot create dataset.")
        sys.exit(1)

    # Clear and create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create YOLO directory structure
    create_yolo_structure(splits, output_dir)

    # Add background (negative) images if available
    bg_count = add_background_images(input_dir, output_dir, seed=seed)
    if bg_count > 0:
        print_info(f"Background images added: {bg_count} (reduces false positives)")

    # Create calibration directory
    calib_count = create_calibration_dir(splits, output_dir)

    # Generate dataset.yaml
    yaml_path = generate_dataset_yaml(output_dir, cats)

    # Generate labels.json
    detection_threshold = cfg["deployment"]["detection_threshold"]
    labels_path = generate_labels_json(cats, detection_threshold)

    # Print statistics
    print_dataset_stats(splits, cats, input_dir, calib_count, yaml_path, labels_path)


if __name__ == "__main__":
    main()
