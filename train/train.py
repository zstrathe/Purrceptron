#!/usr/bin/env python3
"""YOLOv8 training script for cat detection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, resolve_path
from utils.display import print_error, print_header, print_info, print_success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for cat detection"
    )
    add_common_args(parser)
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help="Path to dataset.yaml (default: data/dataset/dataset.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Override model size from config (e.g., yolov8n, yolov8s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--run-name", type=str, default="v1", help="Training run name for versioning"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Override project directory for training output (default: runs/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config_with_args(args)

    model_size = args.model_size or cfg["training"]["model_size"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    patience = cfg["training"]["patience"]
    image_size = cfg["training"]["image_size"]
    lr0 = cfg["training"].get("lr0", 0.01)
    lrf = cfg["training"].get("lrf", 0.01)
    optimizer = cfg["training"].get("optimizer", "SGD")
    dataset_yaml = args.dataset_yaml or resolve_path(
        "data/dataset/dataset.yaml", PROJECT_ROOT
    )

    # Verify dataset exists
    if not dataset_yaml.exists():
        print_error(f"Dataset YAML not found: {dataset_yaml}")
        print_info("Run prepare_dataset.py first to create the training dataset.")
        sys.exit(1)

    print_header("Starting YOLOv8 Training")
    print_info(f"Model: {model_size}")
    print_info(f"Dataset: {dataset_yaml}")
    print_info(f"Epochs: {epochs}, Batch: {batch_size}, Image size: {image_size}")
    print_info(f"LR: {lr0} -> {lrf}, Optimizer: {optimizer}")

    from ultralytics import YOLO

    run_name = args.run_name
    project_dir = args.project or (PROJECT_ROOT / "runs")
    print_info(f"Run: {run_name}")
    print_info(f"Project: {project_dir}")

    if args.resume:
        last_pt = project_dir / run_name / "weights" / "last.pt"
        if not last_pt.exists():
            print_error(f"No checkpoint found at {last_pt}")
            sys.exit(1)
        model = YOLO(str(last_pt))
        print_info(f"Resuming from {last_pt}")
    else:
        model = YOLO(f"{model_size}.pt")
        print_info(f"Starting from pretrained {model_size} COCO weights")

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        patience=patience,
        device=0,
        workers=0,  # WSL2: pin_memory thread causes CUDA OOM with workers>0
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,
        # Fixed-camera augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,       # No rotation — camera is fixed
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,        # No vertical flip
        mosaic=1.0,
        mixup=0.1,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"

    print_header("Training Complete")
    print_success(f"Best model: {best_pt}")
    print_info(f"Results directory: {results.save_dir}")
    print_info("")
    print_info("Next steps:")
    print_info(f"  1. Check results in {results.save_dir}/")
    print_info("     - confusion_matrix.png: see if cats are being confused")
    print_info("     - results.png: training curves")
    print_info("  2. If mAP50 > 0.7: good to export")
    print_info("  3. If mAP50 < 0.5: collect more data or try yolov8m")
    print_info(f"  4. Export: python train/export_onnx.py --model {best_pt}")


if __name__ == "__main__":
    main()
