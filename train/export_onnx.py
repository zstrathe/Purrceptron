#!/usr/bin/env python3
"""Export trained YOLOv8 model to ONNX for Hailo DFC conversion."""
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
        description="Export trained YOLOv8 model to ONNX for Hailo conversion"
    )
    add_common_args(parser)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to best.pt (default: runs/<run-name>/weights/best.pt)",
    )
    parser.add_argument(
        "--run-name", type=str, default="v1", help="Training run name for versioning"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config_with_args(args)

    model_path = args.model or resolve_path(
        f"runs/{args.run_name}/weights/best.pt", PROJECT_ROOT
    )

    if not model_path.exists():
        print_error(f"Model not found: {model_path}")
        print_info("Run train.py first, or specify --model path/to/best.pt")
        sys.exit(1)

    image_size = cfg["training"]["image_size"]

    print_header("Exporting to ONNX")
    print_info(f"Model: {model_path}")
    print_info(f"Image size: {image_size}")
    print_info("Opset: 11 (required for Hailo compatibility)")
    print_info("NMS: disabled (Hailo handles NMS in post-processing)")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    export_path = model.export(
        format="onnx",
        opset=11,           # Required for Hailo DFC
        imgsz=image_size,
        simplify=False,     # Preserve all YOLOv8 output heads
        nms=False,          # Hailo handles NMS separately
    )

    onnx_path = Path(export_path)

    print_success(f"ONNX exported: {onnx_path}")
    print_info(f"File size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    print_info("")
    print_info("Next steps:")
    print_info("  Convert to HEF for Hailo 10H:")
    num_cats = len(cfg["cats"])
    calib_dir = resolve_path("data/dataset/calibration", PROJECT_ROOT)
    print_info(f"  ./train/convert_hef.sh {onnx_path} {num_cats} {calib_dir}")


if __name__ == "__main__":
    main()
