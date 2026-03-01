#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Convert ONNX model to HEF for Hailo 10H
# ═══════════════════════════════════════════════════════════════
#
# Prerequisites:
#   1. Install Hailo Dataflow Compiler:
#      pip install hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl
#   2. Install Hailo Model Zoo (for model scripts):
#      pip install hailo_model_zoo  # or clone from GitHub
#
# Usage:
#   ./train/convert_hef.sh <onnx_path> <num_classes> <calib_image_dir> [hw_arch]
#
# Example:
#   ./train/convert_hef.sh runs/cat_detect/train/weights/best.onnx 3 data/dataset/calibration hailo10h
#
# Note: This can take 30 minutes to several hours depending on model size
# and whether --performance mode is used.
# ═══════════════════════════════════════════════════════════════

ONNX_PATH="${1:?Error: Provide ONNX model path as first argument}"
NUM_CLASSES="${2:?Error: Provide number of classes as second argument}"
CALIB_DIR="${3:?Error: Provide calibration images directory as third argument}"
HW_ARCH="${4:-hailo10h}"

# Validate inputs
if [ ! -f "$ONNX_PATH" ]; then
    echo "[ERROR] ONNX file not found: $ONNX_PATH"
    exit 1
fi

if [ ! -d "$CALIB_DIR" ]; then
    echo "[ERROR] Calibration directory not found: $CALIB_DIR"
    exit 1
fi

CALIB_COUNT=$(find "$CALIB_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
if [ "$CALIB_COUNT" -lt 100 ]; then
    echo "[WARN] Only $CALIB_COUNT calibration images found. Recommend 200-1000 for best quantization."
fi

# Check if hailomz is available
if ! command -v hailomz &> /dev/null; then
    echo "[ERROR] hailomz command not found."
    echo "Install the Hailo Dataflow Compiler and Hailo Model Zoo first."
    echo "See: https://github.com/hailo-ai/hailo_model_zoo"
    exit 1
fi

echo "[INFO] Converting $ONNX_PATH for $HW_ARCH with $NUM_CLASSES classes"
echo "[INFO] Calibration images: $CALIB_COUNT from $CALIB_DIR"
echo "[INFO] This may take 30+ minutes..."

# Run compilation
# --ckpt: path to the ONNX checkpoint
# --hw-arch: target hardware (hailo10h for Hailo 10H / AI HAT+ 2)
# --calib-path: directory of images for INT8 quantization calibration
# --classes: number of output classes (your cat count)
# --performance: optimize for maximum inference throughput
hailomz compile yolov8s \
    --ckpt "$ONNX_PATH" \
    --hw-arch "$HW_ARCH" \
    --calib-path "$CALIB_DIR" \
    --classes "$NUM_CLASSES" \
    --performance

# Find the generated HEF file
HEF_FILE=$(ls -t *.hef 2>/dev/null | head -1)

if [ -z "$HEF_FILE" ]; then
    echo "[ERROR] No HEF file generated. Check the compilation output above for errors."
    exit 1
fi

# Move to models directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
mkdir -p "$PROJECT_ROOT/models"
mv "$HEF_FILE" "$PROJECT_ROOT/models/cat_detector.hef"

echo ""
echo "[OK] HEF file created: $PROJECT_ROOT/models/cat_detector.hef"
echo ""
echo "Transfer to your Raspberry Pi with:"
echo "  scp $PROJECT_ROOT/models/cat_detector.hef pi@raspberrypi:~/cat-detector/models/"
echo "  scp $PROJECT_ROOT/deploy/labels.json pi@raspberrypi:~/cat-detector/deploy/"
