#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Convert ONNX model to HEF for Hailo 10H
# ═══════════════════════════════════════════════════════════════
#
# Runs hailomz locally if available, otherwise falls back to
# Docker (hailo-dfc:latest). See scripts/setup_hailo_docker.sh
# for one-time Docker setup.
#
# Usage:
#   ./train/convert_hef.sh <onnx_path> <num_classes> <calib_dir> [hw_arch] [model_arch]
#
# Example:
#   ./train/convert_hef.sh versions/v1/training/weights/best.onnx 3 versions/v1/dataset/calibration hailo10h yolov8m
#
# Note: This can take 30 minutes to several hours depending on model size.
# ═══════════════════════════════════════════════════════════════

ONNX_PATH="${1:?Error: Provide ONNX model path as first argument}"
NUM_CLASSES="${2:?Error: Provide number of classes as second argument}"
CALIB_DIR="${3:?Error: Provide calibration images directory as third argument}"
HW_ARCH="${4:-hailo10h}"
MODEL_ARCH="${5:-yolov8s}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-detect Docker image: prefer hailo-dfc:latest, fall back to hailo_ai_sw_suite_*
if docker image inspect "hailo-dfc:latest" &>/dev/null 2>&1; then
    DOCKER_IMAGE="hailo-dfc:latest"
else
    DOCKER_IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^hailo_ai_sw_suite_' | head -1 || true)
fi

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

echo "[INFO] Converting $ONNX_PATH for $HW_ARCH with $NUM_CLASSES classes"
echo "[INFO] Model architecture: $MODEL_ARCH"
echo "[INFO] Calibration images: $CALIB_COUNT from $CALIB_DIR"
echo "[INFO] This may take 30+ minutes..."
echo ""

# Compute relative paths (needed for Docker volume mounts)
REL_ONNX=$(realpath --relative-to="$PROJECT_ROOT" "$ONNX_PATH")
REL_CALIB=$(realpath --relative-to="$PROJECT_ROOT" "$CALIB_DIR")

# ── Run compilation (local or Docker) ─────────────────────────
if command -v hailomz &>/dev/null; then
    echo "[INFO] Using local hailomz"
    hailomz compile "$MODEL_ARCH" \
        --ckpt "$ONNX_PATH" \
        --hw-arch "$HW_ARCH" \
        --calib-path "$CALIB_DIR" \
        --performance

elif [ -n "$DOCKER_IMAGE" ] && docker image inspect "$DOCKER_IMAGE" &>/dev/null 2>&1; then
    echo "[INFO] Using Docker image: $DOCKER_IMAGE"
    # Strategy: Trim ONNX to end at detection head Conv outputs, then compile
    # with overwritten alls (change_output_activation + nms_postprocess).
    #
    # Why: Ultralytics ONNX includes Sigmoid/DFL after the Conv layers.
    # Hailo's change_output_activation and nms_postprocess both require raw
    # Conv outputs. Without nms_postprocess, DFL depth_to_space stays on-chip
    # and causes "infeasible" errors.
    docker run --rm \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        bash -c '
            set -euo pipefail

            echo "[INFO] Step 1: Trimming ONNX to detection head Conv outputs..."
            python3 /workspace/.trim_onnx.py \
                "/workspace/'"$REL_ONNX"'" \
                /tmp/model_hailo.onnx

            echo ""
            echo "[INFO] Step 2: Creating NMS config..."
            cat > /tmp/nms_config.json << NMSCFG
{
    "nms_scores_th": 0.2,
    "nms_iou_th": 0.7,
    "image_dims": [640, 640],
    "max_proposals_per_class": 100,
    "classes": '"$NUM_CLASSES"',
    "regression_length": 16,
    "background_removal": false,
    "background_removal_index": 0,
    "bbox_decoders": [
        {"name": "bbox_decoder57", "stride": 8,  "reg_layer": "conv57", "cls_layer": "conv58"},
        {"name": "bbox_decoder70", "stride": 16, "reg_layer": "conv70", "cls_layer": "conv71"},
        {"name": "bbox_decoder82", "stride": 32, "reg_layer": "conv82", "cls_layer": "conv83"}
    ]
}
NMSCFG

            echo "[INFO] Step 3: Overwriting model zoo alls script..."
            # Find and overwrite the alls file so hailomz compile uses our version
            HMZ=/local/workspace/hailo_model_zoo/hailo_model_zoo
            ALLS_PATH=$(find "$HMZ/cfg/alls" -path "*/base/'"$MODEL_ARCH"'.alls" | head -1)
            if [ -z "$ALLS_PATH" ]; then
                echo "[ERROR] alls script not found for '"$MODEL_ARCH"'"
                exit 1
            fi
            echo "[INFO] Overwriting: $ALLS_PATH"
            cat > "$ALLS_PATH" << ALLS
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_config(calibration, batch_size=2)
change_output_activation(conv58, sigmoid)
change_output_activation(conv71, sigmoid)
change_output_activation(conv83, sigmoid)
nms_postprocess("/tmp/nms_config.json", meta_arch=yolov8, engine=cpu)
performance_param(optimize_for_batch=2)
allocator_param(enable_fixer=max_adjcents)
resources_param(max_utilization=1.0, max_memory_utilization=0.85)
ALLS

            echo ""
            echo "[INFO] Step 4: Compiling with hailomz..."
            hailomz compile '"$MODEL_ARCH"' \
                --ckpt /tmp/model_hailo.onnx \
                --hw-arch "'"$HW_ARCH"'" \
                --calib-path "/workspace/'"$REL_CALIB"'" \
                --classes '"$NUM_CLASSES"' \
                --performance
        '

    # Clean up the trim script (it was only needed inside Docker)
    rm -f "$PROJECT_ROOT/.trim_onnx.py"

else
    echo "[ERROR] hailomz not found and Docker image '$DOCKER_IMAGE' not loaded."
    echo ""
    echo "One-time setup:"
    echo "  1. Register at https://hailo.ai/developer-zone/"
    echo "  2. Download the Hailo AI Software Suite (Docker)"
    echo "  3. Run: ./scripts/setup_hailo_docker.sh <path_to_hailo_docker.tar.gz>"
    echo ""
    echo "After setup, the Convert HEF step will work automatically."
    exit 1
fi

# ── Collect output HEF ────────────────────────────────────────
HEF_FILE=$(ls -t "$PROJECT_ROOT"/*.hef "$PROJECT_ROOT"/workspace/*.hef *.hef 2>/dev/null | head -1 || true)

if [ -z "$HEF_FILE" ]; then
    echo "[ERROR] No HEF file generated. Check the compilation output above for errors."
    exit 1
fi

mkdir -p "$PROJECT_ROOT/models"
mv "$HEF_FILE" "$PROJECT_ROOT/models/cat_detector.hef"

echo ""
echo "[OK] HEF file created: $PROJECT_ROOT/models/cat_detector.hef"
echo ""
echo "Transfer to your Raspberry Pi with:"
echo "  scp $PROJECT_ROOT/models/cat_detector.hef pi@raspberrypi:~/cat-detector/models/"
echo "  scp $PROJECT_ROOT/deploy/labels.json pi@raspberrypi:~/cat-detector/deploy/"
