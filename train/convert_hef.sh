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
#   ./train/convert_hef.sh <onnx_path> <num_classes> <calib_dir> [hw_arch] [model_arch] [output_dir]
#
# Example:
#   ./train/convert_hef.sh versions/v1/training/weights/best.onnx 3 versions/v1/dataset/calibration hailo10h yolov8m versions/v1
#
# Note: This can take 30 minutes to several hours depending on model size.
# ═══════════════════════════════════════════════════════════════

ONNX_PATH="${1:?Error: Provide ONNX model path as first argument}"
NUM_CLASSES="${2:?Error: Provide number of classes as second argument}"
CALIB_DIR="${3:?Error: Provide calibration images directory as third argument}"
HW_ARCH="${4:-hailo10h}"
MODEL_ARCH="${5:-yolov8s}"
OUTPUT_DIR="${6:-$PROJECT_ROOT/models}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# YOLOv8 variant conv layer names (for NMS postprocess config)
case "$MODEL_ARCH" in
    yolov8n) CONV_REG=(conv28 conv40 conv53); CONV_CLS=(conv29 conv41 conv54) ;;
    yolov8s) CONV_REG=(conv42 conv55 conv67); CONV_CLS=(conv43 conv56 conv68) ;;
    yolov8m) CONV_REG=(conv57 conv70 conv82); CONV_CLS=(conv58 conv71 conv83) ;;
    yolov8l) CONV_REG=(conv72 conv85 conv97); CONV_CLS=(conv73 conv86 conv98) ;;
    yolov8x) CONV_REG=(conv72 conv85 conv97); CONV_CLS=(conv73 conv86 conv98) ;;
    *)       echo "[ERROR] Unsupported model architecture: $MODEL_ARCH"; exit 1 ;;
esac

# Generate NMS config and model script (avoids broken --classes codepath)
generate_hailo_scripts() {
    local dest_dir="$1"
    cat > "$dest_dir/nms_config.json" << NMSCFG
{
    "nms_scores_th": 0.2,
    "nms_iou_th": 0.7,
    "image_dims": [640, 640],
    "max_proposals_per_class": 100,
    "classes": $NUM_CLASSES,
    "regression_length": 16,
    "background_removal": false,
    "background_removal_index": 0,
    "bbox_decoders": [
        {"name": "bbox_decoder_s8",  "stride": 8,  "reg_layer": "${CONV_REG[0]}", "cls_layer": "${CONV_CLS[0]}"},
        {"name": "bbox_decoder_s16", "stride": 16, "reg_layer": "${CONV_REG[1]}", "cls_layer": "${CONV_CLS[1]}"},
        {"name": "bbox_decoder_s32", "stride": 32, "reg_layer": "${CONV_REG[2]}", "cls_layer": "${CONV_CLS[2]}"}
    ]
}
NMSCFG

    cat > "$dest_dir/model_script.alls" << ALLS
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_config(calibration, batch_size=2)
nms_postprocess("$dest_dir/nms_config.json", meta_arch=yolov8, engine=cpu)
quantization_param(${CONV_CLS[0]}, precision_mode=a16_w16)
quantization_param(${CONV_CLS[1]}, precision_mode=a16_w16)
quantization_param(${CONV_CLS[2]}, precision_mode=a16_w16)
performance_param(optimize_for_batch=2)
ALLS
    echo "[INFO] Generated model script with $NUM_CLASSES classes for $MODEL_ARCH"
}

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
    TRIMMED_ONNX="${ONNX_PATH%.onnx}_trimmed.onnx"
    python3 "$SCRIPT_DIR/trim_onnx.py" "$ONNX_PATH" "$TRIMMED_ONNX"
    generate_hailo_scripts "/tmp"
    HMZ_CFG=$(python3 -c "import hailo_model_zoo, os; print(os.path.join(hailo_model_zoo.__path__[0], 'cfg'))")
    hailomz compile \
        --ckpt "$TRIMMED_ONNX" \
        --calib-path "$CALIB_DIR" \
        --yaml "$HMZ_CFG/networks/$MODEL_ARCH.yaml" \
        --model-script /tmp/model_script.alls \
        --hw-arch "$HW_ARCH"

elif [ -n "$DOCKER_IMAGE" ] && docker image inspect "$DOCKER_IMAGE" &>/dev/null 2>&1; then
    echo "[INFO] Using Docker image: $DOCKER_IMAGE"
    # Pass GPU if available and container toolkit is installed
    GPU_FLAG=""
    if command -v nvidia-smi &>/dev/null \
       && docker run --rm --gpus all "$DOCKER_IMAGE" nvidia-smi &>/dev/null 2>&1; then
        GPU_FLAG="--gpus all"
        echo "[INFO] GPU passthrough enabled"
    fi

    docker run --rm $GPU_FLAG \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        bash -c '
            set -euo pipefail

            # Trim ONNX to raw conv outputs (removes Sigmoid + DFL layers)
            python3 /workspace/train/trim_onnx.py \
                "/workspace/'"$REL_ONNX"'" \
                /tmp/model_trimmed.onnx

            # Generate NMS config and model script inside the container
            cat > /tmp/nms_config.json << '"'"'NMSCFG'"'"'
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
        {"name": "bbox_decoder_s8",  "stride": 8,  "reg_layer": "'"${CONV_REG[0]}"'", "cls_layer": "'"${CONV_CLS[0]}"'"},
        {"name": "bbox_decoder_s16", "stride": 16, "reg_layer": "'"${CONV_REG[1]}"'", "cls_layer": "'"${CONV_CLS[1]}"'"},
        {"name": "bbox_decoder_s32", "stride": 32, "reg_layer": "'"${CONV_REG[2]}"'", "cls_layer": "'"${CONV_CLS[2]}"'"}
    ]
}
NMSCFG

            cat > /tmp/model_script.alls << '"'"'ALLS'"'"'
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_config(calibration, batch_size=2)
nms_postprocess("/tmp/nms_config.json", meta_arch=yolov8, engine=cpu)
quantization_param('"${CONV_CLS[0]}"', precision_mode=a16_w16)
quantization_param('"${CONV_CLS[1]}"', precision_mode=a16_w16)
quantization_param('"${CONV_CLS[2]}"', precision_mode=a16_w16)
performance_param(optimize_for_batch=2)
ALLS

            # Locate HMZ network config inside the Docker image
            HMZ=$(python3 -c "import hailo_model_zoo, os; print(os.path.join(hailo_model_zoo.__path__[0], \"cfg\"))" 2>/dev/null \
                  || echo "/local/workspace/hailo_model_zoo/hailo_model_zoo/cfg")
            YAML_PATH="$HMZ/networks/'"$MODEL_ARCH"'.yaml"

            if [ ! -f "$YAML_PATH" ]; then
                echo "[ERROR] Network config not found: $YAML_PATH"
                echo "[INFO] Available yolov8 configs:"
                ls "$HMZ/networks/"*yolov8* 2>/dev/null || echo "  (none found)"
                exit 1
            fi

            echo "[INFO] Using HMZ config: $YAML_PATH"
            echo "[INFO] Using custom model script with '"$NUM_CLASSES"' classes"
            hailomz compile \
                --ckpt /tmp/model_trimmed.onnx \
                --calib-path "/workspace/'"$REL_CALIB"'" \
                --yaml "$YAML_PATH" \
                --model-script /tmp/model_script.alls \
                --hw-arch "'"$HW_ARCH"'"
        '

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

mkdir -p "$OUTPUT_DIR"
mv "$HEF_FILE" "$OUTPUT_DIR/cat_detector.hef"
echo ""
echo "[OK] HEF file created: $OUTPUT_DIR/cat_detector.hef"

echo ""
echo "Transfer to your Raspberry Pi with:"
echo "  scp $OUTPUT_DIR/cat_detector.hef pi@raspberrypi:~/cat-detector/models/"
echo "  scp $PROJECT_ROOT/deploy/labels.json pi@raspberrypi:~/cat-detector/deploy/"
