#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Transfer trained model to Raspberry Pi
#
# Usage:
#   ./scripts/transfer_model.sh <model_path>              # uses .env for PI target
#   ./scripts/transfer_model.sh <pi_user@pi_host> <model_path>
#
# Supports both .hef (Stage 1 detector) and .pt (Stage 2 classifier).
# Reads defaults from .env (PI_USER, PI_HOST, PI_REMOTE_DIR).
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Parse arguments: if $1 is a file path, treat it as the model; otherwise it's PI_TARGET
if [ $# -ge 2 ]; then
    PI_TARGET="$1"
    MODEL_PATH="$2"
elif [ $# -eq 1 ]; then
    PI_TARGET="${PI_USER:?Set PI_USER in .env}@${PI_HOST:?Set PI_HOST in .env}"
    MODEL_PATH="$1"
else
    PI_TARGET="${PI_USER:?Set PI_USER in .env}@${PI_HOST:?Set PI_HOST in .env}"
    MODEL_PATH="$PROJECT_ROOT/models/cat_detector.hef"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model file not found: $MODEL_PATH"
    exit 1
fi

REMOTE_DIR="${PI_REMOTE_DIR:-~/Repos/cat_detection}"
EXT="${MODEL_PATH##*.}"

# Create remote directories
ssh "$PI_TARGET" "mkdir -p $REMOTE_DIR/models $REMOTE_DIR/deploy"

if [ "$EXT" = "pt" ]; then
    # ── Stage 2: Classifier (.pt) ──
    echo "[INFO] Transferring classifier model to $PI_TARGET:$REMOTE_DIR ..."
    echo "  Model: $MODEL_PATH"
    scp "$MODEL_PATH" "$PI_TARGET:$REMOTE_DIR/models/cat_classifier.pt"
    echo ""
    echo "[OK] Classifier model transferred!"
elif [ "$EXT" = "hef" ]; then
    # ── Stage 1: Detector (.hef) ──
    LABELS_PATH="$PROJECT_ROOT/deploy/labels.json"
    if [ ! -f "$LABELS_PATH" ]; then
        echo "[ERROR] Labels file not found: $LABELS_PATH"
        echo "Run prepare_dataset.py first to generate labels.json."
        exit 1
    fi
    echo "[INFO] Transferring detector model to $PI_TARGET:$REMOTE_DIR ..."
    echo "  HEF:    $MODEL_PATH"
    echo "  Labels: $LABELS_PATH"
    scp "$MODEL_PATH" "$PI_TARGET:$REMOTE_DIR/models/cat_detector.hef"
    scp "$LABELS_PATH" "$PI_TARGET:$REMOTE_DIR/deploy/labels.json"
    echo ""
    echo "[OK] Detector model transferred!"
else
    echo "[ERROR] Unsupported model format: .$EXT (expected .pt or .hef)"
    exit 1
fi

echo ""
echo "On the Pi, run:"
echo "  cd $REMOTE_DIR/hailo-apps && source setup_env.sh"
echo "  cd $REMOTE_DIR"
echo "  python deploy/detect.py"
