#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Transfer trained model to Raspberry Pi
# Usage: ./scripts/transfer_model.sh [pi_user@pi_host] [hef_path]
#
# Reads defaults from .env (PI_USER, PI_HOST, PI_REMOTE_DIR).
# CLI arguments override .env values.
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PI_TARGET="${1:-${PI_USER:?Set PI_USER in .env}@${PI_HOST:?Set PI_HOST in .env}}"

HEF_PATH="${2:-$PROJECT_ROOT/models/cat_detector.hef}"

if [ ! -f "$HEF_PATH" ]; then
    echo "[ERROR] HEF file not found: $HEF_PATH"
    echo "Run convert_hef.sh first to generate the HEF file."
    exit 1
fi

LABELS_PATH="$PROJECT_ROOT/deploy/labels.json"
if [ ! -f "$LABELS_PATH" ]; then
    echo "[ERROR] Labels file not found: $LABELS_PATH"
    echo "Run prepare_dataset.py first to generate labels.json."
    exit 1
fi

REMOTE_DIR="${PI_REMOTE_DIR:-~/Repos/cat_detection}"

echo "[INFO] Transferring model to $PI_TARGET:$REMOTE_DIR ..."
echo "  HEF:    $HEF_PATH"
echo "  Labels: $LABELS_PATH"

# Create remote directories
ssh "$PI_TARGET" "mkdir -p $REMOTE_DIR/models $REMOTE_DIR/deploy"

# Transfer files
scp "$HEF_PATH" "$PI_TARGET:$REMOTE_DIR/models/cat_detector.hef"
scp "$LABELS_PATH" "$PI_TARGET:$REMOTE_DIR/deploy/labels.json"

echo ""
echo "[OK] Model transferred!"
echo ""
echo "On the Pi, run:"
echo "  source ~/hailo-apps/venv_hailo_apps/bin/activate"
echo "  cd $REMOTE_DIR"
echo "  python deploy/detect.py"
