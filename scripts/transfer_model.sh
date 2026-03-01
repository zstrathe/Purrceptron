#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Transfer trained model to Raspberry Pi
# Usage: ./scripts/transfer_model.sh <pi_user@pi_host> [hef_path]
# ═══════════════════════════════════════════════════════════════

PI_TARGET="${1:?Usage: $0 <pi_user@pi_host> [hef_path]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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

echo "[INFO] Transferring model to $PI_TARGET..."
echo "  HEF:    $HEF_PATH"
echo "  Labels: $LABELS_PATH"

# Create remote directories
ssh "$PI_TARGET" "mkdir -p ~/cat-detector/models ~/cat-detector/deploy"

# Transfer files
scp "$HEF_PATH" "$PI_TARGET:~/cat-detector/models/cat_detector.hef"
scp "$LABELS_PATH" "$PI_TARGET:~/cat-detector/deploy/labels.json"

echo ""
echo "[OK] Model transferred!"
echo ""
echo "On the Pi, run:"
echo "  source ~/hailo-apps/venv_hailo_apps/bin/activate"
echo "  cd ~/cat-detector"
echo "  python deploy/detect.py --hef-path models/cat_detector.hef --labels-json deploy/labels.json --input /dev/video0 --track-all"
