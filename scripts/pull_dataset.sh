#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Pull training dataset from Raspberry Pi
# Runs export.py on the Pi via SSH, then downloads the archive.
#
# Usage: ./scripts/pull_dataset.sh [pi_user@pi_host] [remote_dir]
#
# Reads defaults from .env (PI_USER, PI_HOST, PI_REMOTE_DIR).
# CLI arguments override .env values.
#
# Examples:
#   ./scripts/pull_dataset.sh                              # uses .env
#   ./scripts/pull_dataset.sh pi@raspberrypi.local
#   ./scripts/pull_dataset.sh pi@192.168.1.50 ~/projects/cat_detection
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PI_TARGET="${1:-${PI_USER:?Set PI_USER in .env}@${PI_HOST:?Set PI_HOST in .env}}"
REMOTE_DIR="${2:-${PI_REMOTE_DIR:-~/Repos/cat_detection}}"
LOCAL_RAW_DIR="${3:-$PROJECT_ROOT/data/raw}"

echo "[INFO] Running export on $PI_TARGET ..."

# Run export.py on the Pi and capture the archive filename
REMOTE_OUTPUT=$(ssh "$PI_TARGET" "
    cd $REMOTE_DIR
    source venv_collect/bin/activate 2>/dev/null || true
    python collect/export.py 2>&1
")

echo "$REMOTE_OUTPUT"

# Extract the archive path from export.py output (looks for data/export/dataset_*.tar.gz)
ARCHIVE_NAME=$(echo "$REMOTE_OUTPUT" | grep -oP 'dataset_\d{8}\.tar\.gz' | tail -1)

if [ -z "$ARCHIVE_NAME" ]; then
    echo "[ERROR] Could not find archive name in export output."
    echo "Try running export.py on the Pi manually first."
    exit 1
fi

REMOTE_ARCHIVE="$REMOTE_DIR/data/export/$ARCHIVE_NAME"

# Download the archive
mkdir -p "$LOCAL_RAW_DIR"
echo ""
echo "[INFO] Downloading $ARCHIVE_NAME ..."
scp "$PI_TARGET:$REMOTE_ARCHIVE" "$PROJECT_ROOT/$ARCHIVE_NAME"

# Extract into data/raw/ where prepare_dataset.py expects it ({cat_name}/image.jpg)
echo "[INFO] Extracting to $LOCAL_RAW_DIR ..."
tar -xzf "$PROJECT_ROOT/$ARCHIVE_NAME" -C "$LOCAL_RAW_DIR"
rm "$PROJECT_ROOT/$ARCHIVE_NAME"

echo ""
echo "[OK] Dataset pulled and extracted to $LOCAL_RAW_DIR"
echo ""
echo "Next step — prepare for training:"
echo "  source venv_train/bin/activate"
echo "  python train/prepare_dataset.py"
