#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Pull training dataset from Raspberry Pi
# Runs export.py on the Pi via SSH, then downloads the archive.
#
# Usage: ./scripts/pull_dataset.sh <pi_user@pi_host> [remote_dir]
#
# Examples:
#   ./scripts/pull_dataset.sh pi@raspberrypi.local
#   ./scripts/pull_dataset.sh pi@192.168.1.50 ~/projects/cat_detection
# ═══════════════════════════════════════════════════════════════

PI_TARGET="${1:?Usage: $0 <pi_user@pi_host> [remote_dir]}"
REMOTE_DIR="${2:-~/Repos/cat_detection}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_EXPORT_DIR="$PROJECT_ROOT/data/export"

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
mkdir -p "$LOCAL_EXPORT_DIR"
echo ""
echo "[INFO] Downloading $ARCHIVE_NAME ..."
scp "$PI_TARGET:$REMOTE_ARCHIVE" "$LOCAL_EXPORT_DIR/$ARCHIVE_NAME"

# Extract
echo "[INFO] Extracting to $LOCAL_EXPORT_DIR ..."
tar -xzf "$LOCAL_EXPORT_DIR/$ARCHIVE_NAME" -C "$LOCAL_EXPORT_DIR"

echo ""
echo "[OK] Dataset pulled successfully!"
echo "  Archive: $LOCAL_EXPORT_DIR/$ARCHIVE_NAME"
echo ""
echo "Next step — prepare for training:"
echo "  python train/prepare_dataset.py"
