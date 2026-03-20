#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Pull training dataset from Raspberry Pi
# Rsyncs data/raw/ from the Pi into the local data/raw/ directory.
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

MAX_RETRIES=5
RETRY_DELAY=5

run_with_retry() {
    local description="$1"
    shift
    for attempt in $(seq 1 $MAX_RETRIES); do
        set +e
        "$@"
        local rc=$?
        set -e
        if [ $rc -eq 0 ]; then
            return 0
        fi
        if [ $rc -eq 255 ] && [ "$attempt" -lt $MAX_RETRIES ]; then
            echo "  [RETRY $attempt/$MAX_RETRIES] $description failed, retrying in ${RETRY_DELAY}s ..."
            sleep $RETRY_DELAY
        else
            echo "[ERROR] $description failed (exit $rc)"
            return $rc
        fi
    done
}

# Resolve ~ for rsync (rsync doesn't expand ~ like SSH does)
REMOTE_HOME=$(run_with_retry "SSH home resolve" ssh "$PI_TARGET" "echo \$HOME")
RESOLVED_DIR="${REMOTE_DIR/#\~/$REMOTE_HOME}"

mkdir -p "$LOCAL_RAW_DIR"
echo "[INFO] Syncing data/raw/ from $PI_TARGET ..."
run_with_retry "rsync pull" rsync -avz --delete --exclude .thumbnails \
    "$PI_TARGET:$RESOLVED_DIR/data/raw/" \
    "$LOCAL_RAW_DIR/"

echo ""
echo "[OK] Dataset synced to $LOCAL_RAW_DIR"
echo ""
echo "Next step: open the training dashboard or run:"
echo "  source venv_train/bin/activate"
echo "  python train/import_captures.py"
