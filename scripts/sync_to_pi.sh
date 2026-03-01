#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Sync project code to Raspberry Pi via rsync over SSH
# Usage: ./scripts/sync_to_pi.sh <pi_user@pi_host> [remote_dir]
#
# Examples:
#   ./scripts/sync_to_pi.sh pi@raspberrypi.local
#   ./scripts/sync_to_pi.sh pi@192.168.1.50 ~/projects/cat_detection
# ═══════════════════════════════════════════════════════════════

PI_TARGET="${1:?Usage: $0 <pi_user@pi_host> [remote_dir]}"
REMOTE_DIR="${2:-~/Repos/cat_detection}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[INFO] Syncing project to $PI_TARGET:$REMOTE_DIR ..."

rsync -avz --delete \
    --exclude-from="$PROJECT_ROOT/.gitignore" \
    --exclude ".git/" \
    "$PROJECT_ROOT/" \
    "$PI_TARGET:$REMOTE_DIR/"

echo ""
echo "[OK] Sync complete!"
echo ""
echo "On the Pi:"
echo "  cd $REMOTE_DIR"
echo "  pip install -r requirements-pi.txt   # first time only"
