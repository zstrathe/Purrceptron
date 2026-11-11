#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Transfer trained classifier model (.pt) to Raspberry Pi
#
# Usage:
#   ./scripts/transfer_model.sh <model_path>              # uses .env for PI target
#   ./scripts/transfer_model.sh <pi_user@pi_host> <model_path>
#
# Reads defaults from .env (PI_USER, PI_HOST, PI_REMOTE_DIR).
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MAX_RETRIES=4
RETRY_DELAY=3

retry() {
    local attempt
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        if "$@"; then
            return 0
        fi
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            echo "[WARN] Attempt $attempt/$MAX_RETRIES failed, retrying in ${RETRY_DELAY}s..."
            sleep "$RETRY_DELAY"
        fi
    done
    echo "[ERROR] Failed after $MAX_RETRIES attempts"
    return 1
}

# Parse arguments: if $1 is a file path, treat it as the model; otherwise it's PI_TARGET
if [ $# -ge 2 ]; then
    PI_TARGET="$1"
    MODEL_PATH="$2"
elif [ $# -eq 1 ]; then
    PI_TARGET="${PI_USER:?Set PI_USER in .env}@${PI_HOST:?Set PI_HOST in .env}"
    MODEL_PATH="$1"
else
    PI_TARGET="${PI_USER:?Set PI_USER in .env}@${PI_HOST:?Set PI_HOST in .env}"
    MODEL_PATH="$PROJECT_ROOT/models/cat_classifier.pt"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model file not found: $MODEL_PATH"
    exit 1
fi

REMOTE_DIR="${PI_REMOTE_DIR:-~/Repos/cat_detection}"

# Create remote directories
retry ssh "$PI_TARGET" "mkdir -p $REMOTE_DIR/models $REMOTE_DIR/deploy"

echo "[INFO] Transferring classifier model to $PI_TARGET:$REMOTE_DIR ..."
echo "  Model: $MODEL_PATH"
retry scp "$MODEL_PATH" "$PI_TARGET:$REMOTE_DIR/models/cat_classifier.pt"
echo ""
echo "[OK] Classifier model transferred!"

echo ""
echo "On the Pi, run:"
echo "  cd $REMOTE_DIR"
echo "  python deploy/detect.py"
