#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Setup Raspberry Pi for cat detection data collection
# Run from the project root: ~/cat-detector/
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[INFO] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-opencv python3-yaml python3-pip python3-venv

# ── Install hailo-apps (Hailo AI detection pipeline) ──────────────────
# Prerequisites: hailort driver + packages must already be installed.
# See: https://github.com/hailo-ai/hailo-apps#prerequisites
HAILO_APPS_DIR="$PROJECT_ROOT/hailo-apps"
if [ ! -d "$HAILO_APPS_DIR" ]; then
    echo "[INFO] Installing hailo-apps..."
    git clone https://github.com/hailo-ai/hailo-apps.git "$HAILO_APPS_DIR"
    cd "$HAILO_APPS_DIR"
    sudo ./install.sh
    cd "$PROJECT_ROOT"
else
    echo "[INFO] hailo-apps already installed at $HAILO_APPS_DIR"
fi

# ── Install project dependencies into hailo-apps venv ─────────────────
# setup_env.sh activates venv_hailo_apps and sets GStreamer/Hailo env vars.
# We use this single venv for everything (dashboard + detection + classifier).
SAVE_PROJECT_ROOT="$PROJECT_ROOT"
echo "[INFO] Installing project dependencies in hailo-apps venv..."
cd "$HAILO_APPS_DIR"
set +u  # setup_env.sh uses variables that may be unbound (e.g. ZSH_VERSION)
source setup_env.sh
set -u
pip install --upgrade pip
pip install -r "$SAVE_PROJECT_ROOT/requirements-pi.txt"
deactivate 2>/dev/null || true
PROJECT_ROOT="$SAVE_PROJECT_ROOT"
cd "$PROJECT_ROOT"

echo "[INFO] Creating data directories..."
mkdir -p data/raw data/trash data/export models

echo "[INFO] Testing camera..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f'[OK] Camera working: {w}x{h}')
    else:
        print('[WARN] Camera opened but cannot read frames')
    cap.release()
else:
    print('[WARN] Cannot open camera at index 0')
    print('       Check: ls /dev/video*')
    print('       Try: --device /dev/video0 or --device /dev/video2')
"

echo ""
echo "[OK] Pi setup complete!"
echo ""
echo "To start the dashboard (includes live detection + review):"
echo "  cd $PROJECT_ROOT"
echo "  source hailo-apps/setup_env.sh"
echo "  python pi_dashboard.py"
echo ""
echo "Use --no-camera for review-only mode without the detection pipeline."
