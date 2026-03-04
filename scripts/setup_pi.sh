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
HAILO_APPS_DIR="$HOME/hailo-apps"
if [ ! -d "$HAILO_APPS_DIR" ]; then
    echo "[INFO] Installing hailo-apps..."
    git clone https://github.com/hailo-ai/hailo-apps.git "$HAILO_APPS_DIR"
    cd "$HAILO_APPS_DIR"
    sudo ./install.sh
    cd "$PROJECT_ROOT"
else
    echo "[INFO] hailo-apps already installed at $HAILO_APPS_DIR"
fi

# Ensure PyYAML is available in hailo-apps venv (needed by detect.py's utils)
# Note: setup_env.sh looks for venv_hailo_apps relative to cwd, so we must cd first.
# It also clobbers PROJECT_ROOT, so we save/restore ours.
SAVE_PROJECT_ROOT="$PROJECT_ROOT"
echo "[INFO] Installing detect.py dependencies in hailo-apps venv..."
cd "$HAILO_APPS_DIR"
set +u  # setup_env.sh uses variables that may be unbound (e.g. ZSH_VERSION)
source setup_env.sh
set -u
pip install "PyYAML>=6.0"
deactivate 2>/dev/null || true
PROJECT_ROOT="$SAVE_PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ── Dashboard virtual environment ─────────────────────────────────────
echo "[INFO] Creating data collection virtual environment..."
cd "$PROJECT_ROOT"
python3 -m venv venv_collect
source venv_collect/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt

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
echo "  source venv_collect/bin/activate"
echo "  python dashboard.py"
echo ""
echo "The dashboard automatically starts detect.py in the hailo-apps environment."
echo "Use --no-camera for review-only mode without the detection pipeline."
