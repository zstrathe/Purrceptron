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
echo "For data collection:"
echo "  source venv_collect/bin/activate"
echo "  python collect/capture.py"
echo ""
echo "For inference (after training):"
echo "  source ~/hailo-apps/venv_hailo_apps/bin/activate"
echo "  python deploy/detect.py --hef-path models/cat_detector.hef --labels-json deploy/labels.json --input /dev/video0"
