#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Setup training machine (WSL2/Linux x86_64)
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[INFO] Creating training virtual environment..."
cd "$PROJECT_ROOT"
python3 -m venv venv_train
source venv_train/bin/activate
pip install --upgrade pip

echo "[INFO] Installing training dependencies..."
pip install -r requirements-training.txt

echo "[INFO] Verifying GPU setup..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'[OK] GPU: {gpu} ({vram:.1f} GB VRAM)')
else:
    print('[WARN] No CUDA GPU detected. Training will use CPU (much slower).')
    print('       Install PyTorch with CUDA support:')
    print('       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
"

echo ""
echo "[OK] Training setup complete!"
echo ""
echo "Activate with: source venv_train/bin/activate"
echo ""
echo "Workflow:"
echo "  1. Extract dataset: tar -xzf dataset_YYYYMMDD.tar.gz -C data/raw/"
echo "  2. Prepare: python train/prepare_dataset.py"
echo "  3. Train:   python train/train.py"
echo "  4. Export:   python train/export_onnx.py"
echo "  5. Convert:  ./train/convert_hef.sh <onnx> <classes> <calib_dir>"
