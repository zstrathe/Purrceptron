#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

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
echo "Workflow (use training dashboard for guided steps):"
echo "  1. Pull captures from Pi"
echo "  2. Label and crop images"
echo "  3. Prepare classifier dataset: python train/prepare_classifier_dataset.py"
echo "  4. Train classifier:           python train/train_classifier.py"
echo "  5. Deploy to Pi:               ./scripts/transfer_model.sh"
