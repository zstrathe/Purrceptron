#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

r"""Training worker for the PC: runs the jobs the Pi dashboard sends.

Headless; all UI is on the Pi dashboard. The Pi's "Local PC" backend sends
training, cross-validation, and auto-detect jobs here, and the Annotate tab's
Auto-detect button asks it for one image's bounding box. Jobs run with this
Python, so start it from the training venv. Runs on Linux (including WSL2)
and Windows. API: remote_training/worker.py.

Usage (WORKER_TOKEN=<long random string> in .env first):
    venv_train/bin/python train_worker.py [--host 0.0.0.0] [--port 5004]   # Linux / WSL2
    venv_win\Scripts\python train_worker.py [--port 5004]                   # Windows
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask

from remote_training.worker import NO_QNNPACK, create_worker_blueprint, load_worker_token
from utils import PROJECT_ROOT
from utils.display import print_error, print_info, print_success, print_warning


def main() -> None:
    parser = argparse.ArgumentParser(description="Training worker for the Pi dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5004, help="Port (default: 5004)")
    args = parser.parse_args()

    if not load_worker_token(PROJECT_ROOT):
        print_error("WORKER_TOKEN is not set. Add a line WORKER_TOKEN=<any long random "
                    "string> to .env,\nand enter the same token on the Pi dashboard's "
                    "Training tab.")
        sys.exit(1)

    app = Flask(__name__)
    worker_bp, jobs = create_worker_blueprint(PROJECT_ROOT, sys.executable)
    app.register_blueprint(worker_bp)

    health = jobs.health()
    if not health.get("ok"):
        print_error(health.get("error", "torch is not usable in this Python"))
        sys.exit(1)
    if not health.get("int8_ok"):
        print_error(NO_QNNPACK + f" (engines here: {health.get('quant_engines')})")
        sys.exit(1)
    gpu = health.get("gpu")
    print_info(f"Python: {sys.executable}")
    print_info(f"torch {health.get('torch_version')}")
    if gpu:
        print_success(f"GPU: {gpu}")
    else:
        print_warning("No CUDA GPU: jobs will run on the CPU (slow)")

    def shutdown(*_args):
        active = jobs.active_job()
        if active:
            print_info(f"\nStopping job {active}...")
            job = jobs.get(active)
            if job:
                jobs.cancel(job)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if hasattr(signal, "SIGBREAK"):          # Windows Ctrl+Break
        signal.signal(signal.SIGBREAK, shutdown)

    print_success(f"Worker listening on http://{args.host}:{args.port}/api/worker")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
