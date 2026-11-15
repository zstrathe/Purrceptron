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
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask

from remote_training.worker import (
    NO_EXECUTORCH, create_worker_blueprint, default_backup_dir, default_jobs_dir,
    load_worker_token, prune_jobs,
)
from utils import PROJECT_ROOT
from utils.display import print_error, print_info, print_success, print_warning


def main() -> None:
    parser = argparse.ArgumentParser(description="Training worker for the Pi dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5004, help="Port (default: 5004)")
    parser.add_argument("--jobs-dir", type=Path, default=None,
                        help="Where job folders go (default: data/worker_jobs, or "
                             "~/.local/share/cat_detection/worker_jobs for a project "
                             "on a Windows drive under WSL)")
    parser.add_argument("--backup-dir", type=Path, default=None,
                        help="Where the Pi's backups are kept (default: data/backups)")
    parser.add_argument("--keep-jobs", type=int, default=10,
                        help="Keep this many recent job folders, delete older ones at "
                             "startup (0 = keep all; default 10)")
    args = parser.parse_args()
    jobs_dir = (args.jobs_dir or default_jobs_dir(PROJECT_ROOT)).expanduser().resolve()
    backup_dir = (args.backup_dir or default_backup_dir(PROJECT_ROOT)).expanduser().resolve()

    if not load_worker_token(PROJECT_ROOT):
        print_error("WORKER_TOKEN is not set. Add a line WORKER_TOKEN=<any long random "
                    "string> to .env,\nand enter the same token on the Pi dashboard's "
                    "Training tab.")
        sys.exit(1)

    app = Flask(__name__)
    worker_bp, jobs = create_worker_blueprint(PROJECT_ROOT, sys.executable, jobs_dir,
                                              backup_dir)
    app.register_blueprint(worker_bp)

    health = jobs.health()
    if not health.get("ok"):
        print_error(health.get("error", "torch is not usable in this Python"))
        sys.exit(1)
    if not health.get("int8_ok"):
        print_error(NO_EXECUTORCH + f" ({health.get('int8_error')})")
        sys.exit(1)
    gpu = health.get("gpu")
    print_info(f"Python: {sys.executable}")
    print_info(f"Jobs folder: {jobs_dir}")
    print_info(f"Backup folder: {backup_dir}")
    removed = prune_jobs(jobs_dir, args.keep_jobs)
    if removed:
        print_info(f"Deleted {len(removed)} old job folder(s), keeping the {args.keep_jobs} newest")
    print_info(f"torch {health.get('torch_version')}, "
               f"ExecuTorch {health.get('executorch_version')}")
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

    # The Pi checks job status every few seconds; don't print a line per request.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print_success(f"Worker listening on http://{args.host}:{args.port}/api/worker")
    print_info("Job output from the Pi's jobs is shown here as it runs.")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
