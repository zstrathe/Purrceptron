# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Run one backup with the settings saved on the Pi dashboard's Backup tab.

    python -m remote_training.backup run [--config configs/project.yaml]

The dashboard starts this at the scheduled time and for "Back up now"; it can
also be run by hand. The last line, BACKUP_RESULT {...}, is a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from remote_training.backup.core import run_backup
from remote_training.backup.destinations import BackupError
from remote_training.backup.manager import RESULT_PREFIX, backup_settings, make_destination
from utils.config import PROJECT_ROOT, load_config
from utils.secrets import SecretsStore


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m remote_training.backup")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Back up now, to the configured destination")
    run.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "project.yaml")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)          # includes runtime.yaml (dashboard settings)
    secrets = SecretsStore(config_path.parent / "secrets.yaml")
    try:
        dest = make_destination(config, secrets)
        summary = run_backup(dest, project_root=PROJECT_ROOT, config=config,
                             config_path=config_path,
                             keep_days=float(backup_settings(config)["keep_days"]),
                             log=lambda line: print(line, flush=True))
    except BackupError as exc:
        print(f"Backup failed: {exc}", flush=True)
        return 1
    print(RESULT_PREFIX + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
