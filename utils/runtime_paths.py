# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Short-lived files that are rewritten often, kept in RAM.

/dev/shm is a RAM disk on Linux, so files written there never reach the SD
card. That matters for files rewritten many times a second, such as the live
preview frame. Where /dev/shm doesn't exist (not Linux), the system temp
folder is used instead.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

_SHM = Path("/dev/shm")
RUNTIME_DIR = (_SHM if _SHM.is_dir() else Path(tempfile.gettempdir())) / "cat_detection"

PREVIEW_JPG = RUNTIME_DIR / "preview.jpg"
PREVIEW_DETECTIONS_JSON = RUNTIME_DIR / "preview_detections.json"
CLASSIFIER_STATUS_JSON = RUNTIME_DIR / "classifier_status.json"


def runtime_dir() -> Path:
    """RUNTIME_DIR, created if needed."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR


def ram_dir_with_room(nbytes: int, margin: float = 1.5) -> Path | None:
    """RUNTIME_DIR if it is a RAM disk with at least nbytes * margin free,
    else None (the caller then uses a folder on disk)."""
    if not _SHM.is_dir():
        return None
    try:
        if shutil.disk_usage(_SHM).free < nbytes * margin:
            return None
    except OSError:
        return None
    return runtime_dir()
