# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Small store for credentials (API tokens) kept out of project.yaml.

Secrets live in configs/secrets.yaml (gitignored), readable only by the
owner, grouped by section: ``{"hf_jobs": {"token": "..."}}``.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml


class SecretsStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        if not self.path.is_file():
            return {}
        with open(self.path) as f:
            return yaml.safe_load(f) or {}

    def get_section(self, section: str) -> dict:
        with self._lock:
            return dict(self._read().get(section) or {})

    def is_set(self, section: str) -> dict[str, bool]:
        """Which keys of a section have a non-empty value (values not exposed)."""
        return {k: bool(v) for k, v in self.get_section(section).items()}

    def update_section(self, section: str, values: dict) -> None:
        """Set keys in a section. A value of None or "" deletes the key."""
        with self._lock:
            data = self._read()
            sec = dict(data.get(section) or {})
            for k, v in values.items():
                if v in (None, ""):
                    sec.pop(k, None)
                else:
                    sec[k] = str(v)
            data[section] = sec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            os.replace(tmp, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
