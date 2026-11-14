# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Interface every training backend implements.

A backend receives a finished job bundle (a .tar.gz built on the Pi), runs
``code/train/run_job.py`` from it somewhere with more compute, and hands back
the results tarball that run_job.py produced. Everything else (packaging,
validation, deploy) is shared and lives in remote_training/manager.py.

To add a backend: subclass TrainingBackend in a new module in this package
and add it to BACKENDS in __init__.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Remote stages a backend reports from poll(). The manager adds the local
# stages (packaging, downloading, validating, ...) around these.
QUEUED = "queued"
RUNNING = "training"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
REMOTE_TERMINAL = {SUCCEEDED, FAILED, CANCELLED}


@dataclass
class Field:
    """One setting shown in the dashboard config form."""
    key: str
    label: str
    type: str = "text"            # text | number | select | password
    default: Any = None
    help: str = ""
    secret: bool = False          # stored in configs/secrets.yaml, never echoed
    required: bool = False
    options: list[str] = field(default_factory=list)  # for type == "select"
    placeholder: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckResult:
    ok: bool
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JobStatus:
    stage: str                    # one of QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED
    message: str = ""
    new_log_lines: list[str] = field(default_factory=list)
    url: str | None = None        # link to the job in the provider's UI, if any


class BackendError(RuntimeError):
    """A backend operation failed with a message meant for the user."""


class TrainingBackend(ABC):
    name: str = ""
    label: str = ""
    description: str = ""
    poll_interval: float = 5.0    # seconds between poll() calls
    # True if detect_image() gives a bounding box within seconds (the Annotate
    # tab's per-image Auto-detect button). Otherwise only batch detect jobs.
    supports_instant_detect: bool = False

    def __init__(self, settings: dict, secrets: dict, context: dict | None = None):
        """settings: non-secret values; secrets: secret values; context:
        facts about the Pi (e.g. ``pi_torch_version``) some backends need."""
        self.settings = settings or {}
        self.secrets = secrets or {}
        self.context = context or {}

    # ── Config ───────────────────────────────────────────────────
    @classmethod
    @abstractmethod
    def config_schema(cls) -> list[Field]:
        """Settings this backend needs, used to render the config form."""

    @classmethod
    def describe(cls) -> dict:
        return {
            "name": cls.name,
            "label": cls.label,
            "description": cls.description,
            "fields": [f.to_dict() for f in cls.config_schema()],
        }

    def setting(self, key: str) -> Any:
        """Value of a setting or secret, falling back to the schema default."""
        for f in self.config_schema():
            if f.key == key:
                src = self.secrets if f.secret else self.settings
                val = src.get(key)
                return f.default if val in (None, "") else val
        raise KeyError(key)

    def missing_settings(self) -> list[str]:
        return [f.label for f in self.config_schema()
                if f.required and self.setting(f.key) in (None, "")]

    # ── Job lifecycle ────────────────────────────────────────────
    @abstractmethod
    def test_connection(self) -> CheckResult:
        """Cheap check that settings and credentials work."""

    @abstractmethod
    def submit(self, job_id: str, bundle_tar: Path) -> dict:
        """Upload the bundle and start the job. Returns a JSON-serialisable
        handle that is persisted and passed back to the other methods, so a
        dashboard restart can resume polling."""

    @abstractmethod
    def poll(self, handle: dict) -> JobStatus:
        """Current remote stage plus log lines not returned before."""

    @abstractmethod
    def cancel(self, handle: dict) -> None:
        """Stop the remote job (best effort)."""

    @abstractmethod
    def fetch(self, handle: dict, dest_tar: Path) -> None:
        """Download the results tarball written by run_job.py to dest_tar."""

    def detect_image(self, image_path: Path) -> dict | None:
        """Best cat bounding box {cx, cy, w, h, confidence} for one image, or
        None. Only for backends with supports_instant_detect."""
        raise BackendError(f"{self.label} has no instant auto-detect")

    def cleanup(self, handle: dict) -> None:
        """Delete remote copies of the job's data. Optional."""
