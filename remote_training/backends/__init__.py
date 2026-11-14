# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Registry of training backends. Add new backends to BACKENDS."""
from __future__ import annotations

from remote_training.backends.base import (
    BackendError, CheckResult, Field, JobStatus, TrainingBackend,
)
from remote_training.backends.hf_jobs import HFJobsBackend
from remote_training.backends.local_pc import LocalPCBackend

BACKENDS: dict[str, type[TrainingBackend]] = {
    LocalPCBackend.name: LocalPCBackend,
    HFJobsBackend.name: HFJobsBackend,
}


def get_backend_class(name: str) -> type[TrainingBackend]:
    try:
        return BACKENDS[name]
    except KeyError:
        raise BackendError(f"Unknown training backend: {name!r}") from None


__all__ = ["BACKENDS", "get_backend_class", "BackendError", "CheckResult",
           "Field", "JobStatus", "TrainingBackend"]
