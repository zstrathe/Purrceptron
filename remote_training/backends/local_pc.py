# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Train on your own PC: the Pi pushes the job to the PC's worker
(train_worker.py, API in remote_training/worker.py) over HTTP on the LAN."""
from __future__ import annotations

from pathlib import Path

from remote_training.backends.base import (
    CANCELLED, FAILED, QUEUED, RUNNING, SUCCEEDED,
    BackendError, CheckResult, Field, JobStatus, TrainingBackend,
)

_STAGE_MAP = {
    "queued": QUEUED,
    "running": RUNNING,
    "succeeded": SUCCEEDED,
    "failed": FAILED,
    "cancelled": CANCELLED,
}


class LocalPCBackend(TrainingBackend):
    name = "local_pc"
    label = "Local PC"
    description = ("Send the job to the PC running train_worker.py. "
                   "Free, uses your own GPU, and images stay on your network.")
    poll_interval = 3.0
    supports_instant_detect = True

    @classmethod
    def config_schema(cls) -> list[Field]:
        return [
            Field("host", "PC address", required=True, placeholder="192.168.1.20",
                  help="LAN IP or hostname of the PC running train_worker.py."),
            Field("port", "Port", type="number", default=5004,
                  help="train_worker.py port (its --port, default 5004)."),
            Field("token", "Worker token", type="password", secret=True, required=True,
                  help="Must match WORKER_TOKEN in the PC's .env."),
        ]

    # ── HTTP helpers ─────────────────────────────────────────────
    def _url(self, path: str) -> str:
        host = str(self.setting("host")).strip()
        if host.startswith(("http://", "https://")):
            base = host.rstrip("/")
        else:
            base = f"http://{host}:{int(self.setting('port'))}"
        return f"{base}/api/worker{path}"

    def _request(self, method: str, path: str, *, timeout: float = 15, **kwargs):
        import requests
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self.setting('token')}"
        try:
            resp = requests.request(method, self._url(path), headers=headers,
                                    timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            raise BackendError(f"Cannot reach the PC worker at {self._url('')}: {exc}") from exc
        if resp.status_code == 401:
            raise BackendError("The PC rejected the worker token. "
                               "Check WORKER_TOKEN in the PC's .env.")
        if resp.status_code == 404 and path == "/health":
            raise BackendError("No worker API at that address. Is train_worker.py "
                               "running there?")
        if not resp.ok:
            try:
                msg = resp.json().get("message", resp.text)
            except ValueError:
                msg = resp.text
            raise BackendError(f"PC worker error ({resp.status_code}): {msg}")
        return resp

    # ── Interface ────────────────────────────────────────────────
    def test_connection(self) -> CheckResult:
        missing = self.missing_settings()
        if missing:
            return CheckResult(False, f"Missing: {', '.join(missing)}")
        try:
            info = self._request("GET", "/health", timeout=5).json()
        except BackendError as exc:
            return CheckResult(False, str(exc))
        if "executorch_version" not in info:
            return CheckResult(False, "The PC is running an older train_worker.py. Update the "
                                      "code on the PC and restart the worker.", info)
        if not info.get("int8_ok"):
            return CheckResult(False, "The PC's training environment can't export ExecuTorch "
                                      f"models ({info.get('int8_error') or 'executorch missing'}). "
                                      "On the PC run: pip install -r requirements-training.txt",
                               info)
        gpu = info.get("gpu") or "no GPU (CPU training, slow)"
        pc_et = str(info.get("executorch_version") or "").split("+")[0]
        msg = f"Connected: torch {info.get('torch_version')}, ExecuTorch {pc_et}, {gpu}"
        if info.get("busy"):
            msg += f". Busy right now: {info['busy']}"
        pi_et = str(self.context.get("pi_executorch_version") or "").split("+")[0]
        if not pi_et:
            msg += (". Warning: ExecuTorch is not installed on the Pi, so the Pi cannot "
                    "check or run a trained model.")
        elif pc_et != pi_et:
            msg += (f". Warning: the PC has ExecuTorch {pc_et} but the Pi has {pi_et}; "
                    "the Pi may not be able to load the model.")
        return CheckResult(True, msg, info)

    def submit(self, job_id: str, bundle_tar: Path) -> dict:
        with open(bundle_tar, "rb") as f:
            self._request("POST", "/jobs", params={"job_id": job_id}, data=f,
                          headers={"Content-Type": "application/gzip"}, timeout=600)
        return {"job_id": job_id, "log_offset": 0}

    def poll(self, handle: dict) -> JobStatus:
        info = self._request("GET", f"/jobs/{handle['job_id']}",
                             params={"since": handle.get("log_offset", 0)}).json()
        lines = info.get("lines", [])
        handle["log_offset"] = info.get("next", handle.get("log_offset", 0) + len(lines))
        return JobStatus(stage=_STAGE_MAP.get(info.get("stage"), RUNNING),
                         message=info.get("message", ""), new_log_lines=lines)

    def cancel(self, handle: dict) -> None:
        self._request("POST", f"/jobs/{handle['job_id']}/cancel")

    def fetch(self, handle: dict, dest_tar: Path) -> None:
        resp = self._request("GET", f"/jobs/{handle['job_id']}/results",
                             stream=True, timeout=120)
        with open(dest_tar, "wb") as f:
            for chunk in resp.iter_content(1 << 16):
                f.write(chunk)

    def detect_image(self, image_path: Path) -> dict | None:
        with open(image_path, "rb") as f:
            # The first call loads the detector on the PC, which takes a while.
            info = self._request("POST", "/detect", data=f,
                                 headers={"Content-Type": "image/jpeg"}, timeout=120).json()
        return info.get("bbox")
