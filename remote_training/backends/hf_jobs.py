# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Train on Hugging Face Jobs (pay-as-you-go cloud GPUs).

Flow: upload the bundle to a private HF Storage Bucket, run a Job that
mounts the job's bucket folder, untars the bundle, and runs run_job.py,
which writes results.tar.gz back to the bucket. The Pi downloads it, then
deletes both files from the bucket.

Requires a Hugging Face account with prepaid credits and a token with write
access (to create the bucket and run Jobs).
"""
from __future__ import annotations

import re
from pathlib import Path

from remote_training.backends.base import (
    CANCELLED, FAILED, QUEUED, RUNNING, SUCCEEDED,
    BackendError, CheckResult, Field, JobStatus, TrainingBackend,
)

_STAGE_MAP = {
    "SCHEDULING": QUEUED,
    "RUNNING": RUNNING,
    "COMPLETED": SUCCEEDED,
    "ERROR": FAILED,
    "CANCELED": CANCELLED,
    "DELETED": CANCELLED,
}

_FLAVORS = ["t4-small", "t4-medium", "l4x1", "a10g-small", "a10g-large", "cpu-upgrade"]

# Extra packages the training scripts need on top of the pytorch image.
# executorch (pinned to the Pi's runtime version) brings torchao, which does
# the quantization; it keeps the image's torch when that is new enough.
_PIP_PACKAGES = "albumentations scikit-learn opencv-python-headless pyyaml"
# Used only when the Pi has no ExecuTorch yet (cross-validation, auto-detect).
DEFAULT_EXECUTORCH = "1.1.0"

_DOCKER_TAGS_URL = "https://hub.docker.com/v2/repositories/pytorch/pytorch/tags"


def open_private_bucket(api, bucket: str) -> None:
    """Create the bucket as private if it doesn't exist, then make sure it IS
    private. create_bucket(exist_ok=True) returns an existing bucket as it is,
    so a bucket that was made public on the website would otherwise still get
    uploads. Raises ValueError if the bucket is not private."""
    api.create_bucket(bucket, private=True, exist_ok=True)
    if not api.bucket_info(bucket).private:
        raise ValueError(
            f"Bucket {bucket} is public. Nothing was uploaded. Make it private in its "
            f"settings on huggingface.co, or choose another bucket name.")


def job_script(mount: str = "/bucket", executorch_version: str = "") -> str:
    """Shell command the Job runs. The job's bucket folder is mounted at
    ``mount`` and holds bundle.tar.gz; results.tar.gz is written back there."""
    executorch = f"executorch=={executorch_version or DEFAULT_EXECUTORCH}"
    return (
        "set -e; mkdir -p /work; "
        f"tar xzf {mount}/bundle.tar.gz -C /work; "
        # Newer pytorch images mark the system Python as externally managed
        # (PEP 668); overriding that is fine in a throwaway job container.
        "PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install -q --no-cache-dir "
        f"--root-user-action=ignore {executorch} {_PIP_PACKAGES}; "
        "python /work/code/train/run_job.py --work /work "
        f"--output-tar {mount}/results.tar.gz"
    )


class HFJobsBackend(TrainingBackend):
    name = "hf_jobs"
    label = "Hugging Face Jobs"
    description = ("Cloud GPU billed per minute from prepaid HF credits "
                   "(a full training is about $0.15). Cropped images are uploaded to a private "
                   "HF bucket and deleted after the model is downloaded.")
    poll_interval = 15.0

    @classmethod
    def config_schema(cls) -> list[Field]:
        return [
            Field("token", "HF token", type="password", secret=True, required=True,
                  help="huggingface.co/settings/tokens. Needs write access "
                       "(creates a bucket and runs Jobs)."),
            Field("namespace", "Namespace", placeholder="(your username)",
                  help="User or organization that runs and pays for the job. "
                       "Blank uses the token's user."),
            Field("flavor", "Hardware", type="select", default="l4x1", options=_FLAVORS,
                  help="l4x1 ($0.80/h) trains a full model in about 12 min for about "
                       "$0.16. CPUs matter more than the GPU here: t4-small costs "
                       "the same per model but takes twice as long, and a10g-small "
                       "is slow and costs twice as much (both have 4 CPUs)."),
            Field("timeout", "Timeout", default="2h",
                  help="The job is stopped after this long, e.g. 90m or 2h."),
            Field("image", "Docker image", placeholder="(auto: matches the Pi's torch)",
                  help="Blank picks a pytorch/pytorch CUDA runtime image with the "
                       "Pi's torch version; the job installs the Pi's ExecuTorch "
                       "version into it, so the Pi can load the model."),
            Field("bucket", "Bucket", placeholder="(auto: <namespace>/cat-training)",
                  help="Private HF Storage Bucket used to move files. Created if missing."),
        ]

    # ── Helpers ──────────────────────────────────────────────────
    def _api(self):
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import disable_progress_bars
        except ImportError:
            raise BackendError("huggingface_hub is not installed on the Pi. "
                               "Run: pip install -r requirements-pi.txt") from None
        disable_progress_bars()     # they would only fill the dashboard's log file
        return HfApi(token=self.setting("token"))

    def _namespace(self, api) -> str:
        ns = (self.setting("namespace") or "").strip()
        if ns:
            return ns
        try:
            return api.whoami()["name"]
        except Exception as exc:
            raise BackendError(f"Hugging Face rejected the token: {exc}") from exc

    def _pi_executorch(self) -> str:
        return str(self.context.get("pi_executorch_version") or "").split("+")[0]

    def _bucket(self, ns: str) -> str:
        return (self.setting("bucket") or "").strip() or f"{ns}/cat-training"

    def resolve_image(self) -> str:
        """Configured image, or a pytorch/pytorch CUDA runtime tag matching the
        Pi's torch version (looked up on Docker Hub)."""
        image = (self.setting("image") or "").strip()
        if image:
            return image
        pi_ver = str(self.context.get("pi_torch_version", "")).split("+")[0]
        if not pi_ver:
            raise BackendError("Could not read the Pi's torch version; set the Docker image.")
        import requests
        try:
            resp = requests.get(_DOCKER_TAGS_URL,
                                params={"name": f"{pi_ver}-cuda", "page_size": 100},
                                timeout=10)
            resp.raise_for_status()
            tags = [t["name"] for t in resp.json().get("results", [])]
        except Exception as exc:
            raise BackendError(f"Could not look up pytorch images on Docker Hub ({exc}). "
                               "Set the Docker image field.") from exc
        pattern = re.compile(rf"^{re.escape(pi_ver)}-cuda(\d+)\.(\d+)-cudnn\d+-runtime$")
        matches = [(tuple(map(int, m.groups())), t) for t in tags if (m := pattern.match(t))]
        if not matches:
            raise BackendError(f"No pytorch/pytorch runtime image for torch {pi_ver}. "
                               "Set the Docker image field to one close to it.")
        # Oldest CUDA build: a CUDA runtime works on newer drivers but not
        # older ones, so this is the most likely to run on HF's GPU hosts.
        return f"pytorch/pytorch:{min(matches)[1]}"

    # ── Interface ────────────────────────────────────────────────
    def test_connection(self) -> CheckResult:
        missing = self.missing_settings()
        if missing:
            return CheckResult(False, f"Missing: {', '.join(missing)}")
        try:
            api = self._api()
            user = api.whoami()["name"]
            ns = self._namespace(api)
            flavor = self.setting("flavor")
            hw = {h.name: h for h in api.list_jobs_hardware()}
            if flavor not in hw:
                return CheckResult(False, f"Unknown hardware '{flavor}'. "
                                          f"Available: {', '.join(sorted(hw))}")
            h = hw[flavor]
            # HF prices per minute (t4-small: $0.006667), which rounds to a
            # misleading $0.01; show the hourly price like HF's pricing page.
            if not h.unit_cost_usd:
                price = "price n/a"
            elif h.unit_label == "minute":
                price = f"${h.unit_cost_usd * 60:.2f}/hour"
            else:
                price = f"${h.unit_cost_usd:.4f}/{h.unit_label}"
            image = self.resolve_image()
        except BackendError as exc:
            return CheckResult(False, str(exc))
        except Exception as exc:
            return CheckResult(False, f"Hugging Face check failed: {exc}")
        return CheckResult(
            True,
            f"Signed in as {user}; jobs run under {ns} on {h.pretty_name} ({price}); "
            f"image {image}, ExecuTorch {self._pi_executorch() or DEFAULT_EXECUTORCH}. "
            "Jobs need prepaid credits on that account.",
            {"user": user, "namespace": ns, "image": image},
        )

    def submit(self, job_id: str, bundle_tar: Path) -> dict:
        from huggingface_hub import Volume
        api = self._api()
        ns = self._namespace(api)
        bucket = self._bucket(ns)
        prefix = f"jobs/{job_id}"
        image = self.resolve_image()
        try:
            open_private_bucket(api, bucket)
            api.batch_bucket_files(bucket, add=[(str(bundle_tar), f"{prefix}/bundle.tar.gz")])
        except Exception as exc:
            raise BackendError(f"Upload to bucket {bucket} failed: {exc}") from exc

        script = job_script(executorch_version=self._pi_executorch())
        try:
            job = api.run_job(
                image=image,
                command=["bash", "-c", script],
                flavor=self.setting("flavor"),
                timeout=self.setting("timeout"),
                volumes=[Volume(type="bucket", source=bucket, mount_path="/bucket",
                                path=prefix, read_only=False)],
                name=f"cat-classifier-{job_id}",
                labels={"app": "cat-detection"},
                namespace=ns,
            )
        except Exception as exc:
            self.cleanup({"bucket": bucket, "prefix": prefix})
            raise BackendError(f"Starting the Hugging Face job failed: {exc}") from exc
        return {"hf_job_id": job.id, "namespace": ns, "bucket": bucket,
                "prefix": prefix, "url": job.url, "image": image, "log_count": 0}

    def poll(self, handle: dict) -> JobStatus:
        api = self._api()
        info = api.inspect_job(job_id=handle["hf_job_id"], namespace=handle["namespace"])
        raw_stage = str(getattr(info.status.stage, "value", info.status.stage))
        stage = _STAGE_MAP.get(raw_stage, RUNNING)
        new_lines: list[str] = []
        if stage != QUEUED:
            try:
                logs = list(api.fetch_job_logs(job_id=handle["hf_job_id"],
                                               namespace=handle["namespace"]))
                new_lines = logs[handle.get("log_count", 0):]
                handle["log_count"] = len(logs)
            except Exception:
                pass  # logs are best effort; the stage is what matters
        return JobStatus(stage=stage, message=info.status.message or raw_stage.title(),
                         new_log_lines=new_lines, url=handle.get("url"))

    def cancel(self, handle: dict) -> None:
        self._api().cancel_job(job_id=handle["hf_job_id"], namespace=handle["namespace"])

    def fetch(self, handle: dict, dest_tar: Path) -> None:
        try:
            self._api().download_bucket_files(
                handle["bucket"], [(f"{handle['prefix']}/results.tar.gz", str(dest_tar))],
                raise_on_missing_files=True,
            )
        except Exception as exc:
            raise BackendError(f"Downloading results from {handle['bucket']} failed: {exc}") from exc

    def cleanup(self, handle: dict) -> None:
        api = self._api()
        try:
            paths = [f.path for f in api.list_bucket_tree(handle["bucket"], prefix=handle["prefix"],
                                                         recursive=True)
                     if getattr(f, "type", "file") == "file"]
            if paths:
                api.batch_bucket_files(handle["bucket"], delete=paths)
        except Exception as exc:
            raise BackendError(f"Could not delete job files from {handle['bucket']}: {exc}") from exc
