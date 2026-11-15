# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Where backups go: the PC worker (Local PC) or a private Hugging Face bucket.

A destination stores blobs and snapshot manifests (see format.py); core.py
decides what to send. Settings use the same Field schema as the training
backends, so the dashboard renders both with one form builder.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from remote_training.backends.base import CheckResult, Field
from remote_training.backup.format import (
    SNAPSHOT_ID_RE, blob_path, decode_manifest, encode_manifest, manifest_hashes,
    manifest_path, select_expired, snapshot_time,
)
from remote_training.backup.store import GC_GRACE_SECONDS


class BackupError(RuntimeError):
    """A message for the user."""


class BackupDestination:
    name = ""
    label = ""
    description = ""
    allows_secrets = False      # configs/secrets.yaml and .env are included

    def __init__(self, settings: dict, secrets: dict):
        self.settings = settings or {}
        self.secrets = secrets or {}

    @classmethod
    def config_schema(cls) -> list[Field]:
        return []

    @classmethod
    def describe(cls) -> dict:
        return {"name": cls.name, "label": cls.label, "description": cls.description,
                "fields": [f.to_dict() for f in cls.config_schema()]}

    def setting(self, key: str):
        field = next((f for f in self.config_schema() if f.key == key), None)
        source = self.secrets if field is not None and field.secret else self.settings
        value = source.get(key)
        if value in (None, "") and field is not None:
            return field.default
        return value

    # Interface used by core.run_backup and the dashboard
    def test(self) -> CheckResult: raise NotImplementedError
    def list_snapshots(self) -> list[dict]: raise NotImplementedError
    def get_manifest(self, snapshot_id: str) -> dict: raise NotImplementedError
    def missing(self, hashes: list[str]) -> list[str]: raise NotImplementedError
    def put_blob(self, sha: str, content: bytes | Path) -> None: raise NotImplementedError
    def flush(self) -> None: pass
    def put_manifest(self, manifest: dict) -> None: raise NotImplementedError
    def prune(self, keep_days: float) -> dict: raise NotImplementedError


# ═════════════════════════════════════════════════════════════════
# Local PC: the worker's backup routes (remote_training/worker.py)
# ═════════════════════════════════════════════════════════════════

class LocalPCDestination(BackupDestination):
    name = "local_pc"
    label = "Local PC"
    description = ("Send backups to the PC running the training worker. They are kept "
                   "in its backup folder (data/backups in the project by default). "
                   "Includes configs/secrets.yaml.")
    allows_secrets = True

    @classmethod
    def config_schema(cls) -> list[Field]:
        return [
            Field("host", "PC address", placeholder="(same as the Training tab)",
                  help="Blank uses the Local PC address from the Training tab."),
            Field("port", "Port", type="number", placeholder="(same as the Training tab)",
                  help="Blank uses the Training tab's port."),
            Field("token", "Worker token", type="password", secret=True,
                  help="Blank uses the Training tab's worker token."),
        ]

    def __init__(self, settings: dict, secrets: dict):
        super().__init__(settings, secrets)
        self._session = None

    def _url(self, path: str) -> str:
        host = str(self.setting("host") or "").strip()
        if not host:
            raise BackupError("Set the PC address (here or on the Training tab).")
        if host.startswith(("http://", "https://")):
            base = host.rstrip("/")
        else:
            base = f"http://{host}:{int(self.setting('port') or 5004)}"
        return f"{base}/api/worker/backup{path}"

    def _request(self, method: str, path: str, *, timeout: float = 60, **kwargs):
        import requests
        if self._session is None:
            self._session = requests.Session()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self.setting('token') or ''}"
        try:
            resp = self._session.request(method, self._url(path), headers=headers,
                                         timeout=timeout, **kwargs)
        except requests.ConnectionError:
            raise BackupError("Can't reach the PC worker. Is it running (scripts/worker.sh), "
                              "and is the address right?") from None
        except requests.Timeout:
            raise BackupError("The PC worker didn't answer in time.") from None
        if resp.status_code == 401:
            raise BackupError("The PC worker rejected the token.")
        if resp.status_code == 404:
            raise BackupError("The PC worker has no backup support. Update its code and "
                              "restart it (scripts/worker.sh rebuild).")
        if resp.status_code >= 400:
            try:
                msg = resp.json().get("message") or resp.text
            except ValueError:
                msg = resp.text
            raise BackupError(f"PC worker: {msg}")
        return resp

    def test(self) -> CheckResult:
        try:
            info = self._request("GET", "/snapshots", timeout=15).json()
        except BackupError as exc:
            return CheckResult(False, str(exc))
        stats = info.get("stats") or {}
        return CheckResult(True, f"Connected. {stats.get('snapshots', 0)} snapshot(s), "
                                 f"{stats.get('stored_bytes', 0) / 1e6:.0f} MB stored on the PC.",
                           {"stats": stats})

    def list_snapshots(self) -> list[dict]:
        snaps = self._request("GET", "/snapshots").json().get("snapshots", [])
        return sorted(snaps, key=lambda s: s["id"])

    def get_manifest(self, snapshot_id: str) -> dict:
        return decode_manifest(self._request("GET", f"/snapshots/{snapshot_id}").content)

    def missing(self, hashes: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(hashes), 20000):
            out += self._request("POST", "/blobs/missing",
                                 json={"hashes": hashes[i:i + 20000]}).json()["missing"]
        return out

    def put_blob(self, sha: str, content: bytes | Path) -> None:
        if isinstance(content, Path):
            with open(content, "rb") as f:
                self._request("PUT", f"/blobs/{sha}", data=f, timeout=3600)
        else:
            self._request("PUT", f"/blobs/{sha}", data=content, timeout=600)

    def put_manifest(self, manifest: dict) -> None:
        self._request("PUT", f"/snapshots/{manifest['id']}", data=encode_manifest(manifest),
                      headers={"Content-Type": "application/gzip"})

    def prune(self, keep_days: float) -> dict:
        return self._request("POST", "/prune", json={"keep_days": keep_days},
                             timeout=600).json()


# ═════════════════════════════════════════════════════════════════
# Hugging Face: a private Storage Bucket
# ═════════════════════════════════════════════════════════════════

_HF_BATCH_FILES = 500
_HF_BATCH_BYTES = 200 * 1024 * 1024


class HFDestination(BackupDestination):
    name = "hf"
    label = "Hugging Face"
    description = ("Send backups to a private Hugging Face Storage Bucket. Private "
                   "storage is free up to your account's quota (100 GB on a free "
                   "account). Tokens and secrets are never uploaded.")
    allows_secrets = False

    @classmethod
    def config_schema(cls) -> list[Field]:
        return [
            Field("token", "HF token", type="password", secret=True,
                  help="Needs write access. Blank uses the token from the Training "
                       "tab's Hugging Face settings."),
            Field("namespace", "Namespace", placeholder="(your username)",
                  help="User or organization that owns the bucket. Blank uses the "
                       "token's user."),
            Field("bucket", "Bucket", placeholder="(auto: <namespace>/cat-backups)",
                  help="Private bucket for the backups. Created if missing."),
        ]

    def __init__(self, settings: dict, secrets: dict):
        super().__init__(settings, secrets)
        self._api_obj = None
        self._bucket_id = None
        self._existing: set[str] | None = None     # blob hashes in the bucket
        self._pending: list[tuple[bytes | str, str]] = []
        self._pending_bytes = 0

    def _api(self):
        if self._api_obj is None:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                raise BackupError("huggingface_hub is not installed on the Pi. "
                                  "Run: pip install -r requirements-pi.txt") from None
            token = self.setting("token")
            if not token:
                raise BackupError("Set a Hugging Face token (here or on the Training tab).")
            self._api_obj = HfApi(token=token)
        return self._api_obj

    def _bucket(self) -> str:
        if self._bucket_id is None:
            api = self._api()
            ns = (self.setting("namespace") or "").strip()
            if not ns:
                try:
                    ns = api.whoami()["name"]
                except Exception as exc:
                    raise BackupError(f"Hugging Face rejected the token: {exc}") from exc
            bucket = (self.setting("bucket") or "").strip() or f"{ns}/cat-backups"
            try:
                api.create_bucket(bucket, private=True, exist_ok=True)
            except Exception as exc:
                raise BackupError(f"Could not create or open bucket {bucket}: {exc}") from exc
            self._bucket_id = bucket
        return self._bucket_id

    def _files(self, prefix: str) -> list:
        return [f for f in self._api().list_bucket_tree(self._bucket(), prefix=prefix,
                                                         recursive=True)
                if getattr(f, "type", "file") == "file"]

    def _batch(self, *, add=None, delete=None) -> None:
        try:
            self._api().batch_bucket_files(self._bucket(), add=add, delete=delete)
        except Exception as exc:
            raise BackupError(f"Hugging Face upload failed: {exc}") from exc

    def test(self) -> CheckResult:
        try:
            bucket = self._bucket()
            n = len(self._snapshot_ids())
        except BackupError as exc:
            return CheckResult(False, str(exc))
        except Exception as exc:
            return CheckResult(False, f"Hugging Face: {exc}")
        return CheckResult(True, f"Connected to private bucket {bucket}. {n} snapshot(s).")

    def _snapshot_ids(self) -> list[str]:
        ids = []
        for f in self._files("snapshots/"):
            name = f.path.rsplit("/", 1)[-1]
            if name.endswith(".json.gz") and SNAPSHOT_ID_RE.match(name[:-8]):
                ids.append(name[:-8])
        return sorted(ids)

    def list_snapshots(self) -> list[dict]:
        return [{"id": i, "created": snapshot_time(i).isoformat()} for i in self._snapshot_ids()]

    def get_manifest(self, snapshot_id: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="backup-") as tmp:
            local = Path(tmp) / "manifest.json.gz"
            self._api().download_bucket_files(
                self._bucket(), [(manifest_path(snapshot_id), str(local))],
                raise_on_missing_files=True)
            return decode_manifest(local.read_bytes())

    def missing(self, hashes: list[str]) -> list[str]:
        if self._existing is None:
            self._existing = {f.path.rsplit("/", 1)[-1] for f in self._files("blobs/")}
        return [h for h in hashes if h not in self._existing]

    def put_blob(self, sha: str, content: bytes | Path) -> None:
        source = str(content) if isinstance(content, Path) else content
        size = content.stat().st_size if isinstance(content, Path) else len(content)
        self._pending.append((source, blob_path(sha)))
        self._pending_bytes += size
        if len(self._pending) >= _HF_BATCH_FILES or self._pending_bytes >= _HF_BATCH_BYTES:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self._batch(add=self._pending)
            if self._existing is not None:
                self._existing |= {dest.rsplit("/", 1)[-1] for _, dest in self._pending}
            self._pending, self._pending_bytes = [], 0

    def put_manifest(self, manifest: dict) -> None:
        self.flush()
        missing = self.missing(sorted(manifest_hashes(manifest)))
        if missing:
            raise BackupError(f"{len(missing)} file(s) of the snapshot were not uploaded")
        self._batch(add=[(encode_manifest(manifest), manifest_path(manifest["id"]))])

    def prune(self, keep_days: float) -> dict:
        ids = self._snapshot_ids()
        expired = select_expired(ids, keep_days)
        referenced: set[str] = set()
        for sid in ids:
            if sid not in expired:
                referenced |= manifest_hashes(self.get_manifest(sid))
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=GC_GRACE_SECONDS)
        unused, freed = [], 0
        for f in self._files("blobs/"):
            uploaded = getattr(f, "uploaded_at", None)
            if (f.path.rsplit("/", 1)[-1] not in referenced
                    and uploaded is not None and uploaded < cutoff):
                unused.append(f.path)
                freed += f.size or 0
        doomed = [manifest_path(s) for s in expired] + unused
        for i in range(0, len(doomed), 1000):
            self._batch(delete=doomed[i:i + 1000])
        if self._existing is not None:
            self._existing -= {p.rsplit("/", 1)[-1] for p in unused}
        return {"deleted_snapshots": expired, "deleted_blobs": len(unused), "freed_bytes": freed}


DESTINATIONS: dict[str, type[BackupDestination]] = {
    LocalPCDestination.name: LocalPCDestination,
    HFDestination.name: HFDestination,
}
