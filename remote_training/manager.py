# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Training job lifecycle on the Pi.

package (crop + split + tar) -> submit to backend -> poll -> fetch results
-> validate with the Pi's torch -> ready -> user deploys (or rolls back).

One job runs at a time. Each job's state is saved to
data/training_jobs/<job_id>/state.json after every change, so restarting the
dashboard resumes polling a job that is still running remotely.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from remote_training.backends import BACKENDS, BackendError, get_backend_class
from remote_training.backends import base as B
from remote_training.bundle import (
    PackagingError, build_bundle, count_trainable_images, pi_torch_version, safe_extract,
)
from utils.config import resolve_path
from utils.secrets import SecretsStore

STAGES = ["packaging", "uploading", "queued", "training", "downloading", "validating", "ready"]
TERMINAL = {"ready", "failed", "cancelled"}
_REMOTE_STAGES = {"uploading", "queued", "training", "downloading", "validating"}

DEFAULT_MIN_IMAGES = 30
_MAX_POLL_FAILURES = 40     # consecutive poll errors (network blips) before giving up


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class TrainingJobManager:
    def __init__(self, project_root: Path, config_path: Path,
                 config_reader: Callable[[], dict], secrets: SecretsStore,
                 python_exe: str = sys.executable,
                 on_model_installed: Callable[[], str | None] | None = None):
        """on_model_installed: called after deploy/rollback replaces the model
        file; returns a status note for the user (e.g. that detection is
        restarting), or None."""
        self.project_root = project_root
        self.config_path = config_path
        self._read_config = config_reader
        self.secrets = secrets
        self.python_exe = python_exe
        self._on_model_installed = on_model_installed
        self.jobs_dir = project_root / "data" / "training_jobs"
        self.log_dir = project_root / "logs" / "training_jobs"
        self.candidates_dir = project_root / "models" / "candidates"

        self._lock = threading.RLock()
        self._job: dict | None = None
        self._log: collections.deque[str] = collections.deque(maxlen=500)
        self._log_file = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._torch_version = pi_torch_version()

        self._resume_latest()

    # ═════════════════════════════════════════════════════════════
    # Settings and backends
    # ═════════════════════════════════════════════════════════════
    def settings(self) -> dict:
        tb = self._read_config().get("training_backend") or {}
        return {
            "backend": tb.get("backend") or "local_pc",
            "min_images_per_cat": int(tb.get("min_images_per_cat") or DEFAULT_MIN_IMAGES),
            "backends": {name: dict(tb.get(name) or {}) for name in BACKENDS},
        }

    def describe(self) -> dict:
        """Everything the config form needs: schemas, saved values, which
        secrets are set (never their values)."""
        s = self.settings()
        return {
            "backend": s["backend"],
            "min_images_per_cat": s["min_images_per_cat"],
            "pi_torch_version": self._torch_version,
            "backends": [
                {**cls.describe(),
                 "values": s["backends"].get(name, {}),
                 "secrets_set": self.secrets.is_set(name)}
                for name, cls in BACKENDS.items()
            ],
        }

    @staticmethod
    def parse_form(name: str, values: dict) -> tuple[dict, dict]:
        """Split submitted form values into (settings, secrets) using the
        backend's schema: unknown keys are dropped and numbers converted.
        Blank secrets are left out, meaning "keep the stored value"."""
        cls = get_backend_class(name)
        settings, secrets = {}, {}
        for f in cls.config_schema():
            if f.key not in values:
                continue
            val = values[f.key]
            if isinstance(val, str):
                val = val.strip()
            if f.secret:
                if val:
                    secrets[f.key] = val
                continue
            if f.type == "number" and val not in (None, ""):
                try:
                    num = float(val)
                    val = int(num) if num.is_integer() else num
                except (TypeError, ValueError):
                    raise BackendError(f"{f.label} must be a number") from None
            if f.type == "select" and val and f.options and val not in f.options:
                raise BackendError(f"{f.label} must be one of {', '.join(f.options)}")
            settings[f.key] = val
        return settings, secrets

    def make_backend(self, name: str | None = None, settings: dict | None = None,
                     secrets: dict | None = None) -> B.TrainingBackend:
        """Backend from saved settings; settings/secrets override saved values
        (used to test unsaved form input)."""
        s = self.settings()
        name = name or s["backend"]
        cls = get_backend_class(name)
        merged = {**s["backends"].get(name, {}), **(settings or {})}
        merged_secrets = {**self.secrets.get_section(name),
                          **{k: v for k, v in (secrets or {}).items() if v}}
        return cls(merged, merged_secrets, {"pi_torch_version": self._torch_version})

    # ═════════════════════════════════════════════════════════════
    # Job state
    # ═════════════════════════════════════════════════════════════
    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _save(self) -> None:
        if not self._job:
            return
        self._job["updated"] = _now()
        d = self._job_dir(self._job["job_id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "state.json.tmp"
        tmp.write_text(json.dumps(self._job, indent=2))
        os.replace(tmp, d / "state.json")

    def _set_stage(self, stage: str, message: str = "") -> None:
        with self._lock:
            self._job["stage"] = stage
            self._job["message"] = message
            self._job.setdefault("stage_times", {})[stage] = _now()
            self._save()
        self._logline(f"── {stage}{': ' + message if message else ''}")

    def _logline(self, line: str) -> None:
        with self._lock:
            self._log.append(line)
            if self._log_file and not self._log_file.closed:
                try:
                    self._log_file.write(line + "\n")
                    self._log_file.flush()
                except OSError:
                    pass

    def _open_log(self, job_id: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_dir / f"{job_id}.log", "a")

    def _close_log(self) -> None:
        if self._log_file:
            try:
                self._log_file.close()
            except OSError:
                pass
        self._log_file = None

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._job) and self._job["stage"] not in TERMINAL

    def _resume_latest(self) -> None:
        if not self.jobs_dir.is_dir():
            return
        states = sorted(self.jobs_dir.glob("*/state.json"))
        if not states:
            return
        try:
            job = json.loads(states[-1].read_text())
        except (OSError, json.JSONDecodeError):
            return
        self._job = job
        log_path = self.log_dir / f"{job['job_id']}.log"
        if log_path.is_file():
            for line in log_path.read_text(errors="replace").splitlines()[-self._log.maxlen:]:
                self._log.append(line)
        if job["stage"] in TERMINAL:
            return
        if job["stage"] in _REMOTE_STAGES and job.get("handle"):
            self._open_log(job["job_id"])
            self._logline("Dashboard restarted; resuming job")
            self._start_thread(resume=True)
        else:
            job["stage"] = "failed"
            job["message"] = "Interrupted by a dashboard restart while packaging; start again."
            self._save()

    # ═════════════════════════════════════════════════════════════
    # Start / cancel
    # ═════════════════════════════════════════════════════════════
    def image_counts(self) -> dict:
        cfg = self._read_config()
        raw_dir = resolve_path(cfg.get("collection", {}).get("output_dir", "data/raw"),
                               self.project_root)
        return count_trainable_images(raw_dir, cfg["cats"])

    def start(self) -> dict:
        with self._lock:
            if self.is_active():
                return {"status": "error", "message": "A training job is already running"}
            s = self.settings()
            try:
                backend = self.make_backend()
            except BackendError as exc:
                return {"status": "error", "message": str(exc)}
            missing = backend.missing_settings()
            if missing:
                return {"status": "error",
                        "message": f"{backend.label} is not set up: missing {', '.join(missing)}"}
            counts = self.image_counts()
            short = [f"{c} ({v['with_bbox']})" for c, v in counts.items()
                     if v["with_bbox"] < s["min_images_per_cat"]]
            if short:
                return {"status": "error",
                        "message": f"Need at least {s['min_images_per_cat']} images with "
                                   f"bounding boxes per cat. Too few: {', '.join(short)}"}

            job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._job = {
                "job_id": job_id,
                "backend": backend.name,
                "backend_label": backend.label,
                "stage": "packaging",
                "message": "",
                "created": _now(),
                "stage_times": {"packaging": _now()},
                "counts": counts,
                "handle": None,
                "url": None,
                "metrics": None,
                "validation": None,
                "deployed_at": None,
            }
            self._log.clear()
            self._cancel.clear()
            self._open_log(job_id)
            self._save()
            self._logline(f"Job {job_id} on {backend.label}")
            self._start_thread(resume=False)
            return {"status": "ok", "job_id": job_id}

    def _start_thread(self, resume: bool) -> None:
        self._thread = threading.Thread(target=self._run, args=(resume,), daemon=True,
                                        name="training-job")
        self._thread.start()

    def cancel(self) -> dict:
        if not self.is_active():
            return {"status": "error", "message": "No running job"}
        self._cancel.set()
        self._logline("Cancel requested")
        return {"status": "ok"}

    # ═════════════════════════════════════════════════════════════
    # Worker thread
    # ═════════════════════════════════════════════════════════════
    def _run(self, resume: bool) -> None:
        job = self._job
        job_dir = self._job_dir(job["job_id"])
        backend = None
        try:
            backend = self.make_backend(job["backend"])
            if not resume:
                bundle_tar, manifest = build_bundle(
                    job["job_id"], job_dir,
                    project_root=self.project_root, config_path=self.config_path,
                    config=self._read_config(), python_exe=self.python_exe,
                    log=self._logline, should_stop=self._cancel.is_set,
                )
                with self._lock:
                    job["split_counts"] = manifest["split_counts"]
                    job["bundle_mb"] = manifest.get("bundle_mb")
                if self._cancel.is_set():
                    raise PackagingError("Cancelled")
                self._set_stage("uploading", f"{manifest.get('bundle_mb')} MB")
                handle = backend.submit(job["job_id"], bundle_tar)
                bundle_tar.unlink(missing_ok=True)
                with self._lock:
                    job["handle"] = handle
                    job["url"] = handle.get("url")
                self._set_stage("queued")

            if not self._poll_until_done(backend):
                return

            self._set_stage("downloading")
            results_tar = job_dir / "results.tar.gz"
            backend.fetch(job["handle"], results_tar)
            results_dir = job_dir / "results"
            shutil.rmtree(results_dir, ignore_errors=True)
            safe_extract(results_tar, results_dir)
            results_tar.unlink(missing_ok=True)
            try:
                backend.cleanup(job["handle"])
            except BackendError as exc:
                self._logline(f"Warning: {exc}")

            self._set_stage("validating")
            self._validate(results_dir)
        except PackagingError as exc:
            if self._cancel.is_set():
                self._set_stage("cancelled")
            else:
                self._set_stage("failed", str(exc))
        except BackendError as exc:
            self._set_stage("failed", str(exc))
        except Exception as exc:  # unexpected: keep the dashboard alive
            self._set_stage("failed", f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(job_dir / "cropped", ignore_errors=True)
            shutil.rmtree(job_dir / "dataset", ignore_errors=True)
            (job_dir / "bundle.tar.gz").unlink(missing_ok=True)
            self._close_log()

    def _poll_until_done(self, backend: B.TrainingBackend) -> bool:
        """Poll until the remote job ends. True if it succeeded."""
        job = self._job
        failures = 0
        cancel_sent = False
        while True:
            if self._cancel.is_set() and not cancel_sent:
                try:
                    backend.cancel(job["handle"])
                    cancel_sent = True
                except Exception as exc:
                    self._logline(f"Cancel failed: {exc}")
            try:
                st = backend.poll(job["handle"])
                failures = 0
            except Exception as exc:
                failures += 1
                self._logline(f"Status check failed ({failures}/{_MAX_POLL_FAILURES}): {exc}")
                if failures >= _MAX_POLL_FAILURES:
                    raise BackendError(f"Lost contact with {backend.label}: {exc}") from exc
                self._cancel.wait(min(backend.poll_interval * 2, 60))
                continue

            for line in st.new_log_lines:
                self._logline(line)
            with self._lock:
                if st.url:
                    job["url"] = st.url
                self._save()    # persists the handle's log offset too

            if st.stage == B.SUCCEEDED:
                return True
            if st.stage == B.CANCELLED or (cancel_sent and st.stage in B.REMOTE_TERMINAL):
                self._set_stage("cancelled")
                if job.get("handle"):
                    try:
                        backend.cleanup(job["handle"])
                    except Exception:
                        pass
                return False
            if st.stage == B.FAILED:
                raise BackendError(f"Training failed on {backend.label}: "
                                   f"{st.message or 'see log'}")
            if st.stage != job["stage"] and st.stage in ("queued", "training"):
                self._set_stage(st.stage)
            self._cancel.wait(backend.poll_interval)

    def _validate(self, results_dir: Path) -> None:
        job = self._job
        model = results_dir / "cat_classifier.pt"
        if not model.is_file():
            raise BackendError("The job finished but returned no model")
        cats = self._read_config()["cats"]
        proc = subprocess.run(
            [self.python_exe, "-m", "remote_training.validate_model",
             "--model", str(model), "--cats", ",".join(cats)],
            cwd=str(self.project_root), capture_output=True, text=True, timeout=300,
        )
        try:
            res = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            res = {"ok": False, "errors": [proc.stderr.strip()[-500:] or "validator crashed"]}
        metrics_path = results_dir / "final_metrics.json"
        with self._lock:
            job["validation"] = res
            if metrics_path.is_file():
                job["metrics"] = json.loads(metrics_path.read_text())
            status_path = results_dir / "status.json"
            if status_path.is_file():
                job["remote_status"] = json.loads(status_path.read_text())
        if not res.get("ok"):
            raise BackendError("Model failed validation on the Pi: " + "; ".join(res["errors"]))
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, self.candidates_dir / f"{job['job_id']}.pt")
        acc = (job.get("metrics") or {}).get("val_acc")
        self._set_stage("ready", f"val accuracy {acc:.1%}" if acc is not None else "")

    # ═════════════════════════════════════════════════════════════
    # Deploy / rollback
    # ═════════════════════════════════════════════════════════════
    def _model_paths(self) -> tuple[Path, Path, Path, Path]:
        cfg = self._read_config()
        model = resolve_path(cfg.get("classifier", {}).get("model_path",
                                                            "models/cat_classifier.pt"),
                             self.project_root)
        meta = model.with_suffix(".meta.json")
        return model, meta, model.with_suffix(".prev.pt"), model.with_suffix(".prev.meta.json")

    def deployed_info(self) -> dict:
        model, meta, prev, _ = self._model_paths()
        info = {"exists": model.is_file(), "has_previous": prev.is_file(), "meta": None}
        if meta.is_file():
            try:
                info["meta"] = json.loads(meta.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        return info

    @staticmethod
    def _install(src: Path, dst: Path) -> None:
        """Copy src over dst atomically with a fresh mtime, so the pipeline's
        model watcher sees the change and restarts detection."""
        tmp = dst.with_name(dst.name + ".incoming")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)

    def deploy(self, job_id: str) -> dict:
        if self.is_active():
            return {"status": "error", "message": "Wait for the running job to finish"}
        state_path = self._job_dir(job_id) / "state.json"
        candidate = self.candidates_dir / f"{job_id}.pt"
        if not state_path.is_file() or not candidate.is_file():
            return {"status": "error", "message": f"No validated model for job {job_id}"}
        job = json.loads(state_path.read_text())
        model, meta, prev, prev_meta = self._model_paths()
        model.parent.mkdir(parents=True, exist_ok=True)
        if model.is_file():
            shutil.copy2(model, prev)
            if meta.is_file():
                shutil.copy2(meta, prev_meta)
            else:
                prev_meta.unlink(missing_ok=True)
        meta.write_text(json.dumps({
            "job_id": job_id, "backend": job.get("backend_label"),
            "deployed_at": _now(), "metrics": job.get("metrics"),
        }, indent=2))
        self._install(candidate, model)
        job["deployed_at"] = _now()
        state_path.write_text(json.dumps(job, indent=2))
        with self._lock:
            if self._job and self._job["job_id"] == job_id:
                self._job["deployed_at"] = job["deployed_at"]
        return {"status": "ok", "message": self._reload_hint()}

    def rollback(self) -> dict:
        """Swap the deployed model with the previous one (pressing it again
        swaps back)."""
        model, meta, prev, prev_meta = self._model_paths()
        if not prev.is_file():
            return {"status": "error", "message": "No previous model to roll back to"}
        # Remember the current pair, install the previous pair, then store the
        # remembered pair as the new "previous".
        swap = model.with_name(model.name + ".swap")
        had_model = model.is_file()
        if had_model:
            shutil.copy2(model, swap)
        current_meta = meta.read_text() if meta.is_file() else None

        self._install(prev, model)
        if prev_meta.is_file():
            shutil.copy2(prev_meta, meta)
        else:
            meta.unlink(missing_ok=True)

        if had_model:
            os.replace(swap, prev)
        if current_meta is not None:
            prev_meta.write_text(current_meta)
        else:
            prev_meta.unlink(missing_ok=True)
        return {"status": "ok", "message": "Rolled back. " + self._reload_hint()}

    def _reload_hint(self) -> str:
        if not self._read_config().get("classifier", {}).get("enabled"):
            return ("Model installed. The classifier is disabled; enable it on the "
                    "Capture tab to use it.")
        note = self._on_model_installed() if self._on_model_installed else None
        return note or "Detection restarts with the new model within a few seconds."


    # ═════════════════════════════════════════════════════════════
    # Status
    # ═════════════════════════════════════════════════════════════
    def status(self, log_lines: int = 200) -> dict:
        with self._lock:
            job = json.loads(json.dumps(self._job)) if self._job else None
            log = list(self._log)[-log_lines:]
        if job:
            job.pop("handle", None)
        return {
            "active": self.is_active(),
            "job": job,
            "log": log,
            "stages": STAGES,
            "deployed": self.deployed_info(),
        }

    def list_jobs(self, limit: int = 20) -> list[dict]:
        out = []
        if not self.jobs_dir.is_dir():
            return out
        for p in sorted(self.jobs_dir.glob("*/state.json"), reverse=True)[:limit]:
            try:
                j = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append({
                "job_id": j["job_id"], "backend": j.get("backend_label"),
                "stage": j.get("stage"), "message": j.get("message"),
                "created": j.get("created"), "deployed_at": j.get("deployed_at"),
                "val_acc": (j.get("metrics") or {}).get("val_acc"),
                "candidate": (self.candidates_dir / f"{j['job_id']}.pt").is_file(),
            })
        return out
