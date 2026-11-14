# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""HTTP job API the training PC exposes for the Pi's local_pc backend.

Mounted on training_dashboard.py under /api/worker. Every request needs
``Authorization: Bearer <WORKER_TOKEN>``; with no WORKER_TOKEN set (in the
environment or the project's .env) the API is disabled.

Each Pi job becomes a normal training version, versions/pi-<job_id>/, so
the training dashboard's metrics charts work on it:

    versions/pi-<job_id>/job/            extracted bundle (manifest, dataset, code)
    versions/pi-<job_id>/training/       results.csv and checkpoints
    versions/pi-<job_id>/cat_classifier.pt
    versions/pi-<job_id>/results.tar.gz  what the Pi downloads
    versions/pi-<job_id>/worker.log

Routes:
    GET  /health                  torch version, GPU, busy state
    POST /jobs?job_id=ID          body: bundle .tar.gz (application/gzip)
    GET  /jobs/ID?since=N         stage, message, log lines from N
    POST /jobs/ID/cancel
    GET  /jobs/ID/results         results.tar.gz
"""
from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from flask import Blueprint, jsonify, request, send_file

from remote_training.bundle import safe_extract

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def load_worker_token(project_root: Path) -> str:
    token = os.environ.get("WORKER_TOKEN", "").strip()
    if token:
        return token
    env = project_root / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("WORKER_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


class WorkerJob:
    def __init__(self, job_id: str, version_dir: Path):
        self.job_id = job_id
        self.version_dir = version_dir
        self.stage = "queued"
        self.message = ""
        self.lines: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.lock = threading.Lock()

    @property
    def results_tar(self) -> Path:
        return self.version_dir / "results.tar.gz"


class WorkerJobs:
    def __init__(self, project_root: Path, python_exe: str,
                 busy_check: Callable[[], str | None] = lambda: None):
        self.project_root = project_root
        self.python_exe = python_exe
        self.busy_check = busy_check
        self.jobs: dict[str, WorkerJob] = {}
        self._lock = threading.Lock()
        self._health_cache: dict | None = None

    def version_dir(self, job_id: str) -> Path:
        return self.project_root / "versions" / f"pi-{job_id}"

    def active_job(self) -> str | None:
        with self._lock:
            for j in self.jobs.values():
                if j.stage in ("queued", "running"):
                    return j.job_id
        return None

    def health(self) -> dict:
        if self._health_cache is None:
            code = ("import json,torch;print(json.dumps({'torch_version':torch.__version__,"
                    "'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) "
                    "if torch.cuda.is_available() else None}))")
            try:
                out = subprocess.run([self.python_exe, "-c", code], capture_output=True,
                                     text=True, timeout=60).stdout.strip().splitlines()[-1]
                self._health_cache = json.loads(out)
            except Exception as exc:
                return {"ok": False, "error": f"Training venv not usable: {exc}"}
        active = self.active_job()
        busy = f"Pi job {active}" if active else self.busy_check()
        return {"ok": True, **self._health_cache, "busy": busy}

    def submit(self, job_id: str, stream) -> WorkerJob:
        busy = self.active_job() or self.busy_check()
        if busy:
            raise RuntimeError(f"PC is busy: {busy}")
        vdir = self.version_dir(job_id)
        if vdir.exists():
            raise RuntimeError(f"Job {job_id} already exists on the PC")
        work = vdir / "job"
        work.mkdir(parents=True)
        bundle = vdir / "bundle.tar.gz"
        with open(bundle, "wb") as f:
            shutil.copyfileobj(stream, f, 1 << 16)
        try:
            safe_extract(bundle, work)
        except Exception as exc:
            shutil.rmtree(vdir, ignore_errors=True)
            raise RuntimeError(f"Bad bundle: {exc}") from exc
        bundle.unlink()

        job = WorkerJob(job_id, vdir)
        with self._lock:
            self.jobs[job_id] = job
        cmd = [self.python_exe, "-u", str(work / "code" / "train" / "run_job.py"),
               "--work", str(work), "--output-dir", str(vdir),
               "--output-tar", str(job.results_tar)]
        job.proc = subprocess.Popen(cmd, cwd=str(work / "code"), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    start_new_session=True)
        job.stage = "running"
        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return job

    def _watch(self, job: WorkerJob) -> None:
        log = open(job.version_dir / "worker.log", "a")
        try:
            assert job.proc and job.proc.stdout
            for line in job.proc.stdout:
                line = line.rstrip()
                with job.lock:
                    job.lines.append(line)
                log.write(line + "\n")
                log.flush()
            rc = job.proc.wait()
        finally:
            log.close()
        with job.lock:
            if job.cancelled:
                job.stage, job.message = "cancelled", "Cancelled"
            elif rc == 0 and job.results_tar.is_file():
                job.stage, job.message = "succeeded", f"Finished in {self._elapsed(job)}"
            else:
                job.stage, job.message = "failed", f"run_job.py exited with code {rc}"

    @staticmethod
    def _elapsed(job: WorkerJob) -> str:
        try:
            st = json.loads((job.version_dir / "status.json").read_text())
            return f"{st['duration_seconds'] / 60:.1f} min"
        except Exception:
            return "?"

    def get(self, job_id: str) -> WorkerJob | None:
        with self._lock:
            job = self.jobs.get(job_id)
        if job:
            return job
        # Not in memory: the dashboard restarted. Rebuild from disk.
        vdir = self.version_dir(job_id)
        if not vdir.is_dir():
            return None
        job = WorkerJob(job_id, vdir)
        log = vdir / "worker.log"
        if log.is_file():
            job.lines = log.read_text(errors="replace").splitlines()
        if job.results_tar.is_file():
            job.stage, job.message = "succeeded", "Finished"
        else:
            job.stage = "failed"
            job.message = "The training dashboard restarted while this job was running"
        with self._lock:
            self.jobs[job_id] = job
        return job

    def cancel(self, job: WorkerJob) -> None:
        with job.lock:
            job.cancelled = True
            proc = job.proc
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(3)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def create_worker_blueprint(project_root: Path, python_exe: str,
                            busy_check: Callable[[], str | None] = lambda: None
                            ) -> tuple[Blueprint, WorkerJobs]:
    bp = Blueprint("worker", __name__, url_prefix="/api/worker")
    jobs = WorkerJobs(project_root, python_exe, busy_check)

    @bp.before_request
    def _auth():
        token = load_worker_token(project_root)
        if not token:
            return jsonify({"status": "error",
                            "message": "Worker API disabled: set WORKER_TOKEN in the PC's .env"}), 403
        sent = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(sent.encode(), token.encode()):
            return jsonify({"status": "error", "message": "Bad token"}), 401
        return None

    def _job_or_404(job_id: str):
        if not _JOB_ID_RE.match(job_id):
            return None, (jsonify({"status": "error", "message": "Bad job id"}), 400)
        job = jobs.get(job_id)
        if not job:
            return None, (jsonify({"status": "error", "message": "Unknown job"}), 404)
        return job, None

    @bp.get("/health")
    def health():
        return jsonify(jobs.health())

    @bp.post("/jobs")
    def submit():
        job_id = request.args.get("job_id", "")
        if not _JOB_ID_RE.match(job_id):
            return jsonify({"status": "error", "message": "Bad job id"}), 400
        try:
            job = jobs.submit(job_id, request.stream)
        except RuntimeError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 409
        return jsonify({"status": "ok", "job_id": job.job_id})

    @bp.get("/jobs/<job_id>")
    def status(job_id):
        job, err = _job_or_404(job_id)
        if err:
            return err
        since = max(0, request.args.get("since", 0, type=int))
        with job.lock:
            lines = job.lines[since:]
            return jsonify({"stage": job.stage, "message": job.message,
                            "lines": lines, "next": since + len(lines)})

    @bp.post("/jobs/<job_id>/cancel")
    def cancel(job_id):
        job, err = _job_or_404(job_id)
        if err:
            return err
        jobs.cancel(job)
        return jsonify({"status": "ok"})

    @bp.get("/jobs/<job_id>/results")
    def results(job_id):
        job, err = _job_or_404(job_id)
        if err:
            return err
        if job.stage != "succeeded" or not job.results_tar.is_file():
            return jsonify({"status": "error", "message": "No results yet"}), 409
        return send_file(str(job.results_tar), mimetype="application/gzip")

    return bp, jobs
