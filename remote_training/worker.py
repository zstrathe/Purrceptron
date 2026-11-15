# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""HTTP job API the training PC exposes for the Pi's local_pc backend.

Served by train_worker.py under /api/worker. Every request needs
``Authorization: Bearer <WORKER_TOKEN>``; with no WORKER_TOKEN set (in the
environment or the project's .env) the API is disabled.

Each job gets a folder in the jobs folder (see default_jobs_dir):

    <jobs>/<job_id>/job/             extracted bundle (manifest, input, code)
    <jobs>/<job_id>/output/          everything run_job.py writes
    <jobs>/<job_id>/results.tar.gz   what the Pi downloads
    <jobs>/<job_id>/worker.log

Routes:
    GET  /health                  torch version, GPU, busy state
    POST /jobs?job_id=ID          body: bundle .tar.gz (application/gzip)
    GET  /jobs/ID?since=N         stage, message, log lines from N
    POST /jobs/ID/cancel
    GET  /jobs/ID/results         results.tar.gz
    POST /detect                  body: one JPEG; returns {"bbox": {cx, cy, w, h,
                                  confidence} | null} (Annotate tab's Auto-detect)
"""
from __future__ import annotations

import hmac
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from remote_training.bundle import safe_extract

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_WINDOWS = os.name == "nt"

NO_QNNPACK = ("This PyTorch build has no QNNPACK quantization engine, so it can't make "
              "INT8 models the Pi runs. Official Windows builds never include it; run "
              "the worker in WSL or Linux.")
_MAX_DETECT_BYTES = 30 * 1024 * 1024


def default_jobs_dir(project_root: Path) -> Path:
    """data/worker_jobs in the project, except for a project on a Windows drive
    under WSL (/mnt/c/...): there thousands of small files unpack and load
    slowly, so jobs go to the Linux disk instead."""
    on_windows_drive = (re.match(r"^/mnt/[a-z]/", str(project_root.resolve())) is not None
                        and "microsoft" in platform.release().lower())
    if on_windows_drive:
        return Path.home() / ".local" / "share" / "cat_detection" / "worker_jobs"
    return project_root / "data" / "worker_jobs"


def prune_jobs(jobs_dir: Path, keep: int) -> list[str]:
    """Delete all but the `keep` most recent job folders. Returns removed ids."""
    if keep <= 0 or not jobs_dir.is_dir():
        return []
    dirs = sorted((d for d in jobs_dir.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    removed = []
    for d in dirs[keep:]:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d.name)
    return removed


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
    def __init__(self, job_id: str, job_dir: Path):
        self.job_id = job_id
        self.job_dir = job_dir
        self.stage = "queued"
        self.message = ""
        self.lines: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.lock = threading.Lock()

    @property
    def results_tar(self) -> Path:
        return self.job_dir / "results.tar.gz"


class WorkerJobs:
    def __init__(self, project_root: Path, python_exe: str, jobs_dir: Path | None = None):
        self.project_root = project_root
        self.python_exe = python_exe
        self.jobs_dir = jobs_dir or default_jobs_dir(project_root)
        self.jobs: dict[str, WorkerJob] = {}
        self._lock = threading.Lock()
        self._health_cache: dict | None = None
        self._detector = None

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def active_job(self) -> str | None:
        with self._lock:
            for j in self.jobs.values():
                if j.stage in ("queued", "running"):
                    return j.job_id
        return None

    def health(self) -> dict:
        if self._health_cache is None:
            code = ("import json,torch;e=list(torch.backends.quantized.supported_engines);"
                    "print(json.dumps({'torch_version':torch.__version__,"
                    "'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) "
                    "if torch.cuda.is_available() else None,"
                    "'quant_engines':e,'int8_ok':'qnnpack' in e}))")
            try:
                out = subprocess.run([self.python_exe, "-c", code], capture_output=True,
                                     text=True, timeout=60).stdout.strip().splitlines()[-1]
                self._health_cache = json.loads(out)
            except Exception as exc:
                return {"ok": False, "error": f"Training venv not usable: {exc}"}
        active = self.active_job()
        busy = f"Pi job {active}" if active else None
        return {"ok": True, **self._health_cache, "busy": busy}

    def submit(self, job_id: str, stream) -> WorkerJob:
        busy = self.active_job()
        if busy:
            raise RuntimeError(f"PC is busy with Pi job {busy}")
        jdir = self.job_dir(job_id)
        if jdir.exists():
            raise RuntimeError(f"Job {job_id} already exists on the PC")
        work = jdir / "job"
        work.mkdir(parents=True)
        bundle = jdir / "bundle.tar.gz"
        with open(bundle, "wb") as f:
            shutil.copyfileobj(stream, f, 1 << 16)
        try:
            safe_extract(bundle, work)
        except Exception as exc:
            shutil.rmtree(jdir, ignore_errors=True)
            raise RuntimeError(f"Bad bundle: {exc}") from exc
        bundle.unlink()

        job = WorkerJob(job_id, jdir)
        with self._lock:
            self.jobs[job_id] = job
        cmd = [self.python_exe, "-u", str(work / "code" / "train" / "run_job.py"),
               "--work", str(work), "--output-dir", str(jdir / "output"),
               "--output-tar", str(job.results_tar)]
        # UTF-8 everywhere: training prints non-ASCII characters, and a piped
        # child on Windows would otherwise encode with the ANSI code page.
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        # Own process group, so cancel stops run_job.py and everything it started.
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if _WINDOWS
                 else {"start_new_session": True})
        job.proc = subprocess.Popen(cmd, cwd=str(work / "code"), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1, env=env, **group)
        job.stage = "running"
        print(f"[{job_id}] === job received from the Pi, starting", flush=True)
        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return job

    def _watch(self, job: WorkerJob) -> None:
        log = open(job.job_dir / "worker.log", "a", encoding="utf-8")
        try:
            assert job.proc and job.proc.stdout
            for line in job.proc.stdout:
                line = line.rstrip()
                with job.lock:
                    job.lines.append(line)
                log.write(line + "\n")
                log.flush()
                print(f"[{job.job_id}] {line}", flush=True)   # live in the worker's terminal
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
        print(f"[{job.job_id}] === {job.stage}: {job.message}", flush=True)

    @staticmethod
    def _elapsed(job: WorkerJob) -> str:
        try:
            st = json.loads((job.job_dir / "output" / "status.json").read_text())
            return f"{st['duration_seconds'] / 60:.1f} min"
        except Exception:
            return "?"

    def get(self, job_id: str) -> WorkerJob | None:
        with self._lock:
            job = self.jobs.get(job_id)
        if job:
            return job
        # Not in memory: the worker restarted. Rebuild from disk.
        jdir = self.job_dir(job_id)
        if not jdir.is_dir():
            return None
        job = WorkerJob(job_id, jdir)
        log = jdir / "worker.log"
        if log.is_file():
            job.lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        if job.results_tar.is_file():
            job.stage, job.message = "succeeded", "Finished"
        else:
            job.stage = "failed"
            job.message = "The worker restarted while this job was running"
        with self._lock:
            self.jobs[job_id] = job
        return job

    def cancel(self, job: WorkerJob) -> None:
        with job.lock:
            job.cancelled = True
            proc = job.proc
        if proc and proc.poll() is None:
            if _WINDOWS:
                # /T: the whole tree (run_job.py, training, DataLoader workers)
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True)
                return
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(3)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def detect(self, image_path: Path) -> dict | None:
        """Best cat bounding box, using train/auto_detect.py in this process
        (the model stays loaded between calls)."""
        if self._detector is None:
            spec = importlib.util.spec_from_file_location(
                "auto_detect", self.project_root / "train" / "auto_detect.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._detector = module
        return self._detector.detect_best(image_path)


def create_worker_blueprint(project_root: Path, python_exe: str,
                            jobs_dir: Path | None = None) -> tuple[Blueprint, WorkerJobs]:
    bp = Blueprint("worker", __name__, url_prefix="/api/worker")
    jobs = WorkerJobs(project_root, python_exe, jobs_dir)

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

    @bp.post("/detect")
    def detect():
        data = request.get_data(cache=False)
        if not data or len(data) > _MAX_DETECT_BYTES:
            return jsonify({"status": "error", "message": "Send one JPEG image"}), 400
        # A closed temp file: Windows can't open a file twice while it is open.
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            bbox = jobs.detect(Path(tmp))
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Auto-detect failed: {exc}"}), 500
        finally:
            Path(tmp).unlink(missing_ok=True)
        return jsonify({"status": "ok", "bbox": bbox})

    return bp, jobs
