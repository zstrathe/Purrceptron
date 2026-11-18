# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Remote job lifecycle on the Pi.

package -> submit to backend -> poll -> fetch results -> finish -> ready.

Three kinds of job share that lifecycle:

    train     finish = validate the model with the Pi's torch; the user then
              deploys it (or rolls back)
    crossval  finish = keep the k-fold summary
    detect    finish = write the auto-detected bounding boxes into sidecars

One job runs at a time. Each job's state is saved to
data/training_jobs/<job_id>/state.json after every change, so restarting the
dashboard resumes polling a job that is still running remotely.
"""
from __future__ import annotations

import collections
import json
import os
import re
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
    TASKS, PackagingError, build_bundle, count_trainable_images, images_missing_bbox,
    pi_executorch_version, pi_torch_version, safe_extract,
)
from remote_training.metrics import read_training_metrics
from utils.config import resolve_path
from utils.secrets import SecretsStore
from utils.sidecars import from_center, has_bbox, set_bbox

_COMMON = ["packaging", "uploading", "queued", "training", "downloading"]
STAGES = {
    "train": _COMMON + ["validating", "ready"],
    "crossval": _COMMON + ["ready"],
    "detect": _COMMON + ["applying", "ready"],
}
KIND_LABELS = {"train": "Training", "crossval": "Cross-validation", "detect": "Auto-detect"}
TERMINAL = {"ready", "failed", "cancelled"}
_REMOTE_STAGES = {"uploading", "queued", "training", "downloading", "validating", "applying"}

DEFAULT_MIN_IMAGES = 30
_MAX_POLL_FAILURES = 40     # consecutive poll errors (network blips) before giving up
_STATE_SAVE_INTERVAL = 60   # seconds between state.json saves while polling (stage
                            # changes save right away); bounds SD card writes


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_EPOCH_RE = re.compile(r"\[P(\d) +(\d+)/(\d+)\]")
_FOLD_RE = re.compile(r"Fold (\d+)/(\d+)")
TRAINING_PHASES = 3     # head warm-up, fine-tune, QAT (train/train_classifier.py)


def parse_progress(lines: list[str]) -> dict | None:
    """How far a job's training is, from its log: the newest "[P2  12/40]"
    epoch line and "Fold 2/5" line (cross-validation). None before training
    prints anything."""
    out: dict = {}
    for line in reversed(lines):
        if "epoch" not in out:
            m = _EPOCH_RE.search(line)
            if m:
                out.update(phase=int(m[1]), phases=TRAINING_PHASES,
                           epoch=int(m[2]), epochs=int(m[3]))
        if "fold" not in out:
            m = _FOLD_RE.search(line)
            if m:
                out.update(fold=int(m[1]), folds=int(m[2]))
        if "epoch" in out and "fold" in out:
            break
    return out or None


def _seconds_between(a: str | None, b: str | None) -> float | None:
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except (TypeError, ValueError):
        return None


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
        self._executorch_version = pi_executorch_version()

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
            "pi_executorch_version": self._executorch_version,
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
        return cls(merged, merged_secrets, {
            "pi_torch_version": self._torch_version,
            "pi_executorch_version": self._executorch_version,
        })

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
    def raw_dir(self) -> Path:
        cfg = self._read_config()
        return resolve_path(cfg.get("collection", {}).get("output_dir", "data/raw"),
                            self.project_root)

    def image_counts(self) -> dict:
        return count_trainable_images(self.raw_dir(), self._read_config()["cats"])

    _COUNTS_TTL = 30.0

    def _cached_image_counts(self) -> dict:
        """image_counts(), at most every 30 s (the status is polled every 2 s)."""
        now = time.monotonic()
        cached = getattr(self, "_counts_cache", None)
        if cached and now - cached[0] < self._COUNTS_TTL:
            return cached[1]
        counts = self.image_counts()
        self._counts_cache = (now, counts)
        return counts

    def new_images(self) -> dict | None:
        """Labeled images (with a bounding box) per cat added since the model
        in use was trained: {cat: n}. None when that isn't known (no model,
        or one installed by hand)."""
        meta = self.deployed_info().get("meta") or {}
        job = self._read_json(self._job_dir(meta["job_id"]) / "state.json") \
            if meta.get("job_id") and self._valid_job_id(meta["job_id"]) else None
        then = (job or {}).get("counts")
        if not then:
            return None
        now = self._cached_image_counts()
        return {cat: max(0, (c.get("with_bbox") or 0) - ((then.get(cat) or {}).get("with_bbox") or 0))
                for cat, c in now.items()}

    def last_duration(self) -> float | None:
        """Seconds the newest successful training job took, start to finish."""
        if not self.jobs_dir.is_dir():
            return None
        for p in sorted(self.jobs_dir.glob("*/state.json"), reverse=True):
            j = self._read_json(p) or {}
            if j.get("kind", "train") == "train" and j.get("stage") == "ready":
                t = j.get("stage_times") or {}
                return _seconds_between(t.get("packaging"), t.get("ready"))
        return None

    def missing_bbox_count(self) -> int:
        return len(images_missing_bbox(self.raw_dir(), self._read_config()["cats"]))

    def start(self, kind: str = "train", cv_folds: int | None = None) -> dict:
        if kind not in TASKS:
            return {"status": "error", "message": f"Unknown job kind {kind!r}"}
        if cv_folds is not None and not 2 <= cv_folds <= 10:
            return {"status": "error", "message": "Folds must be between 2 and 10"}
        with self._lock:
            if self.is_active():
                return {"status": "error",
                        "message": f"A {KIND_LABELS[self._job.get('kind', 'train')].lower()} "
                                   "job is already running"}
            s = self.settings()
            try:
                backend = self.make_backend()
            except BackendError as exc:
                return {"status": "error", "message": str(exc)}
            missing = backend.missing_settings()
            if missing:
                return {"status": "error",
                        "message": f"{backend.label} is not set up: missing {', '.join(missing)}"}
            if kind == "train" and not self._executorch_version:
                return {"status": "error",
                        "message": "ExecuTorch is not installed on the Pi, so it could not "
                                   "check or run the trained model. Install it first (see "
                                   "Software requirements in the README)."}
            counts = self.image_counts()
            short = [f"{c} ({v['with_bbox']})" for c, v in counts.items()
                     if v["with_bbox"] < s["min_images_per_cat"]]
            if short and kind != "detect":
                return {"status": "error",
                        "message": f"Need at least {s['min_images_per_cat']} images with "
                                   f"bounding boxes per cat. Too few: {', '.join(short)}"}

            job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._job = {
                "job_id": job_id,
                "kind": kind,
                "cv_folds": cv_folds,
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
            self._logline(f"{KIND_LABELS[kind]} job {job_id} on {backend.label}")
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
        bundle_tar = None
        try:
            backend = self.make_backend(job["backend"])
            if not resume:
                bundle_tar, manifest = build_bundle(
                    job["job_id"], job_dir,
                    project_root=self.project_root, config_path=self.config_path,
                    config=self._read_config(), python_exe=self.python_exe,
                    log=self._logline, should_stop=self._cancel.is_set,
                    task=job.get("kind", "train"), cv_folds=job.get("cv_folds"),
                )
                with self._lock:
                    for key in ("split_counts", "crop_counts", "cv_folds"):
                        if key in manifest:
                            job[key] = manifest[key]
                    if "images" in manifest:
                        job["detect_images"] = manifest["images"]
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

            kind = job.get("kind", "train")
            self._read_remote_status(results_dir)
            if kind == "train":
                self._set_stage("validating")
                self._validate(results_dir)
            elif kind == "crossval":
                self._finish_crossval(results_dir)
            else:
                self._set_stage("applying")
                self._apply_detections(results_dir)
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
            if bundle_tar is not None:
                bundle_tar.unlink(missing_ok=True)
            (job_dir / "bundle.tar.gz").unlink(missing_ok=True)
            self._close_log()

    def _poll_until_done(self, backend: B.TrainingBackend) -> bool:
        """Poll until the remote job ends. True if it succeeded."""
        job = self._job
        failures = 0
        cancel_sent = False
        last_save = 0.0
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
                if self._cancel.is_set():
                    # Cancel asked for and the backend can't be reached: stop
                    # waiting instead of retrying for minutes.
                    self._logline(f"{backend.label} unreachable ({exc}); the remote job "
                                  "may still be running there.")
                    self._set_stage("cancelled")
                    return False
                failures += 1
                self._logline(f"Status check failed ({failures}/{_MAX_POLL_FAILURES}): {exc}")
                if failures >= _MAX_POLL_FAILURES:
                    raise BackendError(f"Lost contact with {backend.label}: {exc}") from exc
                self._cancel.wait(min(backend.poll_interval * 2, 60))
                continue

            for line in st.new_log_lines:
                self._logline(line)
            with self._lock:
                url_changed = bool(st.url) and st.url != job.get("url")
                if st.url:
                    job["url"] = st.url
                # Persists the handle's log offset too. After a dashboard restart
                # at most _STATE_SAVE_INTERVAL seconds of log lines can repeat.
                if url_changed or time.monotonic() - last_save >= _STATE_SAVE_INTERVAL:
                    self._save()
                    last_save = time.monotonic()

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
                raise BackendError(f"The job failed on {backend.label}: "
                                   f"{st.message or 'see log'}")
            if st.stage != job["stage"] and st.stage in ("queued", "training"):
                self._set_stage(st.stage)
            self._cancel.wait(backend.poll_interval)

    def _validate(self, results_dir: Path) -> None:
        job = self._job
        model = results_dir / "cat_classifier.pte"
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
        if not res.get("ok"):
            raise BackendError("Model failed validation on the Pi: " + "; ".join(res["errors"]))
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, self.candidates_dir / f"{job['job_id']}.pte")
        acc = (job.get("metrics") or {}).get("val_acc")
        self._set_stage("ready", f"val accuracy {acc:.1%}" if acc is not None else "")

    def _read_remote_status(self, results_dir: Path) -> None:
        status_path = results_dir / "status.json"
        if status_path.is_file():
            with self._lock:
                self._job["remote_status"] = json.loads(status_path.read_text())

    def _finish_crossval(self, results_dir: Path) -> None:
        path = results_dir / "cv_summary.json"
        if not path.is_file():
            raise BackendError("The job finished but returned no cross-validation summary")
        cv = json.loads(path.read_text())
        with self._lock:
            self._job["cv"] = cv
        self._set_stage("ready", f"mean accuracy {cv.get('mean_acc', 0):.1%} "
                                 f"± {cv.get('std_acc', 0):.1%} over {cv.get('n_folds')} folds")

    def _apply_detections(self, results_dir: Path) -> None:
        """Write auto-detected bounding boxes into sidecars. Images that got
        a bounding box while the job ran (drawn by hand, say) are left alone.
        An image labeled in Review meanwhile was renamed to <cat>_<timestamp>,
        so it is found again by its timestamp."""
        path = results_dir / "detections.json"
        if not path.is_file():
            raise BackendError("The job finished but returned no detections")
        detections: dict = json.loads(path.read_text())
        raw_dir = self.raw_dir()
        cats = self._read_config()["cats"]
        by_stamp: dict[str, Path] | None = None

        def locate(rel: str) -> Path | None:
            nonlocal by_stamp
            p = raw_dir / rel
            if p.is_file():
                return p
            if by_stamp is None:
                by_stamp = {}
                for folder in (*cats, "unlabeled"):
                    d = raw_dir / folder
                    if d.is_dir():
                        for img in d.glob("*.jpg"):
                            by_stamp.setdefault(img.stem.split("_", 1)[-1], img)
            return by_stamp.get(Path(rel).stem.split("_", 1)[-1])

        result = {"applied": 0, "no_cat": 0, "already_had_bbox": 0, "gone": 0}
        for rel, box in detections.items():
            if not box:
                result["no_cat"] += 1
                continue
            img = locate(rel)
            if img is None:
                result["gone"] += 1
            elif has_bbox(img):
                result["already_had_bbox"] += 1
            else:
                bbox = from_center(box["cx"], box["cy"], box["w"], box["h"])
                if bbox is None:
                    result["no_cat"] += 1
                    continue
                set_bbox(img, bbox, source="auto-detect",
                         confidence=box.get("confidence", 0.0))
                result["applied"] += 1
        with self._lock:
            self._job["detect_result"] = result
        self._logline(f"Bounding boxes written: {result['applied']}; no cat found: "
                      f"{result['no_cat']}; already had one: {result['already_had_bbox']}; "
                      f"image gone: {result['gone']}")
        self._set_stage("ready", f"{result['applied']} bounding boxes added, "
                                 f"{result['no_cat']} images with no cat found")

    # ═════════════════════════════════════════════════════════════
    # Instant auto-detect (backends that support it, e.g. Local PC)
    # ═════════════════════════════════════════════════════════════
    def instant_detect_available(self) -> bool:
        try:
            return get_backend_class(self.settings()["backend"]).supports_instant_detect
        except BackendError:
            return False

    def instant_detect(self, image_path: Path) -> dict | None:
        """Best cat bounding box {cx, cy, w, h, confidence} or None. Raises
        BackendError when the backend can't do it or is unreachable."""
        backend = self.make_backend()
        if not backend.supports_instant_detect:
            raise BackendError(f"{backend.label} runs auto-detect as a batch job only. "
                               "Use 'Auto-detect all missing'.")
        missing = backend.missing_settings()
        if missing:
            raise BackendError(f"{backend.label} is not set up: missing {', '.join(missing)}")
        return backend.detect_image(image_path)

    # ═════════════════════════════════════════════════════════════
    # Deploy / rollback
    # ═════════════════════════════════════════════════════════════
    def _model_paths(self) -> tuple[Path, Path, Path, Path]:
        cfg = self._read_config()
        model = resolve_path(cfg.get("classifier", {}).get("model_path",
                                                            "models/cat_classifier.pte"),
                             self.project_root)
        meta = model.with_suffix(".meta.json")
        prev = model.with_name(f"{model.stem}.prev{model.suffix}")
        return model, meta, prev, model.with_suffix(".prev.meta.json")

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def deployed_info(self) -> dict:
        """The deployed model and the one a rollback would bring back.
        meta / prev_meta: {job_id, backend, deployed_at, metrics}, or None when
        the model was installed some other way."""
        model, meta, prev, prev_meta = self._model_paths()
        return {
            "exists": model.is_file(),
            "has_previous": prev.is_file(),
            "meta": self._read_json(meta) if model.is_file() else None,
            "prev_meta": self._read_json(prev_meta) if prev.is_file() else None,
        }

    def candidates(self) -> list[dict]:
        """Trained models that passed validation and aren't the deployed one,
        newest first: [{job_id, created, backend, metrics}]."""
        deployed = (self.deployed_info().get("meta") or {}).get("job_id")
        out = []
        if not self.candidates_dir.is_dir():
            return out
        for pte in sorted(self.candidates_dir.glob("*.pte"), reverse=True):
            job_id = pte.stem
            if job_id == deployed or not self._valid_job_id(job_id):
                continue
            job = self._read_json(self._job_dir(job_id) / "state.json") or {}
            if not job.get("metrics"):
                continue
            out.append({"job_id": job_id, "created": job.get("created"),
                        "backend": job.get("backend_label"), "metrics": job["metrics"]})
        return out

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
        candidate = self.candidates_dir / f"{job_id}.pte"
        if not state_path.is_file() or not candidate.is_file():
            return {"status": "error", "message": f"No validated model for job {job_id}"}
        job = json.loads(state_path.read_text())
        model, meta, prev, prev_meta = self._model_paths()
        if model.suffix != ".pte":
            return {"status": "error",
                    "message": f"classifier.model_path is {model.name}, but models are now "
                               "ExecuTorch .pte files. Set it to models/cat_classifier.pte in "
                               "configs/project.yaml and restart the dashboard."}
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
            return ("Model installed. Cat identification is off; turn on "
                    "\"Identify cats\" on the Capture tab to use it.")
        note = self._on_model_installed() if self._on_model_installed else None
        return note or "The new model loads within a few seconds (the camera keeps running)."


    # ═════════════════════════════════════════════════════════════
    # Status
    # ═════════════════════════════════════════════════════════════
    def status(self, log_lines: int = 200) -> dict:
        with self._lock:
            job = json.loads(json.dumps(self._job)) if self._job else None
            log = list(self._log)[-log_lines:]
        if job:
            job.pop("handle", None)
            job.pop("detect_images", None)
        kind = (job or {}).get("kind", "train")
        active = self.is_active()
        with self._lock:
            progress = parse_progress(list(self._log)) if active else None
        return {
            "active": active,
            "job": job,
            "log": log,
            "stages": STAGES[kind],
            "progress": progress,
            "deployed": self.deployed_info(),
            "candidates": self.candidates(),
            "new_images": self.new_images(),
            "last_duration": self.last_duration(),
        }

    def job_metrics(self, job_id: str) -> dict:
        """Per-epoch training curves of a finished training job."""
        if not self._valid_job_id(job_id):
            return {"available": False}
        return read_training_metrics(self._job_dir(job_id) / "results" / "training")

    def job_state(self, job_id: str) -> dict | None:
        """Saved state of any job, without internal fields."""
        if not self._valid_job_id(job_id):
            return None
        path = self._job_dir(job_id) / "state.json"
        try:
            job = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        job.pop("handle", None)
        job.pop("detect_images", None)
        return job

    @staticmethod
    def _valid_job_id(job_id: str) -> bool:
        return bool(job_id) and job_id.replace("-", "").isdigit() and len(job_id) <= 32

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
                "job_id": j["job_id"], "kind": j.get("kind", "train"),
                "backend": j.get("backend_label"),
                "stage": j.get("stage"), "message": j.get("message"),
                "created": j.get("created"), "deployed_at": j.get("deployed_at"),
                "val_acc": (j.get("metrics") or {}).get("val_acc"),
                "cv_mean": (j.get("cv") or {}).get("mean_acc"),
                "cv_std": (j.get("cv") or {}).get("std_acc"),
                "cv_folds": j.get("cv_folds"),
                "duration": _seconds_between((j.get("stage_times") or {}).get("packaging"),
                                             (j.get("stage_times") or {}).get("ready")),
                "has_curves": (self._job_dir(j["job_id"]) / "results" / "training"
                               / "results.csv").is_file(),
                "candidate": (self.candidates_dir / f"{j['job_id']}.pte").is_file(),
            })
        return out
