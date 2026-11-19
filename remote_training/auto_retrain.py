# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Automatic retraining: every night at a set time, when enough new images a
person confirmed have come in, train a model on the confirmed images only and
deploy it if it does at least as well as the model in use on the test set
(remote_training/test_set.py), with no cat doing worse.

Settings live in runtime.yaml under ``auto_retrain`` (written as one block);
the schedule's state in data/auto_retrain_state.json.
"""
from __future__ import annotations

import collections
import json
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from remote_training.backup.manager import is_due, last_slot

DEFAULTS = {"enabled": False, "time": "02:00", "min_new_images": 25, "min_test_per_cat": 20}
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
RETRY_AFTER = timedelta(hours=1)


def auto_settings(config: dict) -> dict:
    s = {**DEFAULTS, **(config.get("auto_retrain") or {})}
    s["enabled"] = s.get("enabled") is True
    t = s.get("time")
    if isinstance(t, int) and 0 <= t < 24 * 60:      # unquoted 02:00 in YAML 1.1
        s["time"] = f"{t // 60:02d}:{t % 60:02d}"
    elif not isinstance(t, str) or not _TIME_RE.match(t):
        s["time"] = DEFAULTS["time"]
    for key in ("min_new_images", "min_test_per_cat"):
        try:
            s[key] = max(0, int(s[key]))
        except (TypeError, ValueError):
            s[key] = DEFAULTS[key]
    return s


def parse_form(values: dict) -> dict:
    """Validated settings block from the Training tab form. Raises ValueError."""
    out = dict(DEFAULTS)
    out["enabled"] = values.get("enabled") in (True, "true", "on", "1", 1)
    t = str(values.get("time") or "").strip()
    if not _TIME_RE.match(t):
        raise ValueError("Time must be HH:MM (24-hour)")
    out["time"] = t
    for key, label in (("min_new_images", "New confirmed images"),
                       ("min_test_per_cat", "Test images per cat")):
        raw = values.get(key, DEFAULTS[key])
        try:
            n = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a whole number") from None
        if not 0 <= n <= 100_000:
            raise ValueError(f"{label} must be between 0 and 100000")
        out[key] = n
    return out


def _acc(ev: dict) -> float:
    o = ev.get("overall") or {}
    return (o.get("correct", 0) / o["total"]) if o.get("total") else 0.0


def decide(candidate: dict, deployed: dict | None, *, model_in_use: bool,
           comparable: bool, min_test_per_cat: int) -> tuple[bool, str]:
    """Whether to deploy a new model, and why (shown on the Training tab).

    candidate / deployed: test scores from evaluate_model, on the same test
    images. model_in_use: a model is deployed now. comparable: that model was
    trained without the test set, so its score is a fair comparison."""
    per_cat = candidate.get("per_cat") or {}
    short = [f"{cat} {v.get('total', 0)}" for cat, v in per_cat.items()
             if v.get("total", 0) < min_test_per_cat]
    if not per_cat or short:
        return False, (f"Too few confirmed test images to judge (need {min_test_per_cat} "
                       f"per cat): {', '.join(short) or 'none'}")
    if not model_in_use:
        return True, f"No model in use; test accuracy {_acc(candidate):.1%}"
    if not comparable or deployed is None:
        return False, ("The model in use was trained before the test set existed, so it "
                       "can't be compared fairly. Deploy this one by hand once; later models "
                       "are compared automatically.")
    new_ok = (candidate.get("overall") or {}).get("correct", 0)
    old_ok = (deployed.get("overall") or {}).get("correct", 0)
    if new_ok < old_ok:
        return False, (f"Test accuracy {_acc(candidate):.1%} is below the model in use "
                       f"({_acc(deployed):.1%})")
    for cat, v in per_cat.items():
        old = (deployed.get("per_cat") or {}).get(cat) or {}
        if v.get("correct", 0) < old.get("correct", 0):
            return False, (f"{cat} did worse on the test set: {v.get('correct', 0)} of "
                           f"{v.get('total', 0)} right, against {old.get('correct', 0)} "
                           "with the model in use")
    return True, f"Test accuracy {_acc(deployed):.1%} → {_acc(candidate):.1%}"


def _parse_time(value) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


class AutoRetrainer:
    def __init__(self, project_root: Path, config_reader: Callable[[], dict],
                 config_writer: Callable[[dict], None], training_mgr):
        self._read_config = config_reader
        self._write_config = config_writer
        self.mgr = training_mgr
        self.state_path = project_root / "data" / "auto_retrain_state.json"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._log: collections.deque[str] = collections.deque(maxlen=50)
        self._state: dict = {}
        try:
            self._state = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        training_mgr.on_train_ready = self.on_train_ready

    # ── State ────────────────────────────────────────────────────
    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        os.replace(tmp, self.state_path)

    def _record(self, **fields) -> None:
        with self._lock:
            self._state.update(fields)
            self._save_state()
        if fields.get("last_result"):
            self._log.append(f"{datetime.now():%Y-%m-%d %H:%M} {fields['last_result']}")

    def settings(self) -> dict:
        return auto_settings(self._read_config())

    def save_settings(self, values: dict) -> dict:
        block = parse_form(values)
        was_enabled = self.settings()["enabled"]
        self._write_config(block)
        if block["enabled"] and not was_enabled:
            # First run at the next scheduled time, not the moment it's turned on.
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            self._record(last_success=now)
        return self.status()

    def status(self) -> dict:
        s = self.settings()
        with self._lock:
            state = dict(self._state)
        next_run = None
        if s["enabled"]:
            now = datetime.now().astimezone()
            last_success = _parse_time(state.get("last_success"))
            last_attempt = _parse_time(state.get("last_attempt"))
            if is_due(now, s["time"], last_success, last_attempt):
                next_run = "now"
            else:
                nxt = last_slot(now, s["time"]) + timedelta(days=1)
                if (last_attempt and (last_success is None or last_success < last_attempt)
                        and last_attempt >= last_slot(now, s["time"])):
                    nxt = min(nxt, last_attempt + RETRY_AFTER)
                next_run = nxt.isoformat(timespec="minutes")
        return {**s, "next_run": next_run, "new_confirmed": self.new_confirmed(),
                "log": list(self._log), **state}

    # ── New data ─────────────────────────────────────────────────
    def new_confirmed(self, counts: dict | None = None) -> int:
        """Confirmed images outside the test set added since the newest
        finished training job (any trigger), summed over cats."""
        counts = counts if counts is not None else self.mgr.cached_image_counts()
        then = self.mgr.last_trained_counts() or {}
        return sum(max(0, (c.get("confirmed") or 0) - ((then.get(cat) or {}).get("confirmed") or 0))
                   for cat, c in counts.items())

    # ── One scheduled run ────────────────────────────────────────
    def run_once(self) -> str:
        """Check the conditions and start a job if they hold. Returns what
        happened (also recorded). A skip for lack of new data counts as done
        for tonight; an unreachable backend is retried hourly."""
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        s = self.settings()
        if self.mgr.is_active():
            return "waiting: a job is running"         # checked again in 30 s
        counts = self.mgr.image_counts()
        new = self.new_confirmed(counts)
        if new < s["min_new_images"]:
            msg = f"Skipped: {new} new confirmed images (needs {s['min_new_images']})"
            self._record(last_attempt=now, last_success=now, last_result=msg)
            return msg
        need = self.mgr.settings()["min_images_per_cat"]
        short = [f"{c} ({v.get('confirmed', 0)})" for c, v in counts.items()
                 if (v.get("confirmed") or 0) < need]
        if short:
            msg = (f"Skipped: fewer than {need} confirmed images outside the test set for "
                   + ", ".join(short))
            self._record(last_attempt=now, last_success=now, last_result=msg)
            return msg
        try:
            backend = self.mgr.make_backend()
            missing = backend.missing_settings()
            check = None if missing else backend.test_connection()
        except Exception as exc:
            missing, check = [], None
            msg = f"Not started: {exc}"
            self._record(last_attempt=now, last_result=msg)
            return msg
        if missing:
            msg = f"Not started: {backend.label} is not set up (missing {', '.join(missing)})"
            self._record(last_attempt=now, last_success=now, last_result=msg)
            return msg
        if check is not None and not check.ok:
            msg = f"Not started: {backend.label} unreachable ({check.message}); retrying in an hour"
            self._record(last_attempt=now, last_result=msg)
            return msg
        res = self.mgr.start("train", trigger="auto")
        if res.get("status") != "ok":
            msg = f"Not started: {res.get('message')}"
            self._record(last_attempt=now, last_result=msg)
            return msg
        msg = f"Started job {res['job_id']} ({new} new confirmed images)"
        self._record(last_attempt=now, last_success=now, last_result=msg,
                     last_job=res["job_id"])
        return msg

    # ── After a job ──────────────────────────────────────────────
    def on_train_ready(self, job: dict) -> None:
        """Called by the job manager when a training job's model is ready and
        scored. Deploys an automatic job's model if decide() says so."""
        if job.get("trigger") != "auto":
            return
        te = job.get("test_eval") or {}
        if te.get("error") or not te.get("candidate"):
            ok, why = False, f"Not scored on the test set: {te.get('error') or 'no test images'}"
        else:
            ok, why = decide(te["candidate"], te.get("deployed"),
                             model_in_use=bool(te.get("model_in_use")),
                             comparable=bool(te.get("comparable")),
                             min_test_per_cat=self.settings()["min_test_per_cat"])
        note = ""
        if ok:
            res = self.mgr.deploy(job["job_id"])
            if res.get("status") != "ok":
                ok, why = False, f"{why}, but deploying failed: {res.get('message')}"
            else:
                note = res.get("message") or ""
        result = (f"Deployed {job['job_id']}: {why}" if ok
                  else f"Kept {job['job_id']} as a candidate: {why}")
        self.mgr.record_auto_decision(job["job_id"], deployed=ok, reason=why)
        self._record(last_result=result, last_decision={"job_id": job["job_id"],
                                                        "deployed": ok, "reason": why,
                                                        "note": note})

    # ── Schedule ─────────────────────────────────────────────────
    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="auto-retrain").start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(30):
            try:
                s = self.settings()
                if not s["enabled"]:
                    continue
                with self._lock:
                    last_success = _parse_time(self._state.get("last_success"))
                    last_attempt = _parse_time(self._state.get("last_attempt"))
                if is_due(datetime.now().astimezone(), s["time"], last_success, last_attempt):
                    self.run_once()
            except Exception as exc:       # keep the schedule alive
                self._log.append(f"{datetime.now():%Y-%m-%d %H:%M} Schedule error: {exc}")
