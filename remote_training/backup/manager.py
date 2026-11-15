# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Backup settings, the nightly schedule and the dashboard's Backup tab.

Settings live in runtime.yaml under ``backup`` (written as one block) and
tokens in secrets.yaml (sections backup_local_pc and backup_hf). A blank
address or token falls back to the Training tab's settings for the same
place, so the PC worker or Hugging Face account only has to be set up once.

Runs happen in a low-priority subprocess (python -m remote_training.backup
run), so hashing and uploading never slows the dashboard or detection.
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from remote_training.backup.destinations import DESTINATIONS, BackupDestination, BackupError
from utils.secrets import SecretsStore

DEFAULTS = {"destination": "off", "time": "03:00", "keep_days": 30}
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_SECRETS_SECTION = {"local_pc": "backup_local_pc", "hf": "backup_hf"}
_TRAINING_BACKEND = {"local_pc": "local_pc", "hf": "hf_jobs"}   # same place, Training tab
RETRY_AFTER = timedelta(hours=1)
RESULT_PREFIX = "BACKUP_RESULT "


def backup_settings(config: dict) -> dict:
    bc = {**DEFAULTS, **(config.get("backup") or {})}
    # Hand-edited YAML: unquoted off is read as false, and an unquoted time
    # like 03:00 as a number of minutes (YAML 1.1 sexagesimal).
    if bc.get("destination") in (False, None, ""):
        bc["destination"] = "off"
    t = bc.get("time")
    if isinstance(t, int) and 0 <= t < 24 * 60:
        bc["time"] = f"{t // 60:02d}:{t % 60:02d}"
    elif not isinstance(t, str) or not _TIME_RE.match(t):
        bc["time"] = DEFAULTS["time"]
    return bc


def make_destination(config: dict, secrets: SecretsStore, name: str | None = None,
                     form_settings: dict | None = None,
                     form_secrets: dict | None = None) -> BackupDestination:
    bc = backup_settings(config)
    name = name or bc["destination"]
    if name not in DESTINATIONS:
        raise BackupError("Backups are off. Choose a destination first.")
    settings = {**(bc.get(name) or {}), **(form_settings or {})}
    sec = {**secrets.get_section(_SECRETS_SECTION[name]), **(form_secrets or {})}
    training = _TRAINING_BACKEND[name]
    if name == "local_pc" and not str(settings.get("host") or "").strip():
        tb = (config.get("training_backend") or {}).get(training) or {}
        settings["host"] = tb.get("host", "")
        if settings.get("port") in (None, ""):
            settings["port"] = tb.get("port") or 5004
    if not sec.get("token"):
        sec["token"] = secrets.get_section(training).get("token", "")
    return DESTINATIONS[name](settings, sec)


def parse_form(form: dict) -> tuple[dict, dict, dict]:
    """Validate the Backup tab's form. Returns (general settings, destination
    settings, destination secrets); blank secrets mean "keep the stored one"."""
    name = form.get("destination", "off")
    if name != "off" and name not in DESTINATIONS:
        raise BackupError(f"Unknown destination {name!r}")
    t = str(form.get("time", DEFAULTS["time"])).strip()
    if not _TIME_RE.match(t):
        raise BackupError("Time must be HH:MM (24-hour), e.g. 03:00")
    try:
        keep_days = float(form.get("keep_days", DEFAULTS["keep_days"]))
    except (TypeError, ValueError):
        raise BackupError("Keep days must be a number") from None
    if not 1 <= keep_days <= 3650:
        raise BackupError("Keep days must be between 1 and 3650")
    general = {"destination": name, "time": t,
               "keep_days": int(keep_days) if keep_days.is_integer() else keep_days}
    settings, secrets = {}, {}
    if name in DESTINATIONS:
        values = form.get("values") or {}
        for f in DESTINATIONS[name].config_schema():
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
                except (TypeError, ValueError):
                    raise BackupError(f"{f.label} must be a number") from None
                val = int(num) if num.is_integer() else num
            settings[f.key] = val
    return general, settings, secrets


def last_slot(now: datetime, hhmm: str) -> datetime:
    """The most recent scheduled time at or before now (local time)."""
    h, m = map(int, hhmm.split(":"))
    slot = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return slot if slot <= now else slot - timedelta(days=1)


def is_due(now: datetime, hhmm: str, last_success: datetime | None,
           last_attempt: datetime | None) -> bool:
    """Due when no run succeeded since the last scheduled time: right away for
    a slot not tried yet (the Pi was off, or backups were just turned on),
    then hourly after a failure."""
    slot = last_slot(now, hhmm)
    if last_success is not None and last_success >= slot:
        return False
    if last_attempt is None or last_attempt < slot:
        return True
    return now - last_attempt >= RETRY_AFTER


def _parse_time(value) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


class BackupManager:
    def __init__(self, project_root: Path, config_path: Path,
                 read_config: Callable[[], dict], save_block: Callable[[dict], None],
                 secrets: SecretsStore, python_exe: str,
                 command_prefix: Callable[[dict], list[str]] = lambda cfg: []):
        self.project_root = project_root
        self.config_path = config_path
        self._read_config = read_config
        self._save_block = save_block
        self.secrets = secrets
        self.python_exe = python_exe
        self._prefix = command_prefix
        self.state_path = project_root / "data" / "backup_state.json"
        self.log_dir = project_root / "logs" / "backups"
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._log: collections.deque[str] = collections.deque(maxlen=300)
        self._state = self._load_state()
        self._stop = threading.Event()

    # ── State ────────────────────────────────────────────────────
    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        os.replace(tmp, self.state_path)

    # ── Settings ─────────────────────────────────────────────────
    def describe(self) -> dict:
        config = self._read_config()
        bc = backup_settings(config)
        dests = {}
        for name, cls in DESTINATIONS.items():
            dests[name] = {**cls.describe(), "values": dict(bc.get(name) or {}),
                           "secrets_set": self.secrets.is_set(_SECRETS_SECTION[name])}
        return {"destination": bc["destination"], "time": bc["time"],
                "keep_days": bc["keep_days"], "destinations": dests}

    def save(self, form: dict) -> dict:
        general, settings, secrets = parse_form(form)
        config = self._read_config()
        block = {**backup_settings(config), **general}
        name = general["destination"]
        if name in DESTINATIONS:
            block[name] = {**(block.get(name) or {}), **settings}
        self._save_block(block)
        if secrets:
            self.secrets.update_section(_SECRETS_SECTION[name], secrets)
        return self.describe()

    def test(self, form: dict) -> dict:
        general, settings, secrets = parse_form(form)
        dest = make_destination(self._read_config(), self.secrets, general["destination"],
                                settings, secrets)
        return dest.test().to_dict()

    def snapshots(self) -> list[dict]:
        return make_destination(self._read_config(), self.secrets).list_snapshots()

    # ── Runs ─────────────────────────────────────────────────────
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def run_now(self) -> dict:
        config = self._read_config()
        if backup_settings(config)["destination"] not in DESTINATIONS:
            return {"status": "error", "message": "Backups are off. Choose a destination first."}
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"status": "error", "message": "A backup is already running"}
            started = datetime.now().astimezone()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{started.strftime('%Y%m%d-%H%M%S')}.log"
            cmd = self._prefix(config) + [self.python_exe, "-u", "-m", "remote_training.backup",
                                          "run", "--config", str(self.config_path)]
            self._proc = subprocess.Popen(cmd, cwd=str(self.project_root),
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, bufsize=1)
            self._log.clear()
            self._state["last_attempt"] = started.isoformat(timespec="seconds")
            self._save_state()
            proc = self._proc
        threading.Thread(target=self._follow, args=(proc, log_path), daemon=True,
                         name="backup-run").start()
        return {"status": "ok"}

    def _follow(self, proc: subprocess.Popen, log_path: Path) -> None:
        result = None
        last_line = ""
        with open(log_path, "w") as log_file:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith(RESULT_PREFIX):
                    try:
                        result = json.loads(line[len(RESULT_PREFIX):])
                    except ValueError:
                        pass
                    continue
                last_line = line or last_line
                log_file.write(line + "\n")
                log_file.flush()
                with self._lock:
                    self._log.append(line)
        rc = proc.wait()
        with self._lock:
            finished = datetime.now().astimezone().isoformat(timespec="seconds")
            if rc == 0 and result is not None:
                self._state.update(last_success=finished, last_result=result, last_error=None)
            else:
                self._state["last_error"] = last_line or f"Backup exited with code {rc}"
            self._state["last_finished"] = finished
            self._save_state()

    def status(self) -> dict:
        config = self._read_config()
        bc = backup_settings(config)
        with self._lock:
            state = dict(self._state)
            log = list(self._log)
        running = self.is_running()
        next_run = None
        if bc["destination"] in DESTINATIONS:
            now = datetime.now().astimezone()
            if is_due(now, bc["time"], _parse_time(state.get("last_success")),
                      _parse_time(state.get("last_attempt"))):
                next_run = "now" if not running else None
            else:
                nxt = last_slot(now, bc["time"]) + timedelta(days=1)
                last_attempt = _parse_time(state.get("last_attempt"))
                last_success = _parse_time(state.get("last_success"))
                if (last_attempt and (last_success is None or last_success < last_attempt)
                        and last_attempt >= last_slot(now, bc["time"])):
                    nxt = min(nxt, last_attempt + RETRY_AFTER)     # retry after a failure
                next_run = nxt.isoformat(timespec="minutes")
        return {"destination": bc["destination"], "time": bc["time"],
                "keep_days": bc["keep_days"], "running": running, "next_run": next_run,
                "log": log, **state}

    # ── Schedule ─────────────────────────────────────────────────
    def start(self) -> None:
        threading.Thread(target=self._schedule_loop, daemon=True, name="backup-schedule").start()

    def stop(self) -> None:
        self._stop.set()

    def _schedule_loop(self) -> None:
        while not self._stop.wait(30):
            try:
                bc = backup_settings(self._read_config())
                if bc["destination"] not in DESTINATIONS or self.is_running():
                    continue
                with self._lock:
                    last_success = _parse_time(self._state.get("last_success"))
                    last_attempt = _parse_time(self._state.get("last_attempt"))
                if is_due(datetime.now().astimezone(), bc["time"], last_success, last_attempt):
                    self.run_now()
            except Exception as exc:       # keep the schedule alive
                with self._lock:
                    self._log.append(f"Backup schedule error: {exc}")
