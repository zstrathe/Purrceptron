# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Colored terminal output helpers using ANSI escape codes."""
from __future__ import annotations

import os
import sys


class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def _fmt(prefix: str, color: str, msg: str) -> str:
    if _supports_color():
        return f"{color}{prefix}{Color.RESET} {msg}"
    return f"{prefix} {msg}"


def print_info(msg: str) -> None:
    print(_fmt("[INFO]", Color.CYAN, msg))


def print_success(msg: str) -> None:
    print(_fmt("[OK]", Color.GREEN, msg))


def print_error(msg: str) -> None:
    print(_fmt("[ERROR]", Color.RED, msg), file=sys.stderr)


def print_warning(msg: str) -> None:
    print(_fmt("[WARN]", Color.YELLOW, msg))


def print_header(msg: str) -> None:
    line = "\u2550" * 60
    upper = msg.upper()
    if _supports_color():
        print(f"\n{Color.BOLD}{line}{Color.RESET}")
        print(f"{Color.BOLD}\u2550\u2550\u2550 {upper} \u2550\u2550\u2550{Color.RESET}")
        print(f"{Color.BOLD}{line}{Color.RESET}\n")
    else:
        print(f"\n{line}")
        print(f"=== {upper} ===")
        print(f"{line}\n")
