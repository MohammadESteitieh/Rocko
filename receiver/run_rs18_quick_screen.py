#!/usr/bin/env python3
"""Run the separate exploratory five-duty RS(18,6) quick screen safely."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "transmitter"))
import rs18_quick_screen as quick  # noqa: E402
import run_rs18_experiment as runner  # noqa: E402


def quick_name(when: datetime | None = None) -> str:
    when = when or datetime.now()
    return f"rs18_quick_screen_11V_{when.strftime('%Y%m%d_%H%M%S')}"


def quick_transmitter_command(remote_dir: str, manifest: str, log: str,
                              hardware_note: str) -> str:
    args = [
        "./rs18_quick_screen.py", "--manifest", manifest,
        "--hardware-note", hardware_note, "--voltage", "11", "--distance-m", "3",
        "--execute",
    ]
    invocation = " ".join(shlex.quote(item) for item in args)
    return (
        f"cd {shlex.quote(remote_dir)} || exit 1; {runner.SAFE_GPIO_COMMAND}; "
        f"nohup {invocation} > {shlex.quote(log)} 2>&1 </dev/null & "
        "pid=$!; echo $pid"
    )


def main() -> int:
    runner.experiment = quick
    runner.run_name = quick_name
    runner.transmitter_command = quick_transmitter_command
    runner.DEFAULT_DATA_ROOT = Path(
        "~/Desktop/CU-hakcing-captures/rs18-quick-screen"
    ).expanduser()
    runner.DEFAULT_REMOTE_DIR = "/data/home/qnxuser/rs18-quick-screen-runner"
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
