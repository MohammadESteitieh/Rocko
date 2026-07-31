#!/usr/bin/env python3
"""Exploratory five-frame RS(18,6) screen at 100, 50, 40, 30, and 10%."""

from __future__ import annotations

import rs18_experiment as base
import rs18_experiment_protocol as protocol

DUTIES = (100.0, 50.0, 40.0, 30.0, 10.0)
PIDFILE = "/tmp/rs18_quick_screen.pid"


def experiment_schedule() -> tuple[base.Trial, ...]:
    payload = protocol.PAYLOADS[1]
    frame = protocol.FRAMES[1]
    return tuple(
        base.Trial(1, duty, position, payload, frame)
        for position, duty in enumerate(DUTIES, 1)
    )


def estimated_seconds() -> float:
    return (
        base.INITIAL_WAIT_SECONDS
        + len(DUTIES) * (protocol.FRAME_BITS * protocol.BIT_SECONDS + base.GAP_SECONDS)
        + base.FINAL_WAIT_SECONDS
    )


def main() -> int:
    base.experiment_schedule = experiment_schedule
    base.estimated_seconds = estimated_seconds
    base.PIDFILE = PIDFILE
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
