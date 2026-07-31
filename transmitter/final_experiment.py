#!/usr/bin/env python3
"""Safely transmit the approved 100-frame final experiment on QNX.

The full schedule is 75 short-comparison frames plus 25 final RS(12,6)
frames: five repetitions at each of 100, 50, 25, 10, and commanded-1% duty.
Nothing energizes unless --execute is supplied. Physical runs are frozen at
11 V, 3 m, two coded bits/s, 15 s initial/gaps, and 5 s final coil-off time.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import os
import signal
import sys
import time
from typing import Callable, Sequence

import duty_pair_test as duty_tx
import final_experiment_protocol as protocol
import hardware as hw

INITIAL_WAIT_SECONDS = 15.0
GAP_SECONDS = 15.0
FINAL_WAIT_SECONDS = 5.0
VOLTAGE = 11.0
DISTANCE_M = 3.0
LEGACY_PWM_GPIO = 18
PIDFILE = "/tmp/final_experiment.pid"


@dataclass(frozen=True)
class Trial:
    phase: str
    scheme: str
    duty_percent: float
    repetition: int


def short_schedule() -> tuple[Trial, ...]:
    """Counterbalance duty and candidate order across five repetitions."""
    trials: list[Trial] = []
    schemes = protocol.SHORT_SCHEMES
    duties = protocol.DUTIES
    for repetition in range(1, protocol.REPETITIONS + 1):
        duty_offset = repetition - 1
        duty_order = duties[duty_offset:] + duties[:duty_offset]
        for duty_position, duty in enumerate(duty_order):
            scheme_offset = (duty_position + repetition - 1) % len(schemes)
            scheme_order = schemes[scheme_offset:] + schemes[:scheme_offset]
            trials.extend(
                Trial("short", scheme, duty, repetition) for scheme in scheme_order
            )
    return tuple(trials)


def final_schedule() -> tuple[Trial, ...]:
    """Counterbalance duty order across the five final-code repetitions."""
    trials: list[Trial] = []
    duties = protocol.DUTIES
    for repetition in range(1, protocol.REPETITIONS + 1):
        duty_offset = repetition - 1
        duty_order = duties[duty_offset:] + duties[:duty_offset]
        trials.extend(
            Trial("final", "rs12_6", duty, repetition) for duty in duty_order
        )
    return tuple(trials)


def experiment_schedule(phase: str) -> tuple[Trial, ...]:
    if phase == "short":
        return short_schedule()
    if phase == "final":
        return final_schedule()
    if phase == "all":
        return short_schedule() + final_schedule()
    raise ValueError("phase must be short, final, or all")


def estimated_seconds(trials: Sequence[Trial]) -> float:
    return (
        INITIAL_WAIT_SECONDS
        + sum(len(protocol.FRAMES[trial.scheme]) * protocol.BIT_SECONDS + GAP_SECONDS
              for trial in trials)
        + FINAL_WAIT_SECONDS
    )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def safe_all_off(backend, config: hw.Config) -> bool:
    """Attempt every required low write even if an earlier pin fails."""
    success = True
    for pin in (config.enb_gpio, LEGACY_PWM_GPIO, config.in3_gpio, config.in4_gpio):
        try:
            backend.write_pin(pin, 0)
        except Exception as exc:
            success = False
            print(f"ERROR forcing GPIO{pin} low: {exc}", file=sys.stderr)
    return success


def require_all_off(backend, config: hw.Config) -> None:
    if not safe_all_off(backend, config):
        raise hw.BeaconError("could not verify all four transmitter GPIOs low")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("short", "final", "all"), default="all")
    parser.add_argument("--manifest", help="output CSV path")
    parser.add_argument("--hardware-note", required=True)
    parser.add_argument("--voltage", type=float, default=VOLTAGE)
    parser.add_argument("--distance-m", type=float, default=DISTANCE_M)
    parser.add_argument("--execute", action="store_true",
                        help="required before GPIO can be energized")
    parser.add_argument("--sim", action="store_true", help="record calls without GPIO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None,
         sleep: Callable[[float], None] = time.sleep) -> int:
    args = parse_args(argv)
    note = args.hardware_note.strip()
    if not note or "\n" in note or "\r" in note:
        print("--hardware-note must be non-empty and single-line", file=sys.stderr)
        return 2
    if args.voltage != VOLTAGE or args.distance_m != DISTANCE_M:
        print("physical experiment is frozen at exactly 11 V and 3 m", file=sys.stderr)
        return 2
    trials = experiment_schedule(args.phase)
    print(
        f"Final experiment phase={args.phase} frames={len(trials)} "
        f"rate={1 / protocol.BIT_SECONDS:g} coded bits/s "
        f"estimate={estimated_seconds(trials) / 60:.2f} minutes",
        flush=True,
    )
    for index, trial in enumerate(trials, 1):
        timing_label = " timing-limited" if trial.duty_percent == 1 else ""
        print(
            f"{index:03d} {trial.phase} {trial.scheme} rep={trial.repetition} "
            f"duty={trial.duty_percent:g}%{timing_label}",
            flush=True,
        )
    if not args.execute:
        print("DRY RUN: add --execute only after supply, distance, receiver, and area checks")
        return 0

    manifest_path = args.manifest or time.strftime(
        f"final-experiment-{args.phase}-%Y%m%d_%H%M%S.csv"
    )
    config = replace(hw.Config(pidfile_path=PIDFILE), bit_seconds=protocol.BIT_SECONDS)
    safety_pins = (config.enb_gpio, LEGACY_PWM_GPIO, config.in3_gpio, config.in4_gpio)
    backend = hw.SimBackend() if args.sim else hw.QnxGpioBackend(
        config.gpio_dev, safety_pins
    )
    driver = hw.CoilDriver(backend, config)
    transmitter = duty_tx.DutyFrameTransmitter(driver, config)
    locks = [] if args.sim else [
        hw.SingleInstanceLock(hw.Config().pidfile_path),
        hw.SingleInstanceLock("/tmp/alphabet_beacon.pid"),
        hw.SingleInstanceLock("/tmp/calibration_beacon.pid"),
        hw.SingleInstanceLock(PIDFILE),
    ]
    signal.signal(signal.SIGINT, hw._raise_exit)
    signal.signal(signal.SIGTERM, hw._raise_exit)
    backend_attempted = False
    fields = [
        "sequence", "phase", "scheme", "repetition", "duty_percent",
        "duty_label", "voltage_v", "distance_m", "hardware_note",
        "payload_bits", "coded_body_bits", "frame_bits", "bit_seconds",
        "started_utc", "finished_utc", "duration_s", "gap_after_s",
        "target_pulse_us", "pulse_count", "late_starts", "pulse_min_us",
        "pulse_median_us", "pulse_p95_us", "pulse_max_us",
    ]
    try:
        for lock in locks:
            lock.acquire()
        backend_attempted = True
        backend.open()
        require_all_off(backend, config)
        print(f"COIL OFF initial_wait={INITIAL_WAIT_SECONDS:g}s", flush=True)
        sleep(INITIAL_WAIT_SECONDS)
        with open(manifest_path, "x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            output.flush()
            os.fsync(output.fileno())
            for sequence, trial in enumerate(trials, 1):
                frame = protocol.FRAMES[trial.scheme]
                payload = (protocol.FINAL_PAYLOAD_TEXT if trial.phase == "final"
                           else protocol.SHORT_PAYLOAD_TEXT)
                body = frame[len(protocol.SYNC_TEXT):]
                started_utc = utc_stamp()
                started = time.monotonic()
                print(
                    f"EXPERIMENT {sequence:03d}/{len(trials)} START "
                    f"phase={trial.phase} scheme={trial.scheme} "
                    f"rep={trial.repetition} duty={trial.duty_percent:g}% bits={frame}",
                    flush=True,
                )
                stats = transmitter.transmit_frame(frame, trial.duty_percent)
                elapsed = time.monotonic() - started
                summary = stats.summary()
                writer.writerow({
                    "sequence": sequence,
                    "phase": trial.phase,
                    "scheme": trial.scheme,
                    "repetition": trial.repetition,
                    "duty_percent": f"{trial.duty_percent:g}",
                    "duty_label": ("commanded 1%, timing-limited"
                                   if trial.duty_percent == 1 else
                                   f"commanded {trial.duty_percent:g}%"),
                    "voltage_v": f"{VOLTAGE:g}",
                    "distance_m": f"{DISTANCE_M:g}",
                    "hardware_note": note,
                    "payload_bits": payload,
                    "coded_body_bits": body,
                    "frame_bits": frame,
                    "bit_seconds": f"{protocol.BIT_SECONDS:g}",
                    "started_utc": started_utc,
                    "finished_utc": utc_stamp(),
                    "duration_s": f"{elapsed:.6f}",
                    "gap_after_s": f"{GAP_SECONDS:g}",
                    **{
                        key: (f"{value:.3f}" if isinstance(value, float) else value)
                        for key, value in summary.items()
                    },
                })
                output.flush()
                os.fsync(output.fileno())
                require_all_off(backend, config)
                print(
                    f"EXPERIMENT {sequence:03d} DONE duration={elapsed:.3f}s; "
                    f"COIL OFF gap={GAP_SECONDS:g}s",
                    flush=True,
                )
                sleep(GAP_SECONDS)
        print(f"COIL OFF final_wait={FINAL_WAIT_SECONDS:g}s", flush=True)
        sleep(FINAL_WAIT_SECONDS)
        require_all_off(backend, config)
        print(f"FINAL EXPERIMENT COMPLETE manifest={manifest_path}", flush=True)
        return 0
    except (hw.BeaconError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if backend_attempted:
            safe_all_off(backend, config)
            backend.close()
        for lock in reversed(locks):
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
