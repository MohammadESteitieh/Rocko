#!/usr/bin/env python3
"""Safe QNX transmitter for the separate 45-frame RS(18,6) experiment.

Nothing energizes without --execute. The frozen schedule sends one prospective
payload per repetition across all nine duties, with cyclic duty-order rotations.
GPIO27, GPIO18, GPIO22, and GPIO17 are forced low before, between, and after
frames, including interruption and failure paths.
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
import rs18_experiment_protocol as protocol
import transmitter as hw

INITIAL_WAIT_SECONDS = 15.0
GAP_SECONDS = 15.0
FINAL_WAIT_SECONDS = 5.0
VOLTAGE = 11.0
DISTANCE_M = 3.0
LEGACY_PWM_GPIO = 18
PIDFILE = "/tmp/rs18_experiment.pid"
DUTY_ROTATIONS = (0, 2, 4, 6, 8)


@dataclass(frozen=True)
class Trial:
    repetition: int
    duty_percent: float
    duty_position: int
    payload_bits: str
    frame_bits: str


def experiment_schedule() -> tuple[Trial, ...]:
    trials: list[Trial] = []
    duties = protocol.DUTIES
    for repetition, offset in enumerate(DUTY_ROTATIONS, 1):
        order = duties[offset:] + duties[:offset]
        trials.extend(
            Trial(repetition, duty, position, protocol.PAYLOADS[repetition],
                  protocol.FRAMES[repetition])
            for position, duty in enumerate(order, 1)
        )
    return tuple(trials)


def estimated_seconds() -> float:
    return (
        INITIAL_WAIT_SECONDS
        + len(experiment_schedule()) * (protocol.FRAME_BITS * protocol.BIT_SECONDS
                                        + GAP_SECONDS)
        + FINAL_WAIT_SECONDS
    )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def safe_all_off(backend, config: hw.Config) -> bool:
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
    parser.add_argument("--manifest")
    parser.add_argument("--hardware-note", required=True)
    parser.add_argument("--voltage", type=float, default=VOLTAGE)
    parser.add_argument("--distance-m", type=float, default=DISTANCE_M)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sim", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None,
         sleep: Callable[[float], None] = time.sleep) -> int:
    args = parse_args(argv)
    note = args.hardware_note.strip()
    if not note or "\n" in note or "\r" in note:
        print("--hardware-note must be non-empty and single-line", file=sys.stderr)
        return 2
    if args.voltage != VOLTAGE or args.distance_m != DISTANCE_M:
        print("RS18 experiment is frozen at exactly 11 V and 3 m", file=sys.stderr)
        return 2
    schedule = experiment_schedule()
    print(
        f"RS(18,6) experiment: {len(schedule)} frames, 2 coded bits/s, "
        f"estimate={estimated_seconds() / 60:.2f} minutes",
        flush=True,
    )
    for sequence, trial in enumerate(schedule, 1):
        suffix = " timing-limited" if trial.duty_percent == 1 else ""
        print(
            f"{sequence:02d} rep={trial.repetition} duty={trial.duty_percent:g}% "
            f"position={trial.duty_position}{suffix} payload={trial.payload_bits}",
            flush=True,
        )
    if not args.execute:
        print("DRY RUN: add --execute only after full physical preflight")
        return 0

    manifest_path = args.manifest or time.strftime("rs18-experiment-%Y%m%d_%H%M%S.csv")
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
        hw.SingleInstanceLock("/tmp/final_experiment.pid"),
        hw.SingleInstanceLock(PIDFILE),
    ]
    signal.signal(signal.SIGINT, hw._raise_exit)
    signal.signal(signal.SIGTERM, hw._raise_exit)
    backend_attempted = False
    fields = [
        "sequence", "scheme", "repetition", "duty_position", "duty_percent",
        "duty_label", "voltage_v", "distance_m", "hardware_note",
        "payload_derivation", "payload_bits", "data_symbols", "parity_symbols",
        "coded_body_bits", "frame_bits", "bit_seconds", "started_utc",
        "finished_utc", "duration_s", "gap_after_s", "target_pulse_us",
        "pulse_count", "late_starts", "pulse_min_us", "pulse_median_us",
        "pulse_p95_us", "pulse_max_us",
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
            for sequence, trial in enumerate(schedule, 1):
                started_utc, started = utc_stamp(), time.monotonic()
                print(
                    f"RS18 {sequence:02d}/{len(schedule)} START rep={trial.repetition} "
                    f"duty={trial.duty_percent:g}% bits={trial.frame_bits}",
                    flush=True,
                )
                stats = transmitter.transmit_frame(trial.frame_bits, trial.duty_percent)
                elapsed = time.monotonic() - started
                summary = stats.summary()
                writer.writerow({
                    "sequence": sequence, "scheme": "rs18_6",
                    "repetition": trial.repetition,
                    "duty_position": trial.duty_position,
                    "duty_percent": f"{trial.duty_percent:g}",
                    "duty_label": ("commanded 1%, timing-limited"
                                   if trial.duty_percent == 1 else
                                   f"commanded {trial.duty_percent:g}%"),
                    "voltage_v": f"{VOLTAGE:g}", "distance_m": f"{DISTANCE_M:g}",
                    "hardware_note": note,
                    "payload_derivation": protocol.PAYLOAD_DERIVATION,
                    "payload_bits": trial.payload_bits,
                    "data_symbols": " ".join(map(str, protocol.CODEWORDS[trial.repetition][:6])),
                    "parity_symbols": " ".join(map(str, protocol.CODEWORDS[trial.repetition][6:])),
                    "coded_body_bits": trial.frame_bits[len(protocol.SYNC_TEXT):],
                    "frame_bits": trial.frame_bits,
                    "bit_seconds": f"{protocol.BIT_SECONDS:g}",
                    "started_utc": started_utc, "finished_utc": utc_stamp(),
                    "duration_s": f"{elapsed:.6f}", "gap_after_s": f"{GAP_SECONDS:g}",
                    **{key: (f"{value:.3f}" if isinstance(value, float) else value)
                       for key, value in summary.items()},
                })
                output.flush()
                os.fsync(output.fileno())
                require_all_off(backend, config)
                print(
                    f"RS18 {sequence:02d} DONE duration={elapsed:.3f}s; "
                    f"COIL OFF gap={GAP_SECONDS:g}s",
                    flush=True,
                )
                sleep(GAP_SECONDS)
        print(f"COIL OFF final_wait={FINAL_WAIT_SECONDS:g}s", flush=True)
        sleep(FINAL_WAIT_SECONDS)
        require_all_off(backend, config)
        print(f"RS18 EXPERIMENT COMPLETE manifest={manifest_path}", flush=True)
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
