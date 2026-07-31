#!/usr/bin/env python3
"""Run one safe 12-frame voltage/duty calibration session on QNX.

The operator sets and verifies the bench-supply voltage before invocation. The
script transmits the canonical current ``~A`` Hamming frame three times at each
of 100%, 50%, 25%, and 10% ENB duty in a counterbalanced order. It includes a
15-second initial transmitter-off interval, a 15-second transmitter-off gap
after every frame, and a 5-second final transmitter-off interval.

GPIO27 (ENB), GPIO18 (defensive legacy PWM cleanup), GPIO22 (IN3), and GPIO17
(IN4) are forced low before and after the session, including interruption and
failure paths. Six-volt operation is rejected unless the operator supplies the
explicit opt-in after documenting the L298N logic-power arrangement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import os
import signal
import sys
import time
from typing import Callable

import alphabet_transmitter as alphabet
import duty_pair_test as duty_tx
import transmitter as hw

LETTER = "A"
INITIAL_WAIT_SECONDS = 15.0
GAP_SECONDS = 15.0
FINAL_WAIT_SECONDS = 5.0
ALLOWED_VOLTAGES = (7.0, 8.0, 9.0, 10.0, 11.0)
CONDITIONAL_VOLTAGE = 6.0
DUTY_ROUNDS = (
    (100.0, 50.0, 25.0, 10.0),
    (10.0, 25.0, 50.0, 100.0),
    (50.0, 10.0, 100.0, 25.0),
)
LEGACY_PWM_GPIO = 18
PIDFILE = "/tmp/calibration_beacon.pid"


def calibration_schedule() -> tuple[tuple[int, int, float, str], ...]:
    """Return ``(round, position, duty, letter)`` for the frozen schedule."""
    return tuple(
        (round_index, position, duty, LETTER)
        for round_index, duties in enumerate(DUTY_ROUNDS, 1)
        for position, duty in enumerate(duties, 1)
    )


def parse_duty_sequence(value: str) -> tuple[float, ...]:
    """Parse an explicit one-frame-per-duty exploratory schedule."""
    try:
        duties = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("duty sequence must be comma-separated numbers") from exc
    if not duties or any(not 0 < duty <= 100 for duty in duties):
        raise ValueError("every duty in the sequence must be in (0, 100]")
    if len(set(duties)) != len(duties):
        raise ValueError("duty sequence must not contain duplicates")
    return duties


def exploratory_schedule(
    duties: tuple[float, ...],
) -> tuple[tuple[int, int, float, str], ...]:
    """Return a single declared round without changing the frozen default."""
    return tuple((1, position, duty, LETTER) for position, duty in enumerate(duties, 1))


def validate_voltage(voltage: float, allow_six_volts: bool = False) -> float:
    """Validate the manually measured supply setting against the safety policy."""
    value = float(voltage)
    if value == CONDITIONAL_VOLTAGE:
        if not allow_six_volts:
            raise ValueError(
                "6 V requires --allow-six-volts after the carrier pilot and "
                "L298N logic-power configuration are documented"
            )
        return value
    if value not in ALLOWED_VOLTAGES:
        raise ValueError("voltage must be one of 7, 8, 9, 10, or 11 V")
    return value


def estimated_seconds(
    *,
    initial_wait: float = INITIAL_WAIT_SECONDS,
    gap: float = GAP_SECONDS,
    final_wait: float = FINAL_WAIT_SECONDS,
    schedule: tuple[tuple[int, int, float, str], ...] | None = None,
    bit_seconds: float = duty_tx.BIT_SECONDS,
) -> float:
    """Return total session duration, including a gap after the final frame."""
    for name, value in (
        ("initial wait", initial_wait), ("gap", gap), ("final wait", final_wait)
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if bit_seconds <= 0:
        raise ValueError("bit duration must be positive")
    frame_seconds = duty_tx.FRAME_BITS * bit_seconds
    selected = calibration_schedule() if schedule is None else schedule
    return initial_wait + len(selected) * (frame_seconds + gap) + final_wait


def safe_all_off(backend, config: hw.Config) -> bool:
    """Best-effort low state for every transmitter pin; report full success."""
    success = True
    # Gate the bridge off first, then clear the legacy PWM and polarity inputs.
    for pin in (config.enb_gpio, LEGACY_PWM_GPIO, config.in3_gpio, config.in4_gpio):
        try:
            backend.write_pin(pin, 0)
        except Exception as exc:  # continue clearing the remaining safety pins
            success = False
            print(f"ERROR forcing GPIO{pin} low: {exc}", file=sys.stderr)
    return success


def require_all_off(backend, config: hw.Config) -> None:
    """Force the safe state and fail the session if any pin write fails."""
    if not safe_all_off(backend, config):
        raise hw.BeaconError("could not verify that all transmitter GPIOs were forced low")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voltage", type=float, required=True,
                        help="operator-measured bench-supply voltage")
    parser.add_argument("--distance-m", type=float, required=True,
                        help="rough coil-center to sensor-midpoint distance")
    parser.add_argument("--manifest", help="output CSV path")
    parser.add_argument("--hardware-note", required=True,
                        help="short identifier for the documented hardware configuration")
    parser.add_argument("--allow-six-volts", action="store_true",
                        help="explicit 6 V opt-in after logic-power documentation and pilot")
    parser.add_argument("--initial-wait", type=float, default=INITIAL_WAIT_SECONDS)
    parser.add_argument("--gap", type=float, default=GAP_SECONDS)
    parser.add_argument("--final-wait", type=float, default=FINAL_WAIT_SECONDS)
    parser.add_argument(
        "--duty-sequence",
        help="exploratory comma-separated duties, one ~A frame each; default stays frozen",
    )
    parser.add_argument(
        "--bit-seconds", type=float, choices=(0.5, 1.0, 2.0), default=duty_tx.BIT_SECONDS,
        help="coded-bit duration; 2.0 is frozen, 1.0/0.5 are pilot-only",
    )
    parser.add_argument("--sim", action="store_true", help="record GPIO calls without hardware")
    parser.add_argument("--dry-run", action="store_true", help="validate and print schedule only")
    return parser.parse_args()


def main(sleep: Callable[[float], None] = time.sleep) -> int:
    args = parse_args()
    try:
        voltage = validate_voltage(args.voltage, args.allow_six_volts)
        schedule = (
            exploratory_schedule(parse_duty_sequence(args.duty_sequence))
            if args.duty_sequence is not None
            else calibration_schedule()
        )
        duration = estimated_seconds(
            initial_wait=args.initial_wait,
            gap=args.gap,
            final_wait=args.final_wait,
            schedule=schedule,
            bit_seconds=args.bit_seconds,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.distance_m <= 0:
        print("--distance-m must be positive", file=sys.stderr)
        return 2
    if not args.hardware_note.strip():
        print("--hardware-note must not be blank", file=sys.stderr)
        return 2

    print(
        f"Calibration: {voltage:g} V, distance={args.distance_m:g} m, "
        f"{len(schedule)} frames, estimated {duration / 60:.2f} minutes",
        flush=True,
    )
    print(" ".join(f"R{round_index}:{duty:g}%" for round_index, _, duty, _ in schedule),
          flush=True)
    if args.dry_run:
        return 0

    manifest_path = args.manifest or time.strftime(
        f"calibration_{voltage:g}V_%Y%m%d_%H%M%S.csv"
    )
    config = replace(hw.Config(pidfile_path=PIDFILE), bit_seconds=args.bit_seconds)
    safety_pins = (
        config.enb_gpio, LEGACY_PWM_GPIO, config.in3_gpio, config.in4_gpio
    )
    backend = hw.SimBackend() if args.sim else hw.QnxGpioBackend(
        config.gpio_dev, safety_pins
    )
    driver = hw.CoilDriver(backend, config)
    transmitter = duty_tx.DutyFrameTransmitter(driver, config)
    locks = [] if args.sim else [
        hw.SingleInstanceLock(hw.Config().pidfile_path),
        hw.SingleInstanceLock(alphabet.PIDFILE),
        hw.SingleInstanceLock(PIDFILE),
    ]
    signal.signal(signal.SIGINT, hw._raise_exit)
    signal.signal(signal.SIGTERM, hw._raise_exit)
    backend_attempted = False
    fields = [
        "sequence", "round", "position", "phase", "letter", "duty_percent",
        "voltage_v", "distance_m", "hardware_note", "coded_bits", "started_utc",
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
        print(f"COIL OFF initial_wait={args.initial_wait:g}s", flush=True)
        sleep(args.initial_wait)
        with open(manifest_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for sequence, (round_index, position, duty, letter) in enumerate(schedule, 1):
                bits = alphabet.build_message(letter)
                started_utc = utc_stamp()
                started = time.monotonic()
                print(
                    f"CAL {sequence:02d}/{len(schedule)} START round={round_index} "
                    f"position={position} letter={letter} duty={duty:g}% coded={bits}",
                    flush=True,
                )
                stats = transmitter.transmit_frame(bits, duty)
                elapsed = time.monotonic() - started
                summary = stats.summary()
                writer.writerow({
                    "sequence": sequence,
                    "round": round_index,
                    "position": position,
                    "phase": "calibration",
                    "letter": letter,
                    "duty_percent": duty,
                    "voltage_v": f"{voltage:g}",
                    "distance_m": f"{args.distance_m:g}",
                    "hardware_note": args.hardware_note,
                    "coded_bits": bits,
                    "started_utc": started_utc,
                    "finished_utc": utc_stamp(),
                    "duration_s": f"{elapsed:.6f}",
                    "gap_after_s": f"{args.gap:g}",
                    **{
                        key: (f"{value:.3f}" if isinstance(value, float) else value)
                        for key, value in summary.items()
                    },
                })
                output.flush()
                os.fsync(output.fileno())
                print(
                    f"CAL {sequence:02d} DONE duty={duty:g}% duration={elapsed:.3f}s; "
                    f"COIL OFF gap={args.gap:g}s",
                    flush=True,
                )
                require_all_off(backend, config)
                sleep(args.gap)
        print(f"COIL OFF final_wait={args.final_wait:g}s", flush=True)
        sleep(args.final_wait)
        require_all_off(backend, config)
        print(f"CALIBRATION COMPLETE manifest={manifest_path}", flush=True)
        return 0
    except hw.BeaconError as exc:
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
