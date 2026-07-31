#!/usr/bin/env python3
"""Finite ENB-duty diagnostics using the verified GPIO27 transmitter path.

The default remains the paired 100%/1% A-E diagnostic. ``--dataset`` sends a
30-frame descending-duty dataset: A-E training frames plus one held-out F-J
frame at each of 100%, 50%, 25%, 10%, and 1%. Every frame uses the current
56-second Hamming alphabet protocol and a 15-second coil-off gap. Partial duty
is applied inside each 62.5 ms polarity half-cycle of the 8 Hz tone.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
import os
import signal
import statistics
import sys
import time
from typing import Callable

import alphabet_transmitter as alphabet
import hardware as hw

LETTERS = "ABCDE"
DUTIES = (100.0, 1.0)
DATASET_DUTIES = (100.0, 50.0, 25.0, 10.0, 1.0)
HELD_OUT_LETTERS = "FGHIJ"
GAP_SECONDS = 15.0
ALPHABET_PIDFILE = "/tmp/alphabet_beacon.pid"
FRAME_BITS = 28
BIT_SECONDS = 2.0  # current 0.5-coded-bit/s alphabet contract


def test_schedule() -> tuple[tuple[float, str], ...]:
    """Pair each letter's baseline and low-duty frame to limit time drift."""
    return tuple((duty, letter) for letter in LETTERS for duty in DUTIES)


def dataset_schedule() -> tuple[tuple[float, str], ...]:
    """Five A-E training frames plus one untouched test frame per duty."""
    return tuple(
        (duty, letter)
        for duty, test_letter in zip(DATASET_DUTIES, HELD_OUT_LETTERS)
        for letter in LETTERS + test_letter
    )


def requested_schedule(
    single_duty: float | None = None, letter: str = "A", *, dataset: bool = False
) -> tuple[tuple[float, str], ...]:
    if dataset:
        if single_duty is not None:
            raise ValueError("--dataset and --single-duty are mutually exclusive")
        return dataset_schedule()
    if single_duty is None:
        return test_schedule()
    letter = letter.upper()
    if len(letter) != 1 or letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise ValueError("--letter must be A-Z")
    if not 0 < single_duty <= 100:
        raise ValueError("--single-duty must be in (0, 100]")
    return ((float(single_duty), letter),)


def estimated_seconds(
    gap: float = GAP_SECONDS,
    schedule: tuple[tuple[float, str], ...] | None = None,
) -> float:
    frames = len(schedule if schedule is not None else test_schedule())
    return frames * FRAME_BITS * BIT_SECONDS + max(0, frames - 1) * gap


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass
class PulseStats:
    target_seconds: float
    widths: list[float] = field(default_factory=list)
    late_starts: int = 0

    def summary(self) -> dict[str, float | int]:
        microseconds = [width * 1e6 for width in self.widths]
        return {
            "target_pulse_us": self.target_seconds * 1e6,
            "pulse_count": len(microseconds),
            "late_starts": self.late_starts,
            "pulse_min_us": min(microseconds, default=float("nan")),
            "pulse_median_us": statistics.median(microseconds) if microseconds else float("nan"),
            "pulse_p95_us": percentile(microseconds, 0.95),
            "pulse_max_us": max(microseconds, default=float("nan")),
        }


class DutyFrameTransmitter:
    """Normal 100% frames plus diagnostic ENB-gated partial-duty frames."""

    def __init__(
        self,
        driver: hw.CoilDriver,
        config: hw.Config,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        config.validate()
        self.driver = driver
        self.config = config
        self.monotonic = monotonic
        self.sleep = sleep
        self.full_duty = hw.FrameTransmitter(driver, config, monotonic, sleep)

    def _sleep_until(self, deadline: float) -> None:
        delay = deadline - self.monotonic()
        if delay > 0:
            self.sleep(delay)

    def _pulsed_tone(
        self, symbol_start: float, symbol_deadline: float, duty: float, stats: PulseStats
    ) -> None:
        half_cycle = 1.0 / (2.0 * self.config.carrier_hz)
        pulse_seconds = half_cycle * duty / 100.0
        self.driver.enable(False)
        edge = 0
        while symbol_start + edge * half_cycle < symbol_deadline:
            edge_start = symbol_start + edge * half_cycle
            edge_deadline = min(edge_start + half_cycle, symbol_deadline)
            self._sleep_until(edge_start)
            # Change polarity only while ENB is low, then gate on for the
            # requested width. The logged width measures software timing from
            # completion of ON to invocation of OFF; a scope is still required
            # to establish the physical pin/bridge pulse width.
            self.driver.set_polarity(edge % 2 == 0)
            self.driver.enable(True)
            enabled_at = self.monotonic()
            if enabled_at > edge_start + pulse_seconds:
                stats.late_starts += 1
            self._sleep_until(min(enabled_at + pulse_seconds, edge_deadline))
            off_requested_at = self.monotonic()
            self.driver.enable(False)
            stats.widths.append(max(0.0, off_requested_at - enabled_at))
            self._sleep_until(edge_deadline)
            edge += 1

    def transmit_frame(self, bits: str, duty: float) -> PulseStats:
        if not 0 < duty <= 100:
            raise ValueError("duty must be in (0, 100]")
        half_cycle = 1.0 / (2.0 * self.config.carrier_hz)
        stats = PulseStats(half_cycle * duty / 100.0)
        if duty == 100:
            self.full_duty.transmit_frame(bits)
            return stats

        symbols = hw.regular_manchester(bits)
        half_seconds = self.config.half_symbol_seconds
        start = self.monotonic()
        try:
            for index, tone in enumerate(symbols):
                symbol_start = start + index * half_seconds
                deadline = symbol_start + half_seconds
                if tone:
                    self._pulsed_tone(symbol_start, deadline, duty, stats)
                else:
                    self.driver.enable(False)
                    self._sleep_until(deadline)
        finally:
            self.driver.all_off()
        return stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", type=float, default=GAP_SECONDS)
    parser.add_argument("--manifest", help="output CSV (default: timestamped file here)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--single-duty", type=float,
        help="send one diagnostic frame at this commanded duty instead of the pair test",
    )
    mode.add_argument(
        "--dataset", action="store_true",
        help="descending 30-frame A-E training plus held-out F-J dataset",
    )
    parser.add_argument("--letter", default="A", help="letter for --single-duty (default A)")
    parser.add_argument("--sim", action="store_true", help="record GPIO calls without hardware")
    parser.add_argument("--dry-run", action="store_true", help="print schedule and exit")
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    args = parse_args()
    if args.gap < 0:
        print("--gap must be non-negative", file=sys.stderr)
        return 2
    try:
        schedule = requested_schedule(args.single_duty, args.letter, dataset=args.dataset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    duration = estimated_seconds(args.gap, schedule)
    print(
        f"Diagnostic schedule: {len(schedule)} frames, {args.gap:g}s gaps, "
        f"estimated {duration/60:.2f} minutes",
        flush=True,
    )
    print(" ".join(f"~{letter}@{duty:g}%" for duty, letter in schedule), flush=True)
    if args.dry_run:
        return 0

    manifest_path = args.manifest or time.strftime("duty-pair-%Y%m%d_%H%M%S.csv")
    # Preserve the exact GPIO27 direct-drive path used by run-alphabet.sh;
    # only add ENB pulse gating for requested duties below 100%.
    config = replace(
        hw.Config(pidfile_path=ALPHABET_PIDFILE), bit_seconds=BIT_SECONDS
    )
    pins = (config.in3_gpio, config.in4_gpio, config.enb_gpio)
    backend = hw.SimBackend() if args.sim else hw.QnxGpioBackend(config.gpio_dev, pins)
    driver = hw.CoilDriver(backend, config)
    transmitter = DutyFrameTransmitter(driver, config)
    locks = [] if args.sim else [
        hw.SingleInstanceLock(hw.Config().pidfile_path),
        hw.SingleInstanceLock(ALPHABET_PIDFILE),
    ]
    opened = False
    signal.signal(signal.SIGINT, hw._raise_exit)
    signal.signal(signal.SIGTERM, hw._raise_exit)

    fields = [
        "sequence", "phase", "letter", "duty_percent", "coded_bits", "started_utc",
        "finished_utc", "duration_s", "target_pulse_us", "pulse_count",
        "late_starts", "pulse_min_us", "pulse_median_us", "pulse_p95_us",
        "pulse_max_us",
    ]
    try:
        for lock in locks:
            lock.acquire()
        backend.open()
        opened = True
        with open(manifest_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for sequence, (duty, letter) in enumerate(schedule, 1):
                bits = alphabet.build_message(letter)
                started_utc = utc_stamp()
                started = time.monotonic()
                print(
                    f"TEST {sequence:02d}/{len(schedule)} START header=~ letter={letter} "
                    f"duty={duty:g}% coded={bits}",
                    flush=True,
                )
                stats = transmitter.transmit_frame(bits, duty)
                elapsed = time.monotonic() - started
                summary = stats.summary()
                writer.writerow({
                    "sequence": sequence,
                    "phase": "train" if letter in LETTERS else "test",
                    "letter": letter,
                    "duty_percent": duty,
                    "coded_bits": bits,
                    "started_utc": started_utc,
                    "finished_utc": utc_stamp(),
                    "duration_s": f"{elapsed:.6f}",
                    **{key: (f"{value:.3f}" if isinstance(value, float) else value)
                       for key, value in summary.items()},
                })
                output.flush()
                os.fsync(output.fileno())
                print(
                    f"TEST {sequence:02d} DONE letter={letter} duty={duty:g}% "
                    f"duration={elapsed:.3f}s pulses={summary['pulse_count']} "
                    f"median={summary['pulse_median_us']:.1f}us "
                    f"p95={summary['pulse_p95_us']:.1f}us late={summary['late_starts']}",
                    flush=True,
                )
                if sequence < len(schedule):
                    print(f"COIL OFF gap={args.gap:g}s", flush=True)
                    time.sleep(args.gap)
        print(f"DIAGNOSTIC COMPLETE manifest={manifest_path}", flush=True)
        return 0
    except hw.BeaconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if opened:
            try:
                driver.all_off()
            except Exception as exc:
                print(f"ERROR during coil shutdown: {exc}", file=sys.stderr)
            backend.close()
        for lock in reversed(locks):
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
