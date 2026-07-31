#!/usr/bin/env python3
"""Finite Hamming(7,4), half-baud, hardware-PWM training/test sweep.

Training sends A-E once at each of six randomized duty levels (30 frames).
Testing sends held-out letters F-J at the lowest level (5 frames). GPIO18 must
be wired to L298N ENB; GPIO27 is not a hardware-PWM pin. Nothing is transmitted
unless --execute is supplied.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import signal
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import transmitter as legacy

HEADER = 0x7E
TRAIN_LETTERS = "ABCDE"
TEST_LETTERS = "FGHIJ"
DUTY_LEVELS = (100.0, 18.0, 5.0, 1.3, 0.36, 0.1)
BIT_SECONDS = 2.0                 # 0.5 coded bits/s (50% of previous rate)
HALF_SYMBOL_SECONDS = BIT_SECONDS / 2
CARRIER_HZ = 8.0
PWM_FREQUENCY_HZ = 2 * CARRIER_HZ
PWM_RANGE = 1024
PWM_MAX_COUNT = PWM_RANGE - 1  # QNX ChangeDutyCycleAbs requires value < range
GAP_SECONDS = 15.0
FRAME_BITS = 28
IN3_GPIO = 22
IN4_GPIO = 17
ENB_PWM_GPIO = 18                 # physical rewire from GPIO27 is required
PIDFILES = ("/tmp/beacon.pid", "/tmp/alphabet_beacon.pid")


def byte_bits(value: int) -> str:
    return f"{value:08b}"


def hamming_group(nibble: str) -> str:
    """Standard even-parity [p1,p2,d1,p4,d2,d3,d4] Hamming(7,4)."""
    if len(nibble) != 4 or any(bit not in "01" for bit in nibble):
        raise ValueError("group must contain four binary digits")
    d1, d2, d3, d4 = map(int, nibble)
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    return f"{p1}{p2}{d1}{p4}{d2}{d3}{d4}"


def build_message(letter: str) -> str:
    letter = letter.upper()
    if len(letter) != 1 or letter not in string.ascii_uppercase:
        raise ValueError("letter must be A-Z")
    data = byte_bits(HEADER) + byte_bits(ord(letter))
    return "".join(hamming_group(data[i:i + 4]) for i in range(0, 16, 4))


def duty_count(duty: float) -> int:
    if not 0 <= duty <= 100:
        raise ValueError("duty must be in [0,100]")
    if duty == 0:
        return 0
    return max(1, min(PWM_MAX_COUNT, round(duty * PWM_RANGE / 100)))


def actual_duty(duty: float) -> float:
    # Nominal resolution reported for metadata; received SNR is ground truth.
    return 100.0 if duty >= 100 else 100.0 * duty_count(duty) / PWM_RANGE


@dataclass(frozen=True)
class Trial:
    phase: str
    letter: str
    requested_duty: float


def training_schedule(seed: int = 2707) -> list[Trial]:
    trials = [
        Trial("train", letter, duty)
        for duty in DUTY_LEVELS for letter in TRAIN_LETTERS
    ]
    random.Random(seed).shuffle(trials)
    return trials


def test_schedule() -> list[Trial]:
    return [Trial("test", letter, DUTY_LEVELS[-1]) for letter in TEST_LETTERS]


def requested_schedule(phase: str, seed: int = 2707) -> list[Trial]:
    if phase == "train":
        return training_schedule(seed)
    if phase == "test":
        return test_schedule()
    if phase == "all":
        return training_schedule(seed) + test_schedule()
    raise ValueError("phase must be train, test, or all")


def estimated_seconds(trials: Sequence[Trial], gap: float = GAP_SECONDS) -> float:
    return len(trials) * FRAME_BITS * BIT_SECONDS + max(0, len(trials) - 1) * gap


class HardwarePwmCoil:
    """QNX rpi_gpio hardware PWM on ENB plus ordinary polarity GPIOs."""

    def __init__(self, enb_gpio: int = ENB_PWM_GPIO):
        if enb_gpio not in (12, 13, 18, 19):
            raise ValueError("ENB must use hardware-PWM GPIO 12, 13, 18, or 19")
        self.enb_gpio = enb_gpio
        self.GPIO = None
        self.pwm = None

    def open(self) -> None:
        import rpi_gpio as GPIO  # QNX-only extension
        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        for pin in (IN3_GPIO, IN4_GPIO, self.enb_gpio):
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        self.pwm = GPIO.PWM(self.enb_gpio, PWM_FREQUENCY_HZ, GPIO.PWM.MODE_MS)
        self.pwm.start(0)

    def set_polarity(self, forward: bool) -> None:
        self.GPIO.output(IN3_GPIO, self.GPIO.HIGH if forward else self.GPIO.LOW)
        self.GPIO.output(IN4_GPIO, self.GPIO.LOW if forward else self.GPIO.HIGH)

    def set_duty(self, duty: float) -> None:
        # The QNX percentage API correctly handles the hardware channel's
        # private range, including 100%. ChangeDutyCycleAbs is not portable
        # across the RP1 PWM range selected by the resource manager.
        self.pwm.ChangeDutyCycle(float(duty))

    def all_off(self) -> None:
        if self.GPIO is None:
            return
        if self.pwm is not None:
            try:
                self.pwm.ChangeDutyCycleAbs(0)
            except Exception:
                pass
        self.GPIO.setup(IN3_GPIO, self.GPIO.OUT)
        self.GPIO.setup(IN4_GPIO, self.GPIO.OUT)
        self.GPIO.output(IN3_GPIO, self.GPIO.LOW)
        self.GPIO.output(IN4_GPIO, self.GPIO.LOW)

    def close(self) -> None:
        if self.GPIO is None:
            return
        # PWM.stop() closes the extension's shared /dev/gpio/msg descriptor;
        # subsequent extension calls fail with EBADF. Stop after requesting
        # zero duty, then use fresh per-pin resource-manager file descriptors
        # to leave every physical pin unambiguously configured OUT and LOW.
        self.all_off()
        if self.pwm is not None:
            self.pwm.stop()
        for pin in (self.enb_gpio, IN3_GPIO, IN4_GPIO):
            path = f"/dev/gpio/{pin}"
            for command in (b"out", b"off"):
                fd = os.open(path, os.O_WRONLY)
                try:
                    os.write(fd, command)
                finally:
                    os.close(fd)
        self.pwm = None
        self.GPIO = None


class SimCoil:
    def __init__(self):
        self.duties: list[float] = []
        self.polarities: list[bool] = []
        self.closed = False

    def open(self) -> None:
        pass

    def set_polarity(self, forward: bool) -> None:
        self.polarities.append(forward)

    def set_duty(self, duty: float) -> None:
        self.duties.append(duty)

    def all_off(self) -> None:
        self.duties.append(0.0)

    def close(self) -> None:
        self.all_off()
        self.closed = True


class FrameTransmitter:
    def __init__(
        self, coil, monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.coil = coil
        self.monotonic = monotonic
        self.sleep = sleep

    def _sleep_until(self, deadline: float) -> None:
        delay = deadline - self.monotonic()
        if delay > 0:
            self.sleep(delay)

    def transmit(self, bits: str, duty: float) -> None:
        symbols = legacy.regular_manchester(bits)
        carrier_half = 1.0 / (2 * CARRIER_HZ)
        edges_per_tone = round(HALF_SYMBOL_SECONDS / carrier_half)
        start = self.monotonic()
        try:
            for symbol_index, tone in enumerate(symbols):
                symbol_start = start + symbol_index * HALF_SYMBOL_SECONDS
                deadline = symbol_start + HALF_SYMBOL_SECONDS
                self._sleep_until(symbol_start)
                if not tone:
                    self.coil.set_duty(0)
                    self.coil.all_off()
                    self._sleep_until(deadline)
                    continue
                for edge in range(edges_per_tone):
                    self._sleep_until(symbol_start + edge * carrier_half)
                    self.coil.set_polarity(edge % 2 == 0)
                    if edge == 0:
                        self.coil.set_duty(duty)
                self._sleep_until(deadline)
                self.coil.set_duty(0)
        finally:
            self.coil.all_off()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "test", "all"), default="all")
    parser.add_argument("--seed", type=int, default=2707)
    parser.add_argument("--gap", type=float, default=GAP_SECONDS)
    parser.add_argument("--manifest")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument(
        "--execute", action="store_true",
        help="required to energize GPIO; without it only print the schedule",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    args = parse_args()
    if args.gap < 0:
        print("--gap must be non-negative", file=sys.stderr)
        return 2
    trials = requested_schedule(args.phase, args.seed)
    duration = estimated_seconds(trials, args.gap)
    print(
        f"Hamming sweep: phase={args.phase}, frames={len(trials)}, "
        f"bit-rate={1/BIT_SECONDS:g} coded bit/s, estimate={duration/60:.2f} min"
    )
    for index, trial in enumerate(trials, 1):
        print(
            f"{index:02d} {trial.phase:<5} ~{trial.letter} "
            f"requested={trial.requested_duty:g}% actual={actual_duty(trial.requested_duty):.6f}%"
        )
    if not args.execute:
        print("DRY RUN: add --execute after GPIO18 is wired to ENB")
        return 0

    manifest = Path(args.manifest or time.strftime("hamming-sweep-%Y%m%d_%H%M%S.csv"))
    coil = SimCoil() if args.sim else HardwarePwmCoil()
    transmitter = FrameTransmitter(coil)
    locks = [] if args.sim else [legacy.SingleInstanceLock(path) for path in PIDFILES]
    signal.signal(signal.SIGINT, legacy._raise_exit)
    signal.signal(signal.SIGTERM, legacy._raise_exit)
    opened = False
    fields = [
        "sequence", "phase", "letter", "requested_duty_percent",
        "pwm_count", "actual_duty_percent", "coded_bits", "started_utc",
        "finished_utc", "duration_s",
    ]
    try:
        for lock in locks:
            lock.acquire()
        coil.open()
        opened = True
        with manifest.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for index, trial in enumerate(trials, 1):
                bits = build_message(trial.letter)
                started_utc, started = utc_stamp(), time.monotonic()
                print(
                    f"TX {index:02d}/{len(trials)} {trial.phase} ~{trial.letter} "
                    f"duty={actual_duty(trial.requested_duty):.6f}% bits={bits}",
                    flush=True,
                )
                transmitter.transmit(bits, trial.requested_duty)
                elapsed = time.monotonic() - started
                writer.writerow({
                    "sequence": index,
                    "phase": trial.phase,
                    "letter": trial.letter,
                    "requested_duty_percent": trial.requested_duty,
                    "pwm_count": duty_count(trial.requested_duty),
                    "actual_duty_percent": f"{actual_duty(trial.requested_duty):.9f}",
                    "coded_bits": bits,
                    "started_utc": started_utc,
                    "finished_utc": utc_stamp(),
                    "duration_s": f"{elapsed:.6f}",
                })
                output.flush()
                os.fsync(output.fileno())
                if index < len(trials):
                    coil.all_off()
                    time.sleep(args.gap)
        print(f"SWEEP COMPLETE manifest={manifest}")
        return 0
    except (legacy.BeaconError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if opened:
            try:
                coil.close()
            except Exception as exc:
                print(f"ERROR during coil shutdown: {exc}", file=sys.stderr)
        for lock in reversed(locks):
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
