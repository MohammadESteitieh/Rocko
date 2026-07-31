#!/usr/bin/env python3
"""QNX alphabet transmitter: encoded '~' + A..Z, with 15 s gaps."""

from __future__ import annotations

import argparse
from dataclasses import replace
import signal
import string
import sys
import time

import transmitter as hw

HEADER = 0x7E
GAP_SECONDS = 15.0
PIDFILE = "/tmp/alphabet_beacon.pid"


def byte_bits(value: int) -> str:
    return f"{value:08b}"


def encode_group(nibble: str) -> str:
    """Standard even-parity [p1,p2,d1,p4,d2,d3,d4] Hamming(7,4)."""
    if len(nibble) != 4 or any(bit not in "01" for bit in nibble):
        raise ValueError("group must contain four binary digits")
    d1, d2, d3, d4 = map(int, nibble)
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    return f"{p1}{p2}{d1}{p4}{d2}{d3}{d4}"


def build_message(letter: str) -> str:
    if len(letter) != 1 or letter not in string.ascii_uppercase:
        raise ValueError("letter must be A-Z")
    data = byte_bits(HEADER) + byte_bits(ord(letter))
    return "".join(encode_group(data[index:index + 4]) for index in range(0, 16, 4))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="A", help="first letter A-Z (default A)")
    parser.add_argument("--gap", type=float, default=GAP_SECONDS)
    parser.add_argument("--once", action="store_true", help="send one letter and stop")
    parser.add_argument("--sim", action="store_true", help="do not touch GPIO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = args.start.upper()
    if len(start) != 1 or start not in string.ascii_uppercase or args.gap < 0:
        print("--start must be A-Z and --gap must be non-negative", file=sys.stderr)
        return 2

    config = replace(hw.Config(pidfile_path=PIDFILE), bit_seconds=2.0)
    pins = (config.in3_gpio, config.in4_gpio, config.enb_gpio)
    backend = hw.SimBackend() if args.sim else hw.QnxGpioBackend(config.gpio_dev, pins)
    lock = None if args.sim else hw.SingleInstanceLock(config.pidfile_path)
    driver = hw.CoilDriver(backend, config)
    frame = hw.FrameTransmitter(driver, config)
    signal.signal(signal.SIGINT, hw._raise_exit)
    signal.signal(signal.SIGTERM, hw._raise_exit)

    index = string.ascii_uppercase.index(start)
    try:
        if lock:
            lock.acquire()
        backend.open()
        while True:
            letter = string.ascii_uppercase[index]
            coded = build_message(letter)
            print(
                f"TX header=~ letter={letter} data={byte_bits(HEADER)}{byte_bits(ord(letter))} "
                f"coded={coded}", flush=True,
            )
            frame.transmit_frame(coded)
            print(f"TX complete letter={letter}; coil off; gap={args.gap:g}s", flush=True)
            if args.once:
                break
            time.sleep(args.gap)
            index = (index + 1) % len(string.ascii_uppercase)
        return 0
    except hw.BeaconError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            driver.all_off()
        except Exception as exc:
            print(f"ERROR during coil shutdown: {exc}", file=sys.stderr)
        backend.close()
        if lock:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
