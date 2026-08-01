#!/usr/bin/env python3
"""Rocko receiver — one command to launch the whole surface station.

    python3 receiver/legacy_decoder/rocko_receiver.py
    python3 receiver/legacy_decoder/rocko_receiver.py -p /dev/cu.usbmodem1201
    python3 receiver/legacy_decoder/rocko_receiver.py --replay captures/trial.csv

It finds the USB serial port, opens the live dashboard (raw / bandpass /
carrier-amplitude panes), logs every numbered event to file and on screen, and
auto-decodes each beacon frame in-process against the frozen contract. One
Ctrl+C (or closing the window) stops everything and flushes the capture.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

RECEIVER_DIR = Path(__file__).resolve().parents[1]
if str(RECEIVER_DIR) not in sys.path:
    sys.path.insert(0, str(RECEIVER_DIR))

from eventlog import EventLog
from live_receiver import MIN_PREAMBLE_CONFIDENCE, LiveReceiver
import coded_protocol as protocol
from coded_protocol import BANDWIDTH_HZ, CARRIER_HZ, DEFAULT_SAMPLE_RATE_HZ
from serial_source import autodetect_port, list_candidate_ports

BANNER = r"""
  ____   ___   ____ _  _____
 |  _ \ / _ \ / ___| |/ / _ \    cave explorer safety beacon
 | |_) | | | | |   | ' / | | |   surface receiver
 |  _ <| |_| | |___| . \ |_| |   silence is the alarm
 |_| \_\\___/ \____|_|\_\___/
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-p", "--port", help="serial port (auto-detected if omitted)")
    parser.add_argument("--replay", help="re-stream a recorded t,x,y CSV (no hardware)")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    parser.add_argument("-b", "--baud", type=int, default=115200)
    parser.add_argument("-o", "--output", help="capture CSV path (default: captures/live_<ts>.csv)")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--carrier", type=float, default=CARRIER_HZ)
    parser.add_argument("--bandwidth", type=float, default=BANDWIDTH_HZ)
    parser.add_argument(
        "--bit-seconds", type=float, default=protocol.BIT_SECONDS,
        help="coded-bit duration (default: 2.0 s)",
    )
    parser.add_argument("--silence", type=float, default=5.0,
                        help="seconds of silence that end a beacon and trigger decode")
    parser.add_argument("--tone-confirm", type=float, default=0.3)
    parser.add_argument("--min-separation", type=float, default=2.0)
    parser.add_argument("--min-confidence", type=float, default=MIN_PREAMBLE_CONFIDENCE,
                        help="reject decodes below this preamble score (0-2 scale)")
    parser.add_argument("--plot-seconds", type=float, default=90.0)
    parser.add_argument("--stop-after-decode", action="store_true")
    return parser.parse_args()


def resolve_port(args) -> bool:
    """Fill args.port when running against hardware. Returns False to abort."""
    if args.replay or args.port:
        return True
    detected = autodetect_port()
    if detected:
        args.port = detected
        print(f"  auto-detected serial port: {detected}")
        return True
    candidates = list_candidate_ports()
    print("  no single USB serial port found.")
    if candidates:
        print("  candidates:")
        for port in candidates:
            print(f"    - {port}")
        print("  re-run with:  -p <port>")
    else:
        print("  plug in the Pico, or demo with:  --replay captures/<file>.csv")
    return False


def main() -> int:
    args = parse_args()
    try:
        protocol.configure_bit_seconds(args.bit_seconds)
    except ValueError as exc:
        print(f"  invalid timing: {exc}", file=sys.stderr)
        return 2
    print(BANNER)
    if not resolve_port(args):
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.output:
        args.output = str(Path("captures") / f"live_{stamp}.csv")
    log = EventLog(Path("captures") / f"rocko_{stamp}.log")

    try:
        app = LiveReceiver(args, event_log=log)
    except Exception as exc:
        log.emit("ERROR", f"startup failed: {exc}")
        log.close()
        print(f"  startup failed: {exc}", file=sys.stderr)
        return 1

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n  interrupted — shutting down")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
