#!/usr/bin/env python3
"""Monitor completed frames in a growing descending-duty dataset CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np

import coded_protocol as protocol
import layered_decoder
import slnn_decoder

DUTIES = (100.0, 50.0, 25.0, 10.0, 1.0)
TRAIN = "ABCDE"
TEST = "FGHIJ"
SCHEDULE = tuple(
    (duty, letter, "train" if letter in TRAIN else "test")
    for duty, test_letter in zip(DUTIES, TEST)
    for letter in TRAIN + test_letter
)


def metadata(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            output[key] = value
    return output


def utc_seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args()

    info = metadata(args.metadata)
    expected_offset = utc_seconds(info["transmitter_started_utc"]) - utc_seconds(
        info["capture_started_utc"]
    )
    fs = protocol.DEFAULT_SAMPLE_RATE_HZ
    frame_samples = round(protocol.CODED_BITS * protocol.BIT_SECONDS * fs)
    # Wait for ten seconds of the post-frame gap so coherent covariance uses H0.
    required_tail = round(69 * fs)
    fields = [
        "sequence", "phase", "duty_percent", "expected", "start_offset_s",
        "preamble_score", "layered", "successful_layers", "coherent_slnn",
        "coherent_full_header", "coherent_full_letter", "accepted",
    ]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", newline="", buffering=1) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        sequence = 0
        first_start = None
        while sequence < len(SCHEDULE):
            try:
                data = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
            except (OSError, ValueError):
                time.sleep(args.poll)
                continue
            good = np.isfinite(data["x"]) & np.isfinite(data["y"])
            x, y = data["x"][good], data["y"][good]
            predicted = (
                round(expected_offset * fs) if first_start is None
                else first_start + round(sequence * (56 + 15) * fs)
            )
            if len(x) < predicted + required_tail:
                time.sleep(args.poll)
                continue

            # Search only around the scheduled boundary; expected letter is not
            # used for synchronization or inference.
            lead = round(3 * fs)
            lo = max(0, predicted - lead)
            hi = min(len(x), predicted + frame_samples + round(13 * fs))
            sx, sy = x[lo:hi], y[lo:hi]
            st = np.arange(len(sx), dtype=float) / fs
            try:
                channels = layered_decoder.analytic_channels(sx, sy, fs)
                half = round(fs * protocol.HALF_SYMBOL_SECONDS)
                template = protocol.complex_template(protocol.ENCODED_HEADER, half, fs)
                correlation = sum(
                    layered_decoder.sliding_correlation(channel, template)
                    for channel in channels
                )
                expected_local = predicted - lo
                radius = round(3 * fs)
                search_lo = max(0, expected_local - radius)
                search_hi = min(len(correlation), expected_local + radius + 1)
                local_start = search_lo + int(np.argmax(correlation[search_lo:search_hi]))
                r0, r1 = layered_decoder.matched_observations(
                    channels, local_start, fs
                )
                layers = layered_decoder.decode_observations(r0, r1)
                valid_layers = [layer for layer in layers if layer.success]
                selected = (
                    next((layer for layer in valid_layers if layer.layer == "L4"), None)
                    or (max(valid_layers, key=lambda layer: layer.confidence)
                        if valid_layers else max(layers, key=lambda layer: layer.confidence))
                )
                successful_layers = tuple(layer.layer for layer in valid_layers)
                preamble_score = float(correlation[local_start])
                absolute_start = lo + local_start
                coherent = slnn_decoder.coherent_llrs(channels, local_start, fs)
                restricted = slnn_decoder.decode_alphabet(coherent.llrs)
                full = slnn_decoder.decode_full(coherent.llrs)
            except Exception as exc:
                args.status.write_text(f"Frame {sequence + 1}/30 decode waiting: {exc}\n")
                time.sleep(args.poll)
                continue
            if first_start is None:
                first_start = absolute_start

            duty, expected, phase = SCHEDULE[sequence]
            accepted = (
                selected.success
                and selected.letter == expected
                and full.header == protocol.HEADER_BYTE
                and full.letter == expected
                and restricted.letter == expected
            )
            writer.writerow({
                "sequence": sequence + 1,
                "phase": phase,
                "duty_percent": duty,
                "expected": expected,
                "start_offset_s": f"{absolute_start/fs:.3f}",
                "preamble_score": f"{preamble_score:.6f}",
                "layered": selected.letter,
                "successful_layers": ",".join(successful_layers),
                "coherent_slnn": restricted.letter,
                "coherent_full_header": f"0x{full.header:02X}",
                "coherent_full_letter": full.letter,
                "accepted": int(accepted),
            })
            marker = "PASS" if accepted else "FAIL"
            args.status.write_text(
                f"Frame {sequence + 1:02d}/30  ~{expected}@{duty:g}%  {marker}\n"
                f"Layered={selected.letter} "
                f"layers={','.join(successful_layers) or 'none'}  "
                f"preamble={preamble_score:.2f}/2.00\n"
                f"Coherent-SLNN={restricted.letter}  "
                f"full=0x{full.header:02X}/{full.letter}\n"
            )
            sequence += 1
        args.status.write_text(args.status.read_text() + "Dataset decode monitor complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
