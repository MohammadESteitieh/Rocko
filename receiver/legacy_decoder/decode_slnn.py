#!/usr/bin/env python3
"""Decode a captured frame with the optimal codebook-weight SLNN."""

from __future__ import annotations

import argparse
import numpy as np

import coded_protocol as protocol
import layered_decoder
import slnn_decoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--start-time", type=float, help="receiver timestamp of frame start")
    start.add_argument(
        "--start-offset", type=float,
        help="frame start in seconds after the first CSV sample",
    )
    parser.add_argument("--expected", help="optional known letter A-Z, evaluation only")
    method = parser.add_mutually_exclusive_group()
    method.add_argument(
        "--coherent", action="store_true",
        help="legacy magnitude-based coherent sensor combining",
    )
    method.add_argument(
        "--coherent-llr", action="store_true",
        help="analytical covariance-aware coherent Manchester LLRs",
    )
    args = parser.parse_args()

    data = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
    good = np.isfinite(data["t"]) & np.isfinite(data["x"]) & np.isfinite(data["y"])
    t, x, y = data["t"][good], data["x"][good], data["y"][good]
    fs = layered_decoder.sample_rate(t)
    channels = layered_decoder.analytic_channels(x, y, fs)
    if args.start_time is not None:
        requested_start = args.start_time
        source = "provided timestamp"
    elif args.start_offset is not None:
        requested_start = float(t[0] + args.start_offset)
        source = "provided offset"
    else:
        sync = layered_decoder.decode_capture(t, x, y)
        requested_start = sync.start_time
        source = f"automatic preamble (score {sync.preamble_score:.4f}/2)"
    index = int(np.argmin(np.abs(t - requested_start)))
    frame_samples = round(fs * protocol.BIT_SECONDS) * protocol.CODED_BITS
    if index + frame_samples > len(t):
        raise SystemExit("complete frame does not fit after requested start")

    if args.coherent_llr:
        coherent = slnn_decoder.coherent_llrs(channels, index, fs)
        received = coherent.llrs
        print(
            f"Sensor coherence: tone={coherent.tone_coherence:.4f}, "
            f"silence={coherent.silence_coherence:.4f}"
        )
        print("Soft metric: analytical coherent Manchester LLR")
    elif args.coherent:
        coherent = slnn_decoder.coherent_soft_symbols(channels, index, fs)
        received = coherent.symbols
        print(
            f"Sensor coherence: tone={coherent.tone_coherence:.4f}, "
            f"silence={coherent.silence_coherence:.4f}"
        )
        print("Soft metric: legacy coherent magnitude difference")
    else:
        r0, r1 = layered_decoder.matched_observations(channels, index, fs)
        received = slnn_decoder.soft_symbols(r0, r1)
    expected = args.expected.upper() if args.expected else None
    restricted = slnn_decoder.decode_alphabet(received, expected)
    expected_value = None if expected is None else (protocol.HEADER_BYTE << 8) | ord(expected)
    full = slnn_decoder.decode_full(received, expected_value)

    print(f"Start: {t[index]:.6f}s ({source})")
    for result in (restricted, full):
        valid = result.header == protocol.HEADER_BYTE and 65 <= result.letter_byte <= 90
        rank = "n/a" if result.expected_rank is None else str(result.expected_rank)
        print(
            f"{result.scope:12} header=0x{result.header:02X} "
            f"letter={result.letter} valid={valid} score={result.score:.6f} "
            f"margin={result.margin:.6f} expected-rank={rank}"
        )
    print(
        "Note: the cited optimality proof assumes synchronized BPSK over AWGN; "
        "Manchester matched-filter differences on this link do not satisfy that model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
