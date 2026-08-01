#!/usr/bin/env python3
"""Offline coherent-GNB-SLNN decoder for a recorded t,x,y frame."""

from __future__ import annotations

import argparse
import numpy as np

import coded_protocol as protocol
import hybrid_decoder
import layered_decoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--start-time", type=float)
    start.add_argument("--start-offset", type=float)
    parser.add_argument("--model", help="fitted GNB .npz; default is provisional synthetic model")
    parser.add_argument("--expected", help="known A-Z letter, evaluation only")
    args = parser.parse_args()

    data = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
    good = np.isfinite(data["t"]) & np.isfinite(data["x"]) & np.isfinite(data["y"])
    t, x, y = data["t"][good], data["x"][good], data["y"][good]
    fs = layered_decoder.sample_rate(t)
    channels = layered_decoder.analytic_channels(x, y, fs)
    requested = args.start_time if args.start_time is not None else t[0] + args.start_offset
    index = int(np.argmin(np.abs(t - requested)))
    model = hybrid_decoder.GaussianNaiveBayes.load(args.model) if args.model else None
    result = hybrid_decoder.decode(channels, index, fs, model, args.expected)

    expected = protocol.encode_message(args.expected.upper()) if args.expected else None
    errors = "n/a"
    if expected is not None:
        errors = str(int(np.count_nonzero(np.asarray(result.hard_bits) != expected)))
    print(f"Start: {t[index]:.6f}s")
    print(
        f"Coherence: tone={result.tone_coherence:.4f} "
        f"silence={result.silence_coherence:.4f}"
    )
    print(f"GNB hard-bit errors: {errors}/{protocol.CODED_BITS if expected is not None else 'n/a'}")
    for decoded in (result.restricted, result.full):
        rank = "n/a" if decoded.expected_rank is None else decoded.expected_rank
        print(
            f"{decoded.scope:12} header=0x{decoded.header:02X} "
            f"letter={decoded.letter} margin={decoded.margin:.6f} "
            f"expected-rank={rank}"
        )
    print(f"Accepted: {result.accepted}")
    print("Model: " + (args.model or "provisional synthetic GNB"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
