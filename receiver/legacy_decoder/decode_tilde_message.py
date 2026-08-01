#!/usr/bin/env python3
"""Offline layered decoder for the coded tilde + capital-letter protocol."""

import argparse
import numpy as np
import layered_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    args = parser.parse_args()
    data = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
    valid = np.isfinite(data["t"]) & np.isfinite(data["x"]) & np.isfinite(data["y"])
    result = layered_decoder.decode_capture(
        data["t"][valid], data["x"][valid], data["y"][valid]
    )
    print(f"Message start:  sample {result.start_index}, t={result.start_time:.6f}s")
    print(f"Header score:  {result.preamble_score:.6f}/2")
    for layer in result.layers:
        print(
            f"{layer.layer:9} {'SUCCESS' if layer.success else 'failed ':7} "
            f"header=0x{layer.header:02X} letter={layer.letter} "
            f"parity={'ok' if layer.parity_ok else 'bad'} confidence={layer.confidence:.3f}"
        )
    print(f"Header:         {chr(result.selected.header)}")
    print(f"Letter:         {result.selected.letter}")
    print(f"Selected layer: {result.selected.layer}")
    print(f"Successful:     {', '.join(result.successful_layers) or 'none'}")


if __name__ == "__main__":
    main()
