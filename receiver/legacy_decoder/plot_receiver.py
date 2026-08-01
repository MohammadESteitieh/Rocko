#!/usr/bin/env python3
"""Offline three-pane plot for the coded alphabet receiver."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import coded_protocol as protocol
import layered_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="captures/receiver.csv")
    parser.add_argument("--save")
    args = parser.parse_args()
    data = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
    good = np.isfinite(data["t"]) & np.isfinite(data["x"]) & np.isfinite(data["y"])
    t, x, y = data["t"][good], data["x"][good], data["y"][good]
    fs = layered_decoder.sample_rate(t)
    channels = layered_decoder.analytic_channels(x, y, fs)
    band_x, band_y = np.real(channels[0]), np.real(channels[1])
    amplitude = np.sqrt(np.abs(channels[0])**2 + np.abs(channels[1])**2)
    result = layered_decoder.decode_capture(t, x, y)
    chosen = result.selected

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(t, x, lw=.8, label="Sensor 1")
    axes[0].plot(t, y, lw=.7, alpha=.7, label="Sensor 2")
    axes[0].set_ylabel("Raw ADC"); axes[0].legend()
    axes[1].plot(t, band_x, lw=.8, label="Sensor 1")
    axes[1].plot(t, band_y, lw=.7, alpha=.7, label="Sensor 2")
    axes[1].set_ylabel("7.25-8.75 Hz bandpass"); axes[1].legend()
    axes[2].plot(t, amplitude, lw=1.0, color="#059669")
    axes[2].axvline(result.start_time, color="#f59e0b", lw=1.4)
    axes[2].set_ylabel("Carrier amplitude"); axes[2].set_xlabel("Time (s)")
    for axis in axes: axis.grid(alpha=.22); axis.margins(x=0)
    fig.suptitle(
        f"Header ~ | Letter {chosen.letter} | Layer {chosen.layer} | "
        f"successful: {', '.join(result.successful_layers)}"
    )
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=160); print(f"Saved {args.save}")
    plt.show()


if __name__ == "__main__": main()
