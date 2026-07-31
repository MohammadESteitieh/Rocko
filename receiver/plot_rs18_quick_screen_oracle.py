#!/usr/bin/env python3
"""Plot the explicitly post-hoc timing-oracle RS18 quick-screen diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

METHODS = {
    "primary_xy": ("Two-sensor coherent", "o", "#1f77b4"),
    "sensor_x": ("Sensor x", "s", "#d62728"),
    "sensor_y": ("Sensor y", "^", "#2ca02c"),
    "gao_y_primary": ("Gao: y primary", "D", "#9467bd"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot-data", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    methods = payload["methods"]
    plot_rows = []
    fig, (ax_ber, ax_symbols) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for key, (label, marker, color) in METHODS.items():
        rows = sorted(methods[key], key=lambda row: row["duty"])
        duties = [row["duty"] for row in rows]
        ber = [100.0 * row["raw_bit_errors"] / 90.0 for row in rows]
        symbol_errors = [row["raw_symbol_errors"] for row in rows]
        ax_ber.plot(duties, ber, marker=marker, color=color, linewidth=2,
                    markersize=7, label=label)
        ax_symbols.plot(duties, symbol_errors, marker=marker, color=color,
                        linewidth=2, markersize=7, label=label)
        for row, rate in zip(rows, ber):
            plot_rows.append({"method": key, "label": label, **row,
                              "raw_body_ber_percent": rate})

    ax_ber.set_ylabel("Minimum raw body BER (%)")
    ax_ber.set_ylim(25, 42)
    ax_ber.grid(alpha=0.3)
    ax_ber.legend(ncol=2)
    ax_ber.set_title(
        "RS(18,6) quick screen — post-hoc payload-truth timing oracle\n"
        "Optimistic diagnostic upper bound; not an operational decoder"
    )

    ax_symbols.axhspan(0, 6, color="#2ca02c", alpha=0.10,
                       label="Hard-RS correction region (≤6 symbols)")
    ax_symbols.axhline(6, color="#2ca02c", linestyle="--", linewidth=1.5)
    ax_symbols.set_ylabel("Erroneous RS symbols (of 18)")
    ax_symbols.set_xlabel("Commanded duty (%)")
    ax_symbols.set_ylim(0, 19)
    ax_symbols.set_xticks([10, 30, 40, 50, 100])
    ax_symbols.grid(alpha=0.3)
    ax_symbols.legend(ncol=2)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)

    args.plot_data.parent.mkdir(parents=True, exist_ok=True)
    with args.plot_data.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
