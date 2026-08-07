#!/usr/bin/env python3
"""Open the accepted RS18 signal in a standard Matplotlib pan/zoom window."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "receiver" / "legacy_decoder"),
]

import analyze_final_experiment as base  # noqa: E402
import analyze_rs18_experiment as rs18  # noqa: E402
import export_rs18_table as exporter  # noqa: E402
from interactive_run_viewer import (  # noqa: E402
    ACCEPTED_RUN,
    DEFAULT_ANALYSIS,
    FROZEN_SHA256,
    verify_frozen_file,
)


def read_analysis(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 45 or {int(row["sequence"]) for row in rows} != set(range(1, 46)):
        raise ValueError("analysis must contain sequences 1 through 45")
    return rows


def build_plot(*, capture: Path, metadata: Path, analysis: Path,
               raw_counts: bool = False) -> plt.Figure:
    for key, path in (
        ("capture", capture), ("metadata", metadata), ("analysis", analysis)
    ):
        verify_frozen_file(path, FROZEN_SHA256[key])
    rows = read_analysis(analysis)
    t, x, y = base.load_capture(capture)
    fs = base.sample_rate(t)
    relative_time = np.asarray(t, float) - float(t[0])

    if raw_counts:
        plot_x, plot_y = np.asarray(x, float), np.asarray(y, float)
        unit = "Raw ADC counts"
    else:
        info = base.metadata(metadata)
        rs18.validate_metadata(info)
        off_start, off_stop = rs18.prelaunch_indices(info, fs, len(t))
        means, scales = rs18.fit_off_rms(x, y, off_start, off_stop)
        plot_x = (np.asarray(x, float) - means[0]) / scales[0]
        plot_y = (np.asarray(y, float) - means[1]) / scales[1]
        unit = "Prelaunch-off RMS units"

    figure, axes = plt.subplots(
        2, 1, figsize=(15, 8.5), sharex=True,
        gridspec_kw={"hspace": 0.10},
    )
    figure.canvas.manager.set_window_title(f"RS18 signal browser — {ACCEPTED_RUN}")
    colors = ("#4C78A8", "#E45756")
    for axis, values, label, color in zip(
        axes, (plot_x, plot_y), ("Sensor X", "Sensor Y"), colors
    ):
        axis.plot(relative_time, values, color=color, linewidth=0.45)
        axis.set_ylabel(f"{label}\n{unit}")
        axis.grid(alpha=0.22)
        for row in rows:
            start = float(row["start_offset_s"])
            axis.axvline(start, color="black", linewidth=0.35, alpha=0.14)
        axis.margins(x=0)

    axes[0].set_title(
        f"Accepted run {ACCEPTED_RUN} — 642,136 actual samples at approximately 200 Hz\n"
        "Thin vertical lines are the 45 frozen frame starts"
    )
    axes[-1].set_xlabel("Seconds from capture start")
    axes[-1].set_xlim(float(relative_time[0]), float(relative_time[-1]))
    figure.text(
        0.5, 0.01,
        "Use the standard Matplotlib toolbar: magnifier = box zoom, hand = pan, "
        "Home = full capture, Back/Forward = view history.",
        ha="center", fontsize=9,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.09)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=exporter.DEFAULT_CAPTURE)
    parser.add_argument("--metadata", type=Path, default=exporter.DEFAULT_METADATA)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--raw-counts", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    figure = build_plot(
        capture=args.capture,
        metadata=args.metadata,
        analysis=args.analysis,
        raw_counts=args.raw_counts,
    )
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.snapshot, dpi=160, bbox_inches="tight")
        print(f"Wrote {args.snapshot}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
