#!/usr/bin/env python3
"""Plot dataset samples, physical SNR, and TabFM soft-GMD outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
import soft_list_decoder  # noqa: E402

DEVELOPMENT = (24, 2, 10, 28, 45)
CONFIRMATION = (3, 18, 32, 39)
SEQUENCES = DEVELOPMENT + CONFIRMATION
METHODS = (
    ("baseline", "Primary coherent"),
    ("sensor_y", "Sensor Y"),
    ("tabfm", "TabFM hard RS"),
    ("sensor_y_tabfm_gated", "Sensor-Y/TabFM gate"),
    ("soft_gmd", "TabFM soft GMD/list"),
)
DEFAULT_ANALYSIS = (
    ROOT / "data/captures/rs18-experiment/derived"
    / "rs18_experiment_11V_20260728_091828.analysis.csv"
)
DEFAULT_DEVELOPMENT = (
    ROOT / "data/captures/rs18-experiment/derived/tabfm"
    / "sensor-y-hybrid-frozen-v1"
)
DEFAULT_CONFIRMATION = (
    ROOT / "data/captures/rs18-experiment/derived/tabfm"
    / "soft-list-confirmatory-v1"
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _summary_by_sequence(development: Path, confirmation: Path) -> dict[int, dict]:
    first = json.loads((development / "batch-summary.json").read_text(encoding="utf-8"))
    second = json.loads(
        (confirmation / "soft-list-batch-summary.json").read_text(encoding="utf-8")
    )
    rows = first["frames"] + second["frames"]
    return {int(row["sequence"]): row for row in rows}


def _soft_outcome(sequence: int, root: Path) -> tuple[int, str]:
    predictions = _csv_rows(root / f"rs18-sequence-{sequence:02d}.predictions.csv")
    truth = _csv_rows(root / f"rs18-sequence-{sequence:02d}.truth.csv")
    llrs = soft_list_decoder.probability_llrs([
        float(row["tabfm_probability_one"]) for row in predictions
    ])
    decoded = soft_list_decoder.decode(llrs)
    if decoded["failure"]:
        return 0, "Reject"
    expected = tuple(map(int, truth[0]["payload_bits"]))
    correct = tuple(decoded["payload"]) == expected
    return int(correct), "OK" if correct else "Wrong"


def build_plot_rows(
    analysis: Path, development: Path, confirmation: Path
) -> list[dict[str, object]]:
    analysis_by_sequence = {
        int(row["sequence"]): row for row in _csv_rows(analysis)
    }
    summaries = _summary_by_sequence(development, confirmation)
    output = []
    for sequence in SEQUENCES:
        source = development if sequence in DEVELOPMENT else confirmation
        summary = summaries[sequence]
        physical = analysis_by_sequence[sequence]
        soft_success, soft_label = _soft_outcome(sequence, source)
        row: dict[str, object] = {
            "sequence": sequence,
            "phase": "development" if sequence in DEVELOPMENT else "confirmation",
            "duty_percent": float(summary["duty_percent"]),
            "sensor_y_physical_inband_snr_db": float(
                physical["physical_inband_snr_db_y"]
            ),
            "soft_gmd_success": soft_success,
            "soft_gmd_label": soft_label,
        }
        for key, _label in METHODS[:-1]:
            row[f"{key}_symbol_errors"] = int(summary[f"{key}_symbol_errors"])
            row[f"{key}_success"] = 1 - int(summary[f"{key}_frame_error"])
        output.append(row)
    return output


def waveform(capture: Path, metadata: Path, analysis: Path,
             sequence: int = 2, seconds: float = 8.0
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, x, y = base.load_capture(capture)
    fs = base.sample_rate(t)
    info = base.metadata(metadata)
    rs18.validate_metadata(info)
    off_start, off_stop = rs18.prelaunch_indices(info, fs, len(t))
    means, scales = rs18.fit_off_rms(x, y, off_start, off_stop)
    rows = {int(row["sequence"]): row for row in _csv_rows(analysis)}
    start = round(float(rows[sequence]["start_offset_s"]) * fs)
    stop = min(len(t), start + round(seconds * fs))
    relative = np.arange(stop - start, dtype=float) / fs
    return (
        relative,
        (np.asarray(x[start:stop], float) - means[0]) / scales[0],
        (np.asarray(y[start:stop], float) - means[1]) / scales[1],
    )


def render(rows: list[dict[str, object]], wave: tuple[np.ndarray, np.ndarray, np.ndarray],
           outputs: list[Path], dpi: int) -> None:
    figure = plt.figure(figsize=(14, 9), layout="constrained")
    figure.get_layout_engine().set(rect=(0.0, 0.055, 1.0, 0.89))
    grid = figure.add_gridspec(2, 2, height_ratios=(0.9, 1.15), width_ratios=(0.92, 1.3))
    sample_axis = figure.add_subplot(grid[0, :])
    snr_axis = figure.add_subplot(grid[1, 0])
    decode_axis = figure.add_subplot(grid[1, 1])

    relative, x, y = wave
    for bit in range(16):
        if bit % 2:
            sample_axis.axvspan(bit * 0.5, (bit + 1) * 0.5, color="#EEEEEE", zorder=0)
    sample_axis.plot(relative, x, color="#4C78A8", linewidth=0.8, label="Sensor X")
    sample_axis.plot(relative, y, color="#E45756", linewidth=0.8, alpha=0.85,
                     label="Sensor Y")
    sample_axis.set_xlim(0, 8)
    sample_axis.set_ylabel("ADC sample (prelaunch-off RMS units)")
    sample_axis.set_xlabel("Time from frozen frame boundary (s)")
    sample_axis.set_title("A  Actual 200 Hz dual-sensor samples — sequence 2, sync field, 50% duty")
    sample_axis.grid(axis="y", alpha=0.2)
    sample_axis.legend(loc="upper right", ncol=2)

    x_positions = np.arange(len(rows))
    snr = [float(row["sensor_y_physical_inband_snr_db"]) for row in rows]
    colors = ["#F28E2B" if row["phase"] == "development" else "#4C78A8" for row in rows]
    bars = snr_axis.bar(x_positions, snr, color=colors, edgecolor="white")
    for bar, value in zip(bars, snr):
        snr_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.12,
                      f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    snr_axis.axhline(0, color="black", linewidth=0.8)
    snr_axis.set_xticks(x_positions, [
        f'{row["sequence"]}\n{float(row["duty_percent"]):g}%'
        for row in rows
    ])
    snr_axis.set_xlabel("Sequence and commanded duty")
    snr_axis.set_ylabel("Sensor-Y physical in-band SNR (dB)")
    snr_axis.set_title("B  Signal conditions")
    snr_axis.grid(axis="y", alpha=0.25)
    snr_axis.text(0.02, 0.98, "Orange: development   Blue: prospective confirmation",
                  transform=snr_axis.transAxes, va="top", fontsize=8)

    outcome = np.zeros((len(METHODS), len(rows)), dtype=int)
    annotations: list[list[str]] = []
    for method_index, (key, _label) in enumerate(METHODS):
        labels = []
        for column, row in enumerate(rows):
            if key == "soft_gmd":
                outcome[method_index, column] = int(row["soft_gmd_success"])
                labels.append(str(row["soft_gmd_label"]))
            else:
                outcome[method_index, column] = int(row[f"{key}_success"])
                labels.append(f'{row[f"{key}_symbol_errors"]} sym')
        annotations.append(labels)
    decode_axis.imshow(outcome, aspect="auto", vmin=0, vmax=1,
                       cmap=ListedColormap(["#F4CCCC", "#B6D7A8"]))
    for row_index in range(len(METHODS)):
        for column in range(len(rows)):
            decode_axis.text(column, row_index, annotations[row_index][column],
                             ha="center", va="center", fontsize=8,
                             fontweight="bold" if outcome[row_index, column] else "normal")
    decode_axis.axvline(len(DEVELOPMENT) - 0.5, color="black", linewidth=2)
    decode_axis.set_xticks(x_positions, [f'Seq {row["sequence"]}' for row in rows],
                           rotation=35, ha="right")
    decode_axis.set_yticks(np.arange(len(METHODS)), [label for _, label in METHODS])
    decode_axis.set_title("C  Payload decoding (green = correct; red = failed/rejected)")
    decode_axis.tick_params(length=0)
    decode_axis.text((len(DEVELOPMENT) - 1) / 2, -0.83, "Method development",
                     ha="center", fontsize=9)
    decode_axis.text(len(DEVELOPMENT) + (len(CONFIRMATION) - 1) / 2, -0.83,
                     "Prospective confirmation", ha="center", fontsize=9)

    figure.suptitle(
        "RS18 TabFM soft-GMD pipeline: one exploratory recovery, no confirmed recovery\n"
        "11 V, 3 m, 2 coded bits/s; frozen 12-symbol list and margin-20 acceptance",
        fontsize=15,
    )
    figure.text(
        0.5, 0.012,
        "Sequence 24 passed initial hard RS; the soft-GMD list added sequence 2 during development. All four prospective candidates were safely rejected. "
        "Hard-decoder cells report pre-RS symbol errors; RS(18,6) hard correction limit is 6 symbols.",
        ha="center", fontsize=9,
    )
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_plot_data(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--capture", type=Path, default=exporter.DEFAULT_CAPTURE)
    parser.add_argument("--metadata", type=Path, default=exporter.DEFAULT_METADATA)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--plot-data", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = [path for path in (args.png, args.svg, args.pdf) if path]
    for path in outputs + [args.plot_data]:
        if path.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    rows = build_plot_rows(args.analysis, args.development, args.confirmation)
    write_plot_data(args.plot_data, rows)
    render(rows, waveform(args.capture, args.metadata, args.analysis), outputs, args.dpi)
    for path in outputs + [args.plot_data]:
        base.write_checksum(path)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
