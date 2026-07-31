#!/usr/bin/env python3
"""Plot RS frontend FER against in-band and wide-noise carrier SNR."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DUTIES = (100, 50, 25, 10, 1)
SCHEMES = (
    ("rs5_3", "RS(5,3)", "RS(5,3)"),
    ("rs12_6", "RS(12,6)", "RS(12,6)"),
)
ALGORITHMS = (
    ("unwhitened_hard", "No whitening + hard RS", "#9D9D9D", "X", "-"),
    ("coherent_hard", "Coherent hard", "#4C78A8", "o", "-"),
    ("coherent_gmd", "Coherent GMD", "#F58518", "s", "--"),
    ("gao_gmd", "Gao GMD", "#54A24B", "^", "-."),
    ("duong_gmd", "Duong GMD", "#E45756", "D", ":"),
    ("gao_duong_gmd", "Gao→Duong GMD", "#B279A2", "P", (0, (5, 1))),
)
SNR_METRICS = (
    (
        "median_inband_snr_7_25_8_75hz_db",
        "no_positive_inband_signal_frames",
        "In-band: 7.25–8.75 Hz carrier noise",
    ),
    (
        "median_carrier_snr_vs_0_10hz_noise_db",
        "no_positive_carrier_signal_frames_for_full_snr",
        "Wide/out-of-band-aware: carrier signal vs 0–10 Hz off-noise",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def wilson(errors: int, frames: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = errors / frames
    denominator = 1.0 + z * z / frames
    center = (rate + z * z / (2.0 * frames)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / frames + z * z / (4.0 * frames**2))
    radius /= denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def frontend_errors(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict]:
    groups: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"frames": 0, "errors": 0}
    )
    for row in rows:
        key = (row["algorithm"], row["scheme"], round(float(row["duty_percent"])))
        groups[key]["frames"] += 1
        groups[key]["errors"] += int(row["frame_error"])
    return dict(groups)


def snr_conditions(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, float | int]]:
    labels = {table_label: scheme for scheme, _display, table_label in SCHEMES}
    result: dict[tuple[str, int], dict[str, float | int]] = {}
    for row in rows:
        scheme = labels.get(row["scheme"])
        if scheme is None:
            continue
        duty = round(float(row["duty_percent"]))
        result[(scheme, duty)] = {
            value_column: float(row[value_column])
            for value_column, _missing_column, _label in SNR_METRICS
        } | {
            missing_column: int(row[missing_column])
            for _value_column, missing_column, _label in SNR_METRICS
        }
    expected = {(scheme, duty) for scheme, *_ in SCHEMES for duty in DUTIES}
    missing = expected - set(result)
    if missing:
        raise ValueError(f"SNR table is missing conditions: {sorted(missing)}")
    return result


def build_figure(frontend_rows: list[dict[str, str]], snr_rows: list[dict[str, str]]) -> plt.Figure:
    errors = frontend_errors(frontend_rows)
    snrs = snr_conditions(snr_rows)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), sharey=True,
                             layout="constrained")
    pick_map: dict[object, list[object]] = {}
    method_artists: dict[str, list[object]] = defaultdict(list)
    method_handles: dict[str, object] = {}

    for row_index, (value_column, missing_column, metric_label) in enumerate(SNR_METRICS):
        for column_index, (scheme, scheme_label, _table_label) in enumerate(SCHEMES):
            axis = axes[row_index, column_index]
            for algorithm, label, color, marker, linestyle in ALGORITHMS:
                points = []
                for duty in DUTIES:
                    x_value = float(snrs[(scheme, duty)][value_column])
                    item = errors[(algorithm, scheme, duty)]
                    rate = item["errors"] / item["frames"]
                    low, high = wilson(item["errors"], item["frames"])
                    points.append((x_value, rate, low, high))
                points.sort()
                x = np.asarray([point[0] for point in points])
                y = np.asarray([point[1] for point in points])
                low = np.asarray([point[2] for point in points])
                high = np.asarray([point[3] for point in points])
                line, = axis.plot(
                    x, y, label=label, color=color, marker=marker,
                    linestyle=linestyle, linewidth=2, markersize=6, zorder=3,
                )
                band = axis.fill_between(x, low, high, color=color, alpha=0.06,
                                         linewidth=0, zorder=1)
                method_artists[algorithm].extend((line, band))
                method_handles.setdefault(algorithm, line)

            for duty in DUTIES:
                item = snrs[(scheme, duty)]
                missing = int(item[missing_column])
                suffix = f"\n{missing} no-est." if missing else ""
                axis.annotate(
                    f"{duty}%{suffix}",
                    (float(item[value_column]), 1.0), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color="#333333",
                )
            condition_x = [
                float(snrs[(scheme, duty)][value_column]) for duty in DUTIES
            ]
            if min(condition_x) <= 0 <= max(condition_x):
                axis.axvline(0, color="#555555", linewidth=1, alpha=0.3)
            axis.set_title(f"{scheme_label} — {metric_label}", fontsize=11)
            axis.set_xlabel("Conditional median carrier SNR (dB)")
            axis.set_ylim(-0.04, 1.18)
            axis.grid(True, alpha=0.25)
            if column_index == 0:
                axis.set_ylabel("Observed payload FER")

    legend = fig.legend(
        [method_handles[key] for key, *_ in ALGORITHMS],
        [label for _key, label, *_ in ALGORITHMS],
        loc="outside lower center", ncol=len(ALGORITHMS), fontsize=9,
    )
    for legend_line, (algorithm, *_rest) in zip(legend.get_lines(), ALGORITHMS):
        legend_line.set_picker(6)
        pick_map[legend_line] = method_artists[algorithm]

    fig.suptitle(
        "RS frontend FER against two measured SNR bandwidth definitions\n"
        "Both use only power-subtracted carrier-band excess as signal",
        fontsize=14,
    )

    def toggle(event) -> None:
        targets = pick_map.get(event.artist)
        if not targets:
            return
        visible = not targets[0].get_visible()
        for target in targets:
            target.set_visible(visible)
        event.artist.set_alpha(1.0 if visible else 0.25)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", toggle)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frontends_csv", type=Path)
    parser.add_argument("snr_table_csv", type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not (args.png or args.svg or args.pdf or args.show):
        parser.error("request at least one of --png, --svg, --pdf, or --show")
    return args


def main() -> int:
    args = parse_args()
    outputs = [path for path in (args.png, args.svg, args.pdf) if path]
    for path in outputs:
        if path.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {path}; use --force")
        path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(read_csv(args.frontends_csv), read_csv(args.snr_table_csv))
    for path in outputs:
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {path}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
