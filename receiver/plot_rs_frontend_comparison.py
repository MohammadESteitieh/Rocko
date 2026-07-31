#!/usr/bin/env python3
"""Create editable Matplotlib plots for the frozen RS frontend comparison."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Edit these constants to change ordering, labels, or colors.
DUTY_ORDER = (100, 50, 25, 10, 1)
SCHEMES = (
    ("rs5_3", "RS(5,3)"),
    ("rs12_6", "RS(12,6)"),
)
ALGORITHMS = (
    ("unwhitened_hard", "No whitening + hard RS", "#9D9D9D"),
    ("coherent_hard", "Coherent hard", "#4C78A8"),
    ("coherent_gmd", "Coherent GMD", "#F58518"),
    ("gao_gmd", "Gao GMD", "#54A24B"),
    ("duong_gmd", "Duong GMD", "#E45756"),
    ("gao_duong_gmd", "Gao→Duong GMD", "#B279A2"),
)
FIGURE_SIZE = (13.0, 9.0)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def aggregate(rows: list[dict[str, str]]) -> tuple[dict, dict]:
    conditions: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"frames": 0, "frame_errors": 0}
    )
    overall: dict[str, dict[str, int]] = defaultdict(
        lambda: {"frames": 0, "frame_errors": 0, "failures": 0,
                 "miscorrections": 0}
    )
    for row in rows:
        algorithm = row["algorithm"]
        scheme = row["scheme"]
        duty = round(float(row["duty_percent"]))
        condition = conditions[(algorithm, scheme, duty)]
        condition["frames"] += 1
        condition["frame_errors"] += int(row["frame_error"])
        result = overall[algorithm]
        result["frames"] += 1
        result["frame_errors"] += int(row["frame_error"])
        result["failures"] += int(row["decoder_failure"])
        result["miscorrections"] += int(row["wrong_codeword_miscorrection"])
    return dict(conditions), dict(overall)


def build_figure(rows: list[dict[str, str]]) -> plt.Figure:
    conditions, overall = aggregate(rows)
    fig = plt.figure(figsize=FIGURE_SIZE, layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.78))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    outcome_axis = fig.add_subplot(grid[1, :])

    x = np.arange(len(DUTY_ORDER), dtype=float)
    width = 0.13
    offsets = (np.arange(len(ALGORITHMS)) - (len(ALGORITHMS) - 1) / 2) * width

    for axis, (scheme, scheme_label) in zip(axes, SCHEMES):
        for offset, (algorithm, label, color) in zip(offsets, ALGORITHMS):
            selected = [conditions[(algorithm, scheme, duty)] for duty in DUTY_ORDER]
            rates = [item["frame_errors"] / item["frames"] for item in selected]
            bars = axis.bar(x + offset, rates, width=width, color=color,
                            edgecolor="white", linewidth=0.5, label=label)
            for bar, item in zip(bars, selected):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(bar.get_height() + 0.025, 1.04),
                    f'{item["frame_errors"]}/{item["frames"]}',
                    ha="center", va="bottom", rotation=90, fontsize=7,
                )
        axis.set_title(f"{scheme_label} payload frame-error rate")
        axis.set_xticks(x, [str(duty) for duty in DUTY_ORDER])
        axis.set_xlabel("Commanded duty (%)")
        axis.set_ylabel("Observed FER")
        axis.set_ylim(0, 1.16)
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(ncol=2, fontsize=9, loc="upper left")

    labels = [label for _, label, _ in ALGORITHMS]
    colors = [color for _, _, color in ALGORITHMS]
    y = np.arange(len(ALGORITHMS))
    correct = [overall[key]["frames"] - overall[key]["frame_errors"]
               for key, _, _ in ALGORITHMS]
    wrong = [overall[key]["miscorrections"] for key, _, _ in ALGORITHMS]
    failed = [overall[key]["failures"] for key, _, _ in ALGORITHMS]
    outcome_axis.barh(y, correct, color="#59A14F", label="Correct payload")
    outcome_axis.barh(y, wrong, left=correct, color="#F28E2B",
                      label="Wrong valid codeword")
    outcome_axis.barh(y, failed, left=np.asarray(correct) + np.asarray(wrong),
                      color="#E15759", label="Decoder failure")
    for index, (good, bad, failure) in enumerate(zip(correct, wrong, failed)):
        for left, value in ((0, good), (good, bad), (good + bad, failure)):
            if value:
                outcome_axis.text(left + value / 2, index, str(value),
                                  ha="center", va="center", fontsize=9)
    outcome_axis.set_yticks(y, labels)
    outcome_axis.invert_yaxis()
    outcome_axis.set_xlim(0, 50)
    outcome_axis.set_xlabel("Physical RS frames")
    outcome_axis.set_title("Overall outcome disposition — 50 RS frames per method")
    outcome_axis.grid(axis="x", alpha=0.25)
    outcome_axis.legend(ncol=3, loc="lower right")

    fig.suptitle(
        "Soft-GMD and frozen Gao/Duong frontend comparison\n"
        "11 V, 3 m, 2 coded bits/s; post-hoc offline analysis",
        fontsize=14,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="rs-frontends comparison CSV")
    parser.add_argument("--png", type=Path, help="optional raster output")
    parser.add_argument("--svg", type=Path, help="optional editable vector output")
    parser.add_argument("--pdf", type=Path, help="optional PDF output")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--force", action="store_true", help="overwrite plot outputs")
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

    figure = build_figure(load_rows(args.csv))
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
