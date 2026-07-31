#!/usr/bin/env python3
"""Plot RS frontend FER against measured per-frame carrier SNR."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DUTIES = (100, 50, 25, 10, 1)
SCHEMES = (("rs5_3", "RS(5,3)"), ("rs12_6", "RS(12,6)"))
ALGORITHMS = (
    ("unwhitened_hard", "No whitening + hard RS", "#9D9D9D", "X", "-"),
    ("coherent_hard", "Coherent hard", "#4C78A8", "o", "-"),
    ("coherent_gmd", "Coherent GMD", "#F58518", "s", "--"),
    ("gao_gmd", "Gao GMD", "#54A24B", "^", "-."),
    ("duong_gmd", "Duong GMD", "#E45756", "D", ":"),
    ("gao_duong_gmd", "Gao→Duong GMD", "#B279A2", "P", (0, (5, 1))),
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


def condition_snrs(analysis_rows: list[dict[str, str]]) -> dict[tuple[str, int], dict]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in analysis_rows:
        if row["scheme"] in {key for key, _ in SCHEMES}:
            groups[(row["scheme"], round(float(row["duty_percent"])))].append(row)
    result = {}
    for key, rows in groups.items():
        positive = [
            float(row["inband_snr_db_pooled"])
            for row in rows
            if int(row["positive_inband_power_estimate"])
            and row["inband_snr_db_pooled"]
        ]
        if not positive:
            raise ValueError(f"condition {key} has no positive SNR estimates")
        result[key] = {
            "median_snr_db": float(np.median(positive)),
            "positive_frames": len(positive),
            "frames": len(rows),
            "no_estimate_frames": len(rows) - len(positive),
        }
    return result


def condition_errors(frontend_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict]:
    groups: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"frames": 0, "errors": 0}
    )
    for row in frontend_rows:
        key = (
            row["algorithm"], row["scheme"],
            round(float(row["duty_percent"])),
        )
        groups[key]["frames"] += 1
        groups[key]["errors"] += int(row["frame_error"])
    return dict(groups)


def build_figure(frontend_rows: list[dict[str, str]], analysis_rows: list[dict[str, str]]) -> plt.Figure:
    snrs = condition_snrs(analysis_rows)
    errors = condition_errors(frontend_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.8), sharey=True,
                             layout="constrained")
    pick_map: dict[object, list[object]] = {}
    method_artists: dict[str, list[object]] = defaultdict(list)
    method_handles: dict[str, object] = {}

    for axis, (scheme, scheme_label) in zip(axes, SCHEMES):
        for algorithm, label, color, marker, linestyle in ALGORITHMS:
            points = []
            for duty in DUTIES:
                snr = snrs[(scheme, duty)]["median_snr_db"]
                item = errors[(algorithm, scheme, duty)]
                rate = item["errors"] / item["frames"]
                low, high = wilson(item["errors"], item["frames"])
                points.append((snr, rate, low, high, duty))
            points.sort()
            x = np.asarray([point[0] for point in points])
            y = np.asarray([point[1] for point in points])
            low = np.asarray([point[2] for point in points])
            high = np.asarray([point[3] for point in points])
            line, = axis.plot(
                x, y, label=label, color=color, marker=marker,
                linestyle=linestyle, linewidth=2.0, markersize=7, zorder=3,
            )
            band = axis.fill_between(x, low, high, color=color, alpha=0.07,
                                     linewidth=0, zorder=1)
            method_artists[algorithm].extend((line, band))
            method_handles.setdefault(algorithm, line)

        # Duty labels identify the shared physical condition SNR on each panel.
        for duty in DUTIES:
            item = snrs[(scheme, duty)]
            suffix = (f"\n{item['no_estimate_frames']} no-est."
                      if item["no_estimate_frames"] else "")
            axis.annotate(
                f"{duty}%{suffix}",
                (item["median_snr_db"], 1.0), xytext=(0, 10),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color="#333333",
            )

        axis.axvline(0.0, color="#555555", linewidth=1, alpha=0.35)
        axis.set_title(f"{scheme_label} — five physical frames per point")
        axis.set_xlabel("Conditional median measured carrier SNR (dB)")
        axis.set_ylim(-0.04, 1.17)
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("Observed payload FER")
    legend = fig.legend(
        [method_handles[key] for key, *_ in ALGORITHMS],
        [label for _key, label, *_ in ALGORITHMS],
        loc="outside lower center", ncol=len(ALGORITHMS), fontsize=9,
    )
    for legend_line, (algorithm, *_rest) in zip(legend.get_lines(), ALGORITHMS):
        legend_line.set_picker(6)
        pick_map[legend_line] = method_artists[algorithm]
    fig.suptitle(
        "RS decoder/front-end performance against measured SNR\n"
        "SNR is shared by algorithms; no-positive-power frames remain in FER",
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
    parser.add_argument("analysis_csv", type=Path)
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
    figure = build_figure(read_csv(args.frontends_csv), read_csv(args.analysis_csv))
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
