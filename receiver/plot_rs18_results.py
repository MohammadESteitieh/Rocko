#!/usr/bin/env python3
"""Create truthful RS18 FER and physical-SNR charts from frozen analysis JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "receiver" / "legacy_decoder"))
import analyze_final_experiment as base  # noqa: E402


def plot_data(summary: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    conditions = summary["by_duty"]
    for key in sorted(conditions, key=float):
        item = conditions[key]
        interval = item["fer_wilson_95"]
        gmd = item["exploratory_gmd_fer_wilson_95"]
        normalized = item["normalized_diagnostic_fer_wilson_95"]
        rows.append({
            "duty_percent": float(item["duty_percent"]),
            "frames": int(item["frames"]),
            "hard_frame_errors": int(item["frame_errors"]),
            "hard_fer": float(interval["rate"]),
            "hard_fer_wilson_lower": float(interval["lower"]),
            "hard_fer_wilson_upper": float(interval["upper"]),
            "exploratory_gmd_frame_errors": int(item["exploratory_gmd_frame_errors"]),
            "exploratory_gmd_fer": float(gmd["rate"]),
            "exploratory_gmd_wilson_lower": float(gmd["lower"]),
            "exploratory_gmd_wilson_upper": float(gmd["upper"]),
            "exploratory_gmd_failures": int(item["exploratory_gmd_decoder_failures"]),
            "exploratory_gmd_miscorrections": int(
                item["exploratory_gmd_wrong_codeword_miscorrections"]
            ),
            "off_rms_normalized_diagnostic_frame_errors": int(
                item["normalized_diagnostic_frame_errors"]
            ),
            "off_rms_normalized_diagnostic_fer": float(normalized["rate"]),
            "off_rms_normalized_diagnostic_wilson_lower": float(normalized["lower"]),
            "off_rms_normalized_diagnostic_wilson_upper": float(normalized["upper"]),
            "positive_physical_snr_frames": int(
                item["positive_physical_inband_snr_frames"]
            ),
            "no_positive_physical_power_frames": int(
                item["no_positive_physical_inband_power_frames"]
            ),
            "conditional_median_positive_physical_inband_snr_db": item[
                "conditional_median_positive_physical_inband_snr_db_pooled"
            ],
            "conditional_median_carrier_vs_0_10hz_off_noise_snr_db": item[
                "conditional_median_carrier_vs_0_10hz_off_noise_snr_db_pooled"
            ],
        })
    return rows


def write_plot_data(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict[str, object]], png: Path, svg: Path) -> None:
    duties = np.array([float(row["duty_percent"]) for row in rows])
    hard = np.array([float(row["hard_fer"]) for row in rows])
    hard_low = np.array([float(row["hard_fer_wilson_lower"]) for row in rows])
    hard_high = np.array([float(row["hard_fer_wilson_upper"]) for row in rows])
    gmd = np.array([float(row["exploratory_gmd_fer"]) for row in rows])
    gmd_low = np.array([float(row["exploratory_gmd_wilson_lower"]) for row in rows])
    gmd_high = np.array([float(row["exploratory_gmd_wilson_upper"]) for row in rows])
    normalized = np.array([
        float(row["off_rms_normalized_diagnostic_fer"]) for row in rows
    ])
    normalized_low = np.array([
        float(row["off_rms_normalized_diagnostic_wilson_lower"]) for row in rows
    ])
    normalized_high = np.array([
        float(row["off_rms_normalized_diagnostic_wilson_upper"]) for row in rows
    ])
    snr = np.array([
        np.nan if row["conditional_median_positive_physical_inband_snr_db"] is None
        else float(row["conditional_median_positive_physical_inband_snr_db"])
        for row in rows
    ])
    positive = np.array([int(row["positive_physical_snr_frames"]) for row in rows])

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    top.errorbar(
        duties, hard,
        yerr=np.vstack((hard - hard_low, hard_high - hard)),
        marker="o", capsize=4, linewidth=2, label="Primary hard RS(18,6)",
    )
    top.errorbar(
        duties, gmd,
        yerr=np.vstack((gmd - gmd_low, gmd_high - gmd)),
        marker="^", capsize=3, linestyle=":",
        label="Exploratory truth-free GMD (not primary)",
    )
    top.errorbar(
        duties, normalized,
        yerr=np.vstack((normalized - normalized_low, normalized_high - normalized)),
        marker="s", capsize=3, linestyle="--",
        label="Off-RMS-normalized coherent diagnostic",
    )
    top.set_ylabel("Frame error rate")
    top.set_ylim(-0.05, 1.08)
    top.grid(alpha=0.25)
    top.legend(loc="best")
    top.set_title("RS(18,6), five frames per duty; bars are Wilson 95% intervals")
    for duty, value, errors in zip(duties, hard, [r["hard_frame_errors"] for r in rows]):
        top.annotate(f"{errors}/5", (duty, value), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=8)

    bottom.plot(duties, snr, marker="o", linewidth=2, color="tab:green")
    bottom.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    bottom.set_ylabel("Physical in-band SNR (dB)")
    bottom.set_xlabel("Commanded duty (%)")
    bottom.grid(alpha=0.25)
    bottom.set_title("Conditional median where positive signal power exists")
    for duty, value, count in zip(duties, snr, positive):
        if np.isfinite(value):
            bottom.annotate(f"{count}/5", (duty, value), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=8)
        else:
            bottom.annotate("0/5 positive", (duty, 0), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=8)
    bottom.set_xticks(duties)
    figure.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--plot-data", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(path.exists() for path in (args.png, args.svg, args.plot_data)):
        print("ERROR: refusing to overwrite RS18 plot outputs", file=sys.stderr)
        return 2
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = plot_data(summary)
    write_plot_data(args.plot_data, rows)
    render(rows, args.png, args.svg)
    for path in (args.plot_data, args.png, args.svg):
        digest = base.write_checksum(path)
        print(f"Wrote {path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
