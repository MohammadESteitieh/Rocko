#!/usr/bin/env python3
"""Combine final-experiment error rates with carrier SNR noise bandwidths."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import signal

DUTIES = (100, 50, 25, 10, 1)
SERIES = (
    ("short", "uncoded", "Uncoded-11"),
    ("short", "hamming15_11", "Hamming(15,11)"),
    ("short", "rs5_3", "RS(5,3)"),
    ("final", "rs12_6", "RS(12,6)"),
)
CENTRAL_GAP_LEAD_SECONDS = 2.5
CENTRAL_GAP_SECONDS = 10.0


def optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def snr_db(signal_power: float, noise_power: float) -> float | None:
    if signal_power <= 0 or noise_power <= 0:
        return None
    return 10.0 * math.log10(signal_power / noise_power)


def display(value: float | None, missing: int) -> str:
    if value is None:
        return "no positive estimate"
    suffix = f"; {missing} no-estimate" if missing else ""
    return f"{value:.2f} dB{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("analysis_csv", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.csv.exists() or args.markdown.exists():
        raise FileExistsError("refusing to overwrite output")
    capture = np.atleast_1d(np.genfromtxt(args.capture, delimiter=",", names=True))
    t = np.asarray(capture["t"], float)
    x = np.asarray(capture["x"], float)
    y = np.asarray(capture["y"], float)
    good = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[good], x[good], y[good]
    fs = 1.0 / float(np.median(np.diff(t)))
    with args.analysis_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 100:
        raise ValueError("analysis CSV must contain exactly 100 frames")

    gap_samples = round(CENTRAL_GAP_SECONDS * fs)
    gap_slices: list[slice] = []
    for row in rows:
        start = round(float(row["start_offset_s"]) * fs)
        frame_samples = round(int(row["frame_bits"]) * 0.5 * fs)
        gap_start = start + frame_samples + round(CENTRAL_GAP_LEAD_SECONDS * fs)
        gap_slices.append(slice(gap_start, gap_start + gap_samples))
    off_indices = np.concatenate([
        np.arange(window.start, window.stop) for window in gap_slices
    ])
    baselines = (float(np.mean(x[off_indices])), float(np.mean(y[off_indices])))
    sos = signal.butter(4, 10.0, btype="lowpass", fs=fs, output="sos")
    limited = (
        signal.sosfiltfilt(sos, x - baselines[0]),
        signal.sosfiltfilt(sos, y - baselines[1]),
    )

    for row, window in zip(rows, gap_slices):
        full_noise = sum(float(np.mean(channel[window] ** 2)) for channel in limited)
        inband_active = (
            float(row["inband_active_power_x"]) + float(row["inband_active_power_y"])
        )
        inband_noise = (
            float(row["inband_off_power_x"]) + float(row["inband_off_power_y"])
        )
        carrier_signal = inband_active - inband_noise
        full_snr = snr_db(carrier_signal, full_noise)
        inband_snr = optional_float(row["inband_snr_db_pooled"])
        if full_snr is not None and inband_snr is not None and full_snr > inband_snr + 1e-6:
            raise ValueError("full-band-noise SNR exceeds in-band-noise SNR")
        row["carrier_snr_vs_0_10hz_noise_db"] = full_snr

    output: list[dict[str, object]] = []
    for phase, scheme, label in SERIES:
        for duty in DUTIES:
            selected = [
                row for row in rows
                if row["phase"] == phase and row["scheme"] == scheme
                and float(row["duty_percent"]) == duty
            ]
            inband = [optional_float(row["inband_snr_db_pooled"]) for row in selected]
            full = [row["carrier_snr_vs_0_10hz_noise_db"] for row in selected]
            positive_inband = [value for value in inband if value is not None]
            positive_full = [value for value in full if value is not None]
            frame_errors = sum(int(row["frame_error"]) for row in selected)
            body_errors = sum(int(row["coded_body_bit_errors"]) for row in selected)
            body_bits = sum(int(row["frame_bits"]) - 16 for row in selected)
            output.append({
                "scheme": label,
                "duty_percent": duty,
                "payload_frame_errors": frame_errors,
                "frames": len(selected),
                "payload_fer": frame_errors / len(selected),
                "coded_body_bit_errors": body_errors,
                "coded_body_bits": body_bits,
                "coded_body_ber": body_errors / body_bits,
                "median_carrier_snr_vs_0_10hz_noise_db": (
                    float(np.median(positive_full)) if positive_full else None
                ),
                "no_positive_carrier_signal_frames_for_full_snr": (
                    len(selected) - len(positive_full)
                ),
                "median_inband_snr_7_25_8_75hz_db": (
                    float(np.median(positive_inband)) if positive_inband else None
                ),
                "no_positive_inband_signal_frames": len(selected) - len(positive_inband),
            })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("x", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    lines = [
        "# Final error rates with carrier SNR bandwidths",
        "",
        "The full-noise SNR keeps the signal numerator restricted to the",
        "power-subtracted 7.25-8.75 Hz carrier and uses all transmitter-off",
        "noise from zero to 10 Hz in the denominator. It therefore cannot exceed",
        "the in-band SNR. Medians exclude frames with no positive carrier-power",
        "estimate and report their count explicitly.",
        "",
        "| Scheme | Duty | Payload FER | Raw body BER | Carrier SNR vs 0-10 Hz noise | In-band SNR (7.25-8.75 Hz) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            f"| {row['scheme']} | {row['duty_percent']}% | "
            f"{row['payload_frame_errors']}/{row['frames']} "
            f"({100 * float(row['payload_fer']):.1f}%) | "
            f"{100 * float(row['coded_body_ber']):.1f}% | "
            f"{display(row['median_carrier_snr_vs_0_10hz_noise_db'], int(row['no_positive_carrier_signal_frames_for_full_snr']))} | "
            f"{display(row['median_inband_snr_7_25_8_75hz_db'], int(row['no_positive_inband_signal_frames']))} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
