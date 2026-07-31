#!/usr/bin/env python3
"""Plot observed payload FER and raw coded-body BER for the final experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

DUTIES = (100, 50, 25, 10, 1)
SERIES = (
    ("short|uncoded", "Uncoded-11", "#4C78A8", "o"),
    ("short|hamming15_11", "Hamming(15,11)", "#F58518", "s"),
    ("short|rs5_3", "RS(5,3)", "#54A24B", "^"),
    ("final|rs12_6", "RS(12,6)", "#E45756", "D"),
)


def condition(summary: dict, prefix: str, duty: int) -> dict:
    return summary["by_phase_scheme_duty"][f"{prefix}|{duty}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    fig, (fer_axis, ber_axis) = plt.subplots(2, 1, figsize=(9.2, 8.4), sharex=True)
    x = list(range(len(DUTIES)))

    for prefix, label, color, marker in SERIES:
        selected = [condition(summary, prefix, duty) for duty in DUTIES]
        fer = [item["frame_errors"] / item["frames"] for item in selected]
        lower = [item["primary_payload_fer_wilson_95"]["lower"] for item in selected]
        upper = [item["primary_payload_fer_wilson_95"]["upper"] for item in selected]
        ber = [
            item["coded_body_bit_errors"] / item["coded_body_bits_denominator"]
            for item in selected
        ]
        fer_axis.errorbar(
            x, fer,
            yerr=([value - lo for value, lo in zip(fer, lower)],
                  [hi - value for value, hi in zip(fer, upper)]),
            label=label, color=color, marker=marker, linewidth=2,
            markersize=6, capsize=3,
        )
        ber_axis.plot(x, ber, label=label, color=color, marker=marker,
                      linewidth=2, markersize=6)
        for duty, item, fer_value, ber_value in zip(DUTIES, selected, fer, ber):
            rows.append({
                "scheme": label,
                "duty_percent": duty,
                "payload_frame_errors": item["frame_errors"],
                "frames": item["frames"],
                "payload_fer": fer_value,
                "payload_fer_wilson_95_lower": item["primary_payload_fer_wilson_95"]["lower"],
                "payload_fer_wilson_95_upper": item["primary_payload_fer_wilson_95"]["upper"],
                "decoder_failures": item["decoder_failures"],
                "coded_body_bit_errors": item["coded_body_bit_errors"],
                "coded_body_bits": item["coded_body_bits_denominator"],
                "coded_body_ber": ber_value,
                "median_positive_inband_snr_db": item["conditional_median_positive_inband_snr_db_pooled"],
                "no_positive_snr_frames": item["no_positive_inband_power_frames"],
            })

    fer_axis.set_title("Observed payload frame-error rate (5 physical frames per cell)")
    fer_axis.set_ylabel("Payload FER")
    fer_axis.set_ylim(-0.04, 1.08)
    fer_axis.grid(True, alpha=0.25)
    fer_axis.legend(ncol=2, loc="upper left")
    fer_axis.text(
        0.99, 0.02, "Error bars: Wilson 95% intervals (n=5)",
        transform=fer_axis.transAxes, ha="right", va="bottom", fontsize=9,
    )

    ber_axis.set_title("Raw coded-body bit-error rate before decoding")
    ber_axis.set_ylabel("Raw coded-body BER")
    ber_axis.set_xlabel("Commanded duty (%)")
    ber_axis.set_ylim(-0.02, 0.62)
    ber_axis.set_xticks(x, [str(duty) for duty in DUTIES])
    ber_axis.grid(True, alpha=0.25)

    fig.suptitle(
        "Final experiment error rates — 11 V, 3 m, 2 coded bits/s\n"
        "Manifest-boundary decoding; one fixed payload per scheme",
        fontsize=13,
    )
    fig.tight_layout()
    for output in (args.png, args.svg, args.csv):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    fig.savefig(args.png, dpi=180, bbox_inches="tight")
    fig.savefig(args.svg, bbox_inches="tight")
    plt.close(fig)
    with args.csv.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.png}")
    print(f"Wrote {args.svg}")
    print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
