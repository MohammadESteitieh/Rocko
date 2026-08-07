#!/usr/bin/env python3
"""Evaluate the frozen soft-list rule on preserved development predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_tabfm  # noqa: E402
import soft_list_decoder  # noqa: E402

DEVELOPMENT_SEQUENCES = (24, 2, 10, 28, 45)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for sequence in DEVELOPMENT_SEQUENCES:
        stem = f"rs18-sequence-{sequence:02d}"
        predictions_path = args.results / f"{stem}.predictions.csv"
        truth_path = args.results / f"{stem}.truth.csv"
        predictions = run_tabfm.read_rows(predictions_path)
        truth = run_tabfm.read_rows(truth_path)
        llrs = soft_list_decoder.probability_llrs([
            float(row["tabfm_probability_one"]) for row in predictions
        ])
        decoded = soft_list_decoder.decode(llrs)
        payload_errors = None if decoded["failure"] else sum(
            left != right
            for left, right in zip(
                decoded["payload"], map(int, truth[0]["payload_bits"])
            )
        )
        rows.append({
            "sequence": sequence,
            "repetition": int(truth[0]["repetition"]),
            "duty_percent": float(truth[0]["duty_percent"]),
            "decoder_failure": int(decoded["failure"]),
            "frame_error": int(bool(decoded["failure"] or payload_errors)),
            "wrong_codeword_miscorrection": int(
                not decoded["failure"] and bool(payload_errors)
            ),
            "payload_bit_errors_conditional_on_decode": payload_errors,
            "mode": decoded["mode"],
            "candidate_count": decoded["candidate_count"],
            "margin": decoded["margin"],
            "selected_erasures": list(decoded["selected_erasures"]),
            "attempt_count": decoded["attempt_count"],
            "predictions_sha256": run_tabfm.sha256_file(predictions_path),
            "truth_sha256": run_tabfm.sha256_file(truth_path),
        })

    artifact = {
        "scope": "method-development; post-hoc, not confirmatory",
        "development_sequences": list(DEVELOPMENT_SEQUENCES),
        "pool_size": soft_list_decoder.FROZEN_POOL_SIZE,
        "acceptance_margin": soft_list_decoder.FROZEN_ACCEPTANCE_MARGIN,
        "soft_list_decoder_sha256": run_tabfm.sha256_file(
            HERE / "soft_list_decoder.py"
        ),
        "frames": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
