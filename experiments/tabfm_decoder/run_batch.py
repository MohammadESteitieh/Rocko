#!/usr/bin/env python3
"""Run the frozen Sensor-Y/TabFM experiment with one loaded model."""

from __future__ import annotations

import argparse
import csv
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import run_tabfm as runner  # noqa: E402

EXPLORATORY_SEQUENCE = 24
CONFIRMATORY_SEQUENCES = (2, 10, 28, 45)
FROZEN_SEQUENCES = (EXPLORATORY_SEQUENCE,) + CONFIRMATORY_SEQUENCES
FROZEN_FEATURE_SET = "sensor-y-hybrid"
FROZEN_CONFIDENCE_THRESHOLD = 0.75
FROZEN_BACKEND = "jax"
FROZEN_CHECKPOINT_REVISION = "d5e74033fcf257699fab013e2cdd7edf424ff904"


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def write_sidecar(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{runner.sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sequence in FROZEN_SEQUENCES:
        subprocess.run([
            sys.executable, str(HERE / "export_rs18_table.py"),
            "--query-sequence", str(sequence),
            "--feature-set", FROZEN_FEATURE_SET,
            "--output-dir", str(args.output_dir),
        ], check=True)

    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        repo_id="google/tabfm-1.0.0-jax",
        revision=FROZEN_CHECKPOINT_REVISION,
        allow_patterns=["classification/**"],
    ))
    if snapshot.name != FROZEN_CHECKPOINT_REVISION:
        raise ValueError("downloaded TabFM snapshot does not match frozen revision")
    model = runner.load_model(
        FROZEN_BACKEND, snapshot / "classification"
    )
    import jax
    software = {
        "rocko_git_revision": git_revision(),
        "run_batch_sha256": runner.sha256_file(Path(__file__)),
        "run_tabfm_sha256": runner.sha256_file(HERE / "run_tabfm.py"),
        "exporter_sha256": runner.sha256_file(HERE / "export_rs18_table.py"),
        "python": sys.version,
        "tabfm_version": package_metadata.version("tabfm"),
        "jax_version": package_metadata.version("jax"),
        "jaxlib_version": package_metadata.version("jaxlib"),
        "tabfm_checkpoint_revision": FROZEN_CHECKPOINT_REVISION,
        "jax_devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "remote_bootstrap_sha256": os.environ.get(
            "ROCKO_REMOTE_BOOTSTRAP_SHA256"
        ),
    }
    contract = {
        "exploratory_sequence": EXPLORATORY_SEQUENCE,
        "confirmatory_sequences": list(CONFIRMATORY_SEQUENCES),
        "feature_set": FROZEN_FEATURE_SET,
        "confidence_threshold": FROZEN_CONFIDENCE_THRESHOLD,
        "backend": FROZEN_BACKEND,
        "checkpoint_revision": FROZEN_CHECKPOINT_REVISION,
        "context_seed": 2026,
    }

    summaries = []
    for sequence in FROZEN_SEQUENCES:
        stem = f"rs18-sequence-{sequence:02d}"
        table_path = args.output_dir / f"{stem}.context-query.csv"
        truth_path = args.output_dir / f"{stem}.truth.csv"
        metadata_path = args.output_dir / f"{stem}.metadata.json"
        prediction_path = args.output_dir / f"{stem}.predictions.csv"
        summary_path = args.output_dir / f"{stem}.summary.json"

        rows = runner.read_rows(table_path)
        metadata = runner.validate_export(
            table_path, truth_path, metadata_path, rows
        )
        context = [row for row in rows if row["role"] != "query"]
        query = [row for row in rows if row["role"] == "query"]
        truth = runner.read_rows(truth_path)
        probability_one = runner.predict_probabilities(
            model, rows, list(metadata["feature_columns"])
        )
        summary = runner.evaluate(
            probability_one, query, truth, FROZEN_CONFIDENCE_THRESHOLD
        )
        summary.update({
            "sequence": sequence,
            "analysis_role": (
                "exploratory-threshold-motivation"
                if sequence == EXPLORATORY_SEQUENCE else "confirmatory"
            ),
            "repetition": int(truth[0]["repetition"]),
            "duty_percent": float(truth[0]["duty_percent"]),
            "backend": FROZEN_BACKEND,
            "n_estimators": 1,
            "context_rows": len(context),
            "context_seed": metadata["context_seed"],
            "feature_set": metadata["feature_set"],
            "feature_count": len(metadata["feature_columns"]),
            "model": "google/tabfm-1.0.0-jax",
            "table_sha256": metadata["table_sha256"],
            "truth_sha256": metadata["truth_sha256"],
            "export_metadata_sha256": runner.sha256_file(metadata_path),
            "source": metadata["source"],
            "software": software,
            "experiment_contract": contract,
        })
        runner.write_predictions(
            prediction_path, query, truth, probability_one,
            FROZEN_CONFIDENCE_THRESHOLD,
        )
        summary["predictions_sha256"] = runner.sha256_file(prediction_path)
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        write_sidecar(prediction_path)
        write_sidecar(summary_path)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    prefixes = ("baseline", "sensor_y", "tabfm", "sensor_y_tabfm_gated")
    aggregate = {}
    confirmatory = [
        summary for summary in summaries if summary["analysis_role"] == "confirmatory"
    ]
    for prefix in prefixes:
        aggregate[prefix] = {
            "confirmatory_frames": len(confirmatory),
            "bit_errors": sum(row[f"{prefix}_bit_errors"] for row in confirmatory),
            "symbol_errors": sum(row[f"{prefix}_symbol_errors"] for row in confirmatory),
            "decoder_failures": sum(
                row[f"{prefix}_decoder_failure"] for row in confirmatory
            ),
            "frame_errors": sum(row[f"{prefix}_frame_error"] for row in confirmatory),
            "wrong_codeword_miscorrections": sum(
                row[f"{prefix}_wrong_codeword_miscorrection"]
                for row in confirmatory
            ),
        }

    aggregate_json = args.output_dir / "batch-summary.json"
    aggregate_csv = args.output_dir / "batch-summary.csv"
    aggregate_json.write_text(
        json.dumps({
            "experiment_contract": contract,
            "software": software,
            "confirmatory_aggregate": aggregate,
            "frames": summaries,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    flat = [
        {key: value for key, value in summary.items()
         if key not in ("source", "software", "experiment_contract")}
        for summary in summaries
    ]
    with aggregate_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    write_sidecar(aggregate_json)
    write_sidecar(aggregate_csv)
    print(aggregate_json)
    print(aggregate_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
