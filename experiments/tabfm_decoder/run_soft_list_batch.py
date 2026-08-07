#!/usr/bin/env python3
"""Run the frozen TabFM soft-list confirmation batch."""

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

import run_batch as common  # noqa: E402
import run_tabfm as runner  # noqa: E402
import soft_list_decoder  # noqa: E402

METHOD_DEVELOPMENT_SEQUENCES = (24, 2, 10, 28, 45)
METHOD_DEVELOPMENT_SUMMARY_SHA256 = (
    "8c3c850438196f287c2b40c5343a4c9509c62f7795dd32272281575f577f9f38"
)
CONFIRMATORY_SEQUENCES = (3, 18, 32, 39)
FROZEN_FEATURE_SET = "sensor-y-hybrid"
FROZEN_BACKEND = "jax"
FROZEN_CHECKPOINT_REVISION = common.FROZEN_CHECKPOINT_REVISION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sequence in CONFIRMATORY_SEQUENCES:
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
    model = runner.load_model(FROZEN_BACKEND, snapshot / "classification")

    import jax
    software = {
        "rocko_git_revision": common.git_revision(),
        "run_soft_list_batch_sha256": runner.sha256_file(Path(__file__)),
        "run_tabfm_sha256": runner.sha256_file(HERE / "run_tabfm.py"),
        "soft_list_decoder_sha256": runner.sha256_file(
            HERE / "soft_list_decoder.py"
        ),
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
        "method_development_sequences": list(METHOD_DEVELOPMENT_SEQUENCES),
        "method_development_summary_sha256": METHOD_DEVELOPMENT_SUMMARY_SHA256,
        "confirmatory_sequences": list(CONFIRMATORY_SEQUENCES),
        "feature_set": FROZEN_FEATURE_SET,
        "backend": FROZEN_BACKEND,
        "checkpoint_revision": FROZEN_CHECKPOINT_REVISION,
        "context_seed": 2026,
        "diagnostic_confidence_gate_threshold": 0.75,
        "soft_list_pool_size": soft_list_decoder.FROZEN_POOL_SIZE,
        "soft_list_acceptance_margin": (
            soft_list_decoder.FROZEN_ACCEPTANCE_MARGIN
        ),
    }

    summaries = []
    for sequence in CONFIRMATORY_SEQUENCES:
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
        summary = runner.evaluate(probability_one, query, truth)
        summary.update({
            "sequence": sequence,
            "analysis_role": "soft-list-confirmatory",
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
        runner.write_predictions(prediction_path, query, truth, probability_one)
        summary["predictions_sha256"] = runner.sha256_file(prediction_path)
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        common.write_sidecar(prediction_path)
        common.write_sidecar(summary_path)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    ordinary_prefixes = (
        "baseline", "sensor_y", "tabfm", "sensor_y_tabfm_gated"
    )
    aggregate = {}
    for prefix in ordinary_prefixes:
        aggregate[prefix] = {
            "confirmatory_frames": len(summaries),
            "bit_errors": sum(row[f"{prefix}_bit_errors"] for row in summaries),
            "symbol_errors": sum(
                row[f"{prefix}_symbol_errors"] for row in summaries
            ),
            "frame_errors": sum(row[f"{prefix}_frame_error"] for row in summaries),
            "wrong_codeword_miscorrections": sum(
                row[f"{prefix}_wrong_codeword_miscorrection"] for row in summaries
            ),
        }
    aggregate["tabfm_soft_list"] = {
        "confirmatory_frames": len(summaries),
        "decoder_failures": sum(
            row["tabfm_soft_list_decoder_failure"] for row in summaries
        ),
        "frame_errors": sum(
            row["tabfm_soft_list_frame_error"] for row in summaries
        ),
        "wrong_codeword_miscorrections": sum(
            row["tabfm_soft_list_wrong_codeword_miscorrection"]
            for row in summaries
        ),
        "accepted_frames": sum(
            not row["tabfm_soft_list_decoder_failure"] for row in summaries
        ),
    }

    aggregate_json = args.output_dir / "soft-list-batch-summary.json"
    aggregate_csv = args.output_dir / "soft-list-batch-summary.csv"
    aggregate_json.write_text(json.dumps({
        "experiment_contract": contract,
        "software": software,
        "confirmatory_aggregate": aggregate,
        "frames": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    flat = [
        {key: value for key, value in summary.items()
         if key not in ("source", "software", "experiment_contract")}
        for summary in summaries
    ]
    with aggregate_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    common.write_sidecar(aggregate_json)
    common.write_sidecar(aggregate_csv)
    print(aggregate_json)
    print(aggregate_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
