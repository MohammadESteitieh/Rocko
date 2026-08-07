#!/usr/bin/env python3
"""Retrospectively validate fixed-context decoding on one-message CSVs."""

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

import decode_capture as model_loader  # noqa: E402
import decode_single_csv as decoder  # noqa: E402
import single_frame_pipeline as pipeline  # noqa: E402
import soft_list_decoder  # noqa: E402

EXPECTED_SEQUENCES = tuple(range(28, 46))
EXPECTED_FILES = len(EXPECTED_SEQUENCES) * 2
EXPECTED_SCOPE = (
    "retrospective pipeline validation; payload absent from reference context, "
    "not method-level confirmation"
)
EXPECTED_CONTRACT = {
    "first_sample_is_sync_start": True,
    "timestamp_origin_is_semantically_ignored": True,
    "message_seconds": 53.0,
    "required_off_tail_seconds": 12.5,
    "sample_rate_hz": 200.0,
    "exact_samples_from_frame_start": 13100,
    "autonomous_synchronization": False,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        check=False, capture_output=True, text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def validate_bundle_contract(bundle: Path
                             ) -> tuple[dict[int, dict[str, dict[str, str]]], dict[str, str]]:
    bundle = bundle.resolve()
    bundle_metadata_path = bundle / "bundle-metadata.json"
    bundle_metadata_hash = pipeline.verify_sidecar(bundle_metadata_path)
    bundle_metadata = json.loads(bundle_metadata_path.read_text(encoding="utf-8"))
    if bundle_metadata.get("artifact") != "rocko-single-frame-validation-bundle-v1":
        raise ValueError("bundle metadata has the wrong artifact type")
    if (bundle_metadata.get("scope") != EXPECTED_SCOPE
            or bundle_metadata.get("contract") != EXPECTED_CONTRACT):
        raise ValueError("bundle scope or input contract is not frozen")
    if (bundle_metadata.get("held_out_logical_frames") != len(EXPECTED_SEQUENCES)
            or bundle_metadata.get("timestamp_variants_per_frame") != 2
            or bundle_metadata.get("signal_csv_files") != EXPECTED_FILES):
        raise ValueError("bundle metadata has the wrong frozen scope")

    expected_path = bundle / "expected-results.csv"
    expected_hash = pipeline.verify_sidecar(expected_path)
    if bundle_metadata.get("expected_results_sha256") != expected_hash:
        raise ValueError("expected-results hash is not bound by bundle metadata")
    context_path = bundle / "tabfm-reference-context.csv"
    context_hash = pipeline.verify_sidecar(context_path)
    if bundle_metadata.get("reference_context_sha256") != context_hash:
        raise ValueError("reference context hash is not bound by bundle metadata")
    context_metadata_path = bundle / "tabfm-reference-context.metadata.json"
    context_metadata_hash = pipeline.verify_sidecar(context_metadata_path)
    if bundle_metadata.get("reference_context_metadata_sha256") != context_metadata_hash:
        raise ValueError("context metadata hash is not bound by bundle metadata")
    pipeline.load_reference_context(context_path, context_metadata_path)
    readme_path = bundle / "README.md"
    readme_hash = pipeline.verify_sidecar(readme_path)
    if bundle_metadata.get("readme_sha256") != readme_hash:
        raise ValueError("README hash is not bound by bundle metadata")

    expected = read_rows(expected_path)
    if len(expected) != EXPECTED_FILES:
        raise ValueError("expected-results must contain exactly 36 timestamp variants")
    by_sequence: dict[int, dict[str, dict[str, str]]] = {}
    seen_paths: set[str] = set()
    for row in expected:
        if row.get("validation_role") != "payload-reference-held-out-retrospective":
            raise ValueError("expected-results has the wrong validation role")
        sequence = int(row["sequence"])
        repetition = int(row["repetition"])
        expected_repetition = 4 if sequence <= 36 else 5
        if sequence not in EXPECTED_SEQUENCES or repetition != expected_repetition:
            raise ValueError("expected-results violates the frozen sequence split")
        time_origin = row["time_origin"]
        if time_origin not in ("relative", "absolute"):
            raise ValueError("invalid timestamp variant")
        relative_path = Path(row["filename"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("signal path must remain inside the bundle")
        path = (bundle / relative_path).resolve()
        if bundle not in path.parents or str(relative_path) in seen_paths:
            raise ValueError("signal path escapes or duplicates within the bundle")
        seen_paths.add(str(relative_path))
        actual = pipeline.verify_sidecar(path)
        if actual != row["file_sha256"]:
            raise ValueError(f"signal hash mismatch for {row['filename']}")
        variants = by_sequence.setdefault(sequence, {})
        if time_origin in variants:
            raise ValueError("duplicate timestamp variant")
        variants[time_origin] = row

    if tuple(sorted(by_sequence)) != EXPECTED_SEQUENCES:
        raise ValueError("bundle does not contain the exact frozen sequences")
    invariant_fields = (
        "sequence", "repetition", "duty_percent", "payload_bits",
        "coded_body_bits", "sample_count", "sample_rate_hz",
        "frame_start_seconds", "message_seconds", "required_off_tail_seconds",
        "validation_role",
    )
    for sequence, variants in by_sequence.items():
        if set(variants) != {"relative", "absolute"}:
            raise ValueError("every sequence requires both timestamp variants")
        if any(variants["relative"][key] != variants["absolute"][key]
               for key in invariant_fields):
            raise ValueError(f"timestamp variants disagree for sequence {sequence}")

    expected_files = {
        "README.md", "README.md.sha256",
        "bundle-metadata.json", "bundle-metadata.json.sha256",
        "expected-results.csv", "expected-results.csv.sha256",
        "tabfm-reference-context.csv", "tabfm-reference-context.csv.sha256",
        "tabfm-reference-context.metadata.json",
        "tabfm-reference-context.metadata.json.sha256",
    }
    for row in expected:
        expected_files.add(row["filename"])
        expected_files.add(row["filename"] + ".sha256")
    actual_files = {
        str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("bundle contains missing or unexpected files")

    return by_sequence, {
        "bundle_metadata_sha256": bundle_metadata_hash,
        "expected_results_sha256": expected_hash,
        "reference_context_sha256": context_hash,
        "reference_context_metadata_sha256": context_metadata_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, nargs="?", default=decoder.DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    by_sequence, bundle_hashes = validate_bundle_contract(args.bundle)
    sequences = list(EXPECTED_SEQUENCES)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        sequences = sequences[:args.limit]

    model = model_loader._load_pinned_model(args.checkpoint)
    import jax
    frame_results = []
    for sequence in sequences:
        variants = by_sequence[sequence]
        relative_path = args.bundle / variants["relative"]["filename"]
        absolute_path = args.bundle / variants["absolute"]["filename"]
        relative_rows = pipeline.extract_rows(relative_path)
        absolute_rows = pipeline.extract_rows(absolute_path)
        relative_digest = decoder._feature_digest(relative_rows)
        absolute_digest = decoder._feature_digest(absolute_rows)
        if relative_digest != absolute_digest:
            raise ValueError(
                f"timestamp origin changed extracted features for sequence {sequence}"
            )

        decoded = decoder.decode_single_csv(
            relative_path,
            context_csv=args.bundle / "tabfm-reference-context.csv",
            context_metadata=args.bundle / "tabfm-reference-context.metadata.json",
            model=model,
        )
        # Evaluator-only truth is accessed here, after decoder output, and is never
        # supplied to decode_single_csv or TabFM.
        truth = variants["relative"]["payload_bits"]
        payload_errors = None if not decoded["accepted"] else sum(
            left != right for left, right in zip(decoded["payload_bits"], truth)
        )
        frame_results.append({
            "sequence": sequence,
            "repetition": int(variants["relative"]["repetition"]),
            "duty_percent": float(variants["relative"]["duty_percent"]),
            "expected_payload_bits": truth,
            "accepted": int(decoded["accepted"]),
            "correct_payload": int(decoded["accepted"] and payload_errors == 0),
            "rejected": int(not decoded["accepted"]),
            "wrong_accepted_payload": int(
                decoded["accepted"] and bool(payload_errors)
            ),
            "payload_bit_errors_conditional_on_accept": payload_errors,
            "decoded_payload_bits": decoded["payload_bits"],
            "mode": decoded["mode"],
            "candidate_count": decoded["candidate_count"],
            "score_margin": decoded["score_margin"],
            "selected_erasures": decoded["selected_erasures"],
            "baseline_hard_rs": decoded["baseline_hard_rs"],
            "sensor_y_hard_rs": decoded["sensor_y_hard_rs"],
            "tabfm_hard_rs": decoded["tabfm_hard_rs"],
            "timestamp_variant_features_identical": 1,
            "query_feature_sha256": relative_digest,
            "relative_csv": variants["relative"]["filename"],
            "relative_sha256": variants["relative"]["file_sha256"],
            "absolute_csv": variants["absolute"]["filename"],
            "absolute_sha256": variants["absolute"]["file_sha256"],
        })
        print(json.dumps(frame_results[-1], indent=2), flush=True)

    aggregate = {
        "frames": len(frame_results),
        "correct_payloads": sum(row["correct_payload"] for row in frame_results),
        "rejections": sum(row["rejected"] for row in frame_results),
        "wrong_accepted_payloads": sum(
            row["wrong_accepted_payload"] for row in frame_results
        ),
        "timestamp_variant_feature_mismatches": 0,
    }
    pinned_verified = args.checkpoint is None
    artifact = {
        "scope": (
            "retrospective fixed-context pipeline validation; query payloads are "
            "absent from the reference context, but some query frames informed "
            "earlier method development and this is not method-level confirmation"
        ),
        "truth_separation": (
            "expected truth remains in the evaluator and is never passed to "
            "decode_single_csv, TabFM, or the RS candidate search"
        ),
        "full_frozen_scope": args.limit is None,
        "bundle_hashes": bundle_hashes,
        "experiment_contract": {
            "reference_sequences": list(pipeline.REFERENCE_CONTEXT_SEQUENCES),
            "reference_repetitions": list(pipeline.REFERENCE_CONTEXT_REPETITIONS),
            "query_repetitions": list(pipeline.HELD_OUT_REPETITIONS),
            "context_rows": pipeline.REFERENCE_BODY_ROWS,
            "query_sync_rows": pipeline.QUERY_SYNC_ROWS,
            "sample_rate_hz": pipeline.NOMINAL_SAMPLE_RATE_HZ,
            "pool_size": soft_list_decoder.FROZEN_POOL_SIZE,
            "acceptance_margin": soft_list_decoder.FROZEN_ACCEPTANCE_MARGIN,
            "autonomous_synchronization": False,
        },
        "model": {
            "repository": model_loader.MODEL_REPOSITORY if pinned_verified else "local checkpoint override",
            "checkpoint_revision": model_loader.CHECKPOINT_REVISION if pinned_verified else None,
            "checkpoint_path": None if args.checkpoint is None else str(args.checkpoint.resolve()),
            "pinned_checkpoint_verified": pinned_verified,
        },
        "software": {
            "rocko_git_revision": git_revision(),
            "validator_sha256": pipeline.sha256_file(Path(__file__)),
            "decoder_sha256": pipeline.sha256_file(HERE / "decode_single_csv.py"),
            "pipeline_sha256": pipeline.sha256_file(HERE / "single_frame_pipeline.py"),
            "python": sys.version,
            "tabfm_version": package_metadata.version("tabfm"),
            "jax_version": package_metadata.version("jax"),
            "jaxlib_version": package_metadata.version("jaxlib"),
            "jax_devices": [str(device) for device in jax.devices()],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "remote_bootstrap_sha256": os.environ.get(
                "ROCKO_REMOTE_BOOTSTRAP_SHA256"
            ),
        },
        "aggregate": aggregate,
        "frames": frame_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    pipeline.write_sidecar(args.output)
    csv_path = args.output.with_suffix(".csv")
    flat_rows = [{
        key: value for key, value in row.items()
        if key not in ("baseline_hard_rs", "sensor_y_hard_rs", "tabfm_hard_rs")
    } for row in frame_results]
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    pipeline.write_sidecar(csv_path)
    print(args.output)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
