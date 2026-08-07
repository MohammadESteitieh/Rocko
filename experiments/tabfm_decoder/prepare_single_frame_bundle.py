#!/usr/bin/env python3
"""Build held-out one-message CSVs, fixed TabFM context, and ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "receiver" / "legacy_decoder"),
]

import analyze_final_experiment as base  # noqa: E402
import export_rs18_table as exporter  # noqa: E402
import interactive_run_viewer as accepted  # noqa: E402
import single_frame_pipeline as pipeline  # noqa: E402

DEFAULT_MANIFEST = exporter.DEFAULT_MANIFEST
DEFAULT_OUTPUT = (
    ROOT / "data/captures/rs18-experiment/derived/tabfm/single-frame-bundle-v1"
)
EXPECTED_COLUMNS = [
    "filename", "file_sha256", "time_origin", "sequence", "repetition",
    "duty_percent", "payload_bits", "coded_body_bits", "sample_count",
    "sample_rate_hz", "frame_start_seconds", "message_seconds",
    "required_off_tail_seconds", "validation_role",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def prepare_output_directory(path: Path, *, force: bool) -> None:
    if path.is_symlink():
        raise ValueError("refusing to replace a symlinked output directory")
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"refusing to overwrite {path}; pass --force")
        marker = path / "bundle-metadata.json"
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "--force may replace only a previously generated single-frame bundle"
            ) from error
        if metadata.get("artifact") != "rocko-single-frame-validation-bundle-v1":
            raise ValueError(
                "--force may replace only a previously generated single-frame bundle"
            )
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_signal(path: Path, t: np.ndarray, x: np.ndarray, y: np.ndarray,
                 *, relative: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(t, float)
    if relative:
        times = times - times[0]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("t", "x", "y"))
        for timestamp, first, second in zip(times, x, y):
            writer.writerow((f"{timestamp:.6f}", int(round(first)), int(round(second))))
    return pipeline.write_sidecar(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=exporter.DEFAULT_CAPTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=exporter.DEFAULT_METADATA)
    parser.add_argument("--analysis", type=Path, default=accepted.DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for key, path in (
        ("capture", args.capture), ("metadata", args.metadata),
        ("analysis", args.analysis),
    ):
        accepted.verify_frozen_file(path, accepted.FROZEN_SHA256[key])
    metadata_info = base.metadata(args.metadata)
    manifest_hash = exporter._verify_recorded_hash(args.manifest, metadata_info)
    manifest = read_csv(args.manifest)
    analysis = read_csv(args.analysis)
    by_manifest = {int(row["sequence"]): row for row in manifest}
    by_analysis = {int(row["sequence"]): row for row in analysis}
    if set(by_manifest) != set(range(1, 46)) or set(by_analysis) != set(range(1, 46)):
        raise ValueError("source run must contain all 45 frames")

    prepare_output_directory(args.output_dir, force=args.force)

    t, x, y = base.load_capture(args.capture)
    fs = round(base.sample_rate(t), 6)
    required_samples = round(pipeline.REQUIRED_SECONDS * fs)

    extracted_reference: list[dict[str, object]] = []
    for sequence in pipeline.REFERENCE_CONTEXT_SEQUENCES:
        manifest_row = by_manifest[sequence]
        start = round(float(by_analysis[sequence]["start_offset_s"]) * fs)
        stop = start + required_samples
        rows = pipeline.extract_rows_from_arrays(
            t[start:stop] - t[start], x[start:stop], y[start:stop], fs,
            sequence=sequence,
            repetition=int(manifest_row["repetition"]),
            duty_percent=float(manifest_row["duty_percent"]),
            frame_text=manifest_row["frame_bits"],
        )
        extracted_reference.extend(row for row in rows if row["section"] == "body")
    selected = exporter._balanced_sample(
        extracted_reference, pipeline.REFERENCE_BODY_ROWS, pipeline.CONTEXT_SEED
    )
    selected = [dict(row, role="historical") for row in selected]
    context_path = args.output_dir / "tabfm-reference-context.csv"
    pipeline.write_rows(context_path, selected)
    context_hash = pipeline.write_sidecar(context_path)

    expected_rows: list[dict[str, object]] = []
    for sequence in range(1, 46):
        manifest_row = by_manifest[sequence]
        repetition = int(manifest_row["repetition"])
        if repetition not in pipeline.HELD_OUT_REPETITIONS:
            continue
        start = round(float(by_analysis[sequence]["start_offset_s"]) * fs)
        stop = start + required_samples
        if stop > len(t):
            raise ValueError(f"frame {sequence} does not contain the required tail")
        for time_origin, relative in (("relative", True), ("absolute", False)):
            filename = f"frames/{time_origin}/rs18-sequence-{sequence:02d}.csv"
            path = args.output_dir / filename
            digest = write_signal(
                path, t[start:stop], x[start:stop], y[start:stop], relative=relative
            )
            expected_rows.append({
                "filename": filename,
                "file_sha256": digest,
                "time_origin": time_origin,
                "sequence": sequence,
                "repetition": repetition,
                "duty_percent": float(manifest_row["duty_percent"]),
                "payload_bits": manifest_row["payload_bits"],
                "coded_body_bits": manifest_row["coded_body_bits"],
                "sample_count": required_samples,
                "sample_rate_hz": f"{fs:.12g}",
                "frame_start_seconds": 0.0,
                "message_seconds": protocol_message_seconds(),
                "required_off_tail_seconds": (
                    pipeline.POST_FRAME_LEAD_SECONDS
                    + pipeline.POST_FRAME_NOISE_SECONDS
                ),
                "validation_role": "payload-reference-held-out-retrospective",
            })

    expected_path = args.output_dir / "expected-results.csv"
    with expected_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(expected_rows)
    expected_hash = pipeline.write_sidecar(expected_path)

    context_metadata_path = args.output_dir / "tabfm-reference-context.metadata.json"
    context_metadata = {
        "artifact": "rocko-tabfm-fixed-reference-context-v1",
        "feature_set": pipeline.FEATURE_SET,
        "feature_columns": exporter.FEATURE_SETS[pipeline.FEATURE_SET],
        "context_rows": pipeline.REFERENCE_BODY_ROWS,
        "class_rows": {"0": 42, "1": 42},
        "context_seed": pipeline.CONTEXT_SEED,
        "reference_sequences": list(pipeline.REFERENCE_CONTEXT_SEQUENCES),
        "reference_repetitions": list(pipeline.REFERENCE_CONTEXT_REPETITIONS),
        "held_out_repetitions": list(pipeline.HELD_OUT_REPETITIONS),
        "context_sha256": context_hash,
        "extraction": {
            "frame_boundary": "first CSV sample; no autonomous synchronization",
            "message_seconds": protocol_message_seconds(),
            "post_frame_lead_seconds": pipeline.POST_FRAME_LEAD_SECONDS,
            "post_frame_noise_seconds": pipeline.POST_FRAME_NOISE_SECONDS,
            "analytic_padding_seconds": pipeline.ANALYTIC_PADDING_SECONDS,
            "sample_rate_hz": pipeline.NOMINAL_SAMPLE_RATE_HZ,
            "sample_interval_tolerance_seconds": pipeline.SAMPLE_INTERVAL_TOLERANCE_SECONDS,
            "capture_duration_tolerance_seconds": pipeline.CAPTURE_DURATION_TOLERANCE_SECONDS,
            "exact_samples_from_frame_start": required_samples,
            "normalization": "per-file declared trailing off-noise mean and RMS",
        },
        "source": {
            "capture_sha256": accepted.FROZEN_SHA256["capture"],
            "metadata_sha256": accepted.FROZEN_SHA256["metadata"],
            "analysis_sha256": accepted.FROZEN_SHA256["analysis"],
            "manifest_sha256": manifest_hash,
        },
        "software": pipeline.extraction_software_contract(),
        "preparation": {
            "prepare_script_sha256": pipeline.sha256_file(Path(__file__)),
        },
    }
    context_metadata_path.write_text(
        json.dumps(context_metadata, indent=2) + "\n", encoding="utf-8"
    )
    context_metadata_hash = pipeline.write_sidecar(context_metadata_path)

    readme_path = args.output_dir / "README.md"
    readme_path.write_text(
        "# Single-frame TabFM validation bundle v1\n\n"
        "Each signal CSV contains one 53-second RS18 message beginning at its first "
        "sample, followed by a 2.5-second guard and 10-second transmitter-off noise "
        "interval required by the frozen coherent frontend. `relative` files start "
        "at timestamp zero; `absolute` files preserve the original capture clock. "
        "The decoder must produce identical features and decisions for both.\n\n"
        "The fixed 84-row labelled TabFM reference context comes only from the "
        "100%-duty frames for payload repetitions 1–3. Query CSVs contain only "
        "repetitions 4–5, so their payloads are absent from the reference context. "
        "Some query frames informed earlier method development, so this bundle is "
        "retrospective pipeline validation, not method-level confirmation. "
        "`expected-results.csv` is evaluator-only ground truth and must never be "
        "read by the decoder. Autonomous synchronization is outside this bundle's "
        "contract.\n",
        encoding="utf-8",
    )
    readme_hash = pipeline.write_sidecar(readme_path)

    bundle_metadata_path = args.output_dir / "bundle-metadata.json"
    bundle_metadata_path.write_text(json.dumps({
        "artifact": "rocko-single-frame-validation-bundle-v1",
        "scope": "retrospective pipeline validation; payload absent from reference context, not method-level confirmation",
        "held_out_logical_frames": 18,
        "timestamp_variants_per_frame": 2,
        "signal_csv_files": len(expected_rows),
        "expected_results_sha256": expected_hash,
        "reference_context_sha256": context_hash,
        "reference_context_metadata_sha256": context_metadata_hash,
        "readme_sha256": readme_hash,
        "contract": {
            "first_sample_is_sync_start": True,
            "timestamp_origin_is_semantically_ignored": True,
            "message_seconds": protocol_message_seconds(),
            "required_off_tail_seconds": (
                pipeline.POST_FRAME_LEAD_SECONDS
                + pipeline.POST_FRAME_NOISE_SECONDS
            ),
            "sample_rate_hz": pipeline.NOMINAL_SAMPLE_RATE_HZ,
            "exact_samples_from_frame_start": required_samples,
            "autonomous_synchronization": False,
        },
    }, indent=2) + "\n", encoding="utf-8")
    pipeline.write_sidecar(bundle_metadata_path)
    print(args.output_dir)
    return 0


def protocol_message_seconds() -> float:
    import rs18_experiment_protocol as protocol
    return protocol.FRAME_BITS * protocol.BIT_SECONDS


if __name__ == "__main__":
    raise SystemExit(main())
