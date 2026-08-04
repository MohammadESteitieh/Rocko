#!/usr/bin/env python3
"""Export a leakage-controlled RS18 bit table for a TabFM pilot.

This is a frontend experiment, not a complete receiver evaluation. Frame
boundaries reproduce the accepted manifest-clock experiment contract. Query
truth is written to a separate evaluator-only file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]

import analyze_final_experiment as base  # noqa: E402
import analyze_rs18_experiment as rs18  # noqa: E402
import compare_final_rs_frontends as coherent  # noqa: E402
import final_experiment_protocol as gf  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402
import run_rs18_experiment as runner  # noqa: E402

DEFAULT_CAPTURE = ROOT / "data/captures/rs18-experiment/raw/rs18_experiment_11V_20260728_091828.csv"
DEFAULT_MANIFEST = ROOT / "data/captures/rs18-experiment/manifests/qnx/rs18_experiment_11V_20260728_091828.transmitter.csv"
DEFAULT_METADATA = ROOT / "data/captures/rs18-experiment/manifests/rs18_experiment_11V_20260728_091828.metadata.txt"

IDENTITY_COLUMNS = [
    "role", "sequence", "repetition", "duty_percent", "section",
    "frame_bit_index", "body_bit_index", "target_bit",
]
FEATURE_COLUMNS = [
    "position", "symbol_index", "bit_in_symbol",
    "x_first_i", "x_first_q", "x_second_i", "x_second_q",
    "x_first_abs", "x_second_abs", "x_delta_i", "x_delta_q",
    "x_delta_abs", "x_aligned_i", "x_aligned_q", "x_sync_abs",
    "y_first_i", "y_first_q", "y_second_i", "y_second_q",
    "y_first_abs", "y_second_abs", "y_delta_i", "y_delta_q",
    "y_delta_abs", "y_aligned_i", "y_aligned_q", "y_sync_abs",
    "pooled_aligned_i", "pooled_aligned_q",
]
SENSOR_Y_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "sensor_y_llr", "sensor_y_abs_llr",
]
FEATURE_SETS = {
    "aligned": FEATURE_COLUMNS,
    "sensor-y-hybrid": SENSOR_Y_FEATURE_COLUMNS,
}
DIAGNOSTIC_COLUMNS = [
    "baseline_llr", "baseline_bit", "sensor_y_llr", "sensor_y_abs_llr",
    "sensor_y_bit",
]
TABLE_COLUMNS = IDENTITY_COLUMNS + FEATURE_COLUMNS + DIAGNOSTIC_COLUMNS
TRUTH_COLUMNS = [
    "sequence", "repetition", "duty_percent", "body_bit_index",
    "symbol_index", "bit_in_symbol", "target_bit", "baseline_llr",
    "baseline_bit", "sensor_y_llr", "sensor_y_bit", "payload_bits",
]


def _manifest(path: Path) -> list[dict[str, str]]:
    runner.validate_manifest(path)
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _complex_features(prefix: str, first: complex, second: complex,
                      channel: complex) -> dict[str, float]:
    delta = first - second
    aligned = delta * np.conj(channel) / max(abs(channel) ** 2, 1e-12)
    return {
        f"{prefix}_first_i": float(first.real),
        f"{prefix}_first_q": float(first.imag),
        f"{prefix}_second_i": float(second.real),
        f"{prefix}_second_q": float(second.imag),
        f"{prefix}_first_abs": float(abs(first)),
        f"{prefix}_second_abs": float(abs(second)),
        f"{prefix}_delta_i": float(delta.real),
        f"{prefix}_delta_q": float(delta.imag),
        f"{prefix}_delta_abs": float(abs(delta)),
        f"{prefix}_aligned_i": float(aligned.real),
        f"{prefix}_aligned_q": float(aligned.imag),
        f"{prefix}_sync_abs": float(abs(channel)),
    }


def _verify_sidecar(path: Path, actual: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing SHA-256 sidecar for {path.name}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[0] != actual or Path(fields[1]).name != path.name:
        raise ValueError(f"invalid SHA-256 sidecar for {path.name}")


def _verify_recorded_hash(path: Path, info: dict[str, str]) -> str:
    key = f"sha256_{path.name}"
    if key not in info:
        raise ValueError(f"metadata does not bind {path.name} to a SHA-256 digest")
    actual = base.sha256_file(path)
    if actual != info[key]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    _verify_sidecar(path, actual)
    return actual


def extract_rows(capture_path: Path, manifest_path: Path,
                 metadata_path: Path
                 ) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract per-bit observables and provenance for all 45 frames."""
    info = base.metadata(metadata_path)
    rs18.validate_metadata(info)
    capture_digest = _verify_recorded_hash(capture_path, info)
    manifest_digest = _verify_recorded_hash(manifest_path, info)
    metadata_digest = base.sha256_file(metadata_path)
    _verify_sidecar(metadata_path, metadata_digest)
    t, x, y = base.load_capture(capture_path)
    fs = base.sample_rate(t)
    manifest = _manifest(manifest_path)

    off_start, off_stop = rs18.prelaunch_indices(info, fs, len(t))
    means, rms = rs18.fit_off_rms(x, y, off_start, off_stop)
    normalized = tuple(
        signal.hilbert((np.asarray(channel, float) - mean) / scale)
        for channel, mean, scale in zip((x, y), means, rms)
    )
    raw = tuple(
        signal.hilbert(np.asarray(channel, float) - np.median(channel))
        for channel in (x, y)
    )
    starts, clock_correction_s, boundary_sync_score, boundary_sync_errors = (
        rs18.locate_raw_coherent_sync_boundaries(
            manifest, base.utc_seconds(info["capture_started_utc"]), raw, fs
        )
    )

    half = round(fs * protocol.BIT_SECONDS / 2)
    sync_halves = len(protocol.SYNC_TEXT) * 2
    sync_gate = base.manchester_levels(protocol.SYNC_TEXT).astype(bool)
    output: list[dict[str, object]] = []

    for manifest_row, start in zip(manifest, starts):
        frame_text = manifest_row["frame_bits"]
        frame_bits = len(frame_text)
        frame_samples = 2 * frame_bits * half
        gap_stop = (
            start + frame_samples
            + round((base.CENTRAL_GAP_LEAD_SECONDS + base.CENTRAL_GAP_SECONDS) * fs)
        )
        if gap_stop > len(t):
            raise ValueError(f"frame {manifest_row['sequence']} or gap is incomplete")

        normalized_local = tuple(
            channel[start:start + frame_samples] for channel in normalized
        )
        phasors = coherent._extract_phasors(
            normalized_local, 0, 2 * frame_bits, half, fs
        )
        sync = phasors[:sync_halves]
        channel_vector = sync[sync_gate].mean(axis=0) - sync[~sync_gate].mean(axis=0)

        raw_local = tuple(channel[start:gap_stop] for channel in raw)
        baseline_llrs = coherent.coherent_llrs(raw_local, frame_bits, fs)
        sensor_y_llrs = coherent.coherent_llrs((raw_local[1],), frame_bits, fs)
        if len(baseline_llrs) != frame_bits or len(sensor_y_llrs) != frame_bits:
            raise ValueError("coherent frontend returned the wrong number of bits")

        for frame_index, target_character in enumerate(frame_text):
            section = "sync" if frame_index < len(protocol.SYNC_TEXT) else "body"
            body_index = frame_index - len(protocol.SYNC_TEXT)
            if section == "sync":
                position = f"sync_{frame_index:02d}"
                symbol_index: object = "sync"
                bit_in_symbol: object = frame_index
            else:
                position = f"body_{body_index:02d}"
                symbol_index = body_index // gf.FIELD_BITS
                bit_in_symbol = body_index % gf.FIELD_BITS

            first, second = phasors[2 * frame_index:2 * frame_index + 2]
            row: dict[str, object] = {
                "role": "unassigned",
                "sequence": int(manifest_row["sequence"]),
                "repetition": int(manifest_row["repetition"]),
                "duty_percent": float(manifest_row["duty_percent"]),
                "section": section,
                "frame_bit_index": frame_index,
                "body_bit_index": body_index if section == "body" else "",
                "target_bit": int(target_character),
                "position": position,
                "symbol_index": symbol_index,
                "bit_in_symbol": bit_in_symbol,
                "baseline_llr": float(baseline_llrs[frame_index]),
                "baseline_bit": int(baseline_llrs[frame_index] > 0),
                "sensor_y_llr": float(sensor_y_llrs[frame_index]),
                "sensor_y_abs_llr": float(abs(sensor_y_llrs[frame_index])),
                "sensor_y_bit": int(sensor_y_llrs[frame_index] > 0),
            }
            row.update(_complex_features("x", first[0], second[0], channel_vector[0]))
            row.update(_complex_features("y", first[1], second[1], channel_vector[1]))
            pooled = (
                complex(row["x_aligned_i"], row["x_aligned_q"])
                + complex(row["y_aligned_i"], row["y_aligned_q"])
            )
            row["pooled_aligned_i"] = float(pooled.real)
            row["pooled_aligned_q"] = float(pooled.imag)
            output.append(row)
    provenance = {
        "capture_path": str(capture_path.resolve()),
        "capture_sha256": capture_digest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_digest,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": metadata_digest,
        "sample_rate_hz": fs,
        "manifest_clock_correction_s": clock_correction_s,
        "boundary_sync_score": boundary_sync_score,
        "boundary_sync_errors": boundary_sync_errors,
    }
    return output, provenance


def _balanced_sample(rows: Iterable[dict[str, object]], count: int,
                     seed: int) -> list[dict[str, object]]:
    by_bit = {0: [], 1: []}
    for row in rows:
        by_bit[int(row["target_bit"])].append(row)
    rng = random.Random(seed)
    for values in by_bit.values():
        rng.shuffle(values)
    wanted = {0: count // 2, 1: count - count // 2}
    selected = by_bit[0][:wanted[0]] + by_bit[1][:wanted[1]]
    if len(selected) != count:
        raise ValueError("not enough historical rows for balanced context")
    rng.shuffle(selected)
    return selected


def make_prompt(rows: list[dict[str, object]], query_sequence: int,
                context_rows: int = 100, seed: int = 2026
                ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return mixed context/query rows and separately held query truth."""
    target = [row for row in rows if int(row["sequence"]) == query_sequence]
    if not target:
        raise ValueError(f"query sequence {query_sequence} is absent")
    target_repetitions = {int(row["repetition"]) for row in target}
    if len(target_repetitions) != 1:
        raise ValueError("query frame does not have one repetition")
    target_repetition = next(iter(target_repetitions))
    known_sync = [dict(row, role="known_sync") for row in target
                  if row["section"] == "sync"]
    query = [dict(row, role="query", target_bit="") for row in target
             if row["section"] == "body"]
    historical_count = context_rows - len(known_sync)
    if historical_count < 2:
        raise ValueError("context_rows must leave room for both historical labels")
    candidates = [
        row for row in rows
        if row["section"] == "body"
        and int(row["repetition"]) != target_repetition
    ]
    historical = [dict(row, role="historical") for row in _balanced_sample(
        candidates, historical_count, seed
    )]

    truth = []
    target_body = [row for row in target if row["section"] == "body"]
    payload = "".join(
        str(row["target_bit"])
        for row in sorted(target_body, key=lambda row: int(row["body_bit_index"]))
        [:protocol.DATA_SYMBOLS * gf.FIELD_BITS]
    )
    for original in target:
        if original["section"] != "body":
            continue
        truth.append({column: original[column] for column in TRUTH_COLUMNS
                      if column != "payload_bits"})
        truth[-1]["payload_bits"] = payload
    return historical + known_sync + query, truth


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--query-sequence", type=int, required=True)
    parser.add_argument("--context-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--feature-set", choices=tuple(FEATURE_SETS), default="aligned"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/rocko-tabfm"))
    args = parser.parse_args()

    extracted, provenance = extract_rows(
        args.capture, args.manifest, args.metadata
    )
    prompt, truth = make_prompt(
        extracted, args.query_sequence, args.context_rows, args.seed
    )
    stem = f"rs18-sequence-{args.query_sequence:02d}"
    table_path = args.output_dir / f"{stem}.context-query.csv"
    truth_path = args.output_dir / f"{stem}.truth.csv"
    metadata_path = args.output_dir / f"{stem}.metadata.json"
    write_csv(table_path, TABLE_COLUMNS, prompt)
    write_csv(truth_path, TRUTH_COLUMNS, truth)
    metadata_path.write_text(json.dumps({
        "experiment": "tabfm-rs18-bit-frontend-pilot",
        "boundary_scope": "accepted manifest-clock boundary contract; not autonomous",
        "query_sequence": args.query_sequence,
        "query_repetition_held_out_from_historical_context": int(truth[0]["repetition"]),
        "context_rows": sum(row["role"] != "query" for row in prompt),
        "context_seed": args.seed,
        "query_rows": sum(row["role"] == "query" for row in prompt),
        "feature_set": args.feature_set,
        "feature_columns": FEATURE_SETS[args.feature_set],
        "table": str(table_path),
        "table_sha256": base.sha256_file(table_path),
        "truth": str(truth_path),
        "truth_sha256": base.sha256_file(truth_path),
        "source": provenance,
    }, indent=2) + "\n", encoding="utf-8")
    print(table_path)
    print(truth_path)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
