#!/usr/bin/env python3
"""Run TabFM binary bit inference on an exported RS18 context/query table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "experiments" / "tabfm_decoder"),
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]

from export_rs18_table import FEATURE_COLUMNS, FEATURE_SETS  # noqa: E402
import analyze_rs18_experiment as rs18  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402
import soft_list_decoder  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export(table_path: Path, truth_path: Path, metadata_path: Path,
                    rows: list[dict[str, str]]) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("experiment") != "tabfm-rs18-bit-frontend-pilot":
        raise ValueError("metadata does not describe the TabFM RS18 pilot")
    if sha256_file(table_path) != metadata.get("table_sha256"):
        raise ValueError("context/query table SHA-256 does not match metadata")
    if sha256_file(truth_path) != metadata.get("truth_sha256"):
        raise ValueError("truth SHA-256 does not match metadata")
    feature_set = metadata.get("feature_set", "aligned")
    if (feature_set not in FEATURE_SETS
            or metadata.get("feature_columns") != FEATURE_SETS[feature_set]):
        raise ValueError("metadata feature set and columns do not match this runner")
    source = metadata.get("source")
    required_source = {
        "capture_path", "capture_sha256", "manifest_path", "manifest_sha256",
        "metadata_path", "metadata_sha256", "manifest_clock_correction_s",
        "boundary_sync_score", "boundary_sync_errors",
    }
    if not isinstance(source, dict) or not required_source.issubset(source):
        raise ValueError("metadata is missing source provenance")
    for key in ("capture_sha256", "manifest_sha256", "metadata_sha256"):
        value = source[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"invalid source digest {key}")

    allowed_roles = {"historical", "known_sync", "query"}
    roles = {row["role"] for row in rows}
    if not roles.issubset(allowed_roles) or "query" not in roles:
        raise ValueError("table contains an invalid role set")
    context_rows = [row for row in rows if row["role"] != "query"]
    query_rows = [row for row in rows if row["role"] == "query"]
    query_identity = {
        (row["sequence"], row["repetition"], row["duty_percent"])
        for row in query_rows
    }
    if len(query_identity) != 1:
        raise ValueError("all query rows must belong to one frame")
    sequence, repetition, _ = next(iter(query_identity))
    if int(sequence) != int(metadata.get("query_sequence", -1)):
        raise ValueError("query sequence does not match metadata")
    if int(repetition) != int(
        metadata.get("query_repetition_held_out_from_historical_context", -1)
    ):
        raise ValueError("query repetition does not match metadata")
    if (len(context_rows) != int(metadata.get("context_rows", -1))
            or len(query_rows) != int(metadata.get("query_rows", -1))):
        raise ValueError("table row counts do not match metadata")

    known_sync = [row for row in context_rows if row["role"] == "known_sync"]
    historical = [row for row in context_rows if row["role"] == "historical"]
    if len(known_sync) != len(protocol.SYNC_TEXT):
        raise ValueError("context must contain the complete known sync")
    sync_by_index = {int(row["frame_bit_index"]): row for row in known_sync}
    if set(sync_by_index) != set(range(len(protocol.SYNC_TEXT))):
        raise ValueError("known sync indices must be complete and unique")
    for index, expected in enumerate(protocol.SYNC_TEXT):
        row = sync_by_index[index]
        if (row["section"] != "sync" or row["target_bit"] != expected
                or (row["sequence"], row["repetition"], row["duty_percent"])
                != next(iter(query_identity))):
            raise ValueError("known sync rows violate the query-frame contract")
    if any(row["section"] != "body" or row["target_bit"] not in ("0", "1")
           or row["repetition"] == repetition for row in historical):
        raise ValueError("historical context leaks or violates the body-bit contract")
    if any(row["section"] != "body" or row["target_bit"]
           for row in query_rows):
        raise ValueError("query rows must be unlabeled body bits")
    query_indices = [int(row["body_bit_index"]) for row in query_rows]
    if (len(set(query_indices)) != len(query_indices)
            or set(query_indices) != set(range(protocol.CODE_BITS))):
        raise ValueError("query body-bit indices must be complete and unique")
    return metadata


def _validate_query_truth(query_rows: list[dict[str, str]],
                          truth_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    expected_indices = set(range(protocol.CODE_BITS))
    query_by_index = {int(row["body_bit_index"]): row for row in query_rows}
    truth_by_index = {int(row["body_bit_index"]): row for row in truth_rows}
    if (len(query_by_index) != len(query_rows)
            or len(truth_by_index) != len(truth_rows)):
        raise ValueError("query and truth body-bit indices must be unique")
    if set(query_by_index) != expected_indices or set(truth_by_index) != expected_indices:
        raise ValueError("query and truth must each contain body-bit indices 0 through 89")
    identity = ("sequence", "repetition", "duty_percent")
    query_identities = {tuple(row[key] for key in identity) for row in query_rows}
    truth_identities = {tuple(row[key] for key in identity) for row in truth_rows}
    if len(query_identities) != 1 or query_identities != truth_identities:
        raise ValueError("query and truth must describe exactly one identical frame")
    for index in expected_indices:
        if any(query_by_index[index][key] != truth_by_index[index][key]
               for key in identity):
            raise ValueError(f"query and truth identity mismatch at body bit {index}")
    return truth_by_index


def evaluate(probability_one: np.ndarray, query_rows: list[dict[str, str]],
             truth_rows: list[dict[str, str]], confidence_threshold: float = 0.75
             ) -> dict[str, object]:
    if len(probability_one) != len(query_rows) or len(query_rows) != len(truth_rows):
        raise ValueError("prediction, query, and truth row counts must match")
    if not 0.5 < confidence_threshold < 1.0:
        raise ValueError("confidence threshold must be strictly between 0.5 and 1")
    truth_by_index = _validate_query_truth(query_rows, truth_rows)
    indexed_query = sorted(
        enumerate(query_rows), key=lambda item: int(item[1]["body_bit_index"])
    )
    ordered_query = [row for _, row in indexed_query]
    ordered_probabilities = np.asarray(
        [probability_one[index] for index, _ in indexed_query], dtype=float
    )
    ordered_truth = [truth_by_index[int(row["body_bit_index"])] for row in ordered_query]
    target = np.asarray([int(row["target_bit"]) for row in ordered_truth], dtype=int)
    baseline_llrs = np.asarray(
        [float(row["baseline_llr"]) for row in ordered_truth], dtype=float
    )
    sensor_y_llrs = np.asarray(
        [float(row["sensor_y_llr"]) for row in ordered_truth], dtype=float
    )
    epsilon = 1e-9
    tabfm_llrs = np.log(np.clip(ordered_probabilities, epsilon, 1.0)) - np.log(
        np.clip(1.0 - ordered_probabilities, epsilon, 1.0)
    )
    confident = ((ordered_probabilities >= confidence_threshold)
                 | (ordered_probabilities <= 1.0 - confidence_threshold))
    gated_llrs = np.where(confident, tabfm_llrs, sensor_y_llrs)
    payload = tuple(int(bit) for bit in ordered_truth[0]["payload_bits"])

    def metrics(prefix: str, llrs: np.ndarray) -> dict[str, object]:
        bits = llrs > 0
        decoded = rs18.hard_rs18_decode(llrs)
        payload_errors = None if decoded["failure"] else sum(
            left != right for left, right in zip(decoded["payload"], payload)
        )
        frame_error = int(bool(decoded["failure"] or payload_errors))
        miscorrection = int(not decoded["failure"] and bool(payload_errors))
        return {
            f"{prefix}_bit_errors": int(np.count_nonzero(bits != target)),
            f"{prefix}_symbol_errors": int(sum(
                np.any(bits[index:index + 5] != target[index:index + 5])
                for index in range(0, len(target), 5)
            )),
            f"{prefix}_decoder_failure": int(decoded["failure"]),
            f"{prefix}_frame_error": frame_error,
            f"{prefix}_wrong_codeword_miscorrection": miscorrection,
            f"{prefix}_corrected_symbol_count": int(
                decoded["corrected_symbol_count"]
            ),
            f"{prefix}_payload_bit_errors_conditional_on_decode": payload_errors,
        }

    result: dict[str, object] = {
        "query_bits": len(target),
        "confidence_threshold": confidence_threshold,
        "confidence_gate_applied_bits": int(np.count_nonzero(confident)),
        "confidence_gate_changed_sensor_y_bits": int(np.count_nonzero(
            confident & ((tabfm_llrs > 0) != (sensor_y_llrs > 0))
        )),
    }
    result.update(metrics("baseline", baseline_llrs))
    result.update(metrics("sensor_y", sensor_y_llrs))
    result.update(metrics("tabfm", tabfm_llrs))
    result.update(metrics("sensor_y_tabfm_gated", gated_llrs))

    list_result = soft_list_decoder.decode(tabfm_llrs)
    list_payload_errors = None if list_result["failure"] else sum(
        left != right for left, right in zip(list_result["payload"], payload)
    )
    result.update({
        "tabfm_soft_list_decoder_failure": int(list_result["failure"]),
        "tabfm_soft_list_frame_error": int(
            bool(list_result["failure"] or list_payload_errors)
        ),
        "tabfm_soft_list_wrong_codeword_miscorrection": int(
            not list_result["failure"] and bool(list_payload_errors)
        ),
        "tabfm_soft_list_payload_bit_errors_conditional_on_decode": (
            list_payload_errors
        ),
        "tabfm_soft_list_mode": list_result["mode"],
        "tabfm_soft_list_candidate_count": list_result["candidate_count"],
        "tabfm_soft_list_margin": list_result["margin"],
        "tabfm_soft_list_selected_erasures": list(
            list_result["selected_erasures"]
        ),
        "tabfm_soft_list_corrected_symbol_count": list_result[
            "corrected_symbol_count"
        ],
        "tabfm_soft_list_attempt_count": list_result["attempt_count"],
    })
    return result


def write_predictions(path: Path, query_rows: list[dict[str, str]],
                      truth_rows: list[dict[str, str]],
                      probability_one: np.ndarray,
                      confidence_threshold: float = 0.75) -> None:
    truth = {int(row["body_bit_index"]): row for row in truth_rows}
    fields = [
        "sequence", "body_bit_index", "symbol_index", "bit_in_symbol",
        "target_bit", "baseline_bit", "baseline_llr", "sensor_y_bit",
        "sensor_y_llr", "tabfm_probability_one", "tabfm_bit",
        "confidence_gate_applied", "sensor_y_tabfm_gated_bit",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(query_rows, probability_one):
            actual = truth[int(row["body_bit_index"])]
            confident = (probability >= confidence_threshold
                         or probability <= 1.0 - confidence_threshold)
            tabfm_bit = int(probability >= 0.5)
            sensor_y_bit = int(actual["sensor_y_bit"])
            writer.writerow({
                "sequence": row["sequence"],
                "body_bit_index": row["body_bit_index"],
                "symbol_index": row["symbol_index"],
                "bit_in_symbol": row["bit_in_symbol"],
                "target_bit": actual["target_bit"],
                "baseline_bit": actual["baseline_bit"],
                "baseline_llr": actual["baseline_llr"],
                "sensor_y_bit": sensor_y_bit,
                "sensor_y_llr": actual["sensor_y_llr"],
                "tabfm_probability_one": f"{float(probability):.12g}",
                "tabfm_bit": tabfm_bit,
                "confidence_gate_applied": int(confident),
                "sensor_y_tabfm_gated_bit": tabfm_bit if confident else sensor_y_bit,
            })


def load_model(backend: str, checkpoint_path: Path | None = None):
    import tabfm
    if backend == "jax":
        return tabfm.tabfm_v1_0_0_jax.load(
            model_type="classification",
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        )
    if backend == "pytorch":
        return tabfm.tabfm_v1_0_0_pytorch.load(
            model_type="classification",
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        )
    raise ValueError(f"unsupported backend {backend!r}")


def predict_probabilities(model, rows: list[dict[str, str]],
                          feature_columns: list[str]) -> np.ndarray:
    import pandas as pd
    import tabfm
    context_rows = [row for row in rows if row["role"] != "query"]
    query_rows = [row for row in rows if row["role"] == "query"]

    def dataframe(selected: list[dict[str, str]]):
        frame = pd.DataFrame([
            {name: row[name] for name in feature_columns} for row in selected
        ])
        categorical = ("position", "symbol_index", "bit_in_symbol")
        numeric = [name for name in feature_columns if name not in categorical]
        frame[numeric] = frame[numeric].astype(float)
        for name in categorical:
            if name in feature_columns:
                frame[name] = frame[name].astype(str)
        return frame

    classifier = tabfm.TabFMClassifier(
        model=model,
        n_estimators=1,
        norm_methods="none",
        class_shift=False,
        max_num_rows=None,
        random_state=2026,
    )
    classifier.fit(
        dataframe(context_rows),
        np.asarray([int(row["target_bit"]) for row in context_rows]),
    )
    probabilities = classifier.predict_proba(dataframe(query_rows))
    class_to_column = {
        int(value): index for index, value in enumerate(classifier.classes_)
    }
    return probabilities[:, class_to_column[1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--backend", choices=("jax", "pytorch"), default="jax")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    try:
        import pandas  # noqa: F401
        import tabfm  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "TabFM environment is missing; install requirements-tabfm.txt"
        ) from error

    rows = read_rows(args.table)
    export_metadata = validate_export(
        args.table, args.truth, args.metadata, rows
    )
    context_rows = [row for row in rows if row["role"] != "query"]
    query_rows = [row for row in rows if row["role"] == "query"]
    truth_rows = read_rows(args.truth)
    if not context_rows or not query_rows:
        raise ValueError("table must contain both context and query rows")
    if any(row["target_bit"] not in ("0", "1") for row in context_rows):
        raise ValueError("all context rows require binary targets")
    if any(row["target_bit"] for row in query_rows):
        raise ValueError("query targets must remain blank")

    feature_columns = list(export_metadata["feature_columns"])
    model = load_model(args.backend)
    probability_one = predict_probabilities(model, rows, feature_columns)

    summary = evaluate(
        probability_one, query_rows, truth_rows, args.confidence_threshold
    )
    summary.update({
        "backend": args.backend,
        "n_estimators": 1,
        "context_rows": len(context_rows),
        "feature_set": export_metadata.get("feature_set", "aligned"),
        "feature_count": len(feature_columns),
        "model": "google/tabfm-1.0.0",
        "table_sha256": export_metadata["table_sha256"],
        "truth_sha256": export_metadata["truth_sha256"],
        "export_metadata_sha256": sha256_file(args.metadata),
        "source": export_metadata["source"],
    })
    write_predictions(
        args.output, query_rows, truth_rows, probability_one,
        args.confidence_threshold,
    )
    summary["predictions_sha256"] = sha256_file(args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
