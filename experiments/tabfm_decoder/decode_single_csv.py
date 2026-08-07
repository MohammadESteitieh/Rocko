#!/usr/bin/env python3
"""Decode one start-aligned RS18 signal CSV with fixed-context TabFM soft GMD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]

import analyze_rs18_experiment as rs18  # noqa: E402
import decode_capture as model_loader  # noqa: E402
import export_rs18_table as exporter  # noqa: E402
import run_tabfm  # noqa: E402
import single_frame_pipeline as pipeline  # noqa: E402
import soft_list_decoder  # noqa: E402

DEFAULT_BUNDLE = (
    ROOT / "data/captures/rs18-experiment/derived/tabfm/single-frame-bundle-v1"
)
DEFAULT_CONTEXT = DEFAULT_BUNDLE / "tabfm-reference-context.csv"
DEFAULT_CONTEXT_METADATA = DEFAULT_BUNDLE / "tabfm-reference-context.metadata.json"


def _feature_digest(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for name in exporter.FEATURE_SETS[pipeline.FEATURE_SET]:
            value = row[name]
            rendered = str(value) if name in ("position", "symbol_index", "bit_in_symbol") else f"{float(value):.17g}"
            digest.update(rendered.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _hard_result(llrs: np.ndarray) -> dict[str, object]:
    decoded = rs18.hard_rs18_decode(llrs)
    return {
        "decoder_failure": int(decoded["failure"]),
        "payload_bits": None if decoded["failure"] else "".join(
            str(bit) for bit in decoded["payload"]
        ),
        "corrected_symbol_count": int(decoded["corrected_symbol_count"]),
    }


def decode_single_csv(
    signal_csv: Path,
    *,
    context_csv: Path = DEFAULT_CONTEXT,
    context_metadata: Path = DEFAULT_CONTEXT_METADATA,
    frame_start_seconds: float = 0.0,
    model=None,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    """Decode without reading a manifest or any query ground truth."""
    reference_rows, reference_metadata = pipeline.load_reference_context(
        context_csv, context_metadata
    )
    query_rows = pipeline.extract_rows(
        signal_csv, frame_start_seconds=frame_start_seconds
    )
    prompt = pipeline.build_prompt(reference_rows, query_rows)
    active_model = model if model is not None else model_loader._load_pinned_model(
        checkpoint
    )
    probabilities = np.asarray(run_tabfm.predict_probabilities(
        active_model, prompt, exporter.FEATURE_SETS[pipeline.FEATURE_SET]
    ), dtype=float)
    tabfm_llrs = soft_list_decoder.probability_llrs(probabilities)
    body_rows = query_rows[len(pipeline.protocol.SYNC_TEXT):]
    baseline_llrs = np.asarray([float(row["baseline_llr"]) for row in body_rows])
    sensor_y_llrs = np.asarray([float(row["sensor_y_llr"]) for row in body_rows])
    selected = soft_list_decoder.decode(tabfm_llrs)

    if model is not None:
        model_provenance = {
            "backend": None,
            "model": "injected model object",
            "checkpoint_revision": None,
            "checkpoint_path": None,
            "pinned_checkpoint_verified": False,
        }
    elif checkpoint is not None:
        model_provenance = {
            "backend": "jax",
            "model": "local checkpoint override",
            "checkpoint_revision": None,
            "checkpoint_path": str(checkpoint.resolve()),
            "pinned_checkpoint_verified": False,
        }
    else:
        model_provenance = {
            "backend": "jax",
            "model": model_loader.MODEL_REPOSITORY,
            "checkpoint_revision": model_loader.CHECKPOINT_REVISION,
            "checkpoint_path": None,
            "pinned_checkpoint_verified": True,
        }

    payload = None if selected["failure"] else "".join(
        str(bit) for bit in selected["payload"]
    )
    return {
        "accepted": not bool(selected["failure"]),
        "payload_bits": payload,
        "decoder": "fixed-context-tabfm-soft-gmd-list-rs18",
        "mode": selected["mode"],
        "candidate_count": int(selected["candidate_count"]),
        "score_margin": selected["margin"],
        "selected_erasures": list(selected["selected_erasures"]),
        "corrected_symbol_count": int(selected["corrected_symbol_count"]),
        "list_attempt_count": int(selected["attempt_count"]),
        "pool_size": soft_list_decoder.FROZEN_POOL_SIZE,
        "acceptance_margin": soft_list_decoder.FROZEN_ACCEPTANCE_MARGIN,
        "signal_csv": str(signal_csv.resolve()),
        "signal_sha256": pipeline.sha256_file(signal_csv),
        "frame_start_seconds": frame_start_seconds,
        "query_feature_sha256": _feature_digest(query_rows),
        "reference_context_sha256": reference_metadata["context_sha256"],
        "reference_sequences": reference_metadata["reference_sequences"],
        "reference_repetitions": reference_metadata["reference_repetitions"],
        "baseline_hard_rs": _hard_result(baseline_llrs),
        "sensor_y_hard_rs": _hard_result(sensor_y_llrs),
        "tabfm_hard_rs": _hard_result(tabfm_llrs),
        "tabfm_probability_one": [float(value) for value in probabilities],
        "input_contract": {
            "first_sample_is_sync_start": frame_start_seconds == 0.0,
            "message_seconds": pipeline.protocol.FRAME_BITS * pipeline.protocol.BIT_SECONDS,
            "required_off_tail_seconds": (
                pipeline.POST_FRAME_LEAD_SECONDS
                + pipeline.POST_FRAME_NOISE_SECONDS
            ),
            "autonomous_synchronization": False,
        },
        **model_provenance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signal_csv", type=Path)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--context-metadata", type=Path, default=DEFAULT_CONTEXT_METADATA)
    parser.add_argument("--frame-start-seconds", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-probabilities", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = decode_single_csv(
        args.signal_csv,
        context_csv=args.context,
        context_metadata=args.context_metadata,
        frame_start_seconds=args.frame_start_seconds,
        checkpoint=args.checkpoint,
    )
    if not args.include_probabilities:
        result.pop("tabfm_probability_one", None)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
