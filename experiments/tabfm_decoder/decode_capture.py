#!/usr/bin/env python3
"""Decode one manifest-bounded RS18 frame with TabFM and soft GMD/list RS.

This is a simple offline entry point over the frozen analysis pipeline. It is
not an autonomous receiver: frame boundaries and historical context labels come
from the capture's experiment manifest. Query truth is never passed to TabFM or
to the decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import export_rs18_table as exporter  # noqa: E402
import run_tabfm  # noqa: E402
import soft_list_decoder  # noqa: E402

MODEL_REPOSITORY = "google/tabfm-1.0.0-jax"
CHECKPOINT_REVISION = "d5e74033fcf257699fab013e2cdd7edf424ff904"
FEATURE_SET = "sensor-y-hybrid"
CONTEXT_ROWS = 100
CONTEXT_SEED = 2026


def _load_pinned_model(checkpoint: Path | None):
    checkpoint_path = checkpoint
    if checkpoint_path is None:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(
            repo_id=MODEL_REPOSITORY,
            revision=CHECKPOINT_REVISION,
            allow_patterns=["classification/**"],
        ))
        if snapshot.name != CHECKPOINT_REVISION:
            raise ValueError("downloaded TabFM snapshot does not match pinned revision")
        checkpoint_path = snapshot / "classification"
    return run_tabfm.load_model("jax", checkpoint_path)


def decode_prompt(
    model,
    prompt_rows: Sequence[dict[str, object]],
    *,
    predictor: Callable = run_tabfm.predict_probabilities,
) -> dict[str, object]:
    """Decode an already-extracted leakage-controlled TabFM prompt."""
    allowed_roles = {"historical", "known_sync", "query"}
    roles = {row.get("role") for row in prompt_rows}
    if not roles.issubset(allowed_roles) or "query" not in roles:
        raise ValueError("prompt roles must be historical, known_sync, or query")
    context = [dict(row) for row in prompt_rows if row["role"] != "query"]
    query = sorted(
        (dict(row) for row in prompt_rows if row["role"] == "query"),
        key=lambda row: int(row["body_bit_index"]),
    )
    if len(context) != CONTEXT_ROWS:
        raise ValueError(f"expected {CONTEXT_ROWS} context rows, got {len(context)}")
    if len(query) != 90 or [int(row["body_bit_index"]) for row in query] != list(range(90)):
        raise ValueError("query must contain body-bit indices 0 through 89 exactly once")
    if any(row["target_bit"] not in (0, 1, "0", "1") for row in context):
        raise ValueError("all context rows require binary labels")
    if any(row["target_bit"] not in ("", None) for row in query):
        raise ValueError("query truth must be absent")

    identities = {
        (int(row["sequence"]), int(row["repetition"]), float(row["duty_percent"]))
        for row in query
    }
    if len(identities) != 1:
        raise ValueError("query rows must describe one frame")
    sequence, repetition, duty = next(iter(identities))
    historical = [row for row in context if row["role"] == "historical"]
    known_sync = [row for row in context if row["role"] == "known_sync"]
    if any(row["section"] != "body" for row in historical):
        raise ValueError("historical context must contain only body bits")
    if any(int(row["repetition"]) == repetition for row in historical):
        raise ValueError("historical context leaks the query payload repetition")
    if (len(known_sync) != len(exporter.protocol.SYNC_TEXT)
            or any(row["section"] != "sync" for row in known_sync)
            or {(int(row["sequence"]), int(row["repetition"])) for row in known_sync}
            != {(sequence, repetition)}):
        raise ValueError("known-sync context must be complete and belong to the query")
    if any(row["section"] != "body" for row in query):
        raise ValueError("query must contain only body bits")

    ordered_prompt = context + query
    probabilities = np.asarray(
        predictor(model, ordered_prompt, exporter.FEATURE_SETS[FEATURE_SET]),
        dtype=float,
    )
    llrs = soft_list_decoder.probability_llrs(probabilities)
    decoded = soft_list_decoder.decode(llrs)
    payload = None if decoded["failure"] else "".join(
        str(bit) for bit in decoded["payload"]
    )
    return {
        "accepted": not bool(decoded["failure"]),
        "payload_bits": payload,
        "sequence": sequence,
        "repetition": repetition,
        "duty_percent": duty,
        "decoder": "tabfm-soft-gmd-list-rs18",
        "mode": decoded["mode"],
        "candidate_count": int(decoded["candidate_count"]),
        "score_margin": decoded["margin"],
        "selected_erasures": list(decoded["selected_erasures"]),
        "corrected_symbol_count": int(decoded["corrected_symbol_count"]),
        "list_attempt_count": int(decoded["attempt_count"]),
        "pool_size": soft_list_decoder.FROZEN_POOL_SIZE,
        "acceptance_margin": soft_list_decoder.FROZEN_ACCEPTANCE_MARGIN,
        "tabfm_probability_one": [float(value) for value in probabilities],
    }


def decode_capture(
    query_sequence: int,
    *,
    capture: Path = exporter.DEFAULT_CAPTURE,
    manifest: Path = exporter.DEFAULT_MANIFEST,
    metadata: Path = exporter.DEFAULT_METADATA,
    checkpoint: Path | None = None,
    model=None,
) -> dict[str, object]:
    """Extract and decode one frame from a frozen RS18 experiment capture."""
    extracted, provenance = exporter.extract_rows(capture, manifest, metadata)
    prompt, _evaluator_only_truth = exporter.make_prompt(
        extracted, query_sequence, CONTEXT_ROWS, CONTEXT_SEED
    )
    active_model = model if model is not None else _load_pinned_model(checkpoint)
    result = decode_prompt(active_model, prompt)
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
            "model": MODEL_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_path": None,
            "pinned_checkpoint_verified": True,
        }
    result.update({
        **model_provenance,
        "feature_set": FEATURE_SET,
        "context_rows": CONTEXT_ROWS,
        "context_seed": CONTEXT_SEED,
        "boundary_scope": "frozen manifest-clock boundary; not autonomous",
        "source": provenance,
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--capture", type=Path, default=exporter.DEFAULT_CAPTURE)
    parser.add_argument("--manifest", type=Path, default=exporter.DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=exporter.DEFAULT_METADATA)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-probabilities", action="store_true",
        help="retain the 90 TabFM bit probabilities in JSON output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = decode_capture(
        args.sequence,
        capture=args.capture,
        manifest=args.manifest,
        metadata=args.metadata,
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
