"""Truth-free TabFM reliability list decoder for RS(18,6)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]

import analyze_rs18_experiment as rs18  # noqa: E402
import compare_final_rs_frontends as coherent  # noqa: E402
import final_experiment_protocol as gf  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402

FROZEN_POOL_SIZE = 12
FROZEN_ACCEPTANCE_MARGIN = 20.0


def probability_llrs(probability_one: Sequence[float]) -> np.ndarray:
    probabilities = np.asarray(probability_one, dtype=float)
    if (probabilities.shape != (protocol.CODE_BITS,)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0) or np.any(probabilities > 1.0)):
        raise ValueError("TabFM probabilities must be 90 finite values in [0, 1]")
    epsilon = 1e-9
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def decode(llrs: Sequence[float], *, pool_size: int = FROZEN_POOL_SIZE,
           acceptance_margin: float = FROZEN_ACCEPTANCE_MARGIN
           ) -> dict[str, object]:
    """Decode hard first, then search every erasure subset in a reliability pool."""
    values = np.asarray(llrs, dtype=float)
    if (values.shape != (protocol.CODE_BITS,) or not np.all(np.isfinite(values))):
        raise ValueError("soft-list decoder requires exactly 90 finite LLRs")
    if not 1 <= pool_size <= protocol.PARITY_SYMBOLS:
        raise ValueError("pool size must be between 1 and 12 symbols")
    if not math.isfinite(acceptance_margin) or acceptance_margin < 0:
        raise ValueError("acceptance margin must be finite and nonnegative")

    hard_result = rs18.hard_rs18_decode(values)
    if not hard_result["failure"]:
        return {
            "failure": 0,
            "mode": "hard",
            "payload": tuple(hard_result["payload"]),
            "candidate_count": 1,
            "margin": None,
            "selected_erasures": (),
            "corrected_symbol_count": int(hard_result["corrected_symbol_count"]),
            "attempt_count": 0,
        }

    hard_bits = tuple(int(value > 0.0) for value in values)
    hard_symbols = gf.bits_to_symbols(hard_bits)
    reliability = [
        min(abs(value) for value in values[5 * index:5 * index + 5])
        for index in range(protocol.CODE_SYMBOLS)
    ]
    pool = tuple(sorted(
        range(protocol.CODE_SYMBOLS), key=lambda index: (reliability[index], index)
    )[:pool_size])

    candidates: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {}
    attempts = 0
    for count in range(pool_size + 1):
        for erased in itertools.combinations(pool, count):
            attempts += 1
            try:
                word = coherent.rs_decode_erasures(
                    hard_symbols, protocol.DATA_SYMBOLS,
                    protocol.PARITY_SYMBOLS, erased,
                )
            except (ValueError, coherent.GMDDecodeError):
                continue
            bits = gf.symbols_to_bits(word)
            score = float(sum(
                value if bit else -value for bit, value in zip(bits, values)
            ))
            previous = candidates.get(word)
            if (previous is None
                    or (len(erased), erased) < (len(previous[1]), previous[1])):
                candidates[word] = (score, tuple(erased))

    ranked = sorted(
        ((score, word, erased) for word, (score, erased) in candidates.items()),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    if not ranked:
        return {
            "failure": 1, "mode": "list-rejected", "payload": (),
            "candidate_count": 0, "margin": None, "selected_erasures": (),
            "corrected_symbol_count": 0, "attempt_count": attempts,
        }
    top_score, word, erased = ranked[0]
    margin = None if len(ranked) == 1 else float(top_score - ranked[1][0])
    accepted = margin is not None and margin >= acceptance_margin
    return {
        "failure": int(not accepted),
        "mode": "list" if accepted else "list-rejected",
        "payload": tuple(gf.symbols_to_bits(word[:protocol.DATA_SYMBOLS])) if accepted else (),
        "candidate_count": len(ranked),
        "margin": margin,
        "selected_erasures": erased,
        "corrected_symbol_count": sum(a != b for a, b in zip(hard_symbols, word)),
        "attempt_count": attempts,
    }
