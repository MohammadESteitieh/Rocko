#!/usr/bin/env python3
"""Coherent sensors -> Manchester GNB LLRs -> codebook SLNN.

The bundled model is deliberately provisional: it is fitted to synthetic
matched-filter observations. It exists to exercise the complete inference
path until a model can be fitted from labeled real frames and evaluated on
held-out physical captures.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

import coded_protocol as protocol
import layered_decoder
import slnn_decoder


@dataclass(frozen=True)
class GaussianNaiveBayes:
    means: np.ndarray       # shape (2, features)
    variances: np.ndarray   # shape (2, features)
    log_priors: np.ndarray  # shape (2,)

    @classmethod
    def fit(
        cls, features: Sequence[Sequence[float]], labels: Sequence[int],
        *, variance_smoothing: float = 1e-3,
    ) -> "GaussianNaiveBayes":
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or len(x) != len(y) or x.shape[1] < 1:
            raise ValueError("features must be a nonempty 2-D array matching labels")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("training labels must contain both zero and one")
        scale = max(float(np.max(np.var(x, axis=0))), 1e-9)
        floor = variance_smoothing * scale
        means = np.stack([x[y == label].mean(axis=0) for label in (0, 1)])
        variances = np.stack([
            np.maximum(x[y == label].var(axis=0), floor) for label in (0, 1)
        ])
        # Equal bit priors prevent the fixed tilde/alphabet distribution from
        # becoming an accidental channel bias.
        return cls(means, variances, np.full(2, -np.log(2.0)))

    def log_likelihoods(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.atleast_2d(np.asarray(features, dtype=float))
        if x.shape[1] != self.means.shape[1]:
            raise ValueError(f"expected {self.means.shape[1]} features per observation")
        scores = []
        for label in (0, 1):
            scores.append(
                self.log_priors[label]
                - 0.5 * np.sum(
                    np.log(2 * np.pi * self.variances[label])
                    + (x - self.means[label]) ** 2 / self.variances[label],
                    axis=1,
                )
            )
        return np.stack(scores, axis=1)

    def llrs(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        scores = self.log_likelihoods(features)
        return scores[:, 1] - scores[:, 0]

    def save(self, path: str | Path) -> None:
        np.savez(
            path, means=self.means, variances=self.variances,
            log_priors=self.log_priors,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GaussianNaiveBayes":
        with np.load(path) as data:
            return cls(data["means"], data["variances"], data["log_priors"])


@dataclass(frozen=True)
class HybridResult:
    restricted: slnn_decoder.SLNNResult
    full: slnn_decoder.SLNNResult
    llrs: np.ndarray
    hard_bits: tuple[int, ...]
    accepted: bool
    tone_coherence: float
    silence_coherence: float


@lru_cache(maxsize=1)
def synthetic_gnb() -> GaussianNaiveBayes:
    """Temporary broad-SNR model matching the old synthetic benchmark style."""
    rng = np.random.default_rng(2707)
    count = 20_000
    labels = rng.integers(0, 2, count, dtype=np.int8)
    severity = rng.uniform(0.02, 0.48, count)
    active = np.clip(0.96 - severity + rng.normal(0, 0.10, count), 0, 1.2)
    inactive = np.clip(0.04 + 0.65 * severity + rng.normal(0, 0.10, count), 0, 1.2)
    first = np.where(labels == 1, active, inactive)
    second = np.where(labels == 0, active, inactive)
    return GaussianNaiveBayes.fit(np.column_stack((first, second)), labels)


def coherent_features(
    channels: Sequence[np.ndarray], start: int, fs: float
) -> tuple[np.ndarray, slnn_decoder.CoherentSoftResult]:
    """Return two normalized Manchester-half amplitudes for every coded bit."""
    coherent = slnn_decoder.coherent_soft_symbols(channels, start, fs)
    combined = coherent.weights.conj() @ np.vstack(channels)
    first, second = layered_decoder.matched_observations([combined], start, fs)
    return np.column_stack((first, second)), coherent


def decode(
    channels: Sequence[np.ndarray], start: int, fs: float,
    model: GaussianNaiveBayes | None = None,
    expected_letter: str | None = None,
) -> HybridResult:
    model = model or synthetic_gnb()
    features, coherent = coherent_features(channels, start, fs)
    llrs = model.llrs(features)
    restricted = slnn_decoder.decode_alphabet(llrs, expected_letter)
    expected_value = None
    if expected_letter is not None:
        expected_value = (protocol.HEADER_BYTE << 8) | ord(expected_letter.upper())
    full = slnn_decoder.decode_full(llrs, expected_value)
    accepted = (
        full.header == protocol.HEADER_BYTE
        and 65 <= full.letter_byte <= 90
        and full.letter_byte == restricted.letter_byte
    )
    return HybridResult(
        restricted=restricted,
        full=full,
        llrs=llrs,
        hard_bits=tuple(map(int, llrs > 0)),
        accepted=accepted,
        tone_coherence=coherent.tone_coherence,
        silence_coherence=coherent.silence_coherence,
    )
