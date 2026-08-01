#!/usr/bin/env python3
"""Duong et al. gain-modulated recurrent whitening for receiver features.

This implements the offline covariance-training form of Algorithm 1 from
"Adaptive Whitening in Neural Populations with Gain-modulating Interneurons"
(ICML 2023).  The fixed recurrent frame W is not trained; only interneuron
gains g adapt.  At equilibrium the response is

    y = [I + W diag(g) W.T]^-1 x.

The model whitens instantaneous feature covariance.  It is not a temporal GRU
and does not, by itself, guarantee temporal independence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DuongWhitener:
    mean: np.ndarray
    scale: float
    frame: np.ndarray
    gains: np.ndarray
    matrix: np.ndarray
    iterations: int
    training_error: float

    def transform(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError(f"expected samples with shape (n, {len(self.mean)})")
        return ((values - self.mean) / self.scale) @ self.matrix.T

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            mean=self.mean,
            scale=np.array(self.scale),
            frame=self.frame,
            gains=self.gains,
            matrix=self.matrix,
            iterations=np.array(self.iterations),
            training_error=np.array(self.training_error),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DuongWhitener":
        data = np.load(path)
        return cls(
            mean=data["mean"],
            scale=float(data["scale"]),
            frame=data["frame"],
            gains=data["gains"],
            matrix=data["matrix"],
            iterations=int(data["iterations"]),
            training_error=float(data["training_error"]),
        )


def feature_matrix(channels: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    """Represent two complex analytic sensors as four real neural responses."""
    if len(channels) != 2:
        raise ValueError("exactly two analytic channels are required")
    first, second = map(np.asarray, channels)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("analytic channels must be equal-length vectors")
    return np.column_stack((first.real, first.imag, second.real, second.imag))


def complex_channels(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert four real neural responses back to two analytic channels."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("expected four real features")
    return values[:, 0] + 1j * values[:, 1], values[:, 2] + 1j * values[:, 3]


def random_frame(dimensions: int, *, seed: int = 7) -> np.ndarray:
    """Return the paper's minimal overcomplete fixed frame, K=N(N+1)/2."""
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    projections = dimensions * (dimensions + 1) // 2
    rng = np.random.default_rng(seed)
    frame = rng.standard_normal((dimensions, projections))
    frame /= np.linalg.norm(frame, axis=0, keepdims=True)
    outer_products = np.stack(
        [np.outer(frame[:, i], frame[:, i]).reshape(-1) for i in range(projections)]
    )
    if np.linalg.matrix_rank(outer_products) < projections:
        raise ValueError("fixed frame does not span the symmetric matrix space")
    return frame


def fit(
    samples: np.ndarray,
    *,
    learning_rate: float = 0.1,
    max_iterations: int = 60_000,
    tolerance: float = 1e-5,
    seed: int = 7,
) -> DuongWhitener:
    """Fit gain variables using the paper's offline covariance update.

    A single global scale places the smallest covariance eigenvalue just above
    the unit target.  It improves numerical stability without independently
    standardizing features (which would perform part of the whitening first).
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("need a two-dimensional array with at least two samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("training samples must be finite")
    if learning_rate <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid optimization parameters")

    mean = values.mean(axis=0)
    centered = values - mean
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0:
        raise ValueError("training covariance must be positive definite")
    scale = float(0.95 * np.sqrt(eigenvalues[0]))
    covariance = covariance / scale**2

    dimensions = values.shape[1]
    frame = random_frame(dimensions, seed=seed)
    gains = np.zeros(frame.shape[1])
    identity = np.eye(dimensions)
    error = float("inf")

    for iteration in range(1, max_iterations + 1):
        recurrent = identity + frame @ (gains[:, None] * frame.T)
        if np.linalg.eigvalsh(recurrent)[0] <= 1e-8:
            raise RuntimeError("gain update made the recurrent matrix singular")
        matrix = np.linalg.inv(recurrent)
        output_covariance = matrix @ covariance @ matrix.T
        projected_variances = np.sum(frame * (output_covariance @ frame), axis=0)
        gains = gains + learning_rate * (projected_variances - 1.0)
        error = float(np.linalg.norm(output_covariance - identity, ord="fro") / dimensions)
        if np.max(np.abs(projected_variances - 1.0)) <= tolerance:
            break
    else:
        iteration = max_iterations

    recurrent = identity + frame @ (gains[:, None] * frame.T)
    matrix = np.linalg.inv(recurrent)
    output_covariance = matrix @ covariance @ matrix.T
    error = float(np.linalg.norm(output_covariance - identity, ord="fro") / dimensions)
    return DuongWhitener(mean, scale, frame, gains, matrix, iteration, error)
