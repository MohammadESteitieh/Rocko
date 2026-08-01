#!/usr/bin/env python3
"""Causal temporal-noise predictors for experimental receiver whitening.

Models operate on four real complex-baseband features: I/Q for each sensor.
They predict the current noise from past observations; prediction innovations
are covariance-normalized before conversion back to analytic 8 Hz channels.
These are experimental blind filters and may cancel predictable beacon energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def baseband_features(channels, fs: float, carrier_hz: float = 8.0) -> np.ndarray:
    if len(channels) != 2:
        raise ValueError("exactly two analytic channels are required")
    first, second = map(np.asarray, channels)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("channels must be equal-length vectors")
    oscillator = np.exp(-2j * np.pi * carrier_hz * np.arange(len(first)) / fs)
    first = first * oscillator
    second = second * oscillator
    return np.column_stack((first.real, first.imag, second.real, second.imag))


def analytic_channels(features: np.ndarray, fs: float, carrier_hz: float = 8.0):
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("expected four real baseband features")
    oscillator = np.exp(2j * np.pi * carrier_hz * np.arange(len(values)) / fs)
    return (
        (values[:, 0] + 1j * values[:, 1]) * oscillator,
        (values[:, 2] + 1j * values[:, 3]) * oscillator,
    )


def covariance_whitener(residuals: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    covariance = np.cov(np.asarray(residuals), rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues[-1]) * ridge, 1e-9)
    return vectors @ np.diag(1 / np.sqrt(np.maximum(eigenvalues, floor))) @ vectors.T


def standardize_residuals(residuals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(residuals) @ np.asarray(matrix).T


@dataclass(frozen=True)
class VARModel:
    order: int
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    residual_whitener: np.ndarray

    def innovations(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features) - self.mean) / self.scale
        output = values.copy()
        for lag in range(1, self.order + 1):
            output[self.order:] -= values[self.order - lag:-lag] @ self.coefficients[lag - 1].T
        output[:self.order] = 0
        return standardize_residuals(output, self.residual_whitener)


def fit_var(features: np.ndarray, order: int = 20, ridge: float = 1e-3) -> VARModel:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) <= order:
        raise ValueError("insufficient VAR training samples")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.maximum(scale, 1e-9)
    normalized = (values - mean) / scale
    design = np.concatenate(
        [normalized[order - lag:-lag] for lag in range(1, order + 1)], axis=1
    )
    target = normalized[order:]
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ target)
    coefficients = weights.reshape(order, values.shape[1], values.shape[1]).transpose(0, 2, 1)
    provisional = VARModel(order, mean, scale, coefficients, np.eye(values.shape[1]))
    residuals = provisional.innovations(values)[order:]
    return VARModel(order, mean, scale, coefficients, covariance_whitener(residuals))


@dataclass(frozen=True)
class KalmanModel:
    mean: np.ndarray
    scale: np.ndarray
    transition: np.ndarray
    process_covariance: np.ndarray
    observation_covariance: np.ndarray

    def innovations(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features) - self.mean) / self.scale
        dimensions = values.shape[1]
        identity = np.eye(dimensions)
        state = np.zeros(dimensions)
        covariance = np.eye(dimensions)
        output = np.zeros_like(values)
        for index, observation in enumerate(values):
            prediction = self.transition @ state
            predicted_covariance = (
                self.transition @ covariance @ self.transition.T + self.process_covariance
            )
            innovation = observation - prediction
            innovation_covariance = predicted_covariance + self.observation_covariance
            eigenvalues, vectors = np.linalg.eigh(innovation_covariance)
            inverse_sqrt = vectors @ np.diag(1 / np.sqrt(np.maximum(eigenvalues, 1e-9))) @ vectors.T
            output[index] = inverse_sqrt @ innovation
            gain = np.linalg.solve(innovation_covariance, predicted_covariance).T
            state = prediction + gain @ innovation
            covariance = (identity - gain) @ predicted_covariance
        return output


def fit_kalman(features: np.ndarray, observation_fraction: float = 0.1, ridge: float = 1e-3) -> KalmanModel:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("insufficient Kalman training samples")
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-9)
    normalized = (values - mean) / scale
    previous, current = normalized[:-1], normalized[1:]
    gram = previous.T @ previous + ridge * np.eye(values.shape[1])
    transition = np.linalg.solve(gram, previous.T @ current).T
    residual = current - previous @ transition.T
    process = np.cov(residual, rowvar=False) + ridge * np.eye(values.shape[1])
    observation = observation_fraction * np.cov(normalized, rowvar=False) + ridge * np.eye(values.shape[1])
    return KalmanModel(mean, scale, transition, process, observation)


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GRU/TCN experiments") from exc
    return torch, nn


class NeuralPredictor:
    """Wrapper for a trained PyTorch one-step predictor."""

    def __init__(self, kind: str, network: Any, mean, scale, residual_whitener, receptive_field=1):
        self.kind = kind
        self.network = network
        self.mean = np.asarray(mean)
        self.scale = np.asarray(scale)
        self.residual_whitener = np.asarray(residual_whitener)
        self.receptive_field = receptive_field

    def innovations(self, features: np.ndarray) -> np.ndarray:
        torch, _ = _require_torch()
        values = (np.asarray(features) - self.mean) / self.scale
        tensor = torch.as_tensor(values, dtype=torch.float32)
        self.network.eval()
        with torch.no_grad():
            if self.kind == "gru":
                prediction, _ = self.network(tensor[None])
                prediction = prediction[0].cpu().numpy()
            else:
                prediction = self.network(tensor.T[None])[0].T.cpu().numpy()
        residual = values.copy()
        residual[1:] -= prediction[:-1]
        residual[:max(1, self.receptive_field)] = 0
        return standardize_residuals(residual, self.residual_whitener)

    def save(self, path) -> None:
        torch, _ = _require_torch()
        torch.save({
            "kind": self.kind,
            "state_dict": self.network.state_dict(),
            "mean": self.mean,
            "scale": self.scale,
            "residual_whitener": self.residual_whitener,
            "receptive_field": self.receptive_field,
        }, path)


def _sequence_batches(values, length: int, batch_size: int, generator):
    torch, _ = _require_torch()
    usable = (len(values) - 1) // length * length
    inputs = values[:usable].reshape(-1, length, values.shape[1])
    targets = values[1:usable + 1].reshape(-1, length, values.shape[1])
    order = torch.randperm(len(inputs), generator=generator)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield inputs[indices], targets[indices]


def fit_neural(
    features: np.ndarray,
    *,
    kind: str,
    hidden: int = 16,
    epochs: int = 8,
    sequence_length: int = 256,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 7,
) -> NeuralPredictor:
    torch, nn = _require_torch()
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    values = np.asarray(features, dtype=np.float32)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-6)
    normalized_np = (values - mean) / scale
    normalized = torch.as_tensor(normalized_np, dtype=torch.float32)

    if kind == "gru":
        class GRUNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(4, hidden, batch_first=True)
                self.output = nn.Linear(hidden, 4)
            def forward(self, x, state=None):
                values, state = self.gru(x, state)
                return self.output(values), state
        network = GRUNetwork()
        receptive_field = 1
    elif kind == "tcn":
        class CausalLayer(nn.Module):
            def __init__(self, incoming, outgoing, dilation):
                super().__init__()
                self.pad = 2 * dilation
                self.conv = nn.Conv1d(incoming, outgoing, 3, dilation=dilation)
                self.activation = nn.Tanh()
            def forward(self, x):
                import torch.nn.functional as functional
                return self.activation(self.conv(functional.pad(x, (self.pad, 0))))
        class TCNNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                layers=[];incoming=4
                for dilation in (1, 2, 4, 8):
                    layers.append(CausalLayer(incoming, hidden, dilation));incoming=hidden
                self.layers=nn.Sequential(*layers)
                self.output=nn.Conv1d(hidden,4,1)
            def forward(self,x):return self.output(self.layers(x))
        network = TCNNetwork()
        receptive_field = 1 + 2 * sum((1, 2, 4, 8))
    else:
        raise ValueError("kind must be 'gru' or 'tcn'")

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    network.train()
    for _ in range(epochs):
        for inputs, targets in _sequence_batches(normalized, sequence_length, batch_size, generator):
            optimizer.zero_grad()
            if kind == "gru":
                prediction, _ = network(inputs)
            else:
                prediction = network(inputs.transpose(1, 2)).transpose(1, 2)
            loss = loss_function(prediction, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

    provisional = NeuralPredictor(kind, network, mean, scale, np.eye(4), receptive_field)
    residual = provisional.innovations(values)
    whitener = covariance_whitener(residual[max(receptive_field, 1):])
    return NeuralPredictor(kind, network, mean, scale, whitener, receptive_field)
