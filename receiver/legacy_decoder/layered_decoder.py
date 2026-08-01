#!/usr/bin/env python3
"""Layered Gaussian-Bayes decoder for the Hamming(7,4) alphabet protocol."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import signal

import coded_protocol as p

RIDGE = 1e-3


@dataclass(frozen=True)
class LayerResult:
    layer: str
    coded_bits: tuple[int, ...]
    header: int
    letter_byte: int
    letter: str
    parity_ok: bool
    success: bool
    confidence: float


@dataclass(frozen=True)
class AlphabetDecode:
    start_index: int
    start_time: float
    preamble_score: float
    selected: LayerResult
    layers: tuple[LayerResult, ...]
    successful_layers: tuple[str, ...]


class GaussianClassifier:
    def __init__(self, classes: int):
        self.classes = classes
        self.models = []

    def fit(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, int)
        for label in range(self.classes):
            values = x[y == label]
            mean = values.mean(axis=0)
            cov = np.atleast_2d(np.cov(values, rowvar=False))
            scale = max(float(np.trace(cov)) / len(mean), 1e-8)
            cov += RIDGE * scale * np.eye(len(mean))
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                chol = np.linalg.cholesky(cov + 1e-6 * np.eye(len(mean)))
            self.models.append((mean, chol, 2 * np.log(np.diag(chol)).sum()))
        return self

    def scores(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        output = []
        for mean, chol, logdet in self.models:
            solved = np.linalg.solve(chol, (x - mean).T)
            output.append(-0.5 * (logdet + np.sum(solved**2, axis=0)))
        return np.stack(output, axis=1)


def _check_features(r0, r1):
    """Eight matched features for each of the three Hamming parity checks."""
    groups = len(r0) // 7
    first, second = r0.reshape(groups, 7), r1.reshape(groups, 7)
    checks = []
    for positions in p.HAMMING_CHECKS:
        feature = np.empty((groups, 8))
        feature[:, 0::2] = first[:, list(positions)]
        feature[:, 1::2] = second[:, list(positions)]
        checks.append(feature)
    return np.stack(checks, axis=1).reshape(-1, 8)


def _group_features(r0, r1):
    groups = len(r0) // 7
    features = np.empty((groups, 14))
    features[:, 0::2] = r0.reshape(groups, 7)
    features[:, 1::2] = r1.reshape(groups, 7)
    return features


def _margin(scores):
    ordered = np.sort(scores, axis=1)
    raw = ordered[:, -1] - ordered[:, -2]
    return float(np.mean(raw / (np.std(scores, axis=1) + 1e-9)))


def _normalize_scores(scores):
    scores = np.asarray(scores, float)
    return (scores - scores.max(axis=1, keepdims=True)) / (
        scores.std(axis=1, keepdims=True) + 1e-9
    )


@lru_cache(maxsize=1)
def trained_models():
    """Deterministic pooled training matching the old L1/L2/L3 benchmark.

    Training spans clean through heavily degraded matched-filter observations;
    no captured test message or alphabet value is embedded in the model.
    """
    rng = np.random.default_rng(2707)
    samples_per_class = 1200
    classes = np.repeat(np.arange(16), samples_per_class)
    words = p.GROUP_CODEBOOK[classes]
    n = len(words)
    severity = rng.uniform(0.02, 0.48, (n, 1))
    active = np.clip(0.96 - severity + rng.normal(0, 0.07, (n, 7)), 0, 1.2)
    inactive = np.clip(0.04 + severity * 0.65 + rng.normal(0, 0.07, (n, 7)), 0, 1.2)
    r0 = np.where(words == 1, active, inactive)
    r1 = np.where(words == 0, active, inactive)

    l1 = GaussianClassifier(2).fit(r0.reshape(-1, 1), words.reshape(-1))
    check_y = np.stack([
        4 * words[:, positions[0]]
        + 2 * words[:, positions[1]]
        + words[:, positions[2]]
        for positions in p.HAMMING_CHECKS
    ], axis=1).reshape(-1)
    l2 = GaussianClassifier(8).fit(
        _check_features(r0.reshape(-1), r1.reshape(-1)), check_y
    )
    l3 = GaussianClassifier(16).fit(_group_features(r0.reshape(-1), r1.reshape(-1)), classes)
    return l1, l2, l3


def decode_observations(r0: Sequence[float], r1: Sequence[float]) -> tuple[LayerResult, ...]:
    r0, r1 = np.asarray(r0, float), np.asarray(r1, float)
    if len(r0) != p.CODED_BITS or len(r1) != p.CODED_BITS:
        raise ValueError(f"expected {p.CODED_BITS} coded-bit observations")
    groups = p.GROUPS
    l1, l2, l3 = trained_models()

    # Naive-max: independent bit decisions, with parity checked afterward.
    naive_bits = (r0 > r1).astype(np.int8)
    naive_classes = (
        8 * naive_bits[2::7] + 4 * naive_bits[4::7]
        + 2 * naive_bits[5::7] + naive_bits[6::7]
    )
    naive_conf = float(np.mean(np.abs(r0 - r1)))

    # L1: one-dimensional bit Gaussians, summed over every valid codeword.
    bit_scores = l1.scores(r0.reshape(-1, 1)).reshape(groups, 7, 2)
    l1_scores = np.empty((groups, 16))
    for cls, word in enumerate(p.GROUP_CODEBOOK):
        l1_scores[:, cls] = sum(bit_scores[:, k, word[k]] for k in range(7))
    l1_classes = np.argmax(l1_scores, axis=1)

    # L2: local Gaussian evidence from all three overlapping Hamming checks.
    l2_check_scores = l2.scores(_check_features(r0, r1)).reshape(groups, 3, 8)
    l2_scores = np.empty((groups, 16))
    for cls, word in enumerate(p.GROUP_CODEBOOK):
        score = np.zeros(groups)
        for check_index, positions in enumerate(p.HAMMING_CHECKS):
            label = 4 * word[positions[0]] + 2 * word[positions[1]] + word[positions[2]]
            score += l2_check_scores[:, check_index, label]
        l2_scores[:, cls] = score
    l2_classes = np.argmax(l2_scores, axis=1)

    # L3: one 14-D Gaussian per complete valid seven-bit group.
    l3_scores = l3.scores(_group_features(r0, r1))
    l3_classes = np.argmax(l3_scores, axis=1)

    # L4 hybrid: normalized evidence from L1, L2, L3, and naive matched energy.
    naive_scores = np.empty((groups, 16))
    gr0, gr1 = r0.reshape(groups, 7), r1.reshape(groups, 7)
    for cls, word in enumerate(p.GROUP_CODEBOOK):
        naive_scores[:, cls] = np.where(word, gr0, gr1).sum(axis=1)
    l4_scores = (
        _normalize_scores(l1_scores) + _normalize_scores(l2_scores)
        + _normalize_scores(l3_scores) + 0.5 * _normalize_scores(naive_scores)
    )
    l4_classes = np.argmax(l4_scores, axis=1)

    candidates = (
        ("naive-max", naive_classes, naive_bits, naive_conf),
        ("L1", l1_classes, p.GROUP_CODEBOOK[l1_classes].reshape(-1), _margin(l1_scores)),
        ("L2", l2_classes, p.GROUP_CODEBOOK[l2_classes].reshape(-1), _margin(l2_scores)),
        ("L3", l3_classes, p.GROUP_CODEBOOK[l3_classes].reshape(-1), _margin(l3_scores)),
        ("L4", l4_classes, p.GROUP_CODEBOOK[l4_classes].reshape(-1), _margin(l4_scores)),
    )
    output = []
    for name, classes_out, coded, confidence in candidates:
        header, letter_byte = p.decode_bytes_from_classes(classes_out)
        letter = chr(letter_byte) if 32 <= letter_byte <= 126 else f"\\x{letter_byte:02x}"
        parity_ok = p.parity_valid(coded)
        success = header == p.HEADER_BYTE and 65 <= letter_byte <= 90 and parity_ok
        output.append(LayerResult(
            name, tuple(map(int, coded)), header, letter_byte, letter,
            parity_ok, success, float(confidence),
        ))
    return tuple(output)


def analytic_channels(x, y, fs):
    low, high = p.CARRIER_HZ - p.BANDWIDTH_HZ / 2, p.CARRIER_HZ + p.BANDWIDTH_HZ / 2
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    result = []
    for values in (x, y):
        values = np.asarray(values, float)
        filtered = signal.sosfiltfilt(sos, values - np.median(values))
        result.append(signal.hilbert(filtered))
    return result


def sample_rate(t):
    diff = np.diff(np.asarray(t, float))
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if not len(diff):
        raise ValueError("need increasing timestamps")
    return float(1 / np.median(diff))


def sliding_correlation(z, template):
    corr = signal.fftconvolve(z, np.conj(template[::-1]), mode="valid")
    cumulative = np.r_[0.0, np.cumsum(np.abs(z)**2)]
    energy = cumulative[len(template):] - cumulative[:-len(template)]
    return np.abs(corr)**2 / (np.vdot(template, template).real * energy + 1e-15)


def matched_observations(channels, start, fs):
    half = round(fs * p.HALF_SYMBOL_SECONDS)
    bit_samples = 2 * half
    carrier = np.exp(2j * np.pi * p.CARRIER_HZ * np.arange(bit_samples) / fs)
    r0, r1 = np.zeros(p.CODED_BITS), np.zeros(p.CODED_BITS)
    for bit in range(p.CODED_BITS):
        offset = start + bit * bit_samples
        for z in channels:
            block = z[offset:offset + bit_samples]
            energy = np.sum(np.abs(block)**2)
            denom = np.sqrt(half * energy) + 1e-15
            r0[bit] += abs(np.vdot(carrier[:half], block[:half])) / denom
            r1[bit] += abs(np.vdot(carrier[half:], block[half:])) / denom
    return r0 / len(channels), r1 / len(channels)


def decode_capture(t, x, y) -> AlphabetDecode:
    t = np.asarray(t, float)
    fs = sample_rate(t)
    half = round(fs * p.HALF_SYMBOL_SECONDS)
    bit_samples = 2 * half
    frame_samples = p.CODED_BITS * bit_samples
    if len(t) < frame_samples:
        raise ValueError(f"capture too short: need {frame_samples} samples")
    channels = analytic_channels(x, y, fs)
    header_template = p.complex_template(p.ENCODED_HEADER, half, fs)
    correlation = sum(sliding_correlation(z, header_template) for z in channels)
    valid_length = max(0, len(correlation) - (frame_samples - len(header_template)))
    if valid_length <= 0:
        raise ValueError("no complete frame can follow the header")
    start = int(np.argmax(correlation[:valid_length]))
    r0, r1 = matched_observations(channels, start, fs)
    layers = decode_observations(r0, r1)
    successful = tuple(layer.layer for layer in layers if layer.success)
    valid = [layer for layer in layers if layer.success]
    if not valid:
        # Preserve evidence even when no layer passes the protocol checks.
        selected = max(layers, key=lambda layer: layer.confidence)
    else:
        l4 = next((layer for layer in valid if layer.layer == "L4"), None)
        selected = l4 or max(valid, key=lambda layer: layer.confidence)
    return AlphabetDecode(
        start, float(t[start]), float(correlation[start]), selected, layers, successful
    )


def synthesize_capture(letter="A", *, fs=200.0, lead=2.0, tail=6.0,
                       noise_std=0.03, seed=1):
    coded = p.encode_message(letter)
    half = round(fs * p.HALF_SYMBOL_SECONDS)
    gate = np.repeat(p.manchester_levels(coded), half)
    gate = np.r_[np.zeros(round(lead*fs)), gate, np.zeros(round(tail*fs))]
    t = np.arange(len(gate)) / fs
    rng = np.random.default_rng(seed)
    x = 800 + 160 * gate * np.cos(2*np.pi*p.CARRIER_HZ*t) + rng.normal(0, noise_std*160, len(t))
    y = 1500 + 120 * gate * np.cos(2*np.pi*p.CARRIER_HZ*t + .55) + rng.normal(0, noise_std*120, len(t))
    return t, x, y
