#!/usr/bin/env python3
"""Frozen RS soft-decoder and sensor-front-end comparison on the final run.

The comparison set is declared in ``ALGORITHMS`` before any payload scoring.
Every method reuses the exact manifest boundaries and raw observations from the
primary analysis. Gao transfers and Duong gains are fit independently for each
frame using only that frame's declared central ten-second transmitter-off gap,
then frozen while transforming the active frame. Payload truth is consulted
only after each decoder has returned a candidate or failure.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
TRANSMITTER_DIR = ROOT / "transmitter"
if str(TRANSMITTER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSMITTER_DIR))

import analyze_final_experiment as base  # noqa: E402
import duong_whitener  # noqa: E402
import final_experiment_protocol as protocol  # noqa: E402

ALGORITHMS = (
    "unwhitened_hard",
    "coherent_hard",
    "coherent_gmd",
    "gao_gmd",
    "duong_gmd",
    "gao_duong_gmd",
)
DUONG_TOLERANCE = 1e-5
DUONG_MAX_ITERATIONS = 60_000
DUONG_SEED = 7


class GMDDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class GMDResult:
    payload_bits: tuple[int, ...]
    codeword_symbols: tuple[int, ...]
    erasures: tuple[int, ...]
    score: float
    margin: float
    attempts: int
    candidate_count: int
    corrected_symbol_count: int


@dataclass(frozen=True)
class GaoModel:
    primary_mean: complex
    reference_mean: complex
    transfer: complex
    off_coherence: float
    noise_reduction_db: float

    def transform(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return (
            np.asarray(primary) - self.primary_mean
            - self.transfer * (np.asarray(reference) - self.reference_mean)
        )


def fit_gao_off(primary_off: np.ndarray, reference_off: np.ndarray) -> GaoModel:
    """Complex least-squares reference transfer fit only on declared H0 data."""
    primary = np.asarray(primary_off, dtype=complex)
    reference = np.asarray(reference_off, dtype=complex)
    if primary.ndim != 1 or primary.shape != reference.shape or len(primary) < 2:
        raise ValueError("Gao off channels must be equal vectors with at least two samples")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(reference)):
        raise ValueError("Gao off channels must be finite")
    primary_mean, reference_mean = complex(primary.mean()), complex(reference.mean())
    first, second = primary - primary_mean, reference - reference_mean
    reference_energy = float(np.vdot(second, second).real)
    if reference_energy <= np.finfo(float).eps * len(second):
        raise ValueError("Gao reference off channel has zero variance")
    transfer = complex(np.vdot(second, first) / reference_energy)
    residual = first - transfer * second
    primary_power = float(np.mean(np.abs(first) ** 2))
    reference_power = float(np.mean(np.abs(second) ** 2))
    residual_power = float(np.mean(np.abs(residual) ** 2))
    cross = abs(np.vdot(second, first) / len(first)) ** 2
    coherence = cross / max(primary_power * reference_power, np.finfo(float).tiny)
    reduction = (
        math.inf if residual_power == 0 else
        10.0 * math.log10(primary_power / residual_power)
    )
    return GaoModel(
        primary_mean, reference_mean, transfer,
        float(np.clip(coherence, 0.0, 1.0)), float(reduction),
    )


def iq_features(channels: Sequence[np.ndarray]) -> np.ndarray:
    if len(channels) not in (1, 2):
        raise ValueError("one or two complex channels required")
    values = tuple(np.asarray(channel) for channel in channels)
    if any(value.ndim != 1 or value.shape != values[0].shape for value in values):
        raise ValueError("complex channels must be equal vectors")
    return np.column_stack(
        [component for value in values for component in (value.real, value.imag)]
    )


def complex_channels(features: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (2, 4):
        raise ValueError("whitened features must have two or four columns")
    return tuple(
        values[:, index] + 1j * values[:, index + 1]
        for index in range(0, values.shape[1], 2)
    )


def _extract_phasors(
    channels: Sequence[np.ndarray], start: int, count: int, half: int, fs: float
) -> np.ndarray:
    carrier = np.exp(2j * np.pi * base.CARRIER_HZ * np.arange(half) / fs)
    output = np.empty((count, len(channels)), dtype=complex)
    for index in range(count):
        begin = start + index * half
        for sensor, channel in enumerate(channels):
            block = np.asarray(channel)[begin:begin + half]
            if len(block) != half:
                raise ValueError("incomplete half-symbol")
            output[index, sensor] = np.vdot(carrier, block) / np.sqrt(half)
    return output


def unwhitened_llrs(
    channels: Sequence[np.ndarray], frame_bits: int, fs: float
) -> np.ndarray:
    """Sync-trained coherent LLRs with identity noise covariance."""
    if len(channels) not in (1, 2):
        raise ValueError("unwhitened comparison requires one or two channels")
    half = round(fs * protocol.BIT_SECONDS / 2)
    phasors = _extract_phasors(channels, 0, 2 * frame_bits, half, fs)
    sync_gate = base.manchester_levels(protocol.SYNC_TEXT).astype(bool)
    sync = phasors[:len(sync_gate)]
    channel_vector = sync[sync_gate].mean(axis=0) - sync[~sync_gate].mean(axis=0)
    projected = phasors @ channel_vector.conj()
    return 2.0 * np.real(projected[0::2] - projected[1::2])


def coherent_llrs(
    channels: Sequence[np.ndarray], frame_bits: int, fs: float
) -> np.ndarray:
    """Signed coherent LLRs; sync trains channel and central gap trains noise."""
    if len(channels) not in (1, 2):
        raise ValueError("coherent comparison requires one or two channels")
    half = round(fs * protocol.BIT_SECONDS / 2)
    frame_halves = 2 * frame_bits
    phasors = _extract_phasors(channels, 0, frame_halves, half, fs)
    sync_gate = base.manchester_levels(protocol.SYNC_TEXT).astype(bool)
    sync = phasors[:len(sync_gate)]
    channel_vector = sync[sync_gate].mean(axis=0) - sync[~sync_gate].mean(axis=0)

    noise_start = frame_halves * half + round(base.CENTRAL_GAP_LEAD_SECONDS * fs)
    noise_count = round(base.CENTRAL_GAP_SECONDS * fs / half)
    noise = _extract_phasors(channels, noise_start, noise_count, half, fs)
    centered = noise - noise.mean(axis=0, keepdims=True)
    # With row observations z and projection z @ conj(w), the spatial
    # covariance is E[z z^H], not its transpose.
    covariance = centered.T @ centered.conj() / len(centered)
    ridge = max(float(np.trace(covariance).real) / len(channels), 1e-12) * 1e-6
    covariance = covariance + ridge * np.eye(len(channels))
    try:
        weights = np.linalg.solve(covariance, channel_vector)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(covariance) @ channel_vector
    projected = phasors @ weights.conj()
    return 2.0 * np.real(projected[0::2] - projected[1::2])


def _rs_parameters(scheme: str) -> tuple[int, int, int]:
    if scheme == "rs5_3":
        return 3, 2, 11
    if scheme == "rs12_6":
        return 6, 6, 30
    raise ValueError("comparison supports only rs5_3 and rs12_6")


def rs_decode_erasures(
    received: Sequence[int], data_symbols: int, parity_symbols: int,
    erasures: Sequence[int] = (),
) -> tuple[int, ...]:
    """Generic errors-and-erasures decoder for the shortened GF(32) words."""
    word = tuple(received)
    length = data_symbols + parity_symbols
    if len(word) != length:
        raise ValueError("wrong RS word length")
    protocol.symbols_to_bits(word)
    erased = tuple(int(position) for position in erasures)
    if len(set(erased)) != len(erased) or any(p < 0 or p >= length for p in erased):
        raise ValueError("invalid RS erasure positions")
    if len(erased) > parity_symbols:
        raise GMDDecodeError("too many erasures")
    syndromes = list(base.rs_syndromes(word, parity_symbols))
    if not any(syndromes):
        return word

    reduced = syndromes[:]
    for position in erased:
        location = protocol._gf_pow(length - 1 - position)
        reduced = [
            reduced[index + 1] ^ protocol._gf_mul(location, reduced[index])
            for index in range(len(reduced) - 1)
        ]
    locator = base._berlekamp_massey(reduced)
    unknown_errors = len(locator) - 1
    if 2 * unknown_errors + len(erased) > parity_symbols:
        raise GMDDecodeError("errors and erasures exceed correction capability")

    erasure_set = set(erased)
    error_positions: list[int] = []
    for position in range(length):
        if position in erasure_set:
            continue
        inverse_location = protocol._gf_pow(-(length - 1 - position))
        evaluation, field_power = 0, 1
        for coefficient in locator:
            evaluation ^= protocol._gf_mul(coefficient, field_power)
            field_power = protocol._gf_mul(field_power, inverse_location)
        if evaluation == 0:
            error_positions.append(position)
    if len(error_positions) != unknown_errors:
        raise GMDDecodeError("error locator does not split over shortened word")
    locations = list(erased) + error_positions
    if not locations:
        raise GMDDecodeError("nonzero syndrome without located errors")
    field_locations = [protocol._gf_pow(length - 1 - position) for position in locations]
    matrix = [
        [protocol._gf_pow(protocol._GF_LOG[location] * equation)
         for location in field_locations]
        for equation in range(1, len(locations) + 1)
    ]
    magnitudes = base._solve_linear(matrix, syndromes[:len(locations)])
    corrected = list(word)
    for position, magnitude in zip(locations, magnitudes):
        corrected[position] ^= magnitude
    corrected_word = tuple(corrected)
    if any(base.rs_syndromes(corrected_word, parity_symbols)):
        raise GMDDecodeError("correction did not produce an RS codeword")
    if protocol.rs_encode_symbols(corrected_word[:data_symbols], parity_symbols) != corrected_word:
        raise GMDDecodeError("systematic re-encoding validation failed")
    return corrected_word


def _payload_from_word(
    word: Sequence[int], scheme: str, data_symbols: int
) -> tuple[int, ...]:
    bits = protocol.symbols_to_bits(tuple(word)[:data_symbols])
    if scheme == "rs5_3":
        if bits[11:] != (0, 0, 0, 0):
            raise GMDDecodeError("RS(5,3) application padding is not zero")
        return bits[:11]
    return bits[:30]


def hard_decode(bit_llrs: Sequence[float], scheme: str) -> GMDResult:
    data_symbols, parity_symbols, _ = _rs_parameters(scheme)
    llrs = tuple(float(value) for value in bit_llrs)
    hard = tuple(int(value > 0.0) for value in llrs)
    symbols = protocol.bits_to_symbols(hard)
    try:
        corrected, units = base.rs_decode_symbols(symbols, data_symbols, parity_symbols)
        payload = _payload_from_word(corrected, scheme, data_symbols)
    except (ValueError, base.ReedSolomonError, GMDDecodeError) as exc:
        raise GMDDecodeError(str(exc)) from exc
    bits = protocol.symbols_to_bits(corrected)
    score = sum(value if bit else -value for bit, value in zip(bits, llrs))
    corrected_count = sum(left != right for left, right in zip(symbols, corrected))
    return GMDResult(
        payload, corrected, (), float(score), math.inf, 1, 1, corrected_count
    )


def gmd_decode(bit_llrs: Sequence[float], scheme: str) -> GMDResult:
    """Hard attempt plus least-reliable-symbol erasure prefixes; soft-score only."""
    data_symbols, parity_symbols, _ = _rs_parameters(scheme)
    code_symbols = data_symbols + parity_symbols
    expected_bits = code_symbols * 5
    llrs = tuple(float(value) for value in bit_llrs)
    if len(llrs) != expected_bits or not all(math.isfinite(value) for value in llrs):
        raise ValueError(f"{scheme} requires {expected_bits} finite bit LLRs")
    hard = tuple(int(value > 0.0) for value in llrs)
    hard_symbols = protocol.bits_to_symbols(hard)
    reliability = [
        min(abs(value) for value in llrs[5 * symbol:5 * symbol + 5])
        for symbol in range(code_symbols)
    ]
    order = sorted(range(code_symbols), key=lambda index: (reliability[index], index))
    schedules = [()] + [tuple(order[:count]) for count in range(1, parity_symbols + 1)]
    candidates: dict[tuple[int, ...], tuple[float, tuple[int, ...], tuple[int, ...]]] = {}
    for erased in schedules:
        try:
            corrected = rs_decode_erasures(
                hard_symbols, data_symbols, parity_symbols, erased
            )
            payload = _payload_from_word(corrected, scheme, data_symbols)
        except (ValueError, GMDDecodeError):
            continue
        bits = protocol.symbols_to_bits(corrected)
        score = float(sum(value if bit else -value for bit, value in zip(bits, llrs)))
        if corrected not in candidates:
            candidates[corrected] = (score, tuple(erased), payload)
    if not candidates:
        raise GMDDecodeError("GMD produced no application-valid candidate")
    ranked = sorted(
        ((score, word, erased, payload)
         for word, (score, erased, payload) in candidates.items()),
        key=lambda item: (-item[0], item[1]),
    )
    score, word, erased, payload = ranked[0]
    margin = math.inf if len(ranked) == 1 else score - ranked[1][0]
    corrected_count = sum(left != right for left, right in zip(hard_symbols, word))
    return GMDResult(
        payload, word, erased, score, float(margin), len(schedules),
        len(candidates), corrected_count,
    )


def _frontends(
    local_channels: tuple[np.ndarray, np.ndarray], frame_samples: int, fs: float
) -> tuple[dict[str, tuple[np.ndarray, ...]], dict[str, object]]:
    """Fit all H0-only models, then transform the common local observations."""
    gap_start = frame_samples + round(base.CENTRAL_GAP_LEAD_SECONDS * fs)
    gap_samples = round(base.CENTRAL_GAP_SECONDS * fs)
    gap = slice(gap_start, gap_start + gap_samples)
    sensor_x, sensor_y = local_channels
    # Frozen before scoring: sensor y is Gao primary, sensor x is reference.
    primary, reference = sensor_y, sensor_x
    primary_off, reference_off = primary[gap], reference[gap]

    gao = fit_gao_off(primary_off, reference_off)
    gao_all = gao.transform(primary, reference)

    duong = duong_whitener.fit(
        iq_features((sensor_x[gap], sensor_y[gap])),
        tolerance=DUONG_TOLERANCE, max_iterations=DUONG_MAX_ITERATIONS,
        seed=DUONG_SEED,
    )
    duong_all = complex_channels(duong.transform(iq_features((sensor_x, sensor_y))))

    gao_duong = duong_whitener.fit(
        iq_features((gao_all[gap],)),
        tolerance=DUONG_TOLERANCE, max_iterations=DUONG_MAX_ITERATIONS,
        seed=DUONG_SEED,
    )
    gao_duong_all = complex_channels(
        gao_duong.transform(iq_features((gao_all,)))
    )
    channels = {
        "unwhitened_hard": local_channels,
        "coherent_hard": local_channels,
        "coherent_gmd": local_channels,
        "gao_gmd": (gao_all,),
        "duong_gmd": duong_all,
        "gao_duong_gmd": gao_duong_all,
    }

    # Desired-signal diagnostics use only the known sync, never payload truth.
    half = round(fs * protocol.BIT_SECONDS / 2)
    sync_gate = np.repeat(base.manchester_levels(protocol.SYNC_TEXT), half)
    sync_template = sync_gate * np.exp(
        2j * np.pi * base.CARRIER_HZ * np.arange(len(sync_gate)) / fs
    )
    template_power = float(np.vdot(sync_template, sync_template).real)
    primary_response = np.vdot(
        sync_template, primary[:len(sync_template)] - gao.primary_mean
    ) / template_power
    reference_response = np.vdot(
        sync_template, reference[:len(sync_template)] - gao.reference_mean
    ) / template_power
    residual_response = primary_response - gao.transfer * reference_response
    primary_amplitude = max(float(abs(primary_response)), np.finfo(float).tiny)
    residual_amplitude = float(abs(residual_response))
    diagnostics = {
        "gao_transfer_magnitude": float(abs(gao.transfer)),
        "gao_transfer_phase_radians": float(np.angle(gao.transfer)),
        "gao_off_coherence": gao.off_coherence,
        "gao_noise_reduction_db": gao.noise_reduction_db,
        "gao_orientation": "sensor_y_primary_sensor_x_reference",
        "gao_sync_reference_leakage_ratio": float(abs(reference_response)) / primary_amplitude,
        "gao_sync_desired_amplitude_ratio": residual_amplitude / primary_amplitude,
        "gao_sync_desired_attenuation_db": (
            math.inf if residual_amplitude == 0 else
            20.0 * math.log10(primary_amplitude / residual_amplitude)
        ),
        "duong_iterations": duong.iterations,
        "duong_converged_before_iteration_limit": int(
            duong.iterations < DUONG_MAX_ITERATIONS
        ),
        "duong_training_error": duong.training_error,
        "gao_duong_iterations": gao_duong.iterations,
        "gao_duong_converged_before_iteration_limit": int(
            gao_duong.iterations < DUONG_MAX_ITERATIONS
        ),
        "gao_duong_training_error": gao_duong.training_error,
    }
    return channels, diagnostics


def compare(
    capture_path: Path, manifest_path: Path, metadata_path: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    t, x, y = base.load_capture(capture_path)
    fs = base.sample_rate(t)
    rows = base.load_manifest(manifest_path)
    info = base.metadata(metadata_path)
    raw_analytic = tuple(
        signal.hilbert(np.asarray(channel, float) - np.median(channel))
        for channel in (x, y)
    )
    filtered = tuple(base.bandpass(channel, fs) for channel in (x, y))
    filtered_analytic = tuple(signal.hilbert(channel) for channel in filtered)
    half = round(fs * protocol.BIT_SECONDS / 2)
    template = base.complex_template(protocol.SYNC_TEXT, half, fs)
    correlation = sum(base.sliding_correlation(channel, template)
                      for channel in filtered_analytic)
    starts, correction = base.locate_manifest_boundaries(
        rows, base.utc_seconds(info["capture_started_utc"]), correlation, fs
    )

    decoded_without_truth: list[tuple[dict[str, str], str, GMDResult | None,
                                      np.ndarray, dict[str, object]]] = []
    for row, start in zip(rows, starts):
        if row["scheme"] not in ("rs5_3", "rs12_6"):
            continue
        frame_bits = len(row["frame_bits"])
        frame_samples = round(frame_bits * protocol.BIT_SECONDS * fs)
        local_stop = (
            start + frame_samples + round(base.CENTRAL_GAP_LEAD_SECONDS * fs)
            + round(base.CENTRAL_GAP_SECONDS * fs)
        )
        if local_stop > len(x):
            raise ValueError(f"frame {row['sequence']} central gap is incomplete")
        local = tuple(channel[start:local_stop] for channel in raw_analytic)
        transformed, diagnostics = _frontends(local, frame_samples, fs)
        for algorithm in ALGORITHMS:
            all_llrs = (
                unwhitened_llrs(transformed[algorithm], frame_bits, fs)
                if algorithm == "unwhitened_hard"
                else coherent_llrs(transformed[algorithm], frame_bits, fs)
            )
            body_llrs = all_llrs[len(protocol.SYNC_TEXT):]
            try:
                result = (hard_decode(body_llrs, row["scheme"])
                          if algorithm.endswith("_hard")
                          else gmd_decode(body_llrs, row["scheme"]))
            except GMDDecodeError:
                result = None
            decoded_without_truth.append(
                (row, algorithm, result, body_llrs, diagnostics)
            )

    # Truth enters only here, after the frozen methods have decoded or failed.
    output: list[dict[str, object]] = []
    for row, algorithm, result, body_llrs, diagnostics in decoded_without_truth:
        expected_payload = tuple(map(int, row["payload_bits"]))
        errors = (None if result is None
                  else base.bit_errors(result.payload_bits, expected_payload))
        output.append({
            "sequence": int(row["sequence"]),
            "phase": row["phase"],
            "scheme": row["scheme"],
            "repetition": int(row["repetition"]),
            "duty_percent": float(row["duty_percent"]),
            "algorithm": algorithm,
            "boundary_source": "identical_frozen_manifest_boundaries",
            "manifest_clock_correction_s": correction,
            "off_fit_samples": round(base.CENTRAL_GAP_SECONDS * fs),
            "decoder_failure": int(result is None),
            "payload_bit_errors_conditional_on_decode": errors,
            "payload_bits_evaluated": 0 if result is None else len(expected_payload),
            "frame_error": int(result is None or (errors is not None and errors > 0)),
            "wrong_codeword_miscorrection": int(
                result is not None and errors is not None and errors > 0
            ),
            "decoded_payload_bits": "" if result is None else
                "".join(map(str, result.payload_bits)),
            "gmd_erasures": "" if result is None else
                ",".join(map(str, result.erasures)),
            "gmd_erasure_count": 0 if result is None else len(result.erasures),
            "gmd_candidate_count": 0 if result is None else result.candidate_count,
            "corrected_symbol_count": 0 if result is None else result.corrected_symbol_count,
            "gmd_attempts": 1 if algorithm.endswith("_hard") else
                (_rs_parameters(row["scheme"])[1] + 1),
            "gmd_score": None if result is None else result.score,
            "gmd_margin": None if result is None or not math.isfinite(result.margin)
                else result.margin,
            "minimum_abs_bit_llr": float(np.min(np.abs(body_llrs))),
            **diagnostics,
        })
    details = {
        "sample_rate_hz": fs,
        "manifest_clock_correction_s": correction,
        "rs_frames": len(output) // len(ALGORITHMS),
        "comparison_rows": len(output),
    }
    return output, details


def aggregate(rows: list[dict[str, object]], details: dict[str, object]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = (
            f"{row['algorithm']}|{row['phase']}|{row['scheme']}|"
            f"{float(row['duty_percent']):g}"
        )
        groups.setdefault(key, []).append(row)
    conditions: dict[str, object] = {}
    for key, selected in groups.items():
        errors = sum(int(row["frame_error"]) for row in selected)
        conditional_errors = sum(
            int(row["payload_bit_errors_conditional_on_decode"])
            for row in selected
            if row["payload_bit_errors_conditional_on_decode"] is not None
        )
        conditions[key] = {
            "algorithm": selected[0]["algorithm"],
            "phase": selected[0]["phase"],
            "scheme": selected[0]["scheme"],
            "duty_percent": selected[0]["duty_percent"],
            "frame_errors": errors,
            "frames": len(selected),
            "fer_wilson_95": base.wilson_95(errors, len(selected)),
            "decoder_failures": sum(int(row["decoder_failure"]) for row in selected),
            "wrong_codeword_miscorrections": sum(
                int(row["wrong_codeword_miscorrection"]) for row in selected
            ),
            "payload_bit_errors_conditional_on_decode": conditional_errors,
            "payload_bits_evaluated": sum(
                int(row["payload_bits_evaluated"]) for row in selected
            ),
            "mean_gmd_erasure_count": float(np.mean([
                int(row["gmd_erasure_count"]) for row in selected
            ])),
            "mean_corrected_symbol_count": float(np.mean([
                int(row["corrected_symbol_count"]) for row in selected
            ])),
            "gmd_attempts_per_frame": int(selected[0]["gmd_attempts"]),
            "median_gmd_margin_conditional_multiple_candidates": (
                float(np.median([
                    float(row["gmd_margin"]) for row in selected
                    if row["gmd_margin"] is not None
                ])) if any(row["gmd_margin"] is not None for row in selected) else None
            ),
            "median_gao_noise_reduction_db": float(np.median([
                float(row["gao_noise_reduction_db"]) for row in selected
            ])),
            "median_gao_sync_reference_leakage_ratio": float(np.median([
                float(row["gao_sync_reference_leakage_ratio"]) for row in selected
            ])),
            "median_gao_sync_desired_attenuation_db": float(np.median([
                float(row["gao_sync_desired_attenuation_db"]) for row in selected
            ])),
        }
    algorithms: dict[str, object] = {}
    for algorithm in ALGORITHMS:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        errors = sum(int(row["frame_error"]) for row in selected)
        algorithms[algorithm] = {
            "frame_errors": errors,
            "frames": len(selected),
            "fer_wilson_95": base.wilson_95(errors, len(selected)),
            "decoder_failures": sum(int(row["decoder_failure"]) for row in selected),
            "wrong_codeword_miscorrections": sum(
                int(row["wrong_codeword_miscorrection"]) for row in selected
            ),
        }

    by_sequence_algorithm = {
        (int(row["sequence"]), str(row["algorithm"])): row for row in rows
    }
    paired_groups: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    for row in rows:
        if row["algorithm"] != "coherent_hard":
            continue
        gmd = by_sequence_algorithm[(int(row["sequence"]), "coherent_gmd")]
        key = f"{row['scheme']}|{float(row['duty_percent']):g}"
        paired_groups.setdefault(key, []).append((row, gmd))

    def paired_summary(pairs):
        return {
            "frames": len(pairs),
            "hard_frame_errors": sum(int(hard["frame_error"]) for hard, _ in pairs),
            "gmd_frame_errors": sum(int(gmd["frame_error"]) for _, gmd in pairs),
            "gmd_improvements": sum(
                int(hard["frame_error"] and not gmd["frame_error"])
                for hard, gmd in pairs
            ),
            "gmd_regressions": sum(
                int(not hard["frame_error"] and gmd["frame_error"])
                for hard, gmd in pairs
            ),
            "both_correct": sum(
                int(not hard["frame_error"] and not gmd["frame_error"])
                for hard, gmd in pairs
            ),
            "both_error": sum(
                int(hard["frame_error"] and gmd["frame_error"])
                for hard, gmd in pairs
            ),
            "hard_decoder_failures": sum(
                int(hard["decoder_failure"]) for hard, _ in pairs
            ),
            "gmd_decoder_failures": sum(
                int(gmd["decoder_failure"]) for _, gmd in pairs
            ),
            "gmd_wrong_codeword_miscorrections": sum(
                int(gmd["wrong_codeword_miscorrection"]) for _, gmd in pairs
            ),
        }

    all_pairs = [pair for pairs in paired_groups.values() for pair in pairs]
    paired = {
        "overall": paired_summary(all_pairs),
        "by_scheme_duty": {
            key: paired_summary(pairs) for key, pairs in paired_groups.items()
        },
    }
    fit_rows = [row for row in rows if row["algorithm"] == "coherent_hard"]
    fit_diagnostics = {
        "frames_fit": len(fit_rows),
        "duong_iteration_limit_frames": sum(
            not int(row["duong_converged_before_iteration_limit"])
            for row in fit_rows
        ),
        "gao_duong_iteration_limit_frames": sum(
            not int(row["gao_duong_converged_before_iteration_limit"])
            for row in fit_rows
        ),
        "median_gao_off_coherence": float(np.median([
            float(row["gao_off_coherence"]) for row in fit_rows
        ])),
        "median_gao_noise_reduction_db": float(np.median([
            float(row["gao_noise_reduction_db"]) for row in fit_rows
        ])),
        "median_gao_sync_desired_attenuation_db": float(np.median([
            float(row["gao_sync_desired_attenuation_db"]) for row in fit_rows
        ])),
    }
    return {
        "algorithm_contract": {
            "frozen_algorithms": list(ALGORITHMS),
            "unwhitened_baseline": (
                "sync-trained coherent channel combining with identity noise "
                "covariance, followed by hard algebraic RS decoding"
            ),
            "candidate_selection": "maximum LLR codeword score; payload truth excluded",
            "gmd_schedule": "hard attempt then least-reliable-symbol prefixes 1..parity",
            "symbol_reliability": "minimum absolute bit LLR; position breaks ties",
            "hard_tie": "zero LLR maps to zero",
            "complex_covariance": "E[z z^H] = centered.T @ centered.conj() for row phasors",
            "rs5_padding": "four trailing application bits must be zero",
            "boundaries": "identical manifest timestamps with one frozen first-sync correction",
            "off_fit": (
                "offline/noncausal per-frame post-frame central 10 s only; "
                "models frozen during active frame"
            ),
            "comparison_status": (
                "post-hoc exploratory reanalysis; no untouched confirmation session"
            ),
            "frontend_fit_observations": "DC-centered raw analytic samples",
            "gao_orientation": "sensor y primary, sensor x reference; frozen before scoring",
            "gao_desired_diagnostics": "known sync projection only; payload truth excluded",
            "duong": {
                "method": "Algorithm-1 offline gain fit and analytical equilibrium",
                "tolerance": DUONG_TOLERANCE,
                "max_iterations": DUONG_MAX_ITERATIONS,
                "seed": DUONG_SEED,
                "solver": "analytical equilibrium; explicit recurrence not scored",
            },
            "truth_usage": "final payload-error scoring only",
        },
        **details,
        "by_algorithm": algorithms,
        "frontend_fit_diagnostics": fit_diagnostics,
        "paired_coherent_hard_vs_gmd": paired,
        "by_algorithm_phase_scheme_duty": conditions,
    }


def _format(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: _format(value) for key, value in row.items()}
                         for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.summary.exists():
        print("ERROR: refusing to overwrite comparison outputs", file=sys.stderr)
        return 2
    rows, details = compare(args.capture, args.manifest, args.metadata)
    result = aggregate(rows, details)
    write_csv(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
    csv_hash = base.write_checksum(args.output)
    json_hash = base.write_checksum(args.summary)
    print(f"Wrote {args.output} sha256={csv_hash}")
    print(f"Wrote {args.summary} sha256={json_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
