#!/usr/bin/env python3
"""Offline manifest-boundary analysis for the approved final experiment.

Primary boundaries are QNX manifest timestamps shifted by one clock correction.
That correction is estimated once, from the first 100% frame's known 16-bit
sync correlation within +/-1.25 seconds, then frozen for all later frames.
Autonomous acquisition is deliberately not evaluated.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
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
import final_experiment as experiment  # noqa: E402
import final_experiment_protocol as protocol  # noqa: E402

CARRIER_HZ = 8.0
FILTER_LOW_HZ = 7.25
FILTER_HIGH_HZ = 8.75
CLOCK_SEARCH_RADIUS_SECONDS = 1.25
CENTRAL_GAP_LEAD_SECONDS = 2.5
CENTRAL_GAP_SECONDS = 10.0
ADC_MIN = 0
ADC_12BIT_MAX = 4095
ADC_16BIT_MAX = 65535


class ReedSolomonError(ValueError):
    pass


def metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def utc_seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def load_capture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if not data.dtype.names or not {"t", "x", "y"}.issubset(data.dtype.names):
        raise ValueError("capture must contain t,x,y columns")
    good = np.isfinite(data["t"]) & np.isfinite(data["x"]) & np.isfinite(data["y"])
    t, x, y = data["t"][good], data["x"][good], data["y"][good]
    if len(t) < 2:
        raise ValueError("capture contains too few finite samples")
    return t, x, y


def sample_rate(t: np.ndarray) -> float:
    differences = np.diff(np.asarray(t, dtype=float))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if not len(differences):
        raise ValueError("capture timestamps are not increasing")
    return float(1.0 / np.median(differences))


def bandpass(values: np.ndarray, fs: float) -> np.ndarray:
    sos = signal.butter(
        4, [FILTER_LOW_HZ, FILTER_HIGH_HZ], btype="bandpass", fs=fs, output="sos"
    )
    centered = np.asarray(values, dtype=float) - np.median(values)
    return signal.sosfiltfilt(sos, centered)


def power(values: np.ndarray) -> float:
    samples = np.asarray(values, dtype=float)
    return float(np.mean(samples * samples))


def signal_snr_db(active_power: float, off_power: float) -> float | None:
    """Power-subtracted SNR; None means no defensible positive signal estimate."""
    if not np.isfinite(active_power) or not np.isfinite(off_power) or off_power <= 0:
        return None
    signal_power = active_power - off_power
    if signal_power <= 0:
        return None
    return float(10.0 * math.log10(signal_power / off_power))


def adc_full_scale(*channels: np.ndarray) -> int:
    maximum = max(float(np.max(np.asarray(channel))) for channel in channels)
    return ADC_16BIT_MAX if maximum > ADC_12BIT_MAX else ADC_12BIT_MAX


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum(path: Path) -> str:
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def load_manifest(path: Path, *, require_complete: bool = True) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    validate_manifest(rows, require_complete=require_complete)
    return rows


def validate_manifest(rows: list[dict[str, str]], *, require_complete: bool = True) -> None:
    expected = experiment.experiment_schedule("all")
    if not rows:
        raise ValueError("transmitter manifest is empty")
    if require_complete and len(rows) != 100:
        raise ValueError("final experiment manifest must contain exactly 100 frames")
    if len(rows) > len(expected):
        raise ValueError("final experiment manifest contains too many frames")
    required = {
        "sequence", "phase", "scheme", "repetition", "duty_percent",
        "voltage_v", "distance_m", "payload_bits", "coded_body_bits",
        "frame_bits", "bit_seconds", "started_utc", "finished_utc", "duration_s",
        "gap_after_s", "target_pulse_us", "pulse_count", "late_starts",
        "pulse_min_us", "pulse_median_us", "pulse_p95_us", "pulse_max_us",
    }
    for sequence, (row, trial) in enumerate(zip(rows, expected), 1):
        missing = required.difference(row)
        if missing or any(row.get(key, "") == "" for key in required):
            raise ValueError(
                f"manifest frame {sequence} has missing values "
                f"{sorted(missing or {key for key in required if not row.get(key, '')})}"
            )
        frame = protocol.FRAMES[trial.scheme]
        payload = (protocol.FINAL_PAYLOAD_TEXT if trial.phase == "final"
                   else protocol.SHORT_PAYLOAD_TEXT)
        actual = (
            int(row["sequence"]), row["phase"], row["scheme"],
            int(row["repetition"]), float(row["duty_percent"]),
        )
        wanted = (sequence, trial.phase, trial.scheme, trial.repetition,
                  trial.duty_percent)
        if actual != wanted:
            raise ValueError(f"manifest frame {sequence} violates frozen schedule")
        if (row["frame_bits"] != frame
                or row["coded_body_bits"] != frame[len(protocol.SYNC_TEXT):]
                or row["payload_bits"] != payload):
            raise ValueError(f"manifest frame {sequence} violates golden vectors")
        if (float(row["voltage_v"]), float(row["distance_m"]),
                float(row["bit_seconds"]), float(row["gap_after_s"])) != (
                    11.0, 3.0, 0.5, 15.0
                ):
            raise ValueError(f"manifest frame {sequence} violates physical contract")


def manchester_levels(bits: Sequence[int] | str) -> np.ndarray:
    checked = protocol.require_bits(bits, len(bits), "Manchester bits")
    return np.asarray(
        [level for bit in checked for level in ((1, 0) if bit else (0, 1))],
        dtype=float,
    )


def complex_template(bits: Sequence[int] | str, half_samples: int, fs: float) -> np.ndarray:
    gate = np.repeat(manchester_levels(bits), half_samples)
    time_axis = np.arange(len(gate)) / fs
    return gate * np.exp(2j * np.pi * CARRIER_HZ * time_axis)


def sliding_correlation(channel: np.ndarray, template: np.ndarray) -> np.ndarray:
    correlation = signal.fftconvolve(channel, np.conj(template[::-1]), mode="valid")
    cumulative = np.r_[0.0, np.cumsum(np.abs(channel) ** 2)]
    energy = cumulative[len(template):] - cumulative[:-len(template)]
    return np.abs(correlation) ** 2 / (
        np.vdot(template, template).real * energy + 1e-15
    )


def locate_manifest_boundaries(
    rows: list[dict[str, str]], capture_start_utc: float,
    combined_correlation: np.ndarray, fs: float,
) -> tuple[list[int], float]:
    predictions = [
        round((utc_seconds(row["started_utc"]) - capture_start_utc) * fs)
        for row in rows
    ]
    if float(rows[0]["duty_percent"]) != 100.0:
        raise ValueError("first clock-alignment frame must be the frozen 100% trial")
    radius = round(CLOCK_SEARCH_RADIUS_SECONDS * fs)
    lo = max(0, predictions[0] - radius)
    hi = min(len(combined_correlation), predictions[0] + radius + 1)
    if hi <= lo:
        raise ValueError("first manifest boundary falls outside capture")
    first = lo + int(np.argmax(combined_correlation[lo:hi]))
    correction_samples = first - predictions[0]
    starts = [prediction + correction_samples for prediction in predictions]
    if any(start < 0 for start in starts):
        raise ValueError("corrected manifest boundary falls before capture")
    if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
        raise ValueError("corrected manifest boundaries are not increasing")
    return starts, correction_samples / fs


def _extract_phasors(
    channels: Sequence[np.ndarray], start: int, count: int, half: int, fs: float
) -> np.ndarray:
    carrier = np.exp(2j * np.pi * CARRIER_HZ * np.arange(half) / fs)
    output = np.empty((count, 2), dtype=complex)
    for index in range(count):
        begin = start + index * half
        for sensor, channel in enumerate(channels):
            block = channel[begin:begin + half]
            if len(block) != half:
                raise ValueError("incomplete Manchester half-symbol")
            output[index, sensor] = np.vdot(carrier, block) / np.sqrt(half)
    return output


def complex_covariance(samples: np.ndarray, *, center: bool = True) -> np.ndarray:
    """Return E[z z^H] for row-wise complex observations."""
    values = np.asarray(samples, dtype=complex)
    if values.ndim != 2 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("complex covariance requires finite row-wise observations")
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    return values.T @ values.conj() / len(values)


def coherent_frame_llrs(
    channels: Sequence[np.ndarray], start: int, frame_bits: int, fs: float
) -> tuple[np.ndarray, float, float]:
    """Estimate channel only from known sync and covariance only from central gap."""
    if len(channels) != 2:
        raise ValueError("coherent combining requires exactly two sensors")
    half = round(fs * protocol.BIT_SECONDS / 2)
    halves = 2 * frame_bits
    frame_stop = start + halves * half
    if start < 0 or any(len(channel) < frame_stop for channel in channels):
        raise ValueError("complete frame does not fit in filtered channels")
    phasors = _extract_phasors(channels, start, halves, half, fs)

    sync_gate = manchester_levels(protocol.SYNC_TEXT).astype(bool)
    sync = phasors[:len(sync_gate)]
    tone = sync[sync_gate]
    silence = sync[~sync_gate]
    channel_vector = tone.mean(axis=0) - silence.mean(axis=0)

    noise_start = frame_stop + round(CENTRAL_GAP_LEAD_SECONDS * fs)
    noise_count = round(CENTRAL_GAP_SECONDS * fs / half)
    noise = _extract_phasors(channels, noise_start, noise_count, half, fs)
    covariance = complex_covariance(noise)
    ridge = max(float(np.trace(covariance).real) / 2, 1e-12) * 1e-6
    covariance = covariance + ridge * np.eye(2)
    try:
        weights = np.linalg.solve(covariance, channel_vector)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(covariance) @ channel_vector
    projected = phasors @ weights.conj()
    llrs = 2.0 * np.real(projected[0::2] - projected[1::2])

    def coherence(matrix: np.ndarray) -> float:
        denominator = float(matrix[0, 0].real * matrix[1, 1].real)
        return float(abs(matrix[0, 1]) ** 2 / max(denominator, 1e-15))

    tone_covariance = complex_covariance(tone, center=False)
    return llrs, coherence(tone_covariance), coherence(covariance)


def bit_errors(actual: Sequence[int], expected: Sequence[int]) -> int:
    return sum(int(a) != int(b) for a, b in zip(actual, expected))


def hard_bits(llrs: Sequence[float]) -> tuple[int, ...]:
    """Deterministic hard decision: strictly positive is one; zero is zero."""
    return tuple(int(float(value) > 0.0) for value in llrs)


def hamming_decode(body: Sequence[int]) -> tuple[tuple[int, ...], int, int | None]:
    bits = list(protocol.require_bits(body, 15, "Hamming body"))
    syndrome = 0
    for position, bit in enumerate(bits, 1):
        if bit:
            syndrome ^= position
    corrected_position = None
    if syndrome:
        corrected_position = syndrome
        bits[syndrome - 1] ^= 1
    data = tuple(bits[position - 1] for position in range(1, 16)
                 if position not in (1, 2, 4, 8))
    return data, syndrome, corrected_position


def _gf_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("division by zero in GF(32)")
    if numerator == 0:
        return 0
    return protocol._GF_EXP[(protocol._GF_LOG[numerator]
                             - protocol._GF_LOG[denominator]) % protocol.FIELD_ORDER]


def rs_syndromes(word: Sequence[int], parity_symbols: int) -> tuple[int, ...]:
    result: list[int] = []
    for root in range(1, parity_symbols + 1):
        point = protocol._gf_pow(root)
        value = 0
        for coefficient in word:
            value = protocol._gf_mul(value, point) ^ coefficient
        result.append(value)
    return tuple(result)


def _berlekamp_massey(sequence: Sequence[int]) -> list[int]:
    connection, previous = [1], [1]
    span, offset, discrepancy_scale = 0, 1, 1
    for index in range(len(sequence)):
        discrepancy = sequence[index]
        for degree in range(1, span + 1):
            if degree < len(connection):
                discrepancy ^= protocol._gf_mul(
                    connection[degree], sequence[index - degree]
                )
        if discrepancy == 0:
            offset += 1
            continue
        old = connection[:]
        multiplier = _gf_div(discrepancy, discrepancy_scale)
        required = len(previous) + offset
        if len(connection) < required:
            connection.extend([0] * (required - len(connection)))
        for degree, coefficient in enumerate(previous):
            connection[degree + offset] ^= protocol._gf_mul(multiplier, coefficient)
        if 2 * span <= index:
            span = index + 1 - span
            previous = old
            discrepancy_scale = discrepancy
            offset = 1
        else:
            offset += 1
    while len(connection) > 1 and connection[-1] == 0:
        connection.pop()
    return connection


def _solve_linear(matrix: list[list[int]], values: list[int]) -> list[int]:
    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values)]
    for column in range(size):
        pivot = next((row for row in range(column, size)
                      if augmented[row][column]), None)
        if pivot is None:
            raise ReedSolomonError("singular RS magnitude system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse = _gf_div(1, augmented[column][column])
        augmented[column] = [protocol._gf_mul(value, inverse)
                             for value in augmented[column]]
        for row in range(size):
            if row == column or augmented[row][column] == 0:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value ^ protocol._gf_mul(factor, pivot_value)
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def rs_decode_symbols(
    received: Sequence[int], data_symbols: int, parity_symbols: int
) -> tuple[tuple[int, ...], int]:
    word = tuple(received)
    if len(word) != data_symbols + parity_symbols:
        raise ValueError("wrong RS codeword length")
    protocol.symbols_to_bits(word)
    syndromes = rs_syndromes(word, parity_symbols)
    if not any(syndromes):
        return word, 0
    locator = _berlekamp_massey(syndromes)
    errors = len(locator) - 1
    if errors == 0 or 2 * errors > parity_symbols:
        raise ReedSolomonError("RS errors exceed correction capability")
    positions: list[int] = []
    length = len(word)
    for position in range(length):
        inverse_location = protocol._gf_pow(-(length - 1 - position))
        evaluation, power_value = 0, 1
        for coefficient in locator:
            evaluation ^= protocol._gf_mul(coefficient, power_value)
            power_value = protocol._gf_mul(power_value, inverse_location)
        if evaluation == 0:
            positions.append(position)
    if len(positions) != errors:
        raise ReedSolomonError("RS error locator does not split over codeword")
    locations = [protocol._gf_pow(length - 1 - position) for position in positions]
    matrix = [
        [protocol._gf_pow(protocol._GF_LOG[location] * equation)
         for location in locations]
        for equation in range(1, errors + 1)
    ]
    magnitudes = _solve_linear(matrix, list(syndromes[:errors]))
    corrected = list(word)
    for position, magnitude in zip(positions, magnitudes):
        corrected[position] ^= magnitude
    corrected_word = tuple(corrected)
    if any(rs_syndromes(corrected_word, parity_symbols)):
        raise ReedSolomonError("RS correction did not produce a codeword")
    if protocol.rs_encode_symbols(corrected_word[:data_symbols], parity_symbols) != corrected_word:
        raise ReedSolomonError("RS systematic re-encoding validation failed")
    return corrected_word, errors


def raw_payload_bits(scheme: str, body: Sequence[int]) -> tuple[int, ...]:
    bits = tuple(body)
    if scheme == "uncoded":
        return bits
    if scheme == "hamming15_11":
        return tuple(bits[position - 1] for position in range(1, 16)
                     if position not in (1, 2, 4, 8))
    if scheme == "rs5_3":
        return bits[:11]
    if scheme == "rs12_6":
        return bits[:30]
    raise ValueError(f"unknown scheme {scheme}")


def decode_body(scheme: str, body: Sequence[int]) -> dict[str, object]:
    bits = tuple(body)
    if scheme == "uncoded":
        return {"payload": bits, "failure": 0, "corrected": 0, "syndrome": 0}
    if scheme == "hamming15_11":
        payload, syndrome, position = hamming_decode(bits)
        return {
            "payload": payload, "failure": 0,
            "corrected": int(position is not None), "syndrome": syndrome,
        }
    data_symbols, parity_symbols = ((3, 2) if scheme == "rs5_3" else (6, 6))
    try:
        corrected, count = rs_decode_symbols(
            protocol.bits_to_symbols(bits), data_symbols, parity_symbols
        )
    except (ValueError, ReedSolomonError):
        return {"payload": (), "failure": 1, "corrected": 0, "syndrome": None}
    payload = protocol.symbols_to_bits(corrected[:data_symbols])
    if scheme == "rs5_3":
        if payload[11:] != (0, 0, 0, 0):
            return {
                "payload": (), "failure": 1, "corrected": count,
                "syndrome": None, "application_padding_valid": 0,
            }
        payload = payload[:11]
    return {
        "payload": payload, "failure": 0, "corrected": count,
        "syndrome": None, "application_padding_valid": 1,
    }


def _float_or_none(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def analyze(
    capture_path: Path, manifest_path: Path, metadata_path: Path,
    *, require_complete_manifest: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    t, x, y = load_capture(capture_path)
    fs = sample_rate(t)
    rows = load_manifest(manifest_path, require_complete=require_complete_manifest)
    info = metadata(metadata_path)
    required_metadata = {
        "capture_started_utc", "phase", "expected_frames", "voltage_v",
        "distance_m", "coded_bits_per_second", "outcome", "manifest_outcome",
        "safe_shutdown_outcome",
    }
    missing = required_metadata.difference(info)
    if missing:
        raise ValueError(f"metadata missing required values {sorted(missing)}")
    if require_complete_manifest and (
        info["phase"] != "all" or int(info["expected_frames"]) != 100
        or info["outcome"] != "COMPLETE" or info["manifest_outcome"] != "VALID"
        or info["safe_shutdown_outcome"] != "VERIFIED_WRITES"
    ):
        raise ValueError("metadata does not describe a complete verified 100-frame run")

    filtered = (bandpass(x, fs), bandpass(y, fs))
    filtered_analytic = tuple(signal.hilbert(channel) for channel in filtered)
    # The narrow carrier filter is authoritative for SNR and sync correlation.
    # At two coded bits/s its 1.5 Hz passband excludes the 6/10 Hz Manchester
    # sidebands, so applying it before half-symbol decisions would smear the
    # envelope. Coherent LLRs therefore use DC-centered raw analytic channels;
    # carrier projection rejects non-8-Hz content, and all learned channel/noise
    # parameters still come only from known sync and the central off gap.
    decode_analytic = tuple(
        signal.hilbert(np.asarray(channel, float) - np.median(channel))
        for channel in (x, y)
    )
    half = round(fs * protocol.BIT_SECONDS / 2)
    sync_template = complex_template(protocol.SYNC_TEXT, half, fs)
    combined_correlation = sum(
        sliding_correlation(channel, sync_template) for channel in filtered_analytic
    )
    starts, clock_correction_s = locate_manifest_boundaries(
        rows, utc_seconds(info["capture_started_utc"]), combined_correlation, fs
    )
    adc_max = adc_full_scale(x, y)
    expected_sync = tuple(map(int, protocol.SYNC_TEXT))
    output: list[dict[str, object]] = []
    previous_started = None
    previous_expected_interval = None

    for row, start in zip(rows, starts):
        frame_text = row["frame_bits"]
        expected_frame = tuple(map(int, frame_text))
        expected_body = expected_frame[len(expected_sync):]
        frame_samples = round(len(expected_frame) * protocol.BIT_SECONDS * fs)
        gap_start = start + frame_samples + round(CENTRAL_GAP_LEAD_SECONDS * fs)
        gap_samples = round(CENTRAL_GAP_SECONDS * fs)
        if start + frame_samples > len(x) or gap_start + gap_samples > len(x):
            raise ValueError(f"frame {row['sequence']} or central gap is incomplete")

        llrs, tone_coherence, off_coherence = coherent_frame_llrs(
            decode_analytic, start, len(expected_frame), fs
        )
        hard = hard_bits(llrs)
        hard_sync = hard[:len(expected_sync)]
        hard_body = hard[len(expected_sync):]
        scheme = row["scheme"]
        expected_payload = tuple(map(int, row["payload_bits"]))
        raw_payload = raw_payload_bits(scheme, hard_body)
        decoded = decode_body(scheme, hard_body)
        decoded_payload = tuple(decoded["payload"])
        decoded_errors = (None if decoded["failure"] else
                          bit_errors(decoded_payload, expected_payload))
        sync_errors = bit_errors(hard_sync, expected_sync)
        body_errors = bit_errors(hard_body, expected_body)
        raw_payload_errors = bit_errors(raw_payload, expected_payload)
        symbol_errors = None
        if scheme.startswith("rs"):
            received_symbols = protocol.bits_to_symbols(hard_body)
            expected_symbols = protocol.bits_to_symbols(expected_body)
            symbol_errors = sum(a != b for a, b in zip(received_symbols, expected_symbols))

        frame_slice = slice(start, start + frame_samples)
        gap_slice = slice(gap_start, gap_start + gap_samples)
        active_power = (power(filtered[0][frame_slice]), power(filtered[1][frame_slice]))
        off_power = (power(filtered[0][gap_slice]), power(filtered[1][gap_slice]))
        snr = tuple(signal_snr_db(active, off)
                    for active, off in zip(active_power, off_power))
        pooled_snr = signal_snr_db(sum(active_power), sum(off_power))
        frame_x, frame_y = x[frame_slice], y[frame_slice]
        clipped_x = (frame_x <= ADC_MIN) | (frame_x >= adc_max)
        clipped_y = (frame_y <= ADC_MIN) | (frame_y >= adc_max)

        started = utc_seconds(row["started_utc"])
        interval = None if previous_started is None else started - previous_started
        interval_error = (None if interval is None else interval - previous_expected_interval)
        expected_duration = len(expected_frame) * protocol.BIT_SECONDS
        duration = float(row["duration_s"])
        output.append({
            "sequence": int(row["sequence"]),
            "phase": row["phase"],
            "scheme": scheme,
            "repetition": int(row["repetition"]),
            "duty_percent": float(row["duty_percent"]),
            "duty_label": row.get("duty_label", ""),
            "frame_bits": len(expected_frame),
            "start_offset_s": start / fs,
            "boundary_source": "manifest_timestamps_frozen_first_sync_correction",
            "manifest_clock_correction_s": clock_correction_s,
            "autonomous_acquisition_evaluated": 0,
            "sync_correlation_score": float(combined_correlation[start]),
            "sample_rate_hz": fs,
            "sync_bit_errors": sync_errors,
            "coded_body_bit_errors": body_errors,
            "raw_frame_bit_errors": sync_errors + body_errors,
            "raw_payload_bit_errors": raw_payload_errors,
            "raw_payload_frame_error": int(raw_payload_errors > 0),
            "coded_body_frame_error": int(body_errors > 0),
            "raw_symbol_errors": symbol_errors,
            "decoder_failure": int(decoded["failure"]),
            "rs5_application_padding_valid": decoded.get("application_padding_valid"),
            "decoder_corrected_units": int(decoded["corrected"]),
            "hamming_syndrome": decoded["syndrome"],
            "decoded_payload_bit_errors_conditional_on_decode": decoded_errors,
            "decoded_payload_bits_evaluated": (
                0 if decoded["failure"] else len(expected_payload)
            ),
            "payload_bits_per_frame": len(expected_payload),
            "decoded_payload_frame_error": int(
                decoded["failure"] or (decoded_errors is not None and decoded_errors > 0)
            ),
            # Primary success is exact decoded payload at the scheduled boundary.
            # Sync hard errors remain diagnostic and do not independently fail it.
            "frame_error": int(
                decoded["failure"] or (decoded_errors is not None and decoded_errors > 0)
            ),
            "sync_inclusive_diagnostic_frame_error": int(
                sync_errors > 0 or decoded["failure"]
                or (decoded_errors is not None and decoded_errors > 0)
            ),
            "decoded_payload_bits": "" if decoded["failure"] else
                "".join(map(str, decoded_payload)),
            "adc_full_scale_assumed": adc_max,
            "clipped_x_percent": 100.0 * float(np.mean(clipped_x)),
            "clipped_y_percent": 100.0 * float(np.mean(clipped_y)),
            "clipped_any_percent": 100.0 * float(np.mean(clipped_x | clipped_y)),
            "inband_active_power_x": active_power[0],
            "inband_off_power_x": off_power[0],
            "inband_snr_db_x": snr[0],
            "inband_active_power_y": active_power[1],
            "inband_off_power_y": off_power[1],
            "inband_snr_db_y": snr[1],
            "inband_snr_db_pooled": pooled_snr,
            "positive_inband_power_estimate": int(pooled_snr is not None),
            "tone_coherence": tone_coherence,
            "off_coherence": off_coherence,
            "manifest_duration_s": duration,
            "expected_duration_s": expected_duration,
            "duration_error_s": duration - expected_duration,
            "start_interval_s": interval,
            "start_interval_error_s": interval_error,
            "target_pulse_us": _float_or_none(row["target_pulse_us"]),
            "pulse_count": int(row["pulse_count"]),
            "late_starts": int(row["late_starts"]),
            "pulse_min_us": _float_or_none(row["pulse_min_us"]),
            "pulse_median_us": _float_or_none(row["pulse_median_us"]),
            "pulse_p95_us": _float_or_none(row["pulse_p95_us"]),
            "pulse_max_us": _float_or_none(row["pulse_max_us"]),
        })
        previous_started = started
        previous_expected_interval = expected_duration + float(row["gap_after_s"])

    details = {
        "sample_rate_hz": fs,
        "manifest_clock_correction_s": clock_correction_s,
        "adc_full_scale_assumed": adc_max,
        "capture_samples": len(t),
        "capture_duration_s": float(t[-1] - t[0]),
    }
    return output, details


def _optional_median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def wilson_95(errors: int, total: int) -> dict[str, float | int | None]:
    """Frame-clustered binomial interval with exact numerator/denominator."""
    if total <= 0 or errors < 0 or errors > total:
        return {"errors": errors, "frames": total, "rate": None,
                "lower": None, "upper": None}
    z = 1.959963984540054
    proportion = errors / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return {
        "errors": errors, "frames": total, "rate": proportion,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def aggregate(rows: list[dict[str, object]], details: dict[str, object]) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for row in rows:
        key = f"{row['phase']}|{row['scheme']}|{float(row['duty_percent']):g}"
        groups.setdefault(key, {"rows": []})["rows"].append(row)
    by_condition: dict[str, object] = {}
    for key, holder in groups.items():
        selected = holder["rows"]
        positive = [float(row["inband_snr_db_pooled"]) for row in selected
                    if row["inband_snr_db_pooled"] is not None]
        symbol_values = [int(row["raw_symbol_errors"]) for row in selected
                         if row["raw_symbol_errors"] is not None]
        frame_errors = sum(int(row["frame_error"]) for row in selected)
        coded_body_frame_errors = sum(
            int(row["coded_body_frame_error"]) for row in selected
        )
        raw_payload_frame_errors = sum(
            int(row["raw_payload_frame_error"]) for row in selected
        )
        by_condition[key] = {
            "phase": selected[0]["phase"],
            "scheme": selected[0]["scheme"],
            "duty_percent": selected[0]["duty_percent"],
            "frames": len(selected),
            "frame_errors": frame_errors,
            "primary_payload_fer_wilson_95": wilson_95(frame_errors, len(selected)),
            "coded_body_frame_errors": coded_body_frame_errors,
            "coded_body_fer_wilson_95": wilson_95(
                coded_body_frame_errors, len(selected)
            ),
            "raw_payload_frame_errors": raw_payload_frame_errors,
            "raw_payload_fer_wilson_95": wilson_95(
                raw_payload_frame_errors, len(selected)
            ),
            "decoded_payload_frame_errors": sum(
                int(row["decoded_payload_frame_error"]) for row in selected
            ),
            "decoder_failures": sum(int(row["decoder_failure"]) for row in selected),
            "sync_bit_errors": sum(int(row["sync_bit_errors"]) for row in selected),
            "coded_body_bit_errors": sum(
                int(row["coded_body_bit_errors"]) for row in selected
            ),
            "coded_body_bits_denominator": sum(
                int(row["frame_bits"]) - len(protocol.SYNC_TEXT) for row in selected
            ),
            "raw_payload_bit_errors": sum(
                int(row["raw_payload_bit_errors"]) for row in selected
            ),
            "raw_payload_bits_denominator": sum(
                int(row["payload_bits_per_frame"]) for row in selected
            ),
            "decoded_payload_bit_errors_conditional_on_decode": sum(
                int(row["decoded_payload_bit_errors_conditional_on_decode"])
                for row in selected
                if row["decoded_payload_bit_errors_conditional_on_decode"] is not None
            ),
            "decoded_payload_bits_evaluated": sum(
                int(row["decoded_payload_bits_evaluated"]) for row in selected
            ),
            "decoded_payload_bit_errors_lower_bound_including_failures": sum(
                int(row["decoded_payload_bit_errors_conditional_on_decode"] or 0)
                for row in selected
            ),
            "decoded_payload_bit_errors_upper_bound_including_failures": sum(
                (int(row["decoded_payload_bit_errors_conditional_on_decode"])
                 if row["decoded_payload_bit_errors_conditional_on_decode"] is not None
                 else len(str(row["decoded_payload_bits"] or
                              (protocol.FINAL_PAYLOAD_TEXT if row["phase"] == "final"
                               else protocol.SHORT_PAYLOAD_TEXT))))
                for row in selected
            ),
            "raw_symbol_errors": sum(symbol_values) if symbol_values else None,
            "raw_symbols_denominator": (
                sum((int(row["frame_bits"]) - len(protocol.SYNC_TEXT)) // 5
                    for row in selected if row["raw_symbol_errors"] is not None)
                if symbol_values else None
            ),
            "positive_inband_snr_frames": len(positive),
            "no_positive_inband_power_frames": len(selected) - len(positive),
            "conditional_median_positive_inband_snr_db_pooled": _optional_median(positive),
            "max_clipped_any_percent": max(
                float(row["clipped_any_percent"]) for row in selected
            ),
            "mean_duration_error_s": float(np.mean([
                float(row["duration_error_s"]) for row in selected
            ])),
            "max_abs_duration_error_s": max(
                abs(float(row["duration_error_s"])) for row in selected
            ),
            "late_starts": sum(int(row["late_starts"]) for row in selected),
        }
    total_frame_errors = sum(int(row["frame_error"]) for row in rows)
    total_coded_body_frame_errors = sum(
        int(row["coded_body_frame_error"]) for row in rows
    )
    total_raw_payload_frame_errors = sum(
        int(row["raw_payload_frame_error"]) for row in rows
    )
    return {
        "analysis_contract": {
            "primary_boundary": "manifest timestamps plus one frozen first-100%-sync correction",
            "clock_search_radius_s": CLOCK_SEARCH_RADIUS_SECONDS,
            "autonomous_acquisition": "not evaluated",
            "filter_hz": [FILTER_LOW_HZ, FILTER_HIGH_HZ],
            "coherent_llr_training": "known 16-bit sync and transmitter-off central gap only",
            "snr": "power-subtracted; null when active power does not exceed off power",
            "golden_vectors_validated": True,
        },
        **details,
        "frames": len(rows),
        "frame_errors": total_frame_errors,
        "primary_payload_fer_wilson_95": wilson_95(total_frame_errors, len(rows)),
        "coded_body_frame_errors": total_coded_body_frame_errors,
        "coded_body_fer_wilson_95": wilson_95(
            total_coded_body_frame_errors, len(rows)
        ),
        "raw_payload_frame_errors": total_raw_payload_frame_errors,
        "raw_payload_fer_wilson_95": wilson_95(
            total_raw_payload_frame_errors, len(rows)
        ),
        "decoded_payload_frame_errors": sum(
            int(row["decoded_payload_frame_error"]) for row in rows
        ),
        "decoder_failures": sum(int(row["decoder_failure"]) for row in rows),
        "sync_bit_errors": sum(int(row["sync_bit_errors"]) for row in rows),
        "coded_body_bit_errors": sum(int(row["coded_body_bit_errors"]) for row in rows),
        "coded_body_bits_denominator": sum(
            int(row["frame_bits"]) - len(protocol.SYNC_TEXT) for row in rows
        ),
        "raw_payload_bit_errors": sum(int(row["raw_payload_bit_errors"]) for row in rows),
        "raw_payload_bits_denominator": sum(
            int(row["payload_bits_per_frame"]) for row in rows
        ),
        "decoded_payload_bit_errors_conditional_on_decode": sum(
            int(row["decoded_payload_bit_errors_conditional_on_decode"])
            for row in rows
            if row["decoded_payload_bit_errors_conditional_on_decode"] is not None
        ),
        "decoded_payload_bits_evaluated": sum(
            int(row["decoded_payload_bits_evaluated"]) for row in rows
        ),
        "decoded_payload_bit_errors_lower_bound_including_failures": sum(
            int(row["decoded_payload_bit_errors_conditional_on_decode"] or 0)
            for row in rows
        ),
        "decoded_payload_bit_errors_upper_bound_including_failures": sum(
            (int(row["decoded_payload_bit_errors_conditional_on_decode"])
             if row["decoded_payload_bit_errors_conditional_on_decode"] is not None
             else (30 if row["phase"] == "final" else 11))
            for row in rows
        ),
        "positive_inband_snr_frames": sum(
            int(row["positive_inband_power_estimate"]) for row in rows
        ),
        "no_positive_inband_power_frames": sum(
            not int(row["positive_inband_power_estimate"]) for row in rows
        ),
        "by_phase_scheme_duty": by_condition,
    }


def format_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no final experiment rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: format_csv_value(value) for key, value in row.items()}
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
        print("ERROR: refusing to overwrite analysis outputs", file=sys.stderr)
        return 2
    rows, details = analyze(args.capture, args.manifest, args.metadata)
    result = aggregate(rows, details)
    write_csv(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
    csv_hash = write_checksum(args.output)
    json_hash = write_checksum(args.summary)
    print(f"Wrote {args.output} sha256={csv_hash}")
    print(f"Wrote {args.summary} sha256={json_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
