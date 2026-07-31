#!/usr/bin/env python3
"""Offline scheduled-boundary analysis for the accepted RS(18,6) run.

Primary boundaries are manifest timestamps shifted by one correction estimated
from the first 100% sync inside a fixed +/-1.25-second window. The correction
is frozen thereafter. Physical SNR is always computed from unnormalized ADC
channels. A separate diagnostic uses per-sensor mean/RMS fitted only on the
captured 120-second prelaunch transmitter-off block and frozen for all frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_DIR = ROOT / "transmitter"
sys.path[:0] = [str(TRANSMITTER_DIR), str(ROOT / "receiver")]
import analyze_final_experiment as base  # noqa: E402
import compare_final_rs_frontends as coherent  # noqa: E402
import final_experiment_protocol as gf  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402
import run_rs18_experiment as runner  # noqa: E402

LOWPASS_NOISE_HZ = 10.0


def validate_metadata(info: dict[str, str]) -> None:
    required = {
        "experiment", "expected_frames", "capture_started_utc",
        "prelaunch_off_started_utc", "prelaunch_off_completed_utc",
        "actual_prelaunch_off_capture_s", "required_prelaunch_off_capture_s",
        "outcome", "transfer_outcome", "manifest_outcome", "safe_shutdown_outcome",
    }
    missing = required.difference(info)
    if missing:
        raise ValueError(f"metadata missing required values {sorted(missing)}")
    if (info["experiment"] != "rs18_6" or int(info["expected_frames"]) != 45
            or float(info["required_prelaunch_off_capture_s"]) != 120.0
            or float(info["actual_prelaunch_off_capture_s"]) < 120.0
            or info["outcome"] != "COMPLETE"
            or info["transfer_outcome"] != "COMPLETE"
            or info["manifest_outcome"] != "VALID"
            or info["safe_shutdown_outcome"] != "VERIFIED_WRITES"):
        raise ValueError("metadata does not describe the accepted complete RS18 run")


def prelaunch_indices(info: dict[str, str], fs: float, samples: int) -> tuple[int, int]:
    capture_start = base.utc_seconds(info["capture_started_utc"])
    off_start = base.utc_seconds(info["prelaunch_off_started_utc"])
    off_stop = base.utc_seconds(info["prelaunch_off_completed_utc"])
    start = max(0, round((off_start - capture_start) * fs))
    stop = min(samples, round((off_stop - capture_start) * fs))
    if stop - start < round(120.0 * fs):
        raise ValueError("raw capture does not contain the declared 120-second prelaunch block")
    return start, stop


def fit_off_rms(
    x: np.ndarray, y: np.ndarray, start: int, stop: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    means, rms = [], []
    for channel in (x, y):
        off = np.asarray(channel[start:stop], dtype=float)
        mean = float(np.mean(off))
        scale = float(np.sqrt(np.mean((off - mean) ** 2)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("prelaunch off RMS must be finite and positive")
        means.append(mean)
        rms.append(scale)
    return (means[0], means[1]), (rms[0], rms[1])


def locate_raw_coherent_sync_boundaries(
    manifest: list[dict[str, str]], capture_start_utc: float,
    raw_analytic: Sequence[np.ndarray], fs: float,
) -> tuple[list[int], float, float, int]:
    """Freeze one payload-truth-free correction from normalized signed sync score."""
    predictions = [
        round((base.utc_seconds(row["started_utc"]) - capture_start_utc) * fs)
        for row in manifest
    ]
    if float(manifest[0]["duty_percent"]) != 100.0:
        raise ValueError("first RS18 clock-alignment frame must be 100% duty")
    radius = round(base.CLOCK_SEARCH_RADIUS_SECONDS * fs)
    expected = np.array([1.0 if bit == "1" else -1.0
                         for bit in protocol.SYNC_TEXT])
    candidates = []
    frame_samples = round(protocol.FRAME_BITS * protocol.BIT_SECONDS * fs)
    tail_samples = round((base.CENTRAL_GAP_LEAD_SECONDS
                          + base.CENTRAL_GAP_SECONDS) * fs)
    for start in range(max(0, predictions[0] - radius), predictions[0] + radius + 1):
        stop = start + frame_samples + tail_samples
        if stop > len(raw_analytic[0]):
            continue
        local = tuple(channel[start:stop] for channel in raw_analytic)
        sync_llrs = coherent.coherent_llrs(local, protocol.FRAME_BITS, fs)[:16]
        denominator = float(np.linalg.norm(sync_llrs) * np.linalg.norm(expected))
        score = float(np.dot(expected, sync_llrs) / max(denominator, 1e-15))
        errors = int(np.count_nonzero((sync_llrs > 0) != (expected > 0)))
        candidates.append((score, start, errors))
    if not candidates:
        raise ValueError("no complete first-frame candidate in sync search window")
    score, first_start, errors = max(candidates, key=lambda item: item[0])
    correction_samples = first_start - predictions[0]
    starts = [prediction + correction_samples for prediction in predictions]
    if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
        raise ValueError("corrected RS18 manifest boundaries are not increasing")
    return starts, correction_samples / fs, score, errors


def rms_normalized_llrs(
    channels: Sequence[np.ndarray], frame_bits: int, fs: float
) -> np.ndarray:
    """MRC diagnostic after frozen off-RMS scaling; sync is the only active fit."""
    half = round(fs * protocol.BIT_SECONDS / 2)
    phasors = coherent._extract_phasors(channels, 0, 2 * frame_bits, half, fs)
    gate = base.manchester_levels(protocol.SYNC_TEXT).astype(bool)
    sync = phasors[:len(gate)]
    channel_vector = sync[gate].mean(axis=0) - sync[~gate].mean(axis=0)
    projected = phasors @ channel_vector.conj()
    return 2.0 * np.real(projected[0::2] - projected[1::2])


def hard_rs18_decode(body_llrs: Sequence[float]) -> dict[str, object]:
    llrs = tuple(float(value) for value in body_llrs)
    if len(llrs) != protocol.CODE_BITS or not all(math.isfinite(value) for value in llrs):
        raise ValueError("RS18 decoder requires exactly 90 finite body LLRs")
    hard_bits = tuple(int(value > 0.0) for value in llrs)
    hard_symbols = gf.bits_to_symbols(hard_bits)
    try:
        corrected, units = base.rs_decode_symbols(
            hard_symbols, protocol.DATA_SYMBOLS, protocol.PARITY_SYMBOLS
        )
    except (ValueError, base.ReedSolomonError):
        return {
            "failure": 1, "payload": (), "hard_bits": hard_bits,
            "hard_symbols": hard_symbols, "corrected_symbols": (),
            "corrected_symbol_count": 0,
        }
    payload = gf.symbols_to_bits(corrected[:protocol.DATA_SYMBOLS])
    return {
        "failure": 0, "payload": payload, "hard_bits": hard_bits,
        "hard_symbols": hard_symbols, "corrected_symbols": corrected,
        "corrected_symbol_count": sum(a != b for a, b in zip(hard_symbols, corrected)),
    }


def exploratory_gmd_rs18_decode(body_llrs: Sequence[float]) -> dict[str, object]:
    """Truth-free GMD diagnostic; hard RS remains the primary decoder."""
    llrs = tuple(float(value) for value in body_llrs)
    if len(llrs) != protocol.CODE_BITS or not all(math.isfinite(value) for value in llrs):
        raise ValueError("RS18 GMD requires exactly 90 finite body LLRs")
    hard = tuple(int(value > 0.0) for value in llrs)
    hard_symbols = gf.bits_to_symbols(hard)
    reliability = [min(abs(value) for value in llrs[5*i:5*i+5])
                   for i in range(protocol.CODE_SYMBOLS)]
    order = sorted(range(protocol.CODE_SYMBOLS),
                   key=lambda index: (reliability[index], index))
    schedules = [()] + [tuple(order[:count])
                        for count in range(1, protocol.PARITY_SYMBOLS + 1)]
    candidates = {}
    for erased in schedules:
        try:
            corrected = coherent.rs_decode_erasures(
                hard_symbols, protocol.DATA_SYMBOLS, protocol.PARITY_SYMBOLS, erased
            )
        except (ValueError, coherent.GMDDecodeError):
            continue
        bits = gf.symbols_to_bits(corrected)
        score = float(sum(value if bit else -value for bit, value in zip(bits, llrs)))
        candidates.setdefault(corrected, (score, erased))
    if not candidates:
        return {"failure": 1, "payload": (), "erasures": (),
                "candidate_count": 0, "margin": None}
    ranked = sorted(
        ((score, word, erased) for word, (score, erased) in candidates.items()),
        key=lambda item: (-item[0], item[1]),
    )
    score, word, erased = ranked[0]
    margin = None if len(ranked) == 1 else float(score - ranked[1][0])
    return {
        "failure": 0, "payload": gf.symbols_to_bits(word[:6]),
        "erasures": tuple(erased), "candidate_count": len(candidates),
        "margin": margin,
    }


def carrier_vs_wide_noise_snr_db(
    inband_active_power: float, inband_off_power: float, wide_off_power: float
) -> float | None:
    """Carrier-only power-subtracted numerator against 0-10 Hz off noise."""
    signal_power = inband_active_power - inband_off_power
    if (not np.isfinite(signal_power) or not np.isfinite(wide_off_power)
            or signal_power <= 0 or wide_off_power <= 0):
        return None
    return float(10.0 * math.log10(signal_power / wide_off_power))


def analyze(
    capture_path: Path, manifest_path: Path, metadata_path: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    t, x, y = base.load_capture(capture_path)
    fs = base.sample_rate(t)
    runner.validate_manifest(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as source:
        manifest = list(csv.DictReader(source))
    info = base.metadata(metadata_path)
    validate_metadata(info)

    off_start, off_stop = prelaunch_indices(info, fs, len(t))
    off_means, off_rms = fit_off_rms(x, y, off_start, off_stop)
    normalized_analytic = tuple(
        signal.hilbert((np.asarray(channel, float) - mean) / rms)
        for channel, mean, rms in zip((x, y), off_means, off_rms)
    )
    raw_analytic = tuple(
        signal.hilbert(np.asarray(channel, float) - np.median(channel))
        for channel in (x, y)
    )

    filtered = tuple(base.bandpass(channel, fs) for channel in (x, y))
    filtered_analytic = tuple(signal.hilbert(channel) for channel in filtered)
    half = round(fs * protocol.BIT_SECONDS / 2)
    sync_template = base.complex_template(protocol.SYNC_TEXT, half, fs)
    correlation = sum(
        base.sliding_correlation(channel, sync_template) for channel in filtered_analytic
    )
    starts, clock_correction, boundary_sync_score, boundary_sync_errors = (
        locate_raw_coherent_sync_boundaries(
            manifest, base.utc_seconds(info["capture_started_utc"]),
            raw_analytic, fs,
        )
    )

    lowpass_sos = signal.butter(4, LOWPASS_NOISE_HZ, btype="lowpass", fs=fs,
                                output="sos")
    wide_noise_channels = tuple(
        signal.sosfiltfilt(lowpass_sos, np.asarray(channel, float) - mean)
        for channel, mean in zip((x, y), off_means)
    )
    adc_max = base.adc_full_scale(x, y)
    expected_sync = tuple(map(int, protocol.SYNC_TEXT))

    decoded_without_truth = []
    for row, start in zip(manifest, starts):
        frame_bits = len(row["frame_bits"])
        frame_samples = round(frame_bits * protocol.BIT_SECONDS * fs)
        gap_start = start + frame_samples + round(base.CENTRAL_GAP_LEAD_SECONDS * fs)
        gap_samples = round(base.CENTRAL_GAP_SECONDS * fs)
        local_stop = gap_start + gap_samples
        if local_stop > len(x):
            raise ValueError(f"frame {row['sequence']} or central gap is incomplete")
        local_start_gap = local_stop - start
        primary_channels = tuple(channel[start:local_stop] for channel in raw_analytic)
        normalized_channels = tuple(
            channel[start:start + frame_samples] for channel in normalized_analytic
        )
        primary_llrs = coherent.coherent_llrs(primary_channels, frame_bits, fs)
        normalized_llrs = rms_normalized_llrs(normalized_channels, frame_bits, fs)
        primary_body_llrs = primary_llrs[len(expected_sync):]
        primary = hard_rs18_decode(primary_body_llrs)
        gmd = exploratory_gmd_rs18_decode(primary_body_llrs)
        normalized = hard_rs18_decode(normalized_llrs[len(expected_sync):])
        decoded_without_truth.append(
            (row, start, frame_samples, gap_start, gap_samples,
             primary_llrs, normalized_llrs, primary, gmd, normalized)
        )

    output: list[dict[str, object]] = []
    previous_started = None
    previous_expected_interval = None
    for (row, start, frame_samples, gap_start, gap_samples,
         primary_llrs, normalized_llrs, primary, gmd, normalized) in decoded_without_truth:
        # Payload/body truth is used only in this final scoring section.
        expected_frame = tuple(map(int, row["frame_bits"]))
        expected_body = expected_frame[len(expected_sync):]
        expected_symbols = gf.bits_to_symbols(expected_body)
        expected_payload = tuple(map(int, row["payload_bits"]))
        hard_frame = tuple(int(value > 0.0) for value in primary_llrs)
        hard_sync, hard_body = hard_frame[:16], tuple(primary["hard_bits"])
        normalized_frame = tuple(int(value > 0.0) for value in normalized_llrs)
        normalized_body = tuple(normalized["hard_bits"])
        payload_errors = (None if primary["failure"] else
                          base.bit_errors(primary["payload"], expected_payload))
        gmd_payload_errors = (None if gmd["failure"] else
                              base.bit_errors(gmd["payload"], expected_payload))
        normalized_payload_errors = (None if normalized["failure"] else
                                     base.bit_errors(normalized["payload"], expected_payload))
        body_errors = base.bit_errors(hard_body, expected_body)
        symbol_errors = sum(
            a != b for a, b in zip(primary["hard_symbols"], expected_symbols)
        )

        frame_slice = slice(start, start + frame_samples)
        gap_slice = slice(gap_start, gap_start + gap_samples)
        active_power = tuple(base.power(channel[frame_slice]) for channel in filtered)
        off_power = tuple(base.power(channel[gap_slice]) for channel in filtered)
        pooled_snr = base.signal_snr_db(sum(active_power), sum(off_power))
        sensor_snr = tuple(base.signal_snr_db(a, n)
                           for a, n in zip(active_power, off_power))
        wide_off = tuple(base.power(channel[gap_slice])
                         for channel in wide_noise_channels)
        carrier_wide = carrier_vs_wide_noise_snr_db(
            sum(active_power), sum(off_power), sum(wide_off)
        )
        frame_x, frame_y = x[frame_slice], y[frame_slice]
        clipped_x = (frame_x <= base.ADC_MIN) | (frame_x >= adc_max)
        clipped_y = (frame_y <= base.ADC_MIN) | (frame_y >= adc_max)

        started = base.utc_seconds(row["started_utc"])
        interval = None if previous_started is None else started - previous_started
        interval_error = None if interval is None else interval - previous_expected_interval
        duration = float(row["duration_s"])
        expected_duration = protocol.FRAME_BITS * protocol.BIT_SECONDS
        output.append({
            "sequence": int(row["sequence"]), "scheme": "rs18_6",
            "repetition": int(row["repetition"]),
            "duty_position": int(row["duty_position"]),
            "duty_percent": float(row["duty_percent"]),
            "payload_bits": row["payload_bits"],
            "boundary_source": "manifest_timestamps_frozen_first_raw_coherent_sync_correction",
            "manifest_clock_correction_s": clock_correction,
            "autonomous_acquisition_evaluated": 0,
            "start_offset_s": start / fs,
            "narrow_sync_correlation_at_frozen_boundary": float(correlation[start]),
            "sync_bit_errors": base.bit_errors(hard_sync, expected_sync),
            "raw_body_bit_errors": body_errors,
            "raw_body_bits_denominator": protocol.CODE_BITS,
            "raw_symbol_errors": symbol_errors,
            "raw_symbols_denominator": protocol.CODE_SYMBOLS,
            "decoder_failure": int(primary["failure"]),
            "corrected_symbol_count": int(primary["corrected_symbol_count"]),
            "decoded_payload_bit_errors_conditional_on_decode": payload_errors,
            "decoded_payload_bits_evaluated": 0 if primary["failure"] else 30,
            "frame_error": int(primary["failure"] or
                               (payload_errors is not None and payload_errors > 0)),
            "decoded_payload_bits": "" if primary["failure"] else
                "".join(map(str, primary["payload"])),
            "exploratory_gmd": 1,
            "exploratory_gmd_decoder_failure": int(gmd["failure"]),
            "exploratory_gmd_payload_bit_errors_conditional_on_decode": gmd_payload_errors,
            "exploratory_gmd_frame_error": int(
                gmd["failure"] or (gmd_payload_errors is not None and gmd_payload_errors > 0)
            ),
            "exploratory_gmd_wrong_codeword_miscorrection": int(
                not gmd["failure"] and gmd_payload_errors is not None
                and gmd_payload_errors > 0
            ),
            "exploratory_gmd_erasure_count": len(gmd["erasures"]),
            "exploratory_gmd_selected_erasures": ",".join(map(str, gmd["erasures"])),
            "exploratory_gmd_candidate_count": int(gmd["candidate_count"]),
            "exploratory_gmd_margin": gmd["margin"],
            "off_rms_normalized_diagnostic": 1,
            "normalized_sync_bit_errors": base.bit_errors(
                normalized_frame[:16], expected_sync
            ),
            "normalized_raw_body_bit_errors": base.bit_errors(
                normalized_body, expected_body
            ),
            "normalized_decoder_failure": int(normalized["failure"]),
            "normalized_payload_bit_errors_conditional_on_decode":
                normalized_payload_errors,
            "normalized_frame_error": int(
                normalized["failure"] or
                (normalized_payload_errors is not None and normalized_payload_errors > 0)
            ),
            "physical_inband_active_power_x": active_power[0],
            "physical_inband_off_power_x": off_power[0],
            "physical_inband_snr_db_x": sensor_snr[0],
            "physical_inband_active_power_y": active_power[1],
            "physical_inband_off_power_y": off_power[1],
            "physical_inband_snr_db_y": sensor_snr[1],
            "physical_inband_snr_db_pooled": pooled_snr,
            "physical_positive_inband_power_estimate": int(pooled_snr is not None),
            "physical_wide_0_10hz_off_power_x": wide_off[0],
            "physical_wide_0_10hz_off_power_y": wide_off[1],
            "physical_carrier_vs_0_10hz_off_noise_snr_db_pooled": carrier_wide,
            "adc_full_scale_assumed": adc_max,
            "clipped_x_percent": 100.0 * float(np.mean(clipped_x)),
            "clipped_y_percent": 100.0 * float(np.mean(clipped_y)),
            "clipped_any_percent": 100.0 * float(np.mean(clipped_x | clipped_y)),
            "manifest_duration_s": duration,
            "expected_duration_s": expected_duration,
            "duration_error_s": duration - expected_duration,
            "start_interval_s": interval,
            "start_interval_error_s": interval_error,
            "target_pulse_us": base._float_or_none(row["target_pulse_us"]),
            "pulse_count": int(row["pulse_count"]),
            "late_starts": int(row["late_starts"]),
            "pulse_median_us": base._float_or_none(row["pulse_median_us"]),
            "pulse_p95_us": base._float_or_none(row["pulse_p95_us"]),
        })
        previous_started = started
        previous_expected_interval = expected_duration + float(row["gap_after_s"])

    details = {
        "sample_rate_hz": fs, "capture_samples": len(t),
        "capture_duration_s": float(t[-1] - t[0]),
        "manifest_clock_correction_s": clock_correction,
        "first_raw_coherent_sync_signed_correlation": boundary_sync_score,
        "first_raw_coherent_sync_bit_errors": boundary_sync_errors,
        "adc_full_scale_assumed": adc_max,
        "prelaunch_off_start_sample": off_start,
        "prelaunch_off_stop_sample": off_stop,
        "prelaunch_off_samples": off_stop - off_start,
        "prelaunch_off_duration_s": (off_stop - off_start) / fs,
        "prelaunch_off_mean_x": off_means[0], "prelaunch_off_mean_y": off_means[1],
        "prelaunch_off_rms_x": off_rms[0], "prelaunch_off_rms_y": off_rms[1],
    }
    return output, details


def aggregate(rows: list[dict[str, object]], details: dict[str, object]) -> dict[str, object]:
    by_duty: dict[str, object] = {}
    for duty in protocol.DUTIES:
        selected = [row for row in rows if float(row["duty_percent"]) == duty]
        errors = sum(int(row["frame_error"]) for row in selected)
        gmd_errors = sum(int(row["exploratory_gmd_frame_error"]) for row in selected)
        normalized_errors = sum(int(row["normalized_frame_error"]) for row in selected)
        positive = [float(row["physical_inband_snr_db_pooled"]) for row in selected
                    if row["physical_inband_snr_db_pooled"] is not None]
        carrier_wide = [
            float(row["physical_carrier_vs_0_10hz_off_noise_snr_db_pooled"])
            for row in selected
            if row["physical_carrier_vs_0_10hz_off_noise_snr_db_pooled"] is not None
        ]
        by_duty[f"{duty:g}"] = {
            "duty_percent": duty, "frames": len(selected),
            "frame_errors": errors,
            "fer_wilson_95": base.wilson_95(errors, len(selected)),
            "decoder_failures": sum(int(row["decoder_failure"]) for row in selected),
            "exploratory_gmd_frame_errors": gmd_errors,
            "exploratory_gmd_fer_wilson_95": base.wilson_95(gmd_errors, len(selected)),
            "exploratory_gmd_decoder_failures": sum(
                int(row["exploratory_gmd_decoder_failure"]) for row in selected
            ),
            "exploratory_gmd_wrong_codeword_miscorrections": sum(
                int(row["exploratory_gmd_wrong_codeword_miscorrection"])
                for row in selected
            ),
            "normalized_diagnostic_frame_errors": normalized_errors,
            "normalized_diagnostic_fer_wilson_95": base.wilson_95(
                normalized_errors, len(selected)
            ),
            "normalized_diagnostic_decoder_failures": sum(
                int(row["normalized_decoder_failure"]) for row in selected
            ),
            "raw_body_bit_errors": sum(int(row["raw_body_bit_errors"])
                                       for row in selected),
            "raw_body_bits": protocol.CODE_BITS * len(selected),
            "raw_symbol_errors": sum(int(row["raw_symbol_errors"])
                                     for row in selected),
            "raw_symbols": protocol.CODE_SYMBOLS * len(selected),
            "positive_physical_inband_snr_frames": len(positive),
            "no_positive_physical_inband_power_frames": len(selected) - len(positive),
            "conditional_median_positive_physical_inband_snr_db_pooled": (
                float(np.median(positive)) if positive else None
            ),
            "carrier_vs_0_10hz_off_noise_frames": len(carrier_wide),
            "conditional_median_carrier_vs_0_10hz_off_noise_snr_db_pooled": (
                float(np.median(carrier_wide)) if carrier_wide else None
            ),
            "max_clipped_any_percent": max(float(row["clipped_any_percent"])
                                           for row in selected),
            "mean_duration_error_s": float(np.mean([
                float(row["duration_error_s"]) for row in selected
            ])),
            "late_starts": sum(int(row["late_starts"]) for row in selected),
        }
    total_errors = sum(int(row["frame_error"]) for row in rows)
    gmd_total = sum(int(row["exploratory_gmd_frame_error"]) for row in rows)
    normalized_total = sum(int(row["normalized_frame_error"]) for row in rows)
    return {
        "analysis_contract": {
            "primary_boundary": "manifest timestamps plus one frozen first-100% normalized signed raw-coherent sync correction within +/-1.25 s",
            "autonomous_acquisition": "not evaluated",
            "complex_covariance": "E[z z^H] using centered.T @ centered.conj()",
            "primary_decoder": "hard bounded-distance RS(18,6); failures count as FER",
            "exploratory_gmd": "truth-free least-reliable-symbol erasure prefixes 0..12; never primary",
            "truth_usage": "varied manifest payload used only after decoding for scoring",
            "physical_snr": "unnormalized 7.25-8.75 Hz power-subtracted active vs central off",
            "carrier_vs_wide_noise": "carrier-only narrowband power-subtracted numerator against 0-10 Hz central off power",
            "normalized_diagnostic": "per-sensor mean/RMS fit only on captured 120-second prelaunch off block, frozen globally; never physical SNR",
        },
        **details,
        "frames": len(rows), "frame_errors": total_errors,
        "fer_wilson_95": base.wilson_95(total_errors, len(rows)),
        "decoder_failures": sum(int(row["decoder_failure"]) for row in rows),
        "exploratory_gmd_frame_errors": gmd_total,
        "exploratory_gmd_fer_wilson_95": base.wilson_95(gmd_total, len(rows)),
        "exploratory_gmd_decoder_failures": sum(
            int(row["exploratory_gmd_decoder_failure"]) for row in rows
        ),
        "exploratory_gmd_wrong_codeword_miscorrections": sum(
            int(row["exploratory_gmd_wrong_codeword_miscorrection"]) for row in rows
        ),
        "normalized_diagnostic_frame_errors": normalized_total,
        "normalized_diagnostic_fer_wilson_95": base.wilson_95(
            normalized_total, len(rows)
        ),
        "normalized_diagnostic_decoder_failures": sum(
            int(row["normalized_decoder_failure"]) for row in rows
        ),
        "raw_body_bit_errors": sum(int(row["raw_body_bit_errors"]) for row in rows),
        "raw_body_bits": protocol.CODE_BITS * len(rows),
        "raw_symbol_errors": sum(int(row["raw_symbol_errors"]) for row in rows),
        "raw_symbols": protocol.CODE_SYMBOLS * len(rows),
        "positive_physical_inband_snr_frames": sum(
            int(row["physical_positive_inband_power_estimate"]) for row in rows
        ),
        "no_positive_physical_inband_power_frames": sum(
            not int(row["physical_positive_inband_power_estimate"]) for row in rows
        ),
        "by_duty": by_duty,
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
        print("ERROR: refusing to overwrite RS18 analysis outputs", file=sys.stderr)
        return 2
    rows, details = analyze(args.capture, args.manifest, args.metadata)
    summary = aggregate(rows, details)
    write_csv(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
    csv_hash = base.write_checksum(args.output)
    json_hash = base.write_checksum(args.summary)
    print(f"Wrote {args.output} sha256={csv_hash}")
    print(f"Wrote {args.summary} sha256={json_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
