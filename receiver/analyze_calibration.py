#!/usr/bin/env python3
"""Summarize synchronization, decoding, clipping, and SNR for calibration data."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal

import coded_protocol as protocol
import layered_decoder
import slnn_decoder

FRAME_SECONDS = protocol.CODED_BITS * protocol.BIT_SECONDS
CENTRAL_GAP_OFFSET_SECONDS = FRAME_SECONDS + 2.5
CENTRAL_GAP_SECONDS = 10.0
INITIAL_WAIT_SECONDS = 15.0
INTERFRAME_SECONDS = FRAME_SECONDS + protocol.INTERFRAME_GAP_SECONDS
FIRST_SEARCH_RADIUS_SECONDS = 10.0
FOLLOWING_SEARCH_RADIUS_SECONDS = 3.0
MANIFEST_CLOCK_SEARCH_RADIUS_SECONDS = 1.25
ADC_MIN = 0
ADC_12BIT_MAX = 4095
ADC_16BIT_MAX = 65535
EXPECTED_SCHEDULE = tuple(
    (round_index, position, duty)
    for round_index, duties in enumerate(
        (
            (100.0, 50.0, 25.0, 10.0),
            (10.0, 25.0, 50.0, 100.0),
            (50.0, 10.0, 100.0, 25.0),
        ),
        1,
    )
    for position, duty in enumerate(duties, 1)
)


def parse_duty_sequence(value: str) -> tuple[float, ...]:
    """Parse the runner's explicit one-frame-per-duty schedule."""
    try:
        duties = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("duty sequence must be comma-separated numbers") from exc
    if not duties or any(not 0 < duty <= 100 for duty in duties):
        raise ValueError("every duty in the sequence must be in (0, 100]")
    if len(set(duties)) != len(duties):
        raise ValueError("duty sequence must not contain duplicates")
    return duties


def schedule_for_duties(duties: tuple[float, ...]) -> tuple[tuple[int, int, float], ...]:
    return tuple((1, position, duty) for position, duty in enumerate(duties, 1))


def metadata(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            output[key] = value
    return output


def utc_seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def signal_snr_db(active_power: float, off_power: float) -> float | None:
    """Return power-subtracted SNR, or None when no positive signal remains."""
    if not np.isfinite(active_power) or not np.isfinite(off_power) or off_power <= 0:
        return None
    signal_power = active_power - off_power
    if signal_power <= 0:
        return None
    return float(10.0 * math.log10(signal_power / off_power))


def power(values: np.ndarray, baseline: float = 0.0) -> float:
    samples = np.asarray(values, dtype=float) - baseline
    return float(np.mean(samples * samples))


def format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def bandpass(values: np.ndarray, fs: float) -> np.ndarray:
    low = protocol.CARRIER_HZ - protocol.BANDWIDTH_HZ / 2
    high = protocol.CARRIER_HZ + protocol.BANDWIDTH_HZ / 2
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    centered = np.asarray(values, dtype=float) - np.median(values)
    return signal.sosfiltfilt(sos, centered)


def load_capture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    required = {"t", "x", "y"}
    if not data.dtype.names or not required.issubset(data.dtype.names):
        raise ValueError("capture must contain t,x,y columns")
    good = np.isfinite(data["t"]) & np.isfinite(data["x"]) & np.isfinite(data["y"])
    return data["t"][good], data["x"][good], data["y"][good]


def validate_manifest(
    rows: list[dict[str, str]],
    *,
    require_complete: bool = True,
    expected_schedule: tuple[tuple[int, int, float], ...] = EXPECTED_SCHEDULE,
) -> None:
    if not rows:
        raise ValueError("transmitter manifest is empty")
    if require_complete and len(rows) != len(expected_schedule):
        raise ValueError(
            f"calibration manifest must contain exactly {len(expected_schedule)} frames"
        )
    if len(rows) > len(expected_schedule):
        raise ValueError("calibration manifest contains too many frames")
    expected_bits = "".join(map(str, protocol.encode_message("A")))
    for index, (row, expected) in enumerate(zip(rows, expected_schedule), 1):
        required = {
            "sequence", "round", "position", "letter", "duty_percent",
            "voltage_v", "coded_bits",
        }
        missing = required.difference(row)
        if missing:
            raise ValueError(f"manifest frame {index} is missing columns {sorted(missing)}")
        actual = (
            int(row["round"]), int(row["position"]), float(row["duty_percent"])
        )
        if int(row["sequence"]) != index or actual != expected:
            raise ValueError(f"manifest frame {index} violates the frozen schedule")
        if row["letter"] != "A" or row["coded_bits"] != expected_bits:
            raise ValueError(f"manifest frame {index} is not the canonical ~A codeword")


def load_manifest(
    path: Path,
    *,
    require_complete: bool = True,
    expected_schedule: tuple[tuple[int, int, float], ...] = EXPECTED_SCHEDULE,
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    validate_manifest(
        rows,
        require_complete=require_complete,
        expected_schedule=expected_schedule,
    )
    return rows


def adc_full_scale(*channels: np.ndarray) -> int:
    """Support the observed 12-bit stream and documented read_u16 stream."""
    maximum = max(float(np.max(np.asarray(channel))) for channel in channels)
    return ADC_16BIT_MAX if maximum > ADC_12BIT_MAX else ADC_12BIT_MAX


def analyze(
    capture_path: Path,
    manifest_path: Path,
    metadata_path: Path,
    *,
    require_complete_manifest: bool = True,
    expected_schedule: tuple[tuple[int, int, float], ...] = EXPECTED_SCHEDULE,
    bit_seconds: float = protocol.BIT_SECONDS,
    use_manifest_boundaries: bool = False,
) -> list[dict[str, object]]:
    t, x, y = load_capture(capture_path)
    fs = layered_decoder.sample_rate(t)
    rows = load_manifest(
        manifest_path,
        require_complete=require_complete_manifest,
        expected_schedule=expected_schedule,
    )
    if bit_seconds <= 0:
        raise ValueError("bit_seconds must be positive")
    frame_seconds = protocol.CODED_BITS * bit_seconds
    half_symbol_seconds = bit_seconds / 2
    interframe_seconds = frame_seconds + protocol.INTERFRAME_GAP_SECONDS
    central_gap_offset_seconds = frame_seconds + 2.5
    info = metadata(metadata_path)
    capture_start = utc_seconds(info["capture_started_utc"])
    transmitter_ack = utc_seconds(info["transmitter_pid_ack_utc"])
    filtered = (bandpass(x, fs), bandpass(y, fs))
    analytic = tuple(signal.hilbert(channel) for channel in filtered)
    half = round(fs * half_symbol_seconds)
    frame_samples = round(frame_seconds * fs)
    gap_samples = round(CENTRAL_GAP_SECONDS * fs)
    template = protocol.complex_template(protocol.ENCODED_HEADER, half, fs)
    correlations = [
        layered_decoder.sliding_correlation(channel, template) for channel in analytic
    ]
    combined_correlation = sum(correlations)

    located: list[tuple[dict[str, str], int, int]] = []
    first_prediction = round(
        (transmitter_ack - capture_start + INITIAL_WAIT_SECONDS) * fs
    )
    manifest_predictions: list[int] | None = None
    manifest_clock_correction = 0
    if use_manifest_boundaries:
        try:
            manifest_predictions = [
                round((utc_seconds(row["started_utc"]) - capture_start) * fs)
                for row in rows
            ]
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "manifest-boundary analysis requires a valid started_utc per frame"
            ) from exc
        radius = round(MANIFEST_CLOCK_SEARCH_RADIUS_SECONDS * fs)
        lo = max(0, manifest_predictions[0] - radius)
        hi = min(len(combined_correlation), manifest_predictions[0] + radius + 1)
        if hi <= lo:
            raise ValueError("first manifest frame predicted outside capture")
        first_start = lo + int(np.argmax(combined_correlation[lo:hi]))
        manifest_clock_correction = first_start - manifest_predictions[0]
    else:
        first_start = None

    for index, row in enumerate(rows):
        if manifest_predictions is not None:
            start = manifest_predictions[index] + manifest_clock_correction
        else:
            predicted = (
                first_prediction
                if first_start is None
                else first_start + round(index * interframe_seconds * fs)
            )
            radius_seconds = (
                FIRST_SEARCH_RADIUS_SECONDS if first_start is None
                else FOLLOWING_SEARCH_RADIUS_SECONDS
            )
            radius = round(radius_seconds * fs)
            lo = max(0, predicted - radius)
            hi = min(len(combined_correlation), predicted + radius + 1)
            if hi <= lo:
                raise ValueError(f"frame {row['sequence']} predicted outside capture")
            start = lo + int(np.argmax(combined_correlation[lo:hi]))
            if first_start is None:
                first_start = start
        if located and start <= located[-1][1]:
            raise ValueError("localized frame starts are not strictly increasing")
        gap_start = start + round(central_gap_offset_seconds * fs)
        if start + frame_samples > len(x) or gap_start + gap_samples > len(x):
            raise ValueError(f"frame {row['sequence']} or central gap is incomplete")
        located.append((row, start, gap_start))

    # One frozen transmitter-off DC baseline per sensor for the entire session.
    off_indices = np.concatenate([
        np.arange(gap_start, gap_start + gap_samples) for _, _, gap_start in located
    ])
    baselines = (float(np.mean(x[off_indices])), float(np.mean(y[off_indices])))
    adc_max = adc_full_scale(x, y)
    output: list[dict[str, object]] = []

    for row, start, gap_start in located:
        frame_slice = slice(start, start + frame_samples)
        gap_slice = slice(gap_start, gap_start + gap_samples)
        broad_active = (power(x[frame_slice], baselines[0]), power(y[frame_slice], baselines[1]))
        broad_off = (power(x[gap_slice], baselines[0]), power(y[gap_slice], baselines[1]))
        inband_active = (power(filtered[0][frame_slice]), power(filtered[1][frame_slice]))
        inband_off = (power(filtered[0][gap_slice]), power(filtered[1][gap_slice]))
        broad_snr = tuple(signal_snr_db(a, n) for a, n in zip(broad_active, broad_off))
        inband_snr = tuple(signal_snr_db(a, n) for a, n in zip(inband_active, inband_off))
        broad_pooled = signal_snr_db(sum(broad_active), sum(broad_off))
        inband_pooled = signal_snr_db(sum(inband_active), sum(inband_off))

        frame_x, frame_y = x[frame_slice], y[frame_slice]
        clipped_x = (frame_x <= ADC_MIN) | (frame_x >= adc_max)
        clipped_y = (frame_y <= ADC_MIN) | (frame_y >= adc_max)
        clipped_any = clipped_x | clipped_y
        coherent = slnn_decoder.coherent_llrs(
            analytic,
            start,
            fs,
            half_symbol_seconds=half_symbol_seconds,
        )
        restricted = slnn_decoder.decode_alphabet(coherent.llrs, expected_letter="A")
        full = slnn_decoder.decode_full(
            coherent.llrs,
            expected_value=(protocol.HEADER_BYTE << 8) | ord("A"),
        )
        accepted = full.header == protocol.HEADER_BYTE and full.letter == "A"
        output.append({
            "sequence": int(row["sequence"]),
            "round": int(row["round"]),
            "position": int(row["position"]),
            "voltage_v": float(row["voltage_v"]),
            "duty_percent": float(row["duty_percent"]),
            "start_offset_s": start / fs,
            "boundary_source": (
                "manifest_timestamps" if manifest_predictions is not None
                else "correlation_schedule"
            ),
            "manifest_clock_correction_s": (
                manifest_clock_correction / fs
                if manifest_predictions is not None else None
            ),
            "preamble_score": float(combined_correlation[start]),
            "decoded_header": f"0x{full.header:02X}",
            "decoded_letter": full.letter,
            "restricted_letter": restricted.letter,
            "accepted_decode_only": int(accepted),
            "expected_rank_full": full.expected_rank,
            "expected_rank_alphabet": restricted.expected_rank,
            "decode_margin": full.margin,
            "adc_full_scale_assumed": adc_max,
            "clipped_x_percent": 100.0 * float(np.mean(clipped_x)),
            "clipped_y_percent": 100.0 * float(np.mean(clipped_y)),
            "clipped_any_percent": 100.0 * float(np.mean(clipped_any)),
            "baseline_x": baselines[0],
            "baseline_y": baselines[1],
            "broadband_active_power_x": broad_active[0],
            "broadband_off_power_x": broad_off[0],
            "broadband_snr_db_x": broad_snr[0],
            "broadband_active_power_y": broad_active[1],
            "broadband_off_power_y": broad_off[1],
            "broadband_snr_db_y": broad_snr[1],
            "broadband_snr_db_pooled": broad_pooled,
            "inband_active_power_x": inband_active[0],
            "inband_off_power_x": inband_off[0],
            "inband_snr_db_x": inband_snr[0],
            "inband_active_power_y": inband_active[1],
            "inband_off_power_y": inband_off[1],
            "inband_snr_db_y": inband_snr[1],
            "inband_snr_db_pooled": inband_pooled,
            "tone_coherence": coherent.tone_coherence,
            "off_coherence": coherent.silence_coherence,
        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no calibration rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: format_optional(value) if value is None or isinstance(value, float) else value
                for key, value in row.items()
            })


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    duties = sorted({float(row["duty_percent"]) for row in rows}, reverse=True)
    by_duty = {}
    for duty in duties:
        selected = [row for row in rows if float(row["duty_percent"]) == duty]
        positive = [
            float(row["inband_snr_db_pooled"])
            for row in selected if row["inband_snr_db_pooled"] is not None
        ]
        by_duty[f"{duty:g}"] = {
            "frames": len(selected),
            "decode_only_accepted": sum(
                int(row["accepted_decode_only"]) for row in selected
            ),
            "positive_inband_snr_frames": len(positive),
            "conditional_median_positive_inband_snr_db_pooled": (
                float(np.median(positive)) if positive else None
            ),
            "max_clipped_any_percent": max(
                float(row["clipped_any_percent"]) for row in selected
            ),
        }
    return {
        "frames": len(rows),
        "decode_only_accepted": sum(
            int(row["accepted_decode_only"]) for row in rows
        ),
        "by_duty": by_duty,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--duty-sequence",
        help="comma-separated one-frame-per-duty schedule used by the runner",
    )
    parser.add_argument(
        "--bit-seconds", type=float, default=protocol.BIT_SECONDS,
        help="coded-bit duration used for this capture",
    )
    parser.add_argument(
        "--manifest-boundaries", action="store_true",
        help="use per-frame manifest timestamps after first-frame clock alignment",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = (
        schedule_for_duties(parse_duty_sequence(args.duty_sequence))
        if args.duty_sequence is not None
        else EXPECTED_SCHEDULE
    )
    rows = analyze(
        args.capture,
        args.manifest,
        args.metadata,
        expected_schedule=expected,
        bit_seconds=args.bit_seconds,
        use_manifest_boundaries=args.manifest_boundaries,
    )
    write_csv(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary(rows), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
