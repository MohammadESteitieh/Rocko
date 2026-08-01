#!/usr/bin/env python3
"""Train temporal H0 models and benchmark them on the physical Hamming run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

import coded_protocol as protocol
import duong_whitener
import layered_decoder
import slnn_decoder
import temporal_whitening as temporal


def repair_saturation(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    bad = (output <= 1) | (output >= 4094)
    if np.any(bad):
        good = ~bad
        output[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), output[good])
    return output


def epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def group_decode(llrs: np.ndarray) -> tuple[int, int]:
    scores = np.asarray(llrs).reshape(4, 7) @ protocol.GROUP_CODEBOOK.T
    classes = np.argmax(scores, axis=1)
    return protocol.decode_bytes_from_classes(classes)


def noise_metrics(features: np.ndarray, max_lag: int = 40) -> dict[str, float]:
    values = np.asarray(features)
    correlations = []
    for lag in range(1, min(max_lag, len(values) // 4) + 1):
        for column in range(values.shape[1]):
            correlations.append(abs(np.corrcoef(values[:-lag, column], values[lag:, column])[0, 1]))
    covariance = np.cov(values, rowvar=False)
    eig = np.linalg.eigvalsh(covariance)
    cross = np.corrcoef(values, rowvar=False) - np.eye(values.shape[1])
    return {
        "mean_abs_temporal_correlation_lags_1_40": float(np.nanmean(correlations)),
        "max_abs_spatial_correlation": float(np.nanmax(np.abs(cross))),
        "covariance_condition": float(eig[-1] / max(eig[0], 1e-15)),
    }


def zca_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    covariance = np.cov(features - mean, rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues[-1]) * 1e-3, 1e-9)
    matrix = vectors @ np.diag(1 / np.sqrt(np.maximum(eigenvalues, floor))) @ vectors.T
    return mean, matrix


def save_linear_models(root: Path, var, kalman) -> None:
    np.savez(
        root / "temporal_var20.npz",
        order=var.order,
        mean=var.mean,
        scale=var.scale,
        coefficients=var.coefficients,
        residual_whitener=var.residual_whitener,
    )
    np.savez(
        root / "temporal_kalman.npz",
        mean=kalman.mean,
        scale=kalman.scale,
        transition=kalman.transition,
        process_covariance=kalman.process_covariance,
        observation_covariance=kalman.observation_covariance,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fs = protocol.DEFAULT_SAMPLE_RATE_HZ
    noise = np.genfromtxt(args.noise, delimiter=",", names=True)
    noise_channels = layered_decoder.analytic_channels(
        repair_saturation(noise["x"]), repair_saturation(noise["y"]), fs
    )
    noise_features = temporal.baseband_features(noise_channels, fs)
    train_end = int(0.70 * len(noise_features))
    validation_end = int(0.85 * len(noise_features))
    train = noise_features[:train_end]
    validation = noise_features[train_end:validation_end]
    test = noise_features[validation_end:]

    zca_mean, zca_matrix = zca_fit(train)
    var = temporal.fit_var(train, order=20)
    kalman = temporal.fit_kalman(train, observation_fraction=0.1)
    gru = temporal.fit_neural(train, kind="gru")
    tcn = temporal.fit_neural(train, kind="tcn")
    save_linear_models(args.models, var, kalman)
    gru.save(args.models / "temporal_gru.pt")
    tcn.save(args.models / "temporal_tcn.pt")

    transformations = {
        "baseline": lambda values: values,
        "zca": lambda values: (values - zca_mean) @ zca_matrix.T,
        "var": var.innovations,
        "kalman": kalman.innovations,
        "gru": gru.innovations,
        "tcn": tcn.innovations,
    }

    # Held-out H0 metrics and zero-observed-false-alarm preamble thresholds.
    template = protocol.complex_template(
        protocol.ENCODED_HEADER, round(fs * protocol.HALF_SYMBOL_SECONDS), fs
    )
    metrics = {}
    h0_thresholds = {}
    for name, transform in transformations.items():
        # Transform the continuous recording before slicing the untouched H0
        # segment so causal model state/history is not reset at the split.
        transformed = transform(noise_features)[validation_end:]
        metrics[name] = noise_metrics(transformed)
        channels = temporal.analytic_channels(transformed, fs)
        correlation = sum(layered_decoder.sliding_correlation(z, template) for z in channels)
        h0_thresholds[name] = float(np.max(correlation))
        metrics[name]["h0_max_preamble"] = h0_thresholds[name]
        metrics[name]["h0_windows"] = int(len(correlation))

    capture = np.genfromtxt(args.capture, delimiter=",", names=True)
    original_channels = layered_decoder.analytic_channels(capture["x"], capture["y"], fs)
    capture_features = temporal.baseband_features(original_channels, fs)
    channel_sets = {
        name: temporal.analytic_channels(transform(capture_features), fs)
        for name, transform in transformations.items()
    }
    correlations = {
        name: sum(layered_decoder.sliding_correlation(z, template) for z in channels)
        for name, channels in channel_sets.items()
    }

    truth = list(csv.DictReader(args.manifest.open()))
    first_index = round(48.180 * fs)
    first_epoch = epoch(truth[0]["started_utc"])
    counts = defaultdict(lambda: defaultdict(int))
    records = []
    for row in truth:
        predicted = first_index + round((epoch(row["started_utc"]) - first_epoch) * fs)
        expected = ord(row["letter"])
        duty = float(row["duty_percent"])
        record = {
            "sequence": row["sequence"],
            "phase": row["phase"],
            "duty_percent": row["duty_percent"],
            "expected": row["letter"],
        }
        for name, channels in channel_sets.items():
            correlation = correlations[name]
            radius = round(3 * fs)
            lo, hi = max(0, predicted - radius), min(len(correlation), predicted + radius + 1)
            peak = lo + int(np.argmax(correlation[lo:hi]))
            for timing, index in (("known", predicted), ("sync", peak)):
                coherent = slnn_decoder.coherent_llrs(channels, index, fs)
                header, payload = group_decode(coherent.llrs)
                ok = header == protocol.HEADER_BYTE and payload == expected
                counts[(name, timing)][duty] += int(ok)
                counts[(name, timing)]["all"] += int(ok)
                record[f"{name}_{timing}_header"] = f"0x{header:02X}"
                record[f"{name}_{timing}_payload"] = f"0x{payload:02X}"
                record[f"{name}_{timing}_ok"] = int(ok)
            score = float(correlation[peak])
            threshold = h0_thresholds[name]
            accepted = bool(record[f"{name}_sync_ok"] and score > threshold)
            legacy_accepted = bool(record[f"{name}_sync_ok"] and score >= 0.8)
            counts[(name, "h0-accepted")][duty] += int(accepted)
            counts[(name, "h0-accepted")]["all"] += int(accepted)
            counts[(name, "legacy-accepted")][duty] += int(legacy_accepted)
            counts[(name, "legacy-accepted")]["all"] += int(legacy_accepted)
            record[f"{name}_preamble"] = score
            record[f"{name}_h0_threshold"] = threshold
            record[f"{name}_h0_accepted"] = int(accepted)
            record[f"{name}_legacy_accepted"] = int(legacy_accepted)
            record[f"{name}_shift_seconds"] = (peak - predicted) / fs
        records.append(record)

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    duties = (100.0, 50.0, 25.0, 10.0, 1.0)
    results = {}
    for name in transformations:
        results[name] = {
            "known_correct": int(counts[(name, "known")]["all"]),
            "sync_correct": int(counts[(name, "sync")]["all"]),
            "h0_accepted_correct": int(counts[(name, "h0-accepted")]["all"]),
            "legacy_accepted_correct": int(counts[(name, "legacy-accepted")]["all"]),
            "by_duty_sync": {str(duty): int(counts[(name, "sync")][duty]) for duty in duties},
            "by_duty_h0_accepted": {
                str(duty): int(counts[(name, "h0-accepted")][duty]) for duty in duties
            },
        }
    summary = {
        "noise_split": {"train_samples": len(train), "validation_samples": len(validation), "test_samples": len(test)},
        "metrics": metrics,
        "results": results,
        "notes": [
            "All temporal models were trained only on the first 70% of the static H0 recording.",
            "Model-specific H0 thresholds are maxima over the untouched final 15% of that recording.",
            "Blind temporal predictors may cancel predictable beacon energy; this benchmark measures that risk.",
            "The physical signal run contains unlabelled human-motion interference.",
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
