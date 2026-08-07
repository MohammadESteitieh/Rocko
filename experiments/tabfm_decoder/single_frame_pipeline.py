"""Manifest-free feature extraction for one start-aligned RS18 message CSV."""

from __future__ import annotations

import csv
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]

import analyze_final_experiment as base  # noqa: E402
import compare_final_rs_frontends as coherent  # noqa: E402
import export_rs18_table as exporter  # noqa: E402
import final_experiment_protocol as gf  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402

FEATURE_SET = "sensor-y-hybrid"
REFERENCE_BODY_ROWS = 84
QUERY_SYNC_ROWS = len(protocol.SYNC_TEXT)
TOTAL_CONTEXT_ROWS = REFERENCE_BODY_ROWS + QUERY_SYNC_ROWS
POST_FRAME_LEAD_SECONDS = base.CENTRAL_GAP_LEAD_SECONDS
POST_FRAME_NOISE_SECONDS = base.CENTRAL_GAP_SECONDS
REQUIRED_SECONDS = (
    protocol.FRAME_BITS * protocol.BIT_SECONDS
    + POST_FRAME_LEAD_SECONDS + POST_FRAME_NOISE_SECONDS
)
ANALYTIC_PADDING_SECONDS = 5.0
REFERENCE_CONTEXT_SEQUENCES = (1, 17, 24)
REFERENCE_CONTEXT_REPETITIONS = (1, 2, 3)
HELD_OUT_REPETITIONS = (4, 5)
CONTEXT_SEED = 2026
NOMINAL_SAMPLE_RATE_HZ = 200.0
SAMPLE_RATE_TOLERANCE_HZ = 0.01
SAMPLE_INTERVAL_TOLERANCE_SECONDS = 5e-6
CAPTURE_DURATION_TOLERANCE_SECONDS = 5e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extraction_software_contract() -> dict[str, str]:
    return {
        "single_frame_pipeline_sha256": sha256_file(Path(__file__)),
        "analyze_final_experiment_sha256": sha256_file(Path(base.__file__)),
        "coherent_frontend_sha256": sha256_file(Path(coherent.__file__)),
        "export_rs18_table_sha256": sha256_file(Path(exporter.__file__)),
        "final_experiment_protocol_sha256": sha256_file(Path(gf.__file__)),
        "rs18_experiment_protocol_sha256": sha256_file(Path(protocol.__file__)),
        "numpy_version": package_metadata.version("numpy"),
        "scipy_version": package_metadata.version("scipy"),
    }


def write_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def verify_sidecar(path: Path) -> str:
    actual = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing SHA-256 sidecar for {path}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[0] != actual or Path(fields[1]).name != path.name:
        raise ValueError(f"invalid SHA-256 sidecar for {path}")
    return actual


def load_signal_csv(path: Path, frame_start_seconds: float = 0.0
                    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Load one message plus its trailing off-noise interval."""
    if not np.isfinite(frame_start_seconds) or frame_start_seconds < 0:
        raise ValueError("frame start must be a finite nonnegative offset")
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if not data.dtype.names or set(data.dtype.names) != {"t", "x", "y"}:
        raise ValueError("single-frame CSV must contain exactly t,x,y columns")
    t = np.asarray(data["t"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("single-frame CSV values must be finite")
    differences = np.diff(t)
    if np.any(differences <= 0):
        raise ValueError("timestamps must be strictly increasing")
    nominal_step = 1.0 / NOMINAL_SAMPLE_RATE_HZ
    if np.max(np.abs(differences - nominal_step)) > SAMPLE_INTERVAL_TOLERANCE_SECONDS:
        raise ValueError("timestamps are not uniformly sampled at 200 Hz")
    median_step = float(np.median(differences))
    measured_fs = 1.0 / median_step
    if abs(measured_fs - NOMINAL_SAMPLE_RATE_HZ) > SAMPLE_RATE_TOLERANCE_HZ:
        raise ValueError(
            f"sample rate {measured_fs:.6f} Hz is incompatible; "
            f"this frozen frontend requires {NOMINAL_SAMPLE_RATE_HZ:.0f} Hz"
        )
    fs = NOMINAL_SAMPLE_RATE_HZ

    frame_start = round(frame_start_seconds * fs)
    required = round(REQUIRED_SECONDS * fs)
    stop = frame_start + required
    if stop != len(t):
        raise ValueError(
            f"single-frame CSV must contain exactly {REQUIRED_SECONDS:.1f} seconds "
            "from frame start (53-second message plus 12.5-second off-noise tail)"
        )
    expected_duration = (len(t) - 1) / NOMINAL_SAMPLE_RATE_HZ
    actual_duration = float(t[-1] - t[0])
    if abs(actual_duration - expected_duration) > CAPTURE_DURATION_TOLERANCE_SECONDS:
        raise ValueError("timestamp duration is inconsistent with 200 Hz sample indexing")
    local_t = t[frame_start:stop]
    local_t = local_t - local_t[0]
    return fs, local_t, x[frame_start:stop], y[frame_start:stop]


def _analytic(values: np.ndarray, center: float, scale: float,
              fs: float) -> np.ndarray:
    normalized = (np.asarray(values, dtype=float) - center) / scale
    padding = min(round(ANALYTIC_PADDING_SECONDS * fs), len(normalized) - 1)
    padded = np.pad(normalized, (padding, padding), mode="reflect")
    analytic = signal.hilbert(padded)
    return analytic[padding:padding + len(normalized)]


def extract_rows_from_arrays(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, fs: float,
    *, sequence: int = 0, repetition: int = 0, duty_percent: float = 0.0,
    frame_text: str | None = None,
) -> list[dict[str, object]]:
    """Extract the same TabFM feature schema without using a manifest boundary."""
    required = round(REQUIRED_SECONDS * fs)
    if any(len(values) < required for values in (t, x, y)):
        raise ValueError("arrays do not contain one frame and its off-noise tail")
    t, x, y = (np.asarray(values[:required]) for values in (t, x, y))
    if frame_text is not None and len(frame_text) != protocol.FRAME_BITS:
        raise ValueError("reference frame truth has the wrong length")

    frame_samples = round(protocol.FRAME_BITS * protocol.BIT_SECONDS * fs)
    noise_start = frame_samples + round(POST_FRAME_LEAD_SECONDS * fs)
    noise_stop = noise_start + round(POST_FRAME_NOISE_SECONDS * fs)
    means, scales = [], []
    for values in (x, y):
        noise = values[noise_start:noise_stop]
        mean = float(np.mean(noise))
        scale = float(np.sqrt(np.mean((noise - mean) ** 2)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("trailing off-noise RMS must be finite and positive")
        means.append(mean)
        scales.append(scale)

    normalized = tuple(
        _analytic(values, mean, scale, fs)
        for values, mean, scale in zip((x, y), means, scales)
    )
    raw = tuple(
        _analytic(values, mean, 1.0, fs)
        for values, mean in zip((x, y), means)
    )
    half = round(fs * protocol.BIT_SECONDS / 2)
    phasors = coherent._extract_phasors(
        normalized, 0, 2 * protocol.FRAME_BITS, half, fs
    )
    sync_gate = base.manchester_levels(protocol.SYNC_TEXT).astype(bool)
    sync = phasors[:len(protocol.SYNC_TEXT) * 2]
    channel_vector = sync[sync_gate].mean(axis=0) - sync[~sync_gate].mean(axis=0)
    baseline_llrs = coherent.coherent_llrs(raw, protocol.FRAME_BITS, fs)
    sensor_y_llrs = coherent.coherent_llrs((raw[1],), protocol.FRAME_BITS, fs)

    output: list[dict[str, object]] = []
    for frame_index in range(protocol.FRAME_BITS):
        section = "sync" if frame_index < len(protocol.SYNC_TEXT) else "body"
        body_index = frame_index - len(protocol.SYNC_TEXT)
        if section == "sync":
            position = f"sync_{frame_index:02d}"
            symbol_index: object = "sync"
            bit_in_symbol: object = frame_index
        else:
            position = f"body_{body_index:02d}"
            symbol_index = body_index // gf.FIELD_BITS
            bit_in_symbol = body_index % gf.FIELD_BITS
        target: object = "" if frame_text is None else int(frame_text[frame_index])
        first, second = phasors[2 * frame_index:2 * frame_index + 2]
        row: dict[str, object] = {
            "role": "unassigned",
            "sequence": sequence,
            "repetition": repetition,
            "duty_percent": duty_percent,
            "section": section,
            "frame_bit_index": frame_index,
            "body_bit_index": body_index if section == "body" else "",
            "target_bit": target,
            "position": position,
            "symbol_index": symbol_index,
            "bit_in_symbol": bit_in_symbol,
            "baseline_llr": float(baseline_llrs[frame_index]),
            "baseline_bit": int(baseline_llrs[frame_index] > 0),
            "sensor_y_llr": float(sensor_y_llrs[frame_index]),
            "sensor_y_abs_llr": float(abs(sensor_y_llrs[frame_index])),
            "sensor_y_bit": int(sensor_y_llrs[frame_index] > 0),
        }
        row.update(exporter._complex_features(
            "x", first[0], second[0], channel_vector[0]
        ))
        row.update(exporter._complex_features(
            "y", first[1], second[1], channel_vector[1]
        ))
        pooled = (
            complex(row["x_aligned_i"], row["x_aligned_q"])
            + complex(row["y_aligned_i"], row["y_aligned_q"])
        )
        row["pooled_aligned_i"] = float(pooled.real)
        row["pooled_aligned_q"] = float(pooled.imag)
        output.append(row)
    return output


def extract_rows(path: Path, *, frame_start_seconds: float = 0.0,
                 sequence: int = 0, repetition: int = 0,
                 duty_percent: float = 0.0,
                 frame_text: str | None = None) -> list[dict[str, object]]:
    fs, t, x, y = load_signal_csv(path, frame_start_seconds)
    return extract_rows_from_arrays(
        t, x, y, fs, sequence=sequence, repetition=repetition,
        duty_percent=duty_percent, frame_text=frame_text,
    )


def build_prompt(reference_rows: Sequence[dict[str, object]],
                 query_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    historical = [dict(row) for row in reference_rows]
    if len(historical) != REFERENCE_BODY_ROWS:
        raise ValueError(f"reference context must contain {REFERENCE_BODY_ROWS} rows")
    if ({int(row["target_bit"]) for row in historical} != {0, 1}
            or sum(int(row["target_bit"]) == 0 for row in historical) != 42
            or any(row["section"] != "body" for row in historical)):
        raise ValueError("reference context must contain 42 rows from each class")
    sync = [dict(row, role="known_sync", target_bit=int(protocol.SYNC_TEXT[index]))
            for index, row in enumerate(query_rows[:QUERY_SYNC_ROWS])]
    body = [dict(row, role="query", target_bit="")
            for row in query_rows[QUERY_SYNC_ROWS:]]
    historical = [dict(row, role="historical") for row in historical]
    if len(sync) != QUERY_SYNC_ROWS or len(body) != protocol.CODE_BITS:
        raise ValueError("query extraction did not produce one complete RS18 frame")
    return historical + sync + body


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=exporter.TABLE_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_reference_context(path: Path, metadata_path: Path
                           ) -> tuple[list[dict[str, str]], dict[str, object]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = verify_sidecar(path)
    verify_sidecar(metadata_path)
    if metadata.get("artifact") != "rocko-tabfm-fixed-reference-context-v1":
        raise ValueError("reference context metadata has the wrong artifact type")
    if metadata.get("context_sha256") != actual:
        raise ValueError("reference context hash does not match metadata")
    if (metadata.get("feature_set") != FEATURE_SET
            or metadata.get("feature_columns") != exporter.FEATURE_SETS[FEATURE_SET]):
        raise ValueError("reference context feature schema is incompatible")
    if (metadata.get("context_rows") != REFERENCE_BODY_ROWS
            or metadata.get("class_rows") != {"0": 42, "1": 42}):
        raise ValueError("reference context row contract is incompatible")
    if (metadata.get("reference_sequences") != list(REFERENCE_CONTEXT_SEQUENCES)
            or metadata.get("reference_repetitions") != list(REFERENCE_CONTEXT_REPETITIONS)
            or metadata.get("held_out_repetitions") != list(HELD_OUT_REPETITIONS)):
        raise ValueError("reference/validation split is not frozen")
    extraction = metadata.get("extraction", {})
    expected_extraction = {
        "frame_boundary": "first CSV sample; no autonomous synchronization",
        "message_seconds": protocol.FRAME_BITS * protocol.BIT_SECONDS,
        "post_frame_lead_seconds": POST_FRAME_LEAD_SECONDS,
        "post_frame_noise_seconds": POST_FRAME_NOISE_SECONDS,
        "analytic_padding_seconds": ANALYTIC_PADDING_SECONDS,
        "sample_rate_hz": NOMINAL_SAMPLE_RATE_HZ,
        "sample_interval_tolerance_seconds": SAMPLE_INTERVAL_TOLERANCE_SECONDS,
        "capture_duration_tolerance_seconds": CAPTURE_DURATION_TOLERANCE_SECONDS,
        "exact_samples_from_frame_start": round(REQUIRED_SECONDS * NOMINAL_SAMPLE_RATE_HZ),
        "normalization": "per-file declared trailing off-noise mean and RMS",
    }
    if extraction != expected_extraction:
        raise ValueError("reference context extraction contract is incompatible")
    if metadata.get("software") != extraction_software_contract():
        raise ValueError("reference context extraction software is incompatible")
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    allowed_pairs = set(zip(
        REFERENCE_CONTEXT_SEQUENCES, REFERENCE_CONTEXT_REPETITIONS
    ))
    if (any(row["role"] != "historical" or row["section"] != "body"
            or float(row["duty_percent"]) != 100.0 for row in rows)
            or {(int(row["sequence"]), int(row["repetition"])) for row in rows}
            != allowed_pairs):
        raise ValueError("reference context row identities are incompatible")
    build_prompt(rows, [
        {
            "section": "sync" if index < QUERY_SYNC_ROWS else "body",
            "target_bit": "", "frame_bit_index": index,
        }
        for index in range(protocol.FRAME_BITS)
    ])
    return rows, metadata
