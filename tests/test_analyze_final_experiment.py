"""Tests for final-experiment boundary, decoding, SNR, and reporting rules."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path[:0] = [
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]
import analyze_final_experiment as analysis  # noqa: E402
import final_experiment as experiment  # noqa: E402
import final_experiment_protocol as protocol  # noqa: E402


def manifest_row(sequence, trial):
    frame = protocol.FRAMES[trial.scheme]
    payload = (protocol.FINAL_PAYLOAD_TEXT if trial.phase == "final"
               else protocol.SHORT_PAYLOAD_TEXT)
    return {
        "sequence": str(sequence), "phase": trial.phase, "scheme": trial.scheme,
        "repetition": str(trial.repetition), "duty_percent": str(trial.duty_percent),
        "duty_label": f"commanded {trial.duty_percent:g}%", "voltage_v": "11",
        "distance_m": "3", "hardware_note": "bench", "payload_bits": payload,
        "coded_body_bits": frame[16:], "frame_bits": frame, "bit_seconds": "0.5",
        "started_utc": "2026-01-01T00:00:05Z",
        "finished_utc": "2026-01-01T00:00:06Z", "duration_s": str(len(frame) * 0.5),
        "gap_after_s": "15", "target_pulse_us": "62500", "pulse_count": "0",
        "late_starts": "0", "pulse_min_us": "nan", "pulse_median_us": "nan",
        "pulse_p95_us": "nan", "pulse_max_us": "nan",
    }


class DecoderTests(unittest.TestCase):
    def test_hamming_corrects_every_single_body_error(self):
        clean = tuple(map(int, protocol.FRAMES["hamming15_11"][16:]))
        expected = tuple(map(int, protocol.SHORT_PAYLOAD_TEXT))
        for index in range(15):
            damaged = list(clean)
            damaged[index] ^= 1
            payload, syndrome, corrected = analysis.hamming_decode(damaged)
            self.assertEqual(payload, expected)
            self.assertEqual((syndrome, corrected), (index + 1, index + 1))

    def test_generic_rs_corrects_to_each_codes_capability(self):
        cases = [
            ((17, 14, 0), 2, (0,)),
            ((17, 14, 2, 10, 14, 31), 6, (0, 4, 11)),
        ]
        for data, parity, positions in cases:
            with self.subTest(parity=parity):
                clean = protocol.rs_encode_symbols(data, parity)
                damaged = list(clean)
                for offset, position in enumerate(positions):
                    damaged[position] ^= offset + 3
                corrected, count = analysis.rs_decode_symbols(damaged, len(data), parity)
                self.assertEqual(corrected, clean)
                self.assertEqual(count, len(positions))

    def test_rs5_application_padding_must_remain_zero(self):
        noncanonical = protocol.rs_encode_symbols((17, 14, 1), 2)
        decoded = analysis.decode_body("rs5_3", protocol.symbols_to_bits(noncanonical))
        self.assertEqual(decoded["failure"], 1)
        self.assertEqual(decoded["application_padding_valid"], 0)

    def test_zero_llr_tie_is_deterministically_zero(self):
        self.assertEqual(analysis.hard_bits((-1.0, -0.0, 0.0, 1.0)), (0, 0, 0, 1))

    def test_raw_systematic_payload_is_separate_from_decoder(self):
        hamming = tuple(map(int, protocol.FRAMES["hamming15_11"][16:]))
        raw = analysis.raw_payload_bits("hamming15_11", hamming)
        self.assertEqual(raw, tuple(map(int, protocol.SHORT_PAYLOAD_TEXT)))
        rs = tuple(map(int, protocol.FRAMES["rs5_3"][16:]))
        self.assertEqual(analysis.raw_payload_bits("rs5_3", rs),
                         tuple(map(int, protocol.SHORT_PAYLOAD_TEXT)))


class ValidationAndMetricTests(unittest.TestCase):
    def test_manifest_requires_exact_frozen_one_hundred_rows(self):
        rows = [manifest_row(index, trial) for index, trial in enumerate(
            experiment.experiment_schedule("all"), 1
        )]
        # Give timestamps increasing values; validation only requires valid fields.
        analysis.validate_manifest(rows)
        rows[88]["frame_bits"] = rows[88]["frame_bits"][:-1] + "0"
        with self.assertRaisesRegex(ValueError, "golden vectors"):
            analysis.validate_manifest(rows)
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            analysis.validate_manifest(rows[:99])

    def test_power_subtracted_snr_never_invents_a_floor(self):
        self.assertAlmostEqual(analysis.signal_snr_db(11, 1), 10)
        self.assertIsNone(analysis.signal_snr_db(1, 1))
        self.assertIsNone(analysis.signal_snr_db(0.5, 1))
        self.assertIsNone(analysis.signal_snr_db(1, 0))

    def test_complex_covariance_uses_z_times_conjugate_transpose(self):
        samples = np.array([
            [1 + 2j, 3 - 1j],
            [-2 + 0.5j, 1 + 4j],
            [0.25 - 3j, -1 + 0.75j],
        ])
        centered = samples - samples.mean(axis=0, keepdims=True)
        expected = centered.T @ centered.conj() / len(centered)
        transposed_convention = centered.conj().T @ centered / len(centered)
        actual = analysis.complex_covariance(samples)
        np.testing.assert_allclose(actual, expected)
        self.assertFalse(np.allclose(actual, transposed_convention))
        np.testing.assert_allclose(actual, actual.conj().T)

    def test_central_gap_covariance_uses_forty_half_symbols(self):
        fs = 200.0
        half = round(fs * protocol.BIT_SECONDS / 2)
        self.assertEqual(round(analysis.CENTRAL_GAP_SECONDS * fs / half), 40)

    def test_aggregate_labels_conditional_ber_and_failure_bounds(self):
        base = {
            "phase": "short", "scheme": "rs5_3", "duty_percent": 10.0,
            "frame_error": 1, "decoded_payload_frame_error": 1,
            "decoder_failure": 1, "sync_bit_errors": 0, "coded_body_bit_errors": 4,
            "raw_payload_bit_errors": 2, "raw_payload_frame_error": 1,
            "coded_body_frame_error": 1, "frame_bits": 41,
            "decoded_payload_bit_errors_conditional_on_decode": None,
            "decoded_payload_bits_evaluated": 0, "decoded_payload_bits": "",
            "payload_bits_per_frame": 11,
            "raw_symbol_errors": 1, "positive_inband_power_estimate": 0,
            "inband_snr_db_pooled": None, "clipped_any_percent": 0.0,
            "duration_error_s": 0.0, "late_starts": 0,
        }
        result = analysis.aggregate([base], {})
        self.assertEqual(result["decoder_failures"], 1)
        self.assertEqual(result["decoded_payload_bit_errors_lower_bound_including_failures"], 0)
        self.assertEqual(result["decoded_payload_bit_errors_upper_bound_including_failures"], 11)
        self.assertEqual(result["no_positive_inband_power_frames"], 1)


class SyntheticEndToEndTests(unittest.TestCase):
    def test_single_manifest_boundary_frame_decodes_without_autonomous_search(self):
        fs = 200.0
        frame = protocol.FRAMES["uncoded"]
        lead, tail = 5.0, 15.0
        half = round(fs * protocol.BIT_SECONDS / 2)
        gate = np.r_[
            np.zeros(round(lead * fs)),
            np.repeat(analysis.manchester_levels(frame), half),
            np.zeros(round(tail * fs)),
        ]
        t = np.arange(len(gate)) / fs
        rng = np.random.default_rng(2707)
        x = 1000 + 500 * gate * np.cos(2 * np.pi * 8 * t) + rng.normal(0, 8, len(t))
        y = 1500 + 350 * gate * np.cos(2 * np.pi * 8 * t + 0.4) + rng.normal(0, 8, len(t))
        trial = experiment.experiment_schedule("all")[0]
        row = manifest_row(1, trial)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture.csv"
            np.savetxt(capture, np.column_stack((t, x, y)), delimiter=",",
                       header="t,x,y", comments="")
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            metadata = root / "metadata.txt"
            metadata.write_text(
                "capture_started_utc=2026-01-01T00:00:00Z\nphase=all\n"
                "expected_frames=100\nvoltage_v=11\ndistance_m=3\n"
                "coded_bits_per_second=2\noutcome=COMPLETE\n"
                "manifest_outcome=VALID\nsafe_shutdown_outcome=VERIFIED_WRITES\n",
                encoding="utf-8",
            )
            rows, details = analysis.analyze(
                capture, manifest, metadata, require_complete_manifest=False
            )
        self.assertEqual(len(rows), 1)
        result = rows[0]
        self.assertEqual(result["boundary_source"],
                         "manifest_timestamps_frozen_first_sync_correction")
        self.assertEqual(result["autonomous_acquisition_evaluated"], 0)
        self.assertEqual(result["decoded_payload_bit_errors_conditional_on_decode"], 0)
        self.assertEqual(result["frame_error"], 0)
        self.assertIsNotNone(result["inband_snr_db_pooled"])
        self.assertAlmostEqual(details["sample_rate_hz"], fs, places=4)


if __name__ == "__main__":
    unittest.main()
