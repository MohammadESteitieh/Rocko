"""Unit tests for calibration SNR and summary reporting."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "receiver"))

import analyze_calibration as analysis  # noqa: E402
import coded_protocol as protocol  # noqa: E402
import layered_decoder  # noqa: E402


class CalibrationAnalysisTests(unittest.TestCase):
    def test_power_subtracted_snr_requires_positive_signal(self):
        self.assertAlmostEqual(analysis.signal_snr_db(11.0, 1.0), 10.0)
        self.assertIsNone(analysis.signal_snr_db(1.0, 1.0))
        self.assertIsNone(analysis.signal_snr_db(0.5, 1.0))
        self.assertIsNone(analysis.signal_snr_db(1.0, 0.0))

    def test_summary_keeps_missing_positive_snr_explicit(self):
        rows = [
            {
                "duty_percent": 10.0,
                "accepted_decode_only": 0,
                "inband_snr_db_pooled": None,
                "clipped_any_percent": 0.0,
            },
            {
                "duty_percent": 10.0,
                "accepted_decode_only": 1,
                "inband_snr_db_pooled": 3.0,
                "clipped_any_percent": 2.0,
            },
            {
                "duty_percent": 25.0,
                "accepted_decode_only": 1,
                "inband_snr_db_pooled": 5.0,
                "clipped_any_percent": 1.0,
            },
        ]
        result = analysis.summary(rows)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["decode_only_accepted"], 2)
        self.assertEqual(result["by_duty"]["10"]["positive_inband_snr_frames"], 1)
        self.assertEqual(
            result["by_duty"]["10"][
                "conditional_median_positive_inband_snr_db_pooled"
            ],
            3.0,
        )
        self.assertEqual(result["by_duty"]["10"]["max_clipped_any_percent"], 2.0)

    def test_csv_writes_missing_snr_as_empty_field(self):
        rows = [{"sequence": 1, "snr": None, "value": 1.25}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            analysis.write_csv(path, rows)
            text = path.read_text()
            self.assertIn("sequence,snr,value", text)
            self.assertIn("1,,1.250000", text)

    def test_adc_scale_supports_observed_12_bit_and_documented_u16_streams(self):
        self.assertEqual(
            analysis.adc_full_scale(np.array([900, 4095]), np.array([1700])),
            4095,
        )
        self.assertEqual(
            analysis.adc_full_scale(np.array([900, 5000]), np.array([1700])),
            65535,
        )

    def test_partial_manifest_is_rejected_for_physical_analysis(self):
        bits = "".join(map(str, protocol.encode_message("A")))
        row = {
            "sequence": "1", "round": "1", "position": "1", "letter": "A",
            "duty_percent": "100", "voltage_v": "11", "coded_bits": bits,
        }
        with self.assertRaisesRegex(ValueError, "exactly 12"):
            analysis.validate_manifest([row])
        analysis.validate_manifest([row], require_complete=False)

    def test_complete_manifest_must_match_every_frozen_schedule_entry(self):
        bits = "".join(map(str, protocol.encode_message("A")))
        rows = [
            {
                "sequence": str(sequence), "round": str(round_index),
                "position": str(position), "letter": "A",
                "duty_percent": str(duty), "voltage_v": "11",
                "coded_bits": bits,
            }
            for sequence, (round_index, position, duty) in enumerate(
                analysis.EXPECTED_SCHEDULE, 1
            )
        ]
        analysis.validate_manifest(rows)
        rows[7]["duty_percent"] = "25"
        with self.assertRaisesRegex(ValueError, "frozen schedule"):
            analysis.validate_manifest(rows)

    def test_synthetic_complete_frame_is_located_measured_and_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t, x, y = layered_decoder.synthesize_capture(
                "A", lead=15.0, tail=15.0, noise_std=0.03
            )
            capture = root / "capture.csv"
            np.savetxt(
                capture, np.column_stack((t, x, y)), delimiter=",",
                header="t,x,y", comments="",
            )
            manifest = root / "manifest.csv"
            fields = [
                "sequence", "round", "position", "letter", "voltage_v",
                "duty_percent", "coded_bits",
            ]
            with manifest.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "sequence": 1, "round": 1, "position": 1, "letter": "A",
                    "voltage_v": 11, "duty_percent": 100,
                    "coded_bits": "".join(map(str, protocol.encode_message("A"))),
                })
            metadata = root / "metadata.txt"
            metadata.write_text(
                "capture_started_utc=2026-01-01T00:00:00Z\n"
                "transmitter_pid_ack_utc=2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            row = analysis.analyze(
                capture, manifest, metadata, require_complete_manifest=False
            )[0]
            self.assertEqual(row["accepted_decode_only"], 1)
            self.assertEqual(row["decoded_header"], "0x7E")
            self.assertEqual(row["decoded_letter"], "A")
            self.assertIsNotNone(row["broadband_snr_db_pooled"])
            self.assertIsNotNone(row["inband_snr_db_pooled"])


if __name__ == "__main__":
    unittest.main()
