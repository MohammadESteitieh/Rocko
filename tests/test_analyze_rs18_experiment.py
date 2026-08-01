"""Tests for accepted-run RS18 boundaries, decoding, normalization, and plots."""

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
import analyze_rs18_experiment as analysis  # noqa: E402
import final_experiment_protocol as gf  # noqa: E402
import plot_rs18_results as plotting  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402


class DecoderTests(unittest.TestCase):
    def _llrs(self, symbols):
        bits = gf.symbols_to_bits(symbols)
        return tuple(8.0 if bit else -8.0 for bit in bits)

    def test_hard_rs18_corrects_six_symbol_errors(self):
        clean = protocol.CODEWORDS[3]
        damaged = list(clean)
        for offset, position in enumerate((0, 2, 5, 8, 12, 17), 1):
            damaged[position] ^= offset + 2
        result = analysis.hard_rs18_decode(self._llrs(damaged))
        self.assertEqual(result["failure"], 0)
        self.assertEqual(result["payload"], tuple(map(int, protocol.PAYLOADS[3])))
        self.assertEqual(result["corrected_symbol_count"], 6)

    def test_exploratory_gmd_clean_decode_is_truth_free(self):
        self.assertNotIn("expected", analysis.exploratory_gmd_rs18_decode.__code__.co_varnames)
        result = analysis.exploratory_gmd_rs18_decode(
            self._llrs(protocol.CODEWORDS[2])
        )
        self.assertEqual(result["failure"], 0)
        self.assertEqual(result["payload"], tuple(map(int, protocol.PAYLOADS[2])))

    def test_each_varied_payload_decodes_without_truth_argument(self):
        self.assertNotIn("expected", analysis.hard_rs18_decode.__code__.co_varnames)
        for repetition in range(1, 6):
            result = analysis.hard_rs18_decode(
                self._llrs(protocol.CODEWORDS[repetition])
            )
            self.assertEqual(result["failure"], 0)
            self.assertEqual(
                result["payload"], tuple(map(int, protocol.PAYLOADS[repetition]))
            )


class BoundaryAndNormalizationTests(unittest.TestCase):
    def test_raw_coherent_sync_scan_recovers_declared_negative_75ms_shift(self):
        fs = 200.0
        actual_start = 1000
        manifest_prediction = actual_start + 15
        half = round(fs * protocol.BIT_SECONDS / 2)
        frame = protocol.FRAMES[1]
        gate = np.r_[
            np.zeros(actual_start),
            np.repeat(analysis.base.manchester_levels(frame), half),
            np.zeros(round(15 * fs)),
        ]
        time_axis = np.arange(len(gate)) / fs
        carrier = np.exp(2j * np.pi * 8 * time_axis)
        channels = (2.0 * gate * carrier, (1.2 + 0.4j) * gate * carrier)
        row = {"started_utc": "2026-01-01T00:00:05.075Z", "duty_percent": "100"}
        starts, correction, score, errors = analysis.locate_raw_coherent_sync_boundaries(
            [row], analysis.base.utc_seconds("2026-01-01T00:00:00Z"), channels, fs
        )
        self.assertEqual(starts, [actual_start])
        self.assertAlmostEqual(correction, -0.075)
        self.assertEqual(errors, 0)
        self.assertGreater(score, 0.99)

    def test_prelaunch_indices_and_rms_use_only_declared_block(self):
        fs = 200.0
        info = {
            "capture_started_utc": "2026-01-01T00:00:00Z",
            "prelaunch_off_started_utc": "2026-01-01T00:00:00Z",
            "prelaunch_off_completed_utc": "2026-01-01T00:02:00.005Z",
        }
        start, stop = analysis.prelaunch_indices(info, fs, 30_000)
        self.assertEqual(start, 0)
        self.assertGreaterEqual(stop - start, 24_000)
        rng = np.random.default_rng(4)
        x = np.r_[rng.normal(10, 2, stop), np.full(1000, 10000)]
        y = np.r_[rng.normal(20, 3, stop), np.full(1000, -10000)]
        means, rms = analysis.fit_off_rms(x, y, start, stop)
        self.assertAlmostEqual(means[0], 10, delta=0.1)
        self.assertAlmostEqual(means[1], 20, delta=0.1)
        self.assertAlmostEqual(rms[0], 2, delta=0.1)
        self.assertAlmostEqual(rms[1], 3, delta=0.1)

    def test_carrier_vs_wide_noise_requires_positive_carrier_numerator(self):
        self.assertAlmostEqual(
            analysis.carrier_vs_wide_noise_snr_db(11, 1, 1), 10
        )
        self.assertIsNone(analysis.carrier_vs_wide_noise_snr_db(1, 1, 1))
        self.assertIsNone(analysis.carrier_vs_wide_noise_snr_db(0.5, 1, 1))
        self.assertIsNone(analysis.carrier_vs_wide_noise_snr_db(2, 1, 0))


class AggregateAndPlotTests(unittest.TestCase):
    def _row(self, duty, error=0, normalized_error=0, positive=True):
        return {
            "duty_percent": float(duty), "frame_error": error,
            "normalized_frame_error": normalized_error,
            "decoder_failure": error, "normalized_decoder_failure": normalized_error,
            "exploratory_gmd_frame_error": error,
            "exploratory_gmd_decoder_failure": 0,
            "exploratory_gmd_wrong_codeword_miscorrection": error,
            "raw_body_bit_errors": error, "raw_symbol_errors": error,
            "physical_inband_snr_db_pooled": 5.0 if positive else None,
            "physical_positive_inband_power_estimate": int(positive),
            "physical_carrier_vs_0_10hz_off_noise_snr_db_pooled": (
                2.0 if positive else None
            ),
            "clipped_any_percent": 0.0, "duration_error_s": 0.001,
            "late_starts": 0,
        }

    def test_failures_remain_in_fer_and_plot_has_exact_denominators(self):
        rows = []
        for duty in protocol.DUTIES:
            rows.extend(self._row(duty, error=int(index == 0),
                                  normalized_error=int(index < 2), positive=index != 4)
                        for index in range(5))
        summary = analysis.aggregate(rows, {})
        item = summary["by_duty"]["100"]
        self.assertEqual((item["frame_errors"], item["frames"]), (1, 5))
        self.assertEqual(item["decoder_failures"], 1)
        self.assertEqual(item["normalized_diagnostic_frame_errors"], 2)
        data = plotting.plot_data(summary)
        selected = next(row for row in data if row["duty_percent"] == 100)
        self.assertEqual(selected["hard_frame_errors"], 1)
        self.assertEqual(selected["frames"], 5)
        self.assertEqual(selected["positive_physical_snr_frames"], 4)

    def test_plot_renderer_writes_png_and_svg(self):
        rows = [{
            "duty_percent": duty, "frames": 5, "hard_frame_errors": 1,
            "hard_fer": 0.2, "hard_fer_wilson_lower": 0.03,
            "hard_fer_wilson_upper": 0.62,
            "exploratory_gmd_frame_errors": 1,
            "exploratory_gmd_fer": 0.2,
            "exploratory_gmd_wilson_lower": 0.03,
            "exploratory_gmd_wilson_upper": 0.62,
            "exploratory_gmd_failures": 0,
            "exploratory_gmd_miscorrections": 1,
            "off_rms_normalized_diagnostic_frame_errors": 2,
            "off_rms_normalized_diagnostic_fer": 0.4,
            "off_rms_normalized_diagnostic_wilson_lower": 0.12,
            "off_rms_normalized_diagnostic_wilson_upper": 0.77,
            "positive_physical_snr_frames": 5,
            "no_positive_physical_power_frames": 0,
            "conditional_median_positive_physical_inband_snr_db": 3.0,
            "conditional_median_carrier_vs_0_10hz_off_noise_snr_db": 1.0,
        } for duty in (1.0, 25.0, 100.0)]
        with tempfile.TemporaryDirectory() as directory:
            png, svg = Path(directory) / "plot.png", Path(directory) / "plot.svg"
            plotting.render(rows, png, svg)
            self.assertGreater(png.stat().st_size, 1000)
            self.assertIn("<svg", svg.read_text(encoding="utf-8")[:500])


if __name__ == "__main__":
    unittest.main()
