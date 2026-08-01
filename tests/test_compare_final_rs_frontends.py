"""Tests for frozen RS GMD and Gao/Duong final-run comparisons."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path[:0] = [
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]
import analyze_final_experiment as base  # noqa: E402
import compare_final_rs_frontends as comparison  # noqa: E402
import final_experiment_protocol as protocol  # noqa: E402


class GenericGMDTests(unittest.TestCase):
    def test_rs12_gmd_uses_reliability_erasures_to_recover_four_errors(self):
        clean = protocol.rs_encode_symbols((17, 14, 2, 10, 14, 31), 6)
        received = list(clean)
        for offset, position in enumerate((0, 1, 2, 3), 1):
            received[position] ^= offset + 2
        received_bits = protocol.symbols_to_bits(received)
        llrs = [8.0 if bit else -8.0 for bit in received_bits]
        for symbol in (0, 1):
            for bit in range(5 * symbol, 5 * symbol + 5):
                llrs[bit] = 0.02 if received_bits[bit] else -0.02
        with self.assertRaises(comparison.GMDDecodeError):
            comparison.hard_decode(llrs, "rs12_6")
        result = comparison.gmd_decode(llrs, "rs12_6")
        self.assertEqual(result.codeword_symbols, clean)
        self.assertEqual(result.payload_bits, tuple(map(int, protocol.FINAL_PAYLOAD_TEXT)))
        self.assertEqual(result.erasures, (0, 1))
        self.assertEqual(result.attempts, 7)

    def test_rs5_padding_is_filtered_before_candidate_ranking(self):
        noncanonical = protocol.rs_encode_symbols((17, 14, 1), 2)
        bits = protocol.symbols_to_bits(noncanonical)
        llrs = tuple(5.0 if bit else -5.0 for bit in bits)
        with self.assertRaisesRegex(comparison.GMDDecodeError, "application-valid"):
            comparison.gmd_decode(llrs, "rs5_3")

    def test_errors_and_erasures_decoder_hits_both_code_bounds(self):
        cases = [
            ((17, 14, 0), 2, (), (0, 1)),
            ((17, 14, 2, 10, 14, 31), 6, (0, 3), (1, 7)),
        ]
        for data, parity, errors, erasures in cases:
            clean = protocol.rs_encode_symbols(data, parity)
            damaged = list(clean)
            for index, position in enumerate(errors + erasures, 1):
                damaged[position] ^= index + 3
            corrected = comparison.rs_decode_erasures(
                damaged, len(data), parity, erasures
            )
            self.assertEqual(corrected, clean)

    def test_zero_llr_is_hard_zero_and_truth_is_not_an_argument(self):
        self.assertNotIn("expected", comparison.gmd_decode.__code__.co_varnames)
        clean = protocol.rs_encode_symbols((17, 14, 0), 2)
        bits = protocol.symbols_to_bits(clean)
        llrs = [4.0 if bit else -4.0 for bit in bits]
        for index, bit in enumerate(bits):
            if bit == 0:
                llrs[index] = 0.0
                break
        result = comparison.gmd_decode(llrs, "rs5_3")
        self.assertEqual(result.codeword_symbols, clean)


class FrontendTests(unittest.TestCase):
    def test_unwhitened_llrs_use_identity_covariance_channel_combining(self):
        fs, frame_bits = 200.0, 41
        samples = round((frame_bits * 0.5 + 12.5) * fs)
        rng = np.random.default_rng(121)
        channels = tuple(
            rng.normal(size=samples) + 1j * rng.normal(size=samples)
            for _ in range(2)
        )
        actual = comparison.unwhitened_llrs(channels, frame_bits, fs)
        half = round(fs * protocol.BIT_SECONDS / 2)
        phasors = comparison._extract_phasors(channels, 0, 2 * frame_bits, half, fs)
        sync_gate = base.manchester_levels(protocol.SYNC_TEXT).astype(bool)
        sync = phasors[:len(sync_gate)]
        channel = sync[sync_gate].mean(axis=0) - sync[~sync_gate].mean(axis=0)
        expected = 2 * np.real(
            (phasors @ channel.conj())[0::2] - (phasors @ channel.conj())[1::2]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_complex_covariance_uses_e_z_z_hermitian_not_its_transpose(self):
        fs, frame_bits = 200.0, 41
        samples = round((frame_bits * 0.5 + 12.5) * fs)
        rng = np.random.default_rng(12)
        latent = rng.normal(size=samples) + 1j * rng.normal(size=samples)
        channels = (
            latent + 0.2 * (rng.normal(size=samples) + 1j * rng.normal(size=samples)),
            (0.3 + 0.8j) * latent
            + 0.2 * (rng.normal(size=samples) + 1j * rng.normal(size=samples)),
        )
        actual = comparison.coherent_llrs(channels, frame_bits, fs)

        half = round(fs * protocol.BIT_SECONDS / 2)
        phasors = comparison._extract_phasors(channels, 0, 2 * frame_bits, half, fs)
        sync_gate = base.manchester_levels(protocol.SYNC_TEXT).astype(bool)
        sync = phasors[:len(sync_gate)]
        channel = sync[sync_gate].mean(axis=0) - sync[~sync_gate].mean(axis=0)
        noise_start = 2 * frame_bits * half + round(base.CENTRAL_GAP_LEAD_SECONDS * fs)
        count = round(base.CENTRAL_GAP_SECONDS * fs / half)
        noise = comparison._extract_phasors(channels, noise_start, count, half, fs)
        centered = noise - noise.mean(axis=0, keepdims=True)
        ridge = max(float(np.trace(centered.T @ centered.conj() / len(centered)).real) / 2,
                    1e-12) * 1e-6

        correct_covariance = centered.T @ centered.conj() / len(centered) + ridge * np.eye(2)
        correct_weights = np.linalg.solve(correct_covariance, channel)
        projected = phasors @ correct_weights.conj()
        expected = 2 * np.real(projected[0::2] - projected[1::2])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

        transposed = centered.conj().T @ centered / len(centered) + ridge * np.eye(2)
        wrong_weights = np.linalg.solve(transposed, channel)
        wrong_projected = phasors @ wrong_weights.conj()
        wrong = 2 * np.real(wrong_projected[0::2] - wrong_projected[1::2])
        self.assertGreater(float(np.max(np.abs(actual - wrong))), 1e-3)

    def test_gao_orientation_and_frozen_transfer(self):
        rng = np.random.default_rng(44)
        reference_x = rng.normal(size=4000) + 1j * rng.normal(size=4000)
        transfer = 1.4 * np.exp(0.3j)
        primary_y = transfer * reference_x + 0.1 * (
            rng.normal(size=4000) + 1j * rng.normal(size=4000)
        )
        model = comparison.fit_gao_off(primary_y, reference_x)
        fitted = model.transfer
        self.assertAlmostEqual(fitted.real, transfer.real, places=2)
        self.assertAlmostEqual(fitted.imag, transfer.imag, places=2)
        model.transform(30 * primary_y, -reference_x)
        self.assertEqual(model.transfer, fitted)
        self.assertGreater(model.noise_reduction_db, 15)

    def test_all_hard_and_gmd_frontends_are_frozen_before_scoring(self):
        self.assertEqual(
            comparison.ALGORITHMS,
            (
                "unwhitened_hard", "coherent_hard", "coherent_gmd",
                "gao_gmd", "duong_gmd", "gao_duong_gmd",
            ),
        )
        fs, frame_bits = 200.0, 41
        samples = round((frame_bits * 0.5 + 12.5) * fs)
        rng = np.random.default_rng(90)
        x = rng.normal(size=samples) + 1j * rng.normal(size=samples)
        y = 0.6 * x + rng.normal(size=samples) + 1j * rng.normal(size=samples)
        transformed, diagnostics = comparison._frontends((x, y),
                                                           round(frame_bits * 0.5 * fs), fs)
        self.assertEqual(tuple(transformed), comparison.ALGORITHMS)
        self.assertEqual(diagnostics["gao_orientation"],
                         "sensor_y_primary_sensor_x_reference")
        self.assertEqual(len(transformed["gao_gmd"]), 1)
        self.assertEqual(len(transformed["duong_gmd"]), 2)
        self.assertTrue(np.isfinite(diagnostics["duong_training_error"]))
        self.assertIn(diagnostics["duong_converged_before_iteration_limit"], (0, 1))
        self.assertEqual(
            diagnostics["gao_duong_converged_before_iteration_limit"], 1
        )


class AggregateTests(unittest.TestCase):
    def test_failures_remain_in_frame_error_denominator(self):
        rows = []
        for algorithm in comparison.ALGORITHMS:
            rows.append({
                "algorithm": algorithm, "phase": "final", "scheme": "rs12_6",
                "sequence": 1, "duty_percent": 25.0,
                "frame_error": 1, "decoder_failure": 1,
                "wrong_codeword_miscorrection": 0,
                "payload_bit_errors_conditional_on_decode": None,
                "payload_bits_evaluated": 0, "gmd_erasure_count": 0,
                "corrected_symbol_count": 0, "gmd_attempts": 1,
                "gmd_margin": None, "gao_off_coherence": 0.25,
                "gao_noise_reduction_db": 1.0,
                "gao_sync_reference_leakage_ratio": 0.5,
                "gao_sync_desired_attenuation_db": 2.0,
                "duong_converged_before_iteration_limit": 1,
                "gao_duong_converged_before_iteration_limit": 1,
            })
        result = comparison.aggregate(rows, {"rs_frames": 1})
        for algorithm in comparison.ALGORITHMS:
            self.assertEqual(result["by_algorithm"][algorithm]["frame_errors"], 1)
            self.assertEqual(result["by_algorithm"][algorithm]["frames"], 1)
            self.assertEqual(result["by_algorithm"][algorithm]["decoder_failures"], 1)


if __name__ == "__main__":
    unittest.main()
