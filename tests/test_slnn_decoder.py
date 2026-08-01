"""Tests for the codebook-weight optimal SLNN decoder."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "receiver" / "legacy_decoder"))

import coded_protocol as protocol  # noqa: E402
import layered_decoder  # noqa: E402
import slnn_decoder  # noqa: E402


class SLNNTests(unittest.TestCase):
    def test_codebook_shapes(self):
        self.assertEqual(slnn_decoder.alphabet_codebook().shape, (26, 28))
        values, codewords = slnn_decoder.full_linear_codebook()
        self.assertEqual(values.shape, (65536,))
        self.assertEqual(codewords.shape, (65536, 28))

    def test_ideal_soft_symbols_decode_in_both_scopes(self):
        target = protocol.encode_message("Q")
        received = 2.0 * target - 1.0
        restricted = slnn_decoder.decode_alphabet(received, "Q")
        full = slnn_decoder.decode_full(received, (0x7E << 8) | ord("Q"))
        for result in (restricted, full):
            self.assertEqual(result.header, 0x7E)
            self.assertEqual(result.letter, "Q")
            self.assertEqual(result.expected_rank, 1)
            self.assertGreater(result.margin, 0)

    def test_clean_manchester_capture_decodes(self):
        t, x, y = layered_decoder.synthesize_capture("M", noise_std=0.02)
        fs = layered_decoder.sample_rate(t)
        channels = layered_decoder.analytic_channels(x, y, fs)
        start = int(np.argmin(np.abs(t - 2.0)))
        r0, r1 = layered_decoder.matched_observations(channels, start, fs)
        received = slnn_decoder.soft_symbols(r0, r1)
        self.assertEqual(slnn_decoder.decode_alphabet(received).letter, "M")
        full = slnn_decoder.decode_full(received)
        self.assertEqual((full.header, full.letter), (0x7E, "M"))

    def test_coherent_combining_decodes_clean_capture(self):
        t, x, y = layered_decoder.synthesize_capture("D", noise_std=0.04)
        fs = layered_decoder.sample_rate(t)
        channels = layered_decoder.analytic_channels(x, y, fs)
        start = int(np.argmin(np.abs(t - 2.0)))
        coherent = slnn_decoder.coherent_soft_symbols(channels, start, fs)
        result = slnn_decoder.decode_alphabet(coherent.symbols, "D")
        self.assertEqual(result.letter, "D")
        self.assertEqual(result.expected_rank, 1)
        self.assertGreaterEqual(coherent.tone_coherence, 0)
        self.assertLessEqual(coherent.tone_coherence, 1.000001)

    def test_analytical_coherent_llrs_decode_clean_capture(self):
        t, x, y = layered_decoder.synthesize_capture(
            "R", noise_std=0.025, tail=15.0, seed=19
        )
        fs = layered_decoder.sample_rate(t)
        channels = layered_decoder.analytic_channels(x, y, fs)
        start = int(np.argmin(np.abs(t - 2.0)))
        coherent = slnn_decoder.coherent_llrs(channels, start, fs)
        restricted = slnn_decoder.decode_alphabet(coherent.llrs, "R")
        full = slnn_decoder.decode_full(coherent.llrs, (0x7E << 8) | ord("R"))
        self.assertEqual((restricted.letter, restricted.expected_rank), ("R", 1))
        self.assertEqual((full.header, full.letter, full.expected_rank), (0x7E, "R", 1))
        self.assertEqual(coherent.llrs.shape, (protocol.CODED_BITS,))
        self.assertEqual(coherent.weights.shape, (2,))
        self.assertEqual(coherent.noise_covariance.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(coherent.llrs)))

    def test_coherent_llrs_require_two_complete_channels(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            slnn_decoder.coherent_llrs([np.zeros(20)], 0, 200.0)
        with self.assertRaisesRegex(ValueError, "complete frame"):
            slnn_decoder.coherent_llrs([np.zeros(20), np.zeros(20)], 0, 200.0)

    def test_soft_symbol_input_shape_is_strict(self):
        with self.assertRaises(ValueError):
            slnn_decoder.soft_symbols(np.zeros(27), np.zeros(27))


if __name__ == "__main__":
    unittest.main()
