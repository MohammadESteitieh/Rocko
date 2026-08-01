"""Tests for coherent-GNB-SLNN inference."""

from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "receiver" / "legacy_decoder"))

import hybrid_decoder  # noqa: E402
import layered_decoder  # noqa: E402


class GaussianNaiveBayesTests(unittest.TestCase):
    def test_fit_likelihood_and_round_trip(self):
        x = np.array([[0.1, .9], [.2, .8], [.9, .1], [.8, .2]])
        y = np.array([0, 0, 1, 1])
        model = hybrid_decoder.GaussianNaiveBayes.fit(x, y)
        self.assertTrue(np.all(model.llrs(x[:2]) < 0))
        self.assertTrue(np.all(model.llrs(x[2:]) > 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = hybrid_decoder.GaussianNaiveBayes.load(path)
            np.testing.assert_allclose(loaded.llrs(x), model.llrs(x))


class HybridTests(unittest.TestCase):
    def test_clean_capture_is_accepted(self):
        t, x, y = layered_decoder.synthesize_capture("R", noise_std=.04)
        fs = layered_decoder.sample_rate(t)
        channels = layered_decoder.analytic_channels(x, y, fs)
        start = int(np.argmin(np.abs(t - 2.0)))
        result = hybrid_decoder.decode(channels, start, fs, expected_letter="R")
        self.assertTrue(result.accepted)
        self.assertEqual(result.restricted.letter, "R")
        self.assertEqual((result.full.header, result.full.letter), (0x7E, "R"))
        self.assertEqual(result.restricted.expected_rank, 1)

    def test_synthetic_model_has_two_manchester_features(self):
        model = hybrid_decoder.synthetic_gnb()
        self.assertEqual(model.means.shape, (2, 2))
        self.assertGreater(model.means[1, 0], model.means[0, 0])
        self.assertLess(model.means[1, 1], model.means[0, 1])


if __name__ == "__main__":
    unittest.main()
