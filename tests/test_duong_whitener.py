"""Tests for the Duong gain-modulated whitening implementation."""

from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "receiver" / "legacy_decoder"))

import duong_whitener as dw  # noqa: E402


class DuongWhitenerTests(unittest.TestCase):
    def test_feature_complex_round_trip(self):
        channels = (
            np.array([1 + 2j, 3 + 4j]),
            np.array([5 + 6j, 7 + 8j]),
        )
        restored = dw.complex_channels(dw.feature_matrix(channels))
        np.testing.assert_allclose(restored[0], channels[0])
        np.testing.assert_allclose(restored[1], channels[1])

    def test_gain_training_whitens_correlated_gaussian_samples(self):
        rng = np.random.default_rng(4)
        mixing = np.array([
            [3.0, 0.0, 0.0, 0.0],
            [0.8, 2.0, 0.0, 0.0],
            [0.4, -0.3, 1.5, 0.0],
            [0.2, 0.1, 0.4, 1.0],
        ])
        samples = rng.standard_normal((20_000, 4)) @ mixing.T
        model = dw.fit(samples, tolerance=5e-5)
        covariance = np.cov(model.transform(samples), rowvar=False)
        np.testing.assert_allclose(covariance, np.eye(4), atol=5e-3)
        self.assertLess(model.training_error, 3e-3)

    def test_model_save_and_load(self):
        rng = np.random.default_rng(3)
        samples = rng.standard_normal((5_000, 2)) @ np.array([[2.0, 0.0], [0.5, 1.0]])
        model = dw.fit(samples, tolerance=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = dw.DuongWhitener.load(path)
        np.testing.assert_allclose(loaded.transform(samples), model.transform(samples))
        self.assertEqual(loaded.iterations, model.iterations)

    def test_rejects_nonfinite_training_data(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            dw.fit(np.array([[0.0, 1.0], [np.nan, 2.0]]))


if __name__ == "__main__":
    unittest.main()
