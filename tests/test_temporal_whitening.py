"""Tests for experimental causal temporal whitening models."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "receiver" / "legacy_decoder"))

import temporal_whitening as tw  # noqa: E402


class TemporalWhiteningTests(unittest.TestCase):
    def test_baseband_round_trip(self):
        fs = 200.0
        time = np.arange(500) / fs
        channels = (
            (1.2 + 0.3j) * np.exp(2j * np.pi * 8 * time),
            (-0.4 + 0.9j) * np.exp(2j * np.pi * 8 * time),
        )
        restored = tw.analytic_channels(tw.baseband_features(channels, fs), fs)
        np.testing.assert_allclose(restored[0], channels[0], atol=1e-12)
        np.testing.assert_allclose(restored[1], channels[1], atol=1e-12)

    def test_var_reduces_predictable_temporal_correlation(self):
        rng = np.random.default_rng(2)
        samples = np.zeros((12_000, 4))
        driving = rng.standard_normal(samples.shape)
        for index in range(1, len(samples)):
            samples[index] = 0.92 * samples[index - 1] + driving[index]
        model = tw.fit_var(samples[:8_000], order=4)
        residual = model.innovations(samples[8_000:])[model.order:]
        original_corr = np.corrcoef(samples[8_001:-1, 0], samples[8_002:, 0])[0, 1]
        residual_corr = np.corrcoef(residual[:-1, 0], residual[1:, 0])[0, 1]
        self.assertGreater(original_corr, 0.8)
        self.assertLess(abs(residual_corr), 0.1)

    def test_kalman_returns_finite_standardized_innovations(self):
        rng = np.random.default_rng(3)
        samples = np.cumsum(rng.normal(size=(2_000, 4)), axis=0)
        model = tw.fit_kalman(samples[:1_500])
        output = model.innovations(samples[1_500:])
        self.assertEqual(output.shape, (500, 4))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_neural_predictors_smoke(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed")
        rng = np.random.default_rng(5)
        samples = rng.standard_normal((1_025, 4)).astype(np.float32)
        for kind in ("gru", "tcn"):
            model = tw.fit_neural(
                samples,
                kind=kind,
                hidden=4,
                epochs=1,
                sequence_length=64,
                batch_size=8,
            )
            output = model.innovations(samples)
            self.assertEqual(output.shape, samples.shape)
            self.assertTrue(np.all(np.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
