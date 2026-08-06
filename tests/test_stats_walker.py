"""Tests for py_tools.stats.walker"""

import numpy as np
import pytest

from py_tools.stats.walker import WalkerRandomSampling


class TestWalkerRandomSampling:
    def test_construction(self):
        sampler = WalkerRandomSampling([1, 2, 3])
        assert sampler.n == 3

    def test_single_sample_in_range(self):
        sampler = WalkerRandomSampling([1, 2, 3])
        k = sampler.random()
        assert 0 <= k < 3

    def test_batch_sample_shape(self):
        sampler = WalkerRandomSampling([1, 2, 3])
        samples = sampler.random(count=50)
        assert samples.shape == (50,)

    def test_with_keys_single(self):
        sampler = WalkerRandomSampling([1, 2, 3], keys=["a", "b", "c"])
        assert sampler.random() in ["a", "b", "c"]

    def test_with_keys_batch(self):
        sampler = WalkerRandomSampling([1, 2, 3], keys=["a", "b", "c"])
        samples = sampler.random(count=20)
        assert all(s in ["a", "b", "c"] for s in samples)

    def test_frequency_proportional_to_weights(self):
        np.random.seed(42)
        sampler = WalkerRandomSampling([1, 4])  # 20% vs 80%
        samples = sampler.random(count=10000)
        np.random.seed(None)
        freq_1 = np.sum(samples == 1) / 10000
        assert np.isclose(freq_1, 0.8, atol=0.02)

    def test_invalid_weights_shape(self):
        with pytest.raises(ValueError):
            WalkerRandomSampling(np.ones((2, 2)))

    def test_accepts_numpy_array(self):
        sampler = WalkerRandomSampling(np.array([1.0, 2.0, 3.0]))
        assert sampler.n == 3
