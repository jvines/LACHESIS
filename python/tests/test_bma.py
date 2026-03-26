"""Tests for Bayesian Model Averaging — TDD."""

from pathlib import Path

import numpy as np
import pytest

from lachesis.bma import BMAResult, bayesian_model_average

FULL_GRID_H5 = Path(__file__).parents[2] / "data" / "mist" / "grids" / "mist_v1.2_vvcrit0.4.h5"


class TestBMA:

    def test_bma_with_two_results(self):
        """BMA should combine two fit results via evidence weighting."""
        # Fake two fit results with known evidence
        r1 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=42)
        r2 = _make_fake_result(logz=1.0, logzerr=0.1, n_samples=100, seed=43)

        bma = bayesian_model_average([r1, r2], names=["MIST", "PARSEC"])
        assert isinstance(bma, BMAResult)

    def test_bma_weights_sum_to_one(self):
        r1 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=42)
        r2 = _make_fake_result(logz=1.0, logzerr=0.1, n_samples=100, seed=43)

        bma = bayesian_model_average([r1, r2])
        assert bma.weights.sum() == pytest.approx(1.0)

    def test_higher_evidence_gets_more_weight(self):
        r1 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=42)
        r2 = _make_fake_result(logz=5.0, logzerr=0.1, n_samples=100, seed=43)

        bma = bayesian_model_average([r1, r2], names=["low_Z", "high_Z"])
        # Model 2 has much higher evidence → should dominate
        assert bma.weights[1] > bma.weights[0]
        assert bma.weights[1] > 0.9

    def test_bma_combined_samples(self):
        """Combined posterior should have samples from both models."""
        r1 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=42)
        r2 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=43)

        bma = bayesian_model_average([r1, r2])
        # Equal evidence → roughly equal samples from each
        assert len(bma.samples) > 0
        assert len(bma.samples) <= 200  # at most all samples

    def test_bma_has_model_labels(self):
        r1 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=42)
        r2 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=43)

        bma = bayesian_model_average([r1, r2], names=["MIST", "PARSEC"])
        # Each combined sample should know which model it came from
        assert "model" in bma.derived
        assert set(bma.derived["model"]).issubset({"MIST", "PARSEC"})

    def test_bma_derived_quantities(self):
        r1 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=42)
        r2 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=43)

        bma = bayesian_model_average([r1, r2])
        assert "initial_mass" in bma.derived
        assert "Teff" in bma.derived

    def test_bma_single_model_degenerates(self):
        """BMA with one model should just return that model's results."""
        r1 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=42)
        bma = bayesian_model_average([r1], names=["MIST"])
        assert bma.weights[0] == pytest.approx(1.0)
        assert len(bma.samples) == 100


def _make_fake_result(logz, logzerr, n_samples, seed):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n_samples, 3))  # eep, age, feh
    derived = {
        "initial_mass": rng.uniform(0.5, 2.0, n_samples),
        "Teff": rng.uniform(4000, 7000, n_samples),
        "log_g": rng.uniform(3.5, 5.0, n_samples),
    }
    return {
        "samples": samples,
        "logz": logz,
        "logzerr": logzerr,
        "derived": derived,
    }
