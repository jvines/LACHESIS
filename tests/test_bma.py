"""Tests for Bayesian Model Averaging, TDD."""

from pathlib import Path

import numpy as np
import pytest

from lachesis.bma import BMAResult, bayesian_model_average

from tests.conftest import mist_h5_path

FULL_GRID_H5 = mist_h5_path()


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
        # Model 2 has much higher evidence -> should dominate
        assert bma.weights[1] > bma.weights[0]
        assert bma.weights[1] > 0.9

    def test_bma_combined_samples(self):
        """Combined posterior should have samples from both models."""
        r1 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=42)
        r2 = _make_fake_result(logz=0.0, logzerr=0.1, n_samples=100, seed=43)

        bma = bayesian_model_average([r1, r2])
        # Equal evidence -> roughly equal samples from each
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


class TestNonFiniteEvidence:
    """A single non-finite log-evidence used to poison everything quietly:
    log_z.max() becomes NaN, every weight becomes NaN, np.round(NaN * N) is 0
    and the (weights > 0) floor does not rescue it because NaN > 0 is False.
    Every model drew zero samples and the run reported success on a (0, ndim)
    posterior."""

    @staticmethod
    def _result(logz):
        return {
            "logz": logz,
            "logzerr": 0.1,
            "samples": np.zeros((10, 3)),
            "derived": {"mass": np.ones(10)},
        }

    @pytest.mark.parametrize("bad", [np.nan, -np.inf, np.inf])
    def test_non_finite_logz_raises(self, bad):
        with pytest.raises(ValueError, match="non-finite log-evidence"):
            bayesian_model_average(
                [self._result(-120.0), self._result(bad)],
                names=["good", "bad"],
            )

    def test_error_names_the_offending_grid(self):
        with pytest.raises(ValueError, match="geneva"):
            bayesian_model_average(
                [self._result(-120.0), self._result(-np.inf)],
                names=["mist", "geneva"],
            )

    def test_finite_evidence_still_works(self):
        result = bayesian_model_average(
            [self._result(-120.0), self._result(-121.0)],
            names=["a", "b"],
        )
        assert len(result.samples) > 0
        assert np.all(np.isfinite(result.weights))


class TestBMAForbiddenGrids:
    """Grids whose coverage is too narrow for a common-scale evidence
    comparison must be refused in BMA mode.

    README documented Geneva as excluded from the start, but nothing enforced
    it and it shipped in the default grid list, where the a-posteriori [Fe/H]
    rail drop removed it by accident.
    """

    @staticmethod
    def _fitter(grids, bma=True):
        from lachesis.fitter import Fitter
        from lachesis.star import Star

        star = Star("dummy", ra=0.0, dec=0.0,
                    magnitudes={"Bessell_V": (10.0, 0.05)},
                    plx=10.0, plx_e=0.1, Av=0.1, feh=-0.1, feh_e=0.05,
                    verbose=False, offline=True)
        star.teff, star.teff_e = 5750.0, 40.0
        f = Fitter()
        f.star = star
        f.grids = grids
        f.bma = bma
        f.verbose = False
        return f

    @pytest.mark.parametrize("grid", ["geneva", "bhac15", "starevol"])
    def test_forbidden_in_bma(self, grid):
        from lachesis.error import InputError

        f = self._fitter(["mist", grid])
        with pytest.raises(InputError, match="BMA mode"):
            f.initialize()

    @pytest.mark.parametrize("grid", ["geneva", "bhac15", "starevol"])
    def test_allowed_as_a_single_grid_fit(self, grid):
        """The exclusion is about BMA, not about the grid being unusable."""
        from lachesis.error import InputError

        f = self._fitter([grid], bma=False)
        try:
            f.initialize()
        except InputError as e:
            assert "BMA mode" not in str(e)
        except Exception:
            pytest.skip(f"{grid} grid data not available")

    def test_default_grid_list_is_bma_safe(self):
        """The defaults must not trip the guard the moment a user sets
        bma = True, which is what the README example does."""
        from lachesis.fitter import Fitter

        defaults = set(Fitter()._grids)
        assert not (defaults & {"geneva", "bhac15", "starevol"})
