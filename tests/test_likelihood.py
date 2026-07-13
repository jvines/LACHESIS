"""Tests for log-likelihood, written BEFORE implementation (TDD)."""

import numpy as np
import pytest

from lachesis.grid.mist import MISTModelGrid
from lachesis.interp import GridInterpolator
from lachesis.likelihood import log_likelihood


@pytest.fixture
def model_interp(full_iso_path):
    mg = MISTModelGrid(full_iso_path.parent)
    return GridInterpolator(mg)


class TestLogLikelihood:
    """Test log-likelihood computation."""

    def test_returns_finite_for_valid_point(self, model_interp):
        """A valid grid point with matching observations should give finite lnL."""
        observed = {"log_Teff": 3.76, "log_g": 4.44}
        uncertainties = {"log_Teff": 0.01, "log_g": 0.1}
        lnl = log_likelihood(
            model_interp, eep=350.0, log_age=9.66, feh=0.0,
            observed=observed, uncertainties=uncertainties,
        )
        assert np.isfinite(lnl)

    def test_returns_neginf_for_oob(self, model_interp):
        """Out-of-bounds parameters should give -inf."""
        observed = {"log_Teff": 3.76}
        uncertainties = {"log_Teff": 0.01}
        lnl = log_likelihood(
            model_interp, eep=9999.0, log_age=9.0, feh=0.0,
            observed=observed, uncertainties=uncertainties,
        )
        assert lnl == -np.inf

    def test_better_match_higher_likelihood(self, model_interp):
        """Closer observations should give higher likelihood."""
        # Get a predicted point
        pred = model_interp(eep=350.0, log_age=9.66, feh=0.0)
        if np.isnan(pred["log_Teff"]):
            pytest.skip("No valid point at this EEP/age")

        uncertainties = {"log_Teff": 0.01, "log_g": 0.1}

        # Perfect match
        obs_perfect = {"log_Teff": pred["log_Teff"], "log_g": pred["log_g"]}
        lnl_perfect = log_likelihood(
            model_interp, eep=350.0, log_age=9.66, feh=0.0,
            observed=obs_perfect, uncertainties=uncertainties,
        )

        # Off by 2 sigma
        obs_off = {
            "log_Teff": pred["log_Teff"] + 0.02,
            "log_g": pred["log_g"] + 0.2,
        }
        lnl_off = log_likelihood(
            model_interp, eep=350.0, log_age=9.66, feh=0.0,
            observed=obs_off, uncertainties=uncertainties,
        )

        assert lnl_perfect > lnl_off

    def test_single_observable(self, model_interp):
        """Should work with just one observable."""
        observed = {"log_Teff": 3.76}
        uncertainties = {"log_Teff": 0.01}
        lnl = log_likelihood(
            model_interp, eep=350.0, log_age=9.66, feh=0.0,
            observed=observed, uncertainties=uncertainties,
        )
        assert np.isfinite(lnl)

    def test_multiple_observables(self, model_interp):
        """Should work with many observables."""
        observed = {
            "log_Teff": 3.76,
            "log_g": 4.44,
            "log_L": 0.0,
            "log_R": 0.0,
        }
        uncertainties = {
            "log_Teff": 0.01,
            "log_g": 0.1,
            "log_L": 0.05,
            "log_R": 0.05,
        }
        lnl = log_likelihood(
            model_interp, eep=350.0, log_age=9.66, feh=0.0,
            observed=observed, uncertainties=uncertainties,
        )
        assert np.isfinite(lnl)

    def test_feh_as_observable(self, model_interp):
        """[Fe/H] can be both a free param and an observable constraint."""
        observed = {"log_Teff": 3.76, "log_g": 4.44}
        uncertainties = {"log_Teff": 0.01, "log_g": 0.1}
        # feh=0.0 with no feh constraint
        lnl1 = log_likelihood(
            model_interp, eep=350.0, log_age=9.66, feh=0.0,
            observed=observed, uncertainties=uncertainties,
        )
        # Add feh as observable, should change the likelihood
        observed2 = {**observed, "feh": 0.0}
        uncertainties2 = {**uncertainties, "feh": 0.1}
        lnl2 = log_likelihood(
            model_interp, eep=350.0, log_age=9.66, feh=0.0,
            observed=observed2, uncertainties=uncertainties2,
        )
        # With perfect feh match, adding feh constraint should make it
        # slightly lower (more terms) but still finite
        assert np.isfinite(lnl2)

    def test_mismatched_keys_raises(self, model_interp):
        """observed and uncertainties must have same keys."""
        with pytest.raises(ValueError):
            log_likelihood(
                model_interp, eep=350.0, log_age=9.66, feh=0.0,
                observed={"log_Teff": 3.76},
                uncertainties={"log_g": 0.1},  # wrong key
            )

    def test_unknown_observable_raises(self, model_interp):
        """Observable not in grid columns should raise."""
        with pytest.raises(ValueError):
            log_likelihood(
                model_interp, eep=350.0, log_age=9.66, feh=0.0,
                observed={"nonexistent_col": 1.0},
                uncertainties={"nonexistent_col": 0.1},
            )
