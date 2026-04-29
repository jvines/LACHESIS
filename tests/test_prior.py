"""Tests for prior transforms and log-prior — written BEFORE implementation."""

import numpy as np
import pytest

from lachesis.prior import IsochronePrior, kroupa_imf


class TestIsochronePrior:
    """Test prior for (eep, log_age, feh) parameter space."""

    def test_construct_default(self):
        prior = IsochronePrior(
            eep_range=(1, 808),
            age_range=(5.0, 10.3),
            feh_range=(-4.0, 0.5),
        )
        assert prior is not None

    def test_prior_transform_maps_unit_cube(self):
        """prior_transform maps [0,1]^3 → physical parameter space."""
        prior = IsochronePrior(
            eep_range=(1, 808),
            age_range=(5.0, 10.3),
            feh_range=(-4.0, 0.5),
        )
        # Corners of unit cube
        lo = prior.prior_transform(np.array([0.0, 0.0, 0.0]))
        hi = prior.prior_transform(np.array([1.0, 1.0, 1.0]))

        assert lo[0] == pytest.approx(1.0)     # eep min
        assert hi[0] == pytest.approx(808.0)   # eep max
        assert lo[1] == pytest.approx(5.0)     # age min
        assert hi[1] == pytest.approx(10.3)    # age max
        assert lo[2] == pytest.approx(-4.0)    # feh min
        assert hi[2] == pytest.approx(0.5)     # feh max

    def test_prior_transform_midpoint(self):
        """Midpoint of unit cube → midpoint of parameter space."""
        prior = IsochronePrior(
            eep_range=(0, 1000),
            age_range=(5.0, 10.0),
            feh_range=(-2.0, 0.0),
        )
        mid = prior.prior_transform(np.array([0.5, 0.5, 0.5]))
        assert mid[0] == pytest.approx(500.0)
        assert mid[1] == pytest.approx(7.5)
        assert mid[2] == pytest.approx(-1.0)

    def test_log_prior_in_bounds(self):
        """In-bounds point should give finite log-prior."""
        prior = IsochronePrior(
            eep_range=(1, 808),
            age_range=(5.0, 10.3),
            feh_range=(-4.0, 0.5),
        )
        lnp = prior.log_prior(eep=400.0, log_age=9.0, feh=0.0)
        assert np.isfinite(lnp)

    def test_log_prior_out_of_bounds(self):
        """Out-of-bounds point should give -inf."""
        prior = IsochronePrior(
            eep_range=(1, 808),
            age_range=(5.0, 10.3),
            feh_range=(-4.0, 0.5),
        )
        assert prior.log_prior(eep=9999.0, log_age=9.0, feh=0.0) == -np.inf
        assert prior.log_prior(eep=400.0, log_age=99.0, feh=0.0) == -np.inf
        assert prior.log_prior(eep=400.0, log_age=9.0, feh=5.0) == -np.inf

    def test_gaussian_feh_prior(self):
        """Gaussian [Fe/H] prior: closer to mean should give higher prior."""
        prior = IsochronePrior(
            eep_range=(1, 808),
            age_range=(5.0, 10.3),
            feh_range=(-4.0, 0.5),
            feh_prior=("gaussian", 0.0, 0.1),  # mean=0, sigma=0.1
        )
        lnp_at_mean = prior.log_prior(eep=400.0, log_age=9.0, feh=0.0)
        lnp_off = prior.log_prior(eep=400.0, log_age=9.0, feh=0.3)
        assert lnp_at_mean > lnp_off

    def test_with_dm_deep(self):
        """When dm_deep is provided, it should affect the prior (IMF weighting)."""
        prior = IsochronePrior(
            eep_range=(1, 808),
            age_range=(5.0, 10.3),
            feh_range=(-4.0, 0.5),
        )
        # Without IMF weighting
        lnp1 = prior.log_prior(eep=400.0, log_age=9.0, feh=0.0)
        # With IMF weighting (needs both initial_mass and dm_deep)
        lnp2 = prior.log_prior(
            eep=400.0, log_age=9.0, feh=0.0,
            initial_mass=1.0, dm_deep=0.01,
        )
        # dm_deep > 0 should give a different prior
        assert lnp1 != lnp2

    def test_kroupa_imf_continuity(self):
        """Kroupa IMF must be continuous at the 0.08 and 0.5 Msun breakpoints."""
        eps = 1e-10
        # Continuity at 0.08
        assert kroupa_imf(0.08 - eps) == pytest.approx(kroupa_imf(0.08), rel=1e-6)
        # Continuity at 0.5
        assert kroupa_imf(0.5 - eps) == pytest.approx(kroupa_imf(0.5), rel=1e-6)

    def test_kroupa_imf_positive_and_decreasing(self):
        """Kroupa IMF should be positive and decreasing for M > 0.08."""
        masses = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
        vals = [kroupa_imf(m) for m in masses]
        for v in vals:
            assert v > 0
        # Should be monotonically decreasing above the low-mass turnover
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1]

    def test_kroupa_imf_zero_for_nonpositive(self):
        assert kroupa_imf(0.0) == 0.0
        assert kroupa_imf(-1.0) == 0.0

    def test_kroupa_imf_selectable(self):
        """Kroupa IMF should be selectable via the imf kwarg."""
        prior = IsochronePrior(
            eep_range=(1, 808),
            age_range=(5.0, 10.3),
            feh_range=(-4.0, 0.5),
            imf="kroupa",
        )
        lnp = prior.log_prior(
            eep=400.0, log_age=9.0, feh=0.0,
            initial_mass=1.0, dm_deep=0.01,
        )
        assert np.isfinite(lnp)

    def test_param_names(self):
        prior = IsochronePrior(
            eep_range=(1, 808),
            age_range=(5.0, 10.3),
            feh_range=(-4.0, 0.5),
        )
        assert prior.param_names == ["eep", "log_age", "feh"]
        assert prior.ndim == 3
