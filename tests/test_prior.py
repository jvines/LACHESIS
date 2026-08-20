"""Tests for prior transforms and log-prior, written BEFORE implementation."""

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
        """prior_transform maps [0,1]^3 -> physical parameter space."""
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
        """Midpoint of unit cube -> midpoint of parameter space."""
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


class TestDegenerateFehPrior:
    """A [Fe/H] posterior from an upstream fit that FIXED the metallicity.

    The column is constant, and scipy's singular-covariance guard fires only
    when the covariance evaluates to exactly zero, which depends on the bit
    pattern of the value and on the sample count. So both halves of that coin
    flip have to be covered: -0.5 used to raise LinAlgError, while -0.13 built
    a KDE whose entire mass fell between two nodes of the tabulated CDF and
    silently pinned every draw at feh_lo.
    """

    RANGE = (-2.0, 0.5)

    def _prior(self, feh_prior):
        return IsochronePrior(
            eep_range=(200, 808), age_range=(8.0, 10.14),
            feh_range=self.RANGE, feh_prior=feh_prior,
        )

    @pytest.mark.parametrize("value", [-0.5, -0.13, 0.0, 0.25, 0.12, -1.47])
    def test_constant_samples_become_a_fixed_prior(self, value):
        with pytest.warns(RuntimeWarning, match="FIXED"):
            prior = self._prior(("kde", np.full(4000, value)))
        assert prior._feh_type == "fixed"
        for u in (0.0, 0.02, 0.5, 0.98, 1.0):
            drawn = prior.prior_transform(np.full(prior.ndim, u))[2]
            assert drawn == pytest.approx(value)
            # The old failure mode: every draw came back as the grid floor.
            assert drawn != pytest.approx(self.RANGE[0])

    @pytest.mark.parametrize("sd", [2e-4, 1e-5])
    def test_unresolvably_narrow_posterior_is_not_a_uniform_ramp(self, sd):
        """Below ~1 CDF cell the tabulation used to collapse to two nodes,
        which np.interp turns into a uniform ramp over the lower half of the
        axis: a posterior at -0.13 drew a mean of -1.07 with a min of -2.0."""
        samples = np.random.default_rng(0).normal(-0.13, sd, 4000)
        with pytest.warns(RuntimeWarning):
            prior = self._prior(("kde", samples))
        drawn = np.array([
            prior.prior_transform(np.full(prior.ndim, u))[2]
            for u in np.linspace(0.0, 1.0, 500)
        ])
        assert drawn.mean() == pytest.approx(-0.13, abs=1e-3)
        assert drawn.min() > -0.2

    def test_resolvable_posterior_still_uses_the_kde(self):
        samples = np.random.default_rng(0).normal(-0.13, 1e-3, 4000)
        prior = self._prior(("kde", samples))
        assert prior._feh_type == "kde"
        drawn = np.array([
            prior.prior_transform(np.full(prior.ndim, u))[2]
            for u in np.linspace(0.001, 0.999, 2000)
        ])
        assert drawn.mean() == pytest.approx(-0.13, abs=1e-4)
        assert drawn.std() == pytest.approx(1e-3, rel=0.3)

    def test_normal_posterior_is_unchanged(self):
        """The tabulation now spans the sample support rather than the whole
        prior range. That must not move a realistic posterior."""
        samples = np.random.default_rng(1).normal(-0.13, 0.05, 4000)
        prior = self._prior(("kde", samples))
        assert prior._feh_type == "kde"
        assert np.all(np.isfinite(prior._feh_cdf_y))
        drawn = np.array([
            prior.prior_transform(np.full(prior.ndim, u))[2]
            for u in np.linspace(0.001, 0.999, 5000)
        ])
        assert drawn.mean() == pytest.approx(samples.mean(), abs=2e-3)
        assert drawn.std() == pytest.approx(samples.std(), rel=0.05)

    def test_all_nan_samples_raise(self):
        from lachesis.error import PriorError
        with pytest.raises(PriorError, match="non-finite"):
            self._prior(("kde", np.full(100, np.nan)))

    def test_samples_outside_coverage_warn_before_going_uniform(self):
        """Silently swapping a measured [Fe/H] for a flat prior also changes
        the evidence normalisation, since the BMA common-scale correction adds
        the prior box only for uniform [Fe/H]."""
        samples = np.random.default_rng(2).normal(-1.5, 0.05, 2000)
        with pytest.warns(RuntimeWarning, match="UNIFORM"):
            prior = IsochronePrior(
                eep_range=(200, 808), age_range=(8.0, 10.14),
                feh_range=(-0.33, 0.5), feh_prior=("kde", samples),
            )
        assert prior._feh_type == "uniform"

    def test_explicit_fixed_prior(self):
        prior = self._prior(("fixed", -0.42))
        assert prior._feh_type == "fixed"
        assert prior.prior_transform(np.full(prior.ndim, 0.7))[2] == pytest.approx(-0.42)
        # A delta carries no prior volume, so it must not shift the evidence.
        lnp_fixed = prior.log_prior(400, 9.5, -0.42, initial_mass=1.0, dm_deep=0.01)
        bare = self._prior(None)
        lnp_uniform = bare.log_prior(400, 9.5, -0.42, initial_mass=1.0, dm_deep=0.01)
        assert lnp_fixed == pytest.approx(lnp_uniform + np.log(2.5))

    @pytest.mark.parametrize("sigma", [0.0, np.float64(0.0), -1.0])
    def test_zero_sigma_gaussian_is_a_delta(self, sigma):
        """N(mu, 0) is a delta. It used to raise ZeroDivisionError for a Python
        float and return NaN for a np.float64."""
        with pytest.warns(RuntimeWarning, match="FIXED"):
            prior = self._prior(("gaussian", -0.13, sigma))
        assert prior._feh_type == "fixed"
        for u in (0.0, 0.5, 1.0):
            assert prior.prior_transform(np.full(prior.ndim, u))[2] == pytest.approx(-0.13)

    def test_narrow_gaussian_is_finite_at_the_cube_edges(self):
        """ndtri saturates once the truncation CDF underflows, and dynesty does
        propose u == 0.0."""
        prior = self._prior(("gaussian", -0.13, 1e-9))
        for u in (0.0, 1e-12, 0.5, 1.0):
            drawn = prior.prior_transform(np.full(prior.ndim, u))[2]
            assert np.isfinite(drawn)
            assert drawn == pytest.approx(-0.13, abs=1e-6)

    @pytest.mark.parametrize("sigma", [0.0, np.float64(0.0), -5.0, np.nan])
    def test_zero_sigma_distance_prior_raises(self, sigma):
        from lachesis.error import PriorError
        with pytest.raises(PriorError, match="positive width"):
            IsochronePrior(
                eep_range=(200, 808), age_range=(8.0, 10.14),
                feh_range=self.RANGE, distance_prior=("normal", 45.2, sigma),
            )

    def test_zero_width_boxes_do_not_make_log_prior_infinite(self):
        """A fixed Av is expressed as a zero-width range, for which
        -log(0) is +inf."""
        prior = IsochronePrior(
            eep_range=(200, 808), age_range=(8.0, 10.14),
            feh_range=self.RANGE, av_range=(0.3, 0.3),
        )
        lnp = prior.log_prior(400, 9.5, -0.1, initial_mass=1.0,
                              dm_deep=0.01, av=0.3)
        assert np.isfinite(lnp)

        flat = IsochronePrior(eep_range=(200, 808), age_range=(8.0, 10.14),
                              feh_range=(0.0, 0.0))
        assert np.isfinite(
            flat.log_prior(400, 9.5, 0.0, initial_mass=1.0, dm_deep=0.01))
