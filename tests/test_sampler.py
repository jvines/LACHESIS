"""Tests for nested sampling integration, TDD."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.mist import MISTModelGrid
from lachesis.interp import GridInterpolator
from lachesis.sampler import IsochroneFitter

from tests.conftest import mist_h5_path

FULL_GRID_H5 = mist_h5_path()


@pytest.fixture(scope="module")
def fitter():
    if FULL_GRID_H5 is None:
        pytest.skip("MIST grid not available")
    mg = MISTModelGrid.from_hdf5(FULL_GRID_H5)
    interp = GridInterpolator(mg)
    return IsochroneFitter(
        interp=interp,
        eep_range=(200, 808),    # ZAMS to TPAGB
        age_range=(8.0, 10.3),   # 100 Myr to 20 Gyr
        feh_range=(-1.0, 0.5),   # reasonable range
    )


class TestIsochroneFitter:

    def test_construct(self, fitter):
        assert fitter is not None

    def test_fit_returns_result(self, fitter):
        """Run a quick fit and get a result dict."""
        observed = {"log_Teff": 3.76, "log_g": 4.44}
        uncertainties = {"log_Teff": 0.01, "log_g": 0.1}
        result = fitter.fit(
            observed=observed,
            uncertainties=uncertainties,
            nlive=50,  # tiny for speed
            dlogz=1.0,  # loose tolerance for speed
        )
        assert "samples" in result
        assert "logz" in result
        assert "logzerr" in result
        assert result["samples"].shape[1] == 3  # eep, log_age, feh

    def test_fit_posterior_in_bounds(self, fitter):
        """Posterior samples should be within prior bounds."""
        observed = {"log_Teff": 3.76, "log_g": 4.44}
        uncertainties = {"log_Teff": 0.01, "log_g": 0.1}
        result = fitter.fit(
            observed=observed,
            uncertainties=uncertainties,
            nlive=50,
            dlogz=1.0,
        )
        samples = result["samples"]
        eep_lo, eep_hi = fitter.prior.eep_lo, fitter.prior.eep_hi
        assert np.all(samples[:, 0] >= eep_lo)
        assert np.all(samples[:, 0] <= eep_hi)
        assert np.all(samples[:, 1] >= 5.0)
        assert np.all(samples[:, 1] <= 10.3)

    def test_fit_derived_quantities(self, fitter):
        """Result should include derived quantities (age, mass, etc.)."""
        observed = {"log_Teff": 3.76, "log_g": 4.44}
        uncertainties = {"log_Teff": 0.01, "log_g": 0.1}
        result = fitter.fit(
            observed=observed,
            uncertainties=uncertainties,
            nlive=50,
            dlogz=1.0,
        )
        assert "derived" in result
        derived = result["derived"]
        for key in ["initial_mass", "Teff", "log_g", "log_L", "radius", "phase"]:
            assert key in derived, f"Missing derived key: {key}"
            assert len(derived[key]) == len(result["samples"])

    def test_evidence_is_finite(self, fitter):
        """Log-evidence should be finite."""
        observed = {"log_Teff": 3.76, "log_g": 4.44}
        uncertainties = {"log_Teff": 0.01, "log_g": 0.1}
        result = fitter.fit(
            observed=observed,
            uncertainties=uncertainties,
            nlive=50,
            dlogz=1.0,
        )
        assert np.isfinite(result["logz"])
        assert np.isfinite(result["logzerr"])

    def test_auto_drop_grids_outside_feh_prior(self):
        """A Gaussian [Fe/H] prior whose ±3σ window is entirely outside a
        grid's [Fe/H] axis must auto-drop that grid before dynesty wastes
        1000 init attempts and raises 'no valid log-likelihood'. The case
        we hit on Gaia_DR3_5413575/8 (halo-like [Fe/H]=−1.47 priors vs
        YAPSI's [-0.75, +0.55] axis).
        """
        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        from lachesis.fitter import Fitter
        from lachesis.star import Star

        # Build a Star with a tight metal-poor feh prior incompatible with YAPSI.
        s = Star("dummy", ra=0.0, dec=0.0, magnitudes={"Bessell_V": (10.0, 0.05)},
                 plx=10.0, plx_e=0.1, Av=0.1, feh=-1.47, feh_e=0.07,
                 verbose=False, offline=True)
        f = Fitter()
        f.star = s
        f.bma = True
        f.grids = ["mist", "yapsi"]
        f.verbose = False
        f.prior_setup = {k: ("default",) for k in ("eep", "log_age", "feh", "dist", "Av")}
        try:
            f.initialize()
        except Exception:
            pytest.skip("Fitter.initialize couldn't run without a full grid set")
        assert "yapsi" not in f.grids, (
            "Fitter must drop YAPSI when feh_prior ±3σ is outside its axis"
        )
        assert "mist" in f.grids, "MIST should survive; covers [-4, +0.5]"

    def test_retries_with_single_bound_on_scipy_kmeans_buffer_error(self, fitter, monkeypatch):
        """dynesty's 'multi' bound calls scipy.cluster.vq.kmeans2 which
        crashes with ``IndexError: Out of bounds on buffer access (axis 0)``
        on degenerate live-point distributions. The fitter must catch this
        specific scipy/dynesty crash and retry with bound='single'.

        This was the deterministic crash on CD_Tau/YAPSI and
        KIC_7871531-combined/BaSTI during the v0.0.7 LACHESIS-I batch.
        """
        import dynesty
        call_log: list[str] = []
        real_factory = dynesty.NestedSampler

        def patched_factory(*args, **kwargs):
            bound = kwargs.get("bound", "multi")
            call_log.append(bound)
            sampler = real_factory(*args, **kwargs)
            if bound != "single":
                def boom(*a, **kw):
                    raise IndexError("Out of bounds on buffer access (axis 0)")
                sampler.run_nested = boom
            return sampler

        monkeypatch.setattr("dynesty.NestedSampler", patched_factory)

        result = fitter.fit(
            observed={"log_Teff": 3.76, "log_g": 4.44},
            uncertainties={"log_Teff": 0.01, "log_g": 0.1},
            nlive=50, dlogz=1.0,
        )
        assert "single" in call_log, (
            f"fitter must retry with bound='single' after scipy kmeans2 "
            f"crash; observed call sequence: {call_log}"
        )
        assert np.isfinite(result["logz"])


class TestFixedFehEndToEnd:
    """Fitter.initialize on a Star loaded from an upstream posterior with a
    FIXED [Fe/H]. Every assertion here failed before: -0.5 raised LinAlgError
    out of gaussian_kde, and -0.13 built a prior that returned feh_lo for
    every draw.

    Deliberately NOT wrapped in try/except -> pytest.skip: a regression here
    must fail, not disappear.
    """

    @staticmethod
    def _star(feh):
        from lachesis.star import Star

        star = Star("dummy", ra=0.0, dec=0.0,
                    magnitudes={"Bessell_V": (10.0, 0.05)},
                    plx=2.0, plx_e=0.02, Av=0.1, verbose=False, offline=True)
        star.teff, star.teff_e = 5750.0, 40.0
        star.logg, star.logg_e = 4.42, 0.03
        star.feh, star.feh_e = feh, None
        star.feh_posterior = np.full(2000, feh)
        star.external_posteriors = {}
        return star

    @pytest.mark.parametrize("feh", [-0.5, -0.13])
    def test_initialize_honours_a_fixed_feh(self, feh):
        from lachesis.fitter import Fitter

        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        f = Fitter()
        f.star = self._star(feh)
        f.grids = ["mist"]
        f.bma = False
        f.verbose = False
        with pytest.warns(RuntimeWarning):
            f.initialize()
        prior = f._fitters["mist"].prior
        assert prior._feh_type == "fixed"
        for u in (0.0, 0.1, 0.5, 0.9, 1.0):
            drawn = prior.prior_transform(np.full(prior.ndim, u))[2]
            assert drawn == pytest.approx(feh)
            assert drawn != pytest.approx(prior.feh_lo)
        # The dimension stays in the vector; dropping it would desynchronise
        # every positional theta index downstream.
        assert "feh" in prior.param_names

    def test_degenerate_external_prior_is_rejected_by_name(self):
        from lachesis.error import InputError
        from lachesis.fitter import Fitter

        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        star = self._star(-0.13)
        star.external_posteriors = {"Teff": np.full(2000, 5750.0)}
        f = Fitter()
        f.star = star
        f.grids = ["mist"]
        f.bma = False
        f.verbose = False
        with pytest.raises(InputError, match="Teff"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.initialize()

    def test_external_kde_tables_have_width(self):
        """lo == hi made pad zero, the 2048-node table collapsed onto a point,
        and the sampler's range check then rejected every proposal."""
        from lachesis.fitter import Fitter

        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        star = self._star(-0.13)
        rng = np.random.default_rng(0)
        star.external_posteriors = {"Teff": rng.normal(5750, 40, 2000)}
        f = Fitter()
        f.star = star
        f.grids = ["mist"]
        f.bma = False
        f.verbose = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.initialize()
        for _param, (grid_x, _log_pdf) in f._fitters["mist"]._external_kdes.items():
            assert grid_x[-1] > grid_x[0]

    def test_unrecognised_feh_prior_head_raises(self):
        from lachesis.error import InputError
        from lachesis.fitter import Fitter

        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        f = Fitter()
        f.star = self._star(-0.13)
        f.grids = ["mist"]
        f.bma = False
        f.verbose = False
        f.prior_setup = {"feh": ("fixd", -0.13)}
        with pytest.raises(InputError, match="Unrecognised"):
            f.initialize()

    def test_explicit_fixed_prior_setup_is_honoured(self):
        """('fixed', v) used to fall through every branch and land on the
        upstream KDE, i.e. the instruction was accepted and ignored."""
        from lachesis.fitter import Fitter

        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        f = Fitter()
        f.star = self._star(-0.13)
        f.grids = ["mist"]
        f.bma = False
        f.verbose = False
        f.prior_setup = {"feh": ("fixed", 0.21)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.initialize()
        prior = f._fitters["mist"].prior
        assert prior._feh_type == "fixed"
        assert prior.prior_transform(np.full(prior.ndim, 0.5))[2] == pytest.approx(0.21)


class TestReviewRegressions:
    """Regressions found by adversarial review of the degenerate-posterior fix.

    Each of these was introduced BY that fix, so they are not covered by the
    tests above.
    """

    @staticmethod
    def _star(feh=-0.13, plx=20.0):
        from lachesis.star import Star

        star = Star("dummy", ra=0.0, dec=0.0,
                    magnitudes={"Bessell_V": (10.0, 0.05)},
                    plx=plx, plx_e=0.1, Av=0.1, verbose=False, offline=True)
        star.teff, star.teff_e = 5750.0, 40.0
        star.logg, star.logg_e = 4.42, 0.03
        star.feh, star.feh_e = feh, None
        star.feh_posterior = np.full(2000, feh)
        star.external_posteriors = {}
        return star

    def _fitter(self, star, **kw):
        from lachesis.fitter import Fitter

        f = Fitter()
        f.star = star
        f.grids = kw.pop("grids", ["mist"])
        f.bma = kw.pop("bma", False)
        f.verbose = False
        for k, v in kw.items():
            setattr(f, k, v)
        return f

    def test_single_metallicity_grid_is_still_rail_dropped(self):
        """The rail bypass must key on the prior TYPE, not on the column
        spread. geneva's feh axis is the single point [0.0], so it produces a
        constant column through a zero-width UNIFORM box and is railed by
        construction. Keying on np.ptp let it into the BMA and put a delta
        atom at exactly 0.0 into the combined [Fe/H] posterior."""
        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        star = self._star()
        star.feh, star.feh_e = None, None
        star.feh_posterior = None
        f = self._fitter(star, grids=["mist", "geneva"], bma=True, n_grid_jobs=1)
        f.setup = ["dynesty", 60, 3.0, "multi", "rwalk", 4, False]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.initialize()
            res = f.fit_bma()
        assert "geneva" in dict(f.dropped_grids)
        pn = f._fitters[f._grids[0]].prior.param_names
        feh = np.asarray(res.samples)[:, pn.index("feh")]
        assert float(np.mean(feh == 0.0)) == 0.0

    def test_coverage_drop_is_recorded_and_warns_when_quiet(self):
        """The drop now fires on the ARIADNE path, so a batch run (verbose
        False) must not lose grids without a warning or a record."""
        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        star = self._star(feh=-1.60)
        f = self._fitter(star, grids=["mist", "yapsi"], bma=True)
        assert f.verbose is False
        with pytest.warns(RuntimeWarning, match="Auto-dropped"):
            f.initialize()
        assert "yapsi" in dict(f.dropped_grids)
        assert "yapsi" not in f.grids

    def test_local_bubble_with_a_fixed_upstream_av_is_not_blocked(self):
        """The degenerate-prior raise fired before the Local Bubble drop, so a
        nearby star with a fixed upstream Av aborted on a prior the next block
        would have deleted anyway."""
        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        star = self._star(plx=20.0)  # 50 pc, inside the bubble
        star.external_posteriors = {"Av": np.full(2000, 0.0)}
        f = self._fitter(star)
        with pytest.warns(RuntimeWarning, match="Local Bubble"):
            f.initialize()
        assert "Av" not in f._fitters["mist"]._external_kdes

    def test_short_but_spread_external_prior_warns_rather_than_aborting(self):
        """Too few samples is a different failure from fixed-upstream, with a
        different remedy, so it must not borrow that message or abort."""
        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        star = self._star()
        star.external_posteriors = {
            "Teff": np.array([5700., 5730., 5750., 5770., 5800.])}
        f = self._fitter(star)
        with pytest.warns(RuntimeWarning, match="too few"):
            f.initialize()
        assert "Teff" not in f._fitters["mist"]._external_kdes

    def test_constant_external_prior_still_raises_by_name(self):
        from lachesis.error import InputError

        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        star = self._star()
        star.external_posteriors = {"Teff": np.full(2000, 5750.0)}
        f = self._fitter(star)
        with pytest.raises(InputError, match="constant at 5750"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.initialize()

    def test_show_priors_does_not_report_a_delta_as_uniform(self, capsys):
        if FULL_GRID_H5 is None:
            pytest.skip("MIST grid not available")
        f = self._fitter(self._star())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.initialize()
        f.show_priors()
        out = capsys.readouterr().out
        assert "FIXED at -0.1300" in out
        assert "U(-2.00, 0.50)" not in out
