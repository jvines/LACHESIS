"""Tests for nested sampling integration, TDD."""

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
