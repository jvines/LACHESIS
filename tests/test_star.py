"""Star construction from an upstream (ARIADNE) posterior."""

import warnings

import numpy as np
import pytest

from lachesis.star import Star


def _ariadne_nc(tmp_path, n=2000, feh=-0.13, **overrides):
    """A minimal ARIADNE-shaped InferenceData file.

    Built with xarray directly rather than az.from_dict, whose signature moved
    between arviz 0.x and 1.0 and which is unpinned here.
    """
    import xarray as xr

    rng = np.random.default_rng(0)
    cols = {
        "teff": rng.normal(5750, 40, (1, n)),
        "logg": rng.normal(4.42, 0.03, (1, n)),
        "z": np.full((1, n), feh),
        "rad": rng.normal(1.02, 0.01, (1, n)),
        "dist": rng.normal(500.0, 5.0, (1, n)),
        "Av": rng.normal(0.2, 0.02, (1, n)),
    }
    cols.update(overrides)
    ds = xr.Dataset(
        {k: (("chain", "draw"), v) for k, v in cols.items()},
        coords={"chain": [0], "draw": np.arange(n)},
    )
    path = tmp_path / "ariadne.nc"
    ds.to_netcdf(str(path), group="posterior", engine="h5netcdf")
    return str(path)


class TestFromAriadneFixedParameter:
    """A parameter FIXED upstream comes back as a constant column, whose
    np.std is 0 for some values and a ~5.6e-17 rounding residue for others.
    Neither is a measurement uncertainty."""

    @pytest.mark.parametrize("feh", [-0.5, -0.13, 0.0, 0.25])
    def test_fixed_feh_yields_no_uncertainty(self, tmp_path, feh):
        path = _ariadne_nc(tmp_path, feh=feh)
        with pytest.warns(RuntimeWarning, match="FIXED"):
            star = Star.from_ariadne(path, starname="T", verbose=False)
        assert star.feh == pytest.approx(feh)
        assert star.feh_e is None
        # observed and uncertainties are two halves of one thing.
        assert "feh" not in star.observed
        assert "feh" not in star.uncertainties
        assert set(star.observed) == set(star.uncertainties)

    @pytest.mark.parametrize("feh", [-0.5, -0.13])
    def test_likelihood_plan_builds(self, tmp_path, feh):
        """The second crash: an exact-zero sigma reached math.log(sigma) and
        raised a bare 'math domain error' well after initialize() succeeded."""
        from lachesis.likelihood import build_likelihood_plan
        from tests.test_likelihood import _StubInterp

        path = _ariadne_nc(tmp_path, feh=feh)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            star = Star.from_ariadne(path, starname="T", verbose=False)
        plan, _has_phot, _const = build_likelihood_plan(
            _StubInterp(), star.observed, star.uncertainties)
        assert plan

    def test_free_feh_still_becomes_an_observable(self, tmp_path):
        rng = np.random.default_rng(3)
        path = _ariadne_nc(tmp_path, z=rng.normal(-0.13, 0.05, (1, 2000)))
        star = Star.from_ariadne(path, starname="T", verbose=False)
        assert star.feh_e is not None and star.feh_e > 0
        assert "feh" in star.observed and "feh" in star.uncertainties

    def test_full_posterior_is_still_carried(self, tmp_path):
        path = _ariadne_nc(tmp_path, feh=-0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            star = Star.from_ariadne(path, starname="T", verbose=False)
        assert star.feh_posterior is not None
        assert float(np.ptp(star.feh_posterior)) == 0.0
        assert set(star.external_posteriors) >= {"Teff", "log_g", "radius"}

    def test_unusable_observables_are_reported(self, tmp_path):
        path = _ariadne_nc(tmp_path, feh=-0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            star = Star.from_ariadne(path, starname="T", verbose=False)
        assert "[Fe/H]" in star.unusable_observables()
