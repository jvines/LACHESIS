"""Tests for nested sampling integration — TDD."""

from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.mist import MISTModelGrid
from lachesis.interp import GridInterpolator
from lachesis.sampler import IsochroneFitter

FULL_GRID_H5 = Path(__file__).parents[2] / "data" / "mist" / "grids" / "mist_v1.2_vvcrit0.4.h5"


@pytest.fixture(scope="module")
def fitter():
    if not FULL_GRID_H5.exists():
        pytest.skip("Full grid not built yet — run build_grid first")
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
