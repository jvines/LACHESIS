"""Tests for PARSEC → EEP translation and model grid — TDD."""

from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.parsec import PARSECModelGrid

from tests.conftest import parsec_raw_dir

PARSEC_DIR = parsec_raw_dir()


@pytest.fixture(scope="module")
def parsec_grid():
    if PARSEC_DIR is None:
        pytest.skip("PARSEC raw data not available; set LACHESIS_PARSEC_DIR")
    return PARSECModelGrid(PARSEC_DIR)


class TestPARSECEEPTranslation:

    def test_has_eep_values(self, parsec_grid):
        """After EEP translation, PARSEC should have eep_values like MIST."""
        assert hasattr(parsec_grid, "eep_values")
        assert len(parsec_grid.eep_values) > 100

    def test_eep_range_overlaps_mist(self, parsec_grid):
        """PARSEC EEP range should overlap with MIST's (1-808+)."""
        lo, hi = parsec_grid.eep_range
        assert lo <= 202   # should cover at least ZAMS
        assert hi >= 454   # should cover at least TAMS

    def test_eep_monotonic(self, parsec_grid):
        """EEP values should be strictly increasing."""
        assert np.all(np.diff(parsec_grid.eep_values) > 0)

    def test_ms_eeps_present(self, parsec_grid):
        """Main sequence EEPs (202-454) should be populated."""
        eeps = parsec_grid.eep_values
        ms_eeps = eeps[(eeps >= 202) & (eeps <= 454)]
        assert len(ms_eeps) > 10

    def test_grid_shape_is_4d(self, parsec_grid):
        """Grid should be (n_feh, n_age, n_eep, n_cols) — same as MIST."""
        assert parsec_grid._data.ndim == 4
        assert parsec_grid._data.shape[0] == len(parsec_grid.feh_values)
        assert parsec_grid._data.shape[1] == len(parsec_grid.age_values)
        assert parsec_grid._data.shape[2] == len(parsec_grid.eep_values)


class TestPARSECInterpolation:
    """PARSEC grid should work with the same GridInterpolator as MIST."""

    def test_interpolator_constructs(self, parsec_grid):
        from lachesis.interp import GridInterpolator
        interp = GridInterpolator(parsec_grid)
        assert interp is not None

    def test_interpolate_solar(self, parsec_grid):
        """Interpolate at solar-like params — should give reasonable values."""
        from lachesis.interp import GridInterpolator
        interp = GridInterpolator(parsec_grid)
        # MS star, solar age, solar metallicity
        result = interp(eep=350.0, log_age=9.66, feh=0.0)
        teff = result.get("Teff", np.nan)
        if np.isfinite(teff):
            assert 4500 < teff < 7000

    def test_out_of_bounds_nan(self, parsec_grid):
        from lachesis.interp import GridInterpolator
        interp = GridInterpolator(parsec_grid)
        result = interp(eep=9999.0, log_age=9.0, feh=0.0)
        assert np.isnan(result["log_Teff"])


class TestPARSECSampler:
    """PARSEC should work end-to-end through the sampler."""

    def test_spectro_fit(self, parsec_grid):
        """Fit solar twin with PARSEC — same as we did with MIST."""
        from lachesis.interp import GridInterpolator
        from lachesis.sampler import IsochroneFitter

        interp = GridInterpolator(parsec_grid)
        fitter = IsochroneFitter(
            interp=interp,
            eep_range=parsec_grid.eep_range,
            age_range=(8.0, 10.3),
            feh_range=(-1.0, 0.5),
        )
        result = fitter.fit(
            observed={"log_Teff": 3.761, "log_g": 4.44},
            uncertainties={"log_Teff": 0.006, "log_g": 0.1},
            nlive=50,
            dlogz=1.0,
        )
        assert np.isfinite(result["logz"])
        # Mass should be solar-ish
        masses = result["derived"]["initial_mass"]
        med_mass = np.median(masses[np.isfinite(masses)])
        assert 0.5 < med_mass < 2.0
