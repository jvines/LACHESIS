"""Tests for PARSEC grid — TDD."""

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


class TestPARSECModelGrid:

    def test_construct(self, parsec_grid):
        assert parsec_grid is not None

    def test_name(self, parsec_grid):
        assert parsec_grid.name == "PARSEC"

    def test_feh_values(self, parsec_grid):
        assert len(parsec_grid.feh_values) == 11
        assert any(abs(f) < 0.01 for f in parsec_grid.feh_values)

    def test_age_values(self, parsec_grid):
        ages = parsec_grid.age_values
        assert len(ages) > 50
        assert ages[0] >= 5.0
        assert ages[-1] <= 10.35

    def test_mass_axis(self, parsec_grid):
        """PARSEC uses mass as the running variable (not EEP)."""
        masses = parsec_grid.mass_values
        assert len(masses) > 50
        assert masses[0] > 0
        assert masses[-1] > masses[0]

    def test_columns(self, parsec_grid):
        for col in ["initial_mass", "star_mass", "log_Teff", "log_g",
                     "log_L", "Teff", "Mbol", "radius", "density", "phase"]:
            assert col in parsec_grid.columns, f"Missing: {col}"

    def test_grid_shape(self, parsec_grid):
        assert parsec_grid._data.ndim == 4  # (feh, age, mass, cols)
        assert parsec_grid._data.shape[0] == 11  # n_feh
        assert parsec_grid._data.shape[1] > 50   # n_age

    def test_physical_values(self, parsec_grid):
        """Spot check solar-like values."""
        ci = {c: i for i, c in enumerate(parsec_grid.columns)}
        # Find age ~9.0, feh ~0, mass ~1
        feh_idx = np.argmin(np.abs(parsec_grid.feh_values))
        age_idx = np.argmin(np.abs(parsec_grid.age_values - 9.0))
        mass_idx = np.argmin(np.abs(parsec_grid.mass_values - 1.0))
        teff = parsec_grid._data[feh_idx, age_idx, mass_idx, ci["Teff"]]
        if np.isfinite(teff):
            assert 4000 < teff < 8000

    def test_hdf5_roundtrip(self, parsec_grid, tmp_path):
        h5 = tmp_path / "parsec_test.h5"
        parsec_grid.to_hdf5(h5)
        pg2 = PARSECModelGrid.from_hdf5(h5)
        assert parsec_grid.columns == pg2.columns
        np.testing.assert_allclose(
            parsec_grid.feh_values, pg2.feh_values
        )
        # Spot check data
        valid = ~np.isnan(parsec_grid._data) & ~np.isnan(pg2._data)
        np.testing.assert_allclose(
            parsec_grid._data[valid], pg2._data[valid], rtol=1e-10
        )

    def test_fill_fraction(self, parsec_grid):
        total = parsec_grid._data.size
        filled = np.sum(~np.isnan(parsec_grid._data))
        frac = filled / total
        assert 0.1 < frac < 0.9
