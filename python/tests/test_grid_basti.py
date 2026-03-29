"""Tests for BaSTI grid parser and model grid."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.basti import _parse_basti_file, BaSTIModelGrid


# -- Fixtures ---------------------------------------------------------

_SAMPLE_BASTI = textwrap.dedent("""\
    #==================================================
    #  log(L/Lo)       logTe          M/Mo(ini)       M/Mo(fin)
    #==================================================
    #AGE=5000 NPTS=4
       -2.5000    3.5100    0.2000    0.2000
       -1.2000    3.6000    0.5000    0.4950
       -0.5000    3.7000    0.8000    0.7800
        0.0000    3.7500    1.0000    0.9500
    #AGE=10000 NPTS=3
       -2.6000    3.5050    0.1800    0.1800
       -1.3000    3.5900    0.4500    0.4450
       -0.6000    3.6900    0.7500    0.7300
""")


@pytest.fixture
def sample_basti(tmp_path):
    """Write a sample BaSTI .dat file and return its path."""
    f = tmp_path / "basti_feh+0.00.dat"
    f.write_text(_SAMPLE_BASTI)
    return f


@pytest.fixture
def sample_grid_dir(tmp_path):
    """Directory with two fake BaSTI .dat files at different [Fe/H]."""
    f0 = tmp_path / "basti_feh+0.00.dat"
    f0.write_text(_SAMPLE_BASTI)

    # Second file at [Fe/H] = -1.0 (same structure, different filename)
    f1 = tmp_path / "basti_feh-1.00.dat"
    f1.write_text(_SAMPLE_BASTI)
    return tmp_path


# -- Parser tests -----------------------------------------------------


class TestParser:
    def test_extracts_feh_from_filename(self, sample_basti):
        result = _parse_basti_file(sample_basti)
        assert result["feh"] == 0.0

    def test_two_ages(self, sample_basti):
        result = _parse_basti_file(sample_basti)
        assert len(result["isochrones"]) == 2
        ages = [a for a, _ in result["isochrones"]]
        assert ages == [5.0, 10.0]

    def test_row_counts(self, sample_basti):
        result = _parse_basti_file(sample_basti)
        age5_data = result["isochrones"][0][1]
        age10_data = result["isochrones"][1][1]
        assert len(age5_data) == 4
        assert len(age10_data) == 3

    def test_log_l_column(self, sample_basti):
        result = _parse_basti_file(sample_basti)
        age5_data = result["isochrones"][0][1]
        assert age5_data[0, 0] == pytest.approx(-2.5)

    def test_log_teff_column(self, sample_basti):
        result = _parse_basti_file(sample_basti)
        age5_data = result["isochrones"][0][1]
        assert age5_data[0, 1] == pytest.approx(3.51)

    def test_mass_ini_column(self, sample_basti):
        result = _parse_basti_file(sample_basti)
        age5_data = result["isochrones"][0][1]
        assert age5_data[0, 2] == pytest.approx(0.2)
        assert age5_data[3, 2] == pytest.approx(1.0)

    def test_mass_fin_column(self, sample_basti):
        result = _parse_basti_file(sample_basti)
        age5_data = result["isochrones"][0][1]
        assert age5_data[1, 3] == pytest.approx(0.495)

    def test_negative_feh(self, tmp_path):
        f = tmp_path / "basti_feh-1.50.dat"
        f.write_text(_SAMPLE_BASTI)
        result = _parse_basti_file(f)
        assert result["feh"] == pytest.approx(-1.5)


# -- Grid tests -------------------------------------------------------


class TestBaSTIModelGrid:
    def test_construction(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        assert grid.name == "BaSTI"

    def test_feh_values(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        assert len(grid.feh_values) == 2
        assert grid.feh_values[0] == pytest.approx(-1.0)
        assert grid.feh_values[1] == pytest.approx(0.0)

    def test_age_values(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        assert len(grid.age_values) == 2
        # log10(5e9) ~ 9.699, log10(10e9) = 10.0
        assert grid.age_values[0] == pytest.approx(np.log10(5e9), abs=0.01)
        assert grid.age_values[1] == pytest.approx(10.0, abs=0.01)

    def test_eep_range(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        lo, hi = grid.eep_range
        assert lo == 0
        assert hi == 3  # max 4 rows -> EEPs 0,1,2,3

    def test_fitting_eep_range_equals_eep_range(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        assert grid.fitting_eep_range == grid.eep_range

    def test_4d_shape(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        assert grid._data.ndim == 4
        n_feh, n_age, n_eep, n_cols = grid._data.shape
        assert n_feh == 2
        assert n_age == 2
        assert n_eep == 4  # max 4 rows in age=5Gyr
        assert n_cols == 14

    def test_columns(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        assert "log_Teff" in grid.columns
        assert "initial_mass" in grid.columns
        assert "dm_deep" in grid.columns
        assert "Teff" in grid.columns
        assert "log_g" in grid.columns

    def test_log_r_computed(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # [Fe/H]=0.0, age=5 Gyr, EEP=0
        fi = 1  # feh=0.0
        ai = 0  # age=5 Gyr
        ei = 0  # first row
        log_r = grid._data[fi, ai, ei, ci["log_R"]]
        log_l = grid._data[fi, ai, ei, ci["log_L"]]
        log_te = grid._data[fi, ai, ei, ci["log_Teff"]]
        expected = 0.5 * log_l + 2.0 * (np.log10(5772) - log_te)
        assert log_r == pytest.approx(expected, abs=1e-6)

    def test_log_g_computed(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi = 1  # feh=0.0
        ai = 0  # age=5 Gyr
        ei = 3  # EEP=3 (M=1.0 Msun)
        log_g = grid._data[fi, ai, ei, ci["log_g"]]
        log_m = np.log10(grid._data[fi, ai, ei, ci["star_mass"]])
        log_r = grid._data[fi, ai, ei, ci["log_R"]]
        expected = np.log10(2.7427e4) + log_m - 2.0 * log_r
        assert log_g == pytest.approx(expected, abs=1e-6)

    def test_teff_derived(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, ai, ei = 1, 0, 0
        teff = grid._data[fi, ai, ei, ci["Teff"]]
        log_teff = grid._data[fi, ai, ei, ci["log_Teff"]]
        assert teff == pytest.approx(10**log_teff, rel=1e-6)

    def test_nan_padding(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # age=10Gyr has only 3 rows; EEP=3 should be NaN
        fi = 0  # feh=-1.0
        ai = 1  # age=10 Gyr
        ei = 3  # beyond the 3 rows for this age
        assert np.isnan(grid._data[fi, ai, ei, ci["log_Teff"]])

    def test_initial_mass_vs_star_mass(self, sample_grid_dir):
        grid = BaSTIModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, ai, ei = 1, 0, 1  # feh=0, age=5, EEP=1
        m_ini = grid._data[fi, ai, ei, ci["initial_mass"]]
        m_fin = grid._data[fi, ai, ei, ci["star_mass"]]
        assert m_ini == pytest.approx(0.5)
        assert m_fin == pytest.approx(0.495)
        # initial >= final (mass loss)
        assert m_ini >= m_fin


# -- HDF5 roundtrip ---------------------------------------------------


class TestHDF5Roundtrip:
    def test_roundtrip(self, sample_grid_dir, tmp_path):
        grid = BaSTIModelGrid(sample_grid_dir)
        h5_path = tmp_path / "test_basti.h5"
        grid.to_hdf5(h5_path)

        loaded = BaSTIModelGrid.from_hdf5(h5_path)
        assert loaded.name == "BaSTI"
        np.testing.assert_array_equal(loaded.feh_values, grid.feh_values)
        np.testing.assert_array_equal(loaded.age_values, grid.age_values)
        np.testing.assert_array_equal(loaded.eep_values, grid.eep_values)

        # Data matches (including NaNs)
        mask = np.isfinite(grid._data) & np.isfinite(loaded._data)
        np.testing.assert_array_almost_equal(
            grid._data[mask], loaded._data[mask]
        )
