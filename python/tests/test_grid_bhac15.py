"""Tests for BHAC15 grid parser and model grid."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.bhac15 import _parse_bhac15_file, BHAC15ModelGrid


# -- Fixtures ---------------------------------------------------------

_SAMPLE_BHAC15 = textwrap.dedent("""\
    BHAC15 models

    L/Ls: log L/Lsun
    g: log g
    R/Rs : R/Rsun
    Li/Li0: ratio of surface lithium abundance to initial abundance

    2MASS

    !  t (Gyr) =   0.0010
    !-----------------------------------------------------------------------------------------------
    ! M/Ms  Teff     L/Ls   g    R/Rs   Li/Li0    Mj     Mh     Mk
    !-----------------------------------------------------------------------------------------------
     0.010  2200.   -2.80  3.50  0.280    1.000   9.50   8.90   8.50
     0.070  2700.   -1.50  3.40  0.850    1.000   6.20   5.60   5.30
     0.100  2900.   -1.20  3.45  1.000    1.000   5.70   5.10   4.80
     0.500  3800.    0.05  3.35  2.400    1.000   3.00   2.15   1.95
     1.000  4400.    0.50  3.46  3.100    1.000   2.00   1.35   1.22
    !-----------------------------------------------------------------------------------------------


    !  t (Gyr) =   1.0000
    !-----------------------------------------------------------------------------------------------
    ! M/Ms  Teff     L/Ls   g    R/Rs   Li/Li0    Mj     Mh     Mk
    !-----------------------------------------------------------------------------------------------
     0.070  2300.   -3.50  5.10  0.110    0.000  12.00  11.00  10.50
     0.100  2800.   -3.10  5.20  0.130    0.000  10.50   9.90   9.60
     0.500  3700.   -1.40  4.80  0.470    0.000   6.60   5.95   5.75
     1.000  5770.    0.00  4.44  1.000    0.300   3.50   3.10   3.00
    !-----------------------------------------------------------------------------------------------


    !  t (Gyr) =  10.0000
    !-----------------------------------------------------------------------------------------------
    ! M/Ms  Teff     L/Ls   g    R/Rs   Li/Li0    Mj     Mh     Mk
    !-----------------------------------------------------------------------------------------------
     0.070  1600.   -4.30  5.38  0.090    0.000  14.20  13.00  12.10
     0.100  2800.   -3.06  5.25  0.124    0.000  10.36   9.77   9.48
     0.500  3690.   -1.43  4.79  0.471    0.000   6.61   5.97   5.74
    !-----------------------------------------------------------------------------------------------
""")


@pytest.fixture
def sample_bhac15(tmp_path):
    """Write a sample BHAC15 isochrone file and return its path."""
    f = tmp_path / "BHAC15_iso.2mass"
    f.write_text(_SAMPLE_BHAC15)
    return f


@pytest.fixture
def sample_grid_dir(tmp_path):
    """Directory with a fake BHAC15 isochrone file."""
    f = tmp_path / "BHAC15_iso.2mass"
    f.write_text(_SAMPLE_BHAC15)
    return tmp_path


# -- Parser tests -----------------------------------------------------


class TestParser:
    def test_three_ages(self, sample_bhac15):
        result = _parse_bhac15_file(sample_bhac15)
        assert len(result) == 3

    def test_age_values(self, sample_bhac15):
        result = _parse_bhac15_file(sample_bhac15)
        ages = [a for a, _ in result]
        assert ages == pytest.approx([0.001, 1.0, 10.0])

    def test_row_counts(self, sample_bhac15):
        result = _parse_bhac15_file(sample_bhac15)
        assert len(result[0][1]) == 5   # 1 Myr: 5 rows
        assert len(result[1][1]) == 4   # 1 Gyr: 4 rows
        assert len(result[2][1]) == 3   # 10 Gyr: 3 rows

    def test_mass_column(self, sample_bhac15):
        result = _parse_bhac15_file(sample_bhac15)
        data_1myr = result[0][1]
        assert data_1myr[0, 0] == pytest.approx(0.01)
        assert data_1myr[-1, 0] == pytest.approx(1.0)

    def test_teff_column_is_linear(self, sample_bhac15):
        result = _parse_bhac15_file(sample_bhac15)
        data_1myr = result[0][1]
        # Teff stored as linear in raw data
        assert data_1myr[0, 1] == pytest.approx(2200.0)

    def test_log_l_column(self, sample_bhac15):
        result = _parse_bhac15_file(sample_bhac15)
        data_1myr = result[0][1]
        assert data_1myr[0, 2] == pytest.approx(-2.80)

    def test_log_g_column(self, sample_bhac15):
        result = _parse_bhac15_file(sample_bhac15)
        data_1myr = result[0][1]
        assert data_1myr[0, 3] == pytest.approx(3.50)

    def test_radius_column_is_linear(self, sample_bhac15):
        result = _parse_bhac15_file(sample_bhac15)
        data_1myr = result[0][1]
        assert data_1myr[0, 4] == pytest.approx(0.280)


# -- Grid tests -------------------------------------------------------


class TestBHAC15ModelGrid:
    def test_construction(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        assert grid.name == "BHAC15"

    def test_solar_metallicity_only(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        assert len(grid.feh_values) == 1
        assert grid.feh_values[0] == pytest.approx(0.0)

    def test_age_values(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        assert len(grid.age_values) == 3
        # log10(1e6) = 6.0, log10(1e9) = 9.0, log10(10e9) = 10.0
        assert grid.age_values[0] == pytest.approx(6.0, abs=0.01)
        assert grid.age_values[1] == pytest.approx(9.0, abs=0.01)
        assert grid.age_values[2] == pytest.approx(10.0, abs=0.01)

    def test_eep_range(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        lo, hi = grid.eep_range
        assert lo == 0
        assert hi == 4  # max 5 rows -> EEPs 0,1,2,3,4

    def test_fitting_eep_range_equals_eep_range(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        assert grid.fitting_eep_range == grid.eep_range

    def test_4d_shape(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        assert grid._data.ndim == 4
        n_feh, n_age, n_eep, n_cols = grid._data.shape
        assert n_feh == 1
        assert n_age == 3
        assert n_eep == 5  # max 5 rows at age=1 Myr
        assert n_cols == 14

    def test_columns(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        assert "log_Teff" in grid.columns
        assert "initial_mass" in grid.columns
        assert "dm_deep" in grid.columns
        assert "Teff" in grid.columns
        assert "log_g" in grid.columns
        assert len(grid.columns) == 14

    def test_log_teff_from_linear(self, sample_grid_dir):
        """Teff in file is linear; grid should store log10."""
        grid = BHAC15ModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # age=1 Myr, EEP=0: Teff=2200
        ai = 0  # youngest age
        ei = 0
        log_teff = grid._data[0, ai, ei, ci["log_Teff"]]
        assert log_teff == pytest.approx(np.log10(2200.0), abs=1e-6)

    def test_teff_derived_matches_log(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        ai, ei = 0, 0
        teff = grid._data[0, ai, ei, ci["Teff"]]
        log_teff = grid._data[0, ai, ei, ci["log_Teff"]]
        assert teff == pytest.approx(10**log_teff, rel=1e-6)

    def test_log_r_from_linear(self, sample_grid_dir):
        """R/Rs in file is linear; grid should store log10."""
        grid = BHAC15ModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        ai = 0  # 1 Myr
        ei = 0  # first row: R=0.280
        log_r = grid._data[0, ai, ei, ci["log_R"]]
        assert log_r == pytest.approx(np.log10(0.280), abs=1e-6)

    def test_log_g_directly_stored(self, sample_grid_dir):
        """log_g comes directly from the file."""
        grid = BHAC15ModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        ai = 0  # 1 Myr
        ei = 0
        log_g = grid._data[0, ai, ei, ci["log_g"]]
        assert log_g == pytest.approx(3.50, abs=1e-6)

    def test_initial_mass_equals_star_mass(self, sample_grid_dir):
        """BHAC15 isochrones: initial_mass == star_mass."""
        grid = BHAC15ModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        mask = np.isfinite(grid._data[0, :, :, ci["initial_mass"]])
        np.testing.assert_array_equal(
            grid._data[0, :, :, ci["initial_mass"]][mask],
            grid._data[0, :, :, ci["star_mass"]][mask],
        )

    def test_nan_padding(self, sample_grid_dir):
        """Ages with fewer rows should be NaN-padded."""
        grid = BHAC15ModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # age=10 Gyr has 3 rows; EEP=3,4 should be NaN
        ai = 2  # 10 Gyr
        assert np.isnan(grid._data[0, ai, 3, ci["log_Teff"]])
        assert np.isnan(grid._data[0, ai, 4, ci["log_Teff"]])

    def test_density_computed(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        ai, ei = 0, 0
        density = grid._data[0, ai, ei, ci["density"]]
        assert np.isfinite(density)
        assert density > 0

    def test_mbol_computed(self, sample_grid_dir):
        grid = BHAC15ModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        ai, ei = 0, 0
        mbol = grid._data[0, ai, ei, ci["Mbol"]]
        log_l = grid._data[0, ai, ei, ci["log_L"]]
        expected = 4.74 - 2.5 * log_l
        assert mbol == pytest.approx(expected, abs=1e-6)


# -- HDF5 roundtrip ---------------------------------------------------


class TestHDF5Roundtrip:
    def test_roundtrip(self, sample_grid_dir, tmp_path):
        grid = BHAC15ModelGrid(sample_grid_dir)
        h5_path = tmp_path / "test_bhac15.h5"
        grid.to_hdf5(h5_path)

        loaded = BHAC15ModelGrid.from_hdf5(h5_path)
        assert loaded.name == "BHAC15"
        np.testing.assert_array_equal(loaded.feh_values, grid.feh_values)
        np.testing.assert_array_equal(loaded.age_values, grid.age_values)
        np.testing.assert_array_equal(loaded.eep_values, grid.eep_values)

        # Data matches (including NaNs)
        mask = np.isfinite(grid._data) & np.isfinite(loaded._data)
        np.testing.assert_array_almost_equal(
            grid._data[mask], loaded._data[mask]
        )
