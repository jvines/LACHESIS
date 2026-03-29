"""Tests for Geneva grid parser and model grid."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.geneva import _parse_geneva_file, GenevaModelGrid


# ── Fixtures ─────────────────────────────────────────────────────

# Minimal Geneva-format file: header + blank + data rows
# Columns: M_ini Z_ini OmOc_ini M logL logTe_c logTe_nc MBol MV U-B B-V B2-V1 r_pol oblat g_pol
_SAMPLE_DAT = textwrap.dedent("""\
 M_ini      Z_ini  OmOc_ini  M       logL     logTe_c  logTe_nc      MBol        MV       U-B       B-V     B2-V1      r_pol   oblat   g_pol

  0.800  0.014000  0.000    0.800   -0.5838    3.6871    3.6871    6.2094    6.6004    0.4946    0.9596    0.5992  5.013E+10  1.0000   4.626
  0.900  0.014000  0.000    0.899   -0.3500    3.7100    3.7100    5.6250    6.0000    0.4000    0.8800    0.5500  5.500E+10  1.0000   4.520
  1.000  0.014000  0.000    1.000   -0.0100    3.7600    3.7600    4.7750    5.0000    0.0300    0.6500    0.4000  6.500E+10  1.0000   4.400
  1.200  0.014000  0.000    1.198    0.3500    3.7900    3.7900    3.8750    4.2000   -0.0800    0.4300    0.2800  8.500E+10  1.0000   4.250
  1.500  0.014000  0.000    1.490    0.7000    3.8300    3.8300    2.9900    3.1000   -0.2000    0.2800    0.1700  1.100E+11  1.0000   4.050
  2.000  0.014000  0.000    1.980    1.1000    3.8900    3.8900    1.9900    2.0000   -0.4000    0.0500    0.0200  1.500E+11  1.0000   3.800
  6.000  0.014000  0.000    5.900    3.5000    4.2000    4.2000   -4.0000   -3.5000   -1.0000   -0.3000   -0.2000  5.000E+11  1.0000   2.500
""")


@pytest.fixture
def sample_dat(tmp_path):
    """Write a sample Geneva isochrone file and return its path."""
    f = tmp_path / "Isochr_Z0.014_Vini0.00_t09.500.dat"
    f.write_text(_SAMPLE_DAT)
    return f


@pytest.fixture
def sample_grid_dir(tmp_path):
    """Directory with two Geneva isochrone files at different ages."""
    f1 = tmp_path / "Isochr_Z0.014_Vini0.00_t09.500.dat"
    f1.write_text(_SAMPLE_DAT)

    # Second file at a different age — fewer rows (older = fewer stars)
    dat2 = textwrap.dedent("""\
 M_ini      Z_ini  OmOc_ini  M       logL     logTe_c  logTe_nc      MBol        MV       U-B       B-V     B2-V1      r_pol   oblat   g_pol

  0.800  0.014000  0.000    0.800   -0.4660    3.6983    3.6983    5.9151    6.2677    0.4384    0.9034    0.5634  5.450E+10  1.0000   4.553
  0.900  0.014000  0.000    0.899   -0.2800    3.7150    3.7150    5.4500    5.8000    0.3800    0.8600    0.5400  5.700E+10  1.0000   4.480
  1.000  0.014000  0.000    0.998    0.0300    3.7620    3.7620    4.6750    4.9000    0.0100    0.6300    0.3900  6.600E+10  1.0000   4.380
  1.200  0.014000  0.000    1.195    0.4000    3.7950    3.7950    3.7500    4.1000   -0.1000    0.4100    0.2700  8.700E+10  1.0000   4.220
""")
    f2 = tmp_path / "Isochr_Z0.014_Vini0.00_t10.000.dat"
    f2.write_text(dat2)
    return tmp_path


# ── Parser tests ─────────────────────────────────────────────────


class TestParser:
    def test_extracts_z(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        assert result["z"] == pytest.approx(0.014)

    def test_extracts_log_age(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        assert result["log_age"] == pytest.approx(9.5)

    def test_row_count_clips_mass(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        # 7 rows in file, but M_ini=6.0 row should be clipped (>5 Msun)
        assert len(result["data"]) == 6

    def test_m_ini_column(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        assert result["data"][0, 0] == pytest.approx(0.8)
        assert result["data"][-1, 0] == pytest.approx(2.0)

    def test_current_mass_column(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        # M_ini=2.0, M=1.980
        assert result["data"][-1, 1] == pytest.approx(1.98)

    def test_log_teff_column(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        # First row: logTe_c = 3.6871
        assert result["data"][0, 3] == pytest.approx(3.6871)

    def test_g_pol_column(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        # First row: g_pol = 4.626
        assert result["data"][0, 4] == pytest.approx(4.626)

    def test_r_pol_column(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        # First row: r_pol = 5.013E+10 cm
        assert result["data"][0, 5] == pytest.approx(5.013e10)

    def test_mbol_column(self, sample_dat):
        result = _parse_geneva_file(sample_dat)
        # First row: MBol = 6.2094
        assert result["data"][0, 6] == pytest.approx(6.2094)

    def test_bad_filename_raises(self, tmp_path):
        bad = tmp_path / "not_a_geneva_file.dat"
        bad.write_text("junk")
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_geneva_file(bad)


# ── Grid tests ───────────────────────────────────────────────────


class TestGenevaModelGrid:
    def test_construction(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        assert grid.name == "Geneva"

    def test_feh_values(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        assert len(grid.feh_values) == 1
        # Z=0.014 -> [Fe/H] = log10(0.014/0.014) = 0.0
        assert grid.feh_values[0] == pytest.approx(0.0)

    def test_age_values(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        assert len(grid.age_values) == 2
        assert grid.age_values[0] == pytest.approx(9.5)
        assert grid.age_values[1] == pytest.approx(10.0)

    def test_eep_range(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        lo, hi = grid.eep_range
        assert lo == 0
        # Max rows = 6 (age=9.5 has 6 rows after clip), so EEPs 0-5
        assert hi == 5

    def test_fitting_eep_range(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        assert grid.fitting_eep_range == grid.eep_range

    def test_4d_shape(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        assert grid._data.ndim == 4
        n_feh, n_age, n_eep, n_cols = grid._data.shape
        assert n_feh == 1
        assert n_age == 2
        assert n_eep == 6
        assert n_cols == 14

    def test_columns(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        assert "log_Teff" in grid.columns
        assert "initial_mass" in grid.columns
        assert "dm_deep" in grid.columns
        assert "Teff" in grid.columns
        assert "Mbol" in grid.columns
        assert "log_R" in grid.columns

    def test_log_r_from_rpol(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # At age=9.5, EEP=0 (first row): r_pol = 5.013E+10 cm
        fi, ai, ei = 0, 0, 0
        log_r = grid._data[fi, ai, ei, ci["log_R"]]
        R_SUN_CM = 6.9566e10
        expected = np.log10(5.013e10 / R_SUN_CM)
        assert log_r == pytest.approx(expected, abs=1e-4)

    def test_teff_derived(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, ai, ei = 0, 0, 0
        teff = grid._data[fi, ai, ei, ci["Teff"]]
        log_teff = grid._data[fi, ai, ei, ci["log_Teff"]]
        assert teff == pytest.approx(10**log_teff, rel=1e-6)

    def test_mbol_from_file(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, ai, ei = 0, 0, 0
        mbol = grid._data[fi, ai, ei, ci["Mbol"]]
        # Should use the value from the file directly
        assert mbol == pytest.approx(6.2094, abs=1e-3)

    def test_nan_padding(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # Age=10.0 has only 4 rows, so EEP=4 and EEP=5 should be NaN
        fi, ai, ei = 0, 1, 5
        assert np.isnan(grid._data[fi, ai, ei, ci["log_Teff"]])

    def test_mass_clipping(self, sample_grid_dir):
        grid = GenevaModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # No initial_mass should exceed 5.0
        masses = grid._data[:, :, :, ci["initial_mass"]]
        valid = masses[np.isfinite(masses)]
        assert np.all(valid <= 5.0)

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GenevaModelGrid(tmp_path)


class TestHDF5Roundtrip:
    def test_roundtrip(self, sample_grid_dir, tmp_path):
        grid = GenevaModelGrid(sample_grid_dir)
        h5_path = tmp_path / "test_geneva.h5"
        grid.to_hdf5(h5_path)

        loaded = GenevaModelGrid.from_hdf5(h5_path)
        assert loaded.name == "Geneva"
        np.testing.assert_array_equal(loaded.feh_values, grid.feh_values)
        np.testing.assert_array_equal(loaded.age_values, grid.age_values)
        np.testing.assert_array_equal(loaded.eep_values, grid.eep_values)

        # Data matches (including NaNs)
        mask = np.isfinite(grid._data) & np.isfinite(loaded._data)
        np.testing.assert_array_almost_equal(
            grid._data[mask], loaded._data[mask]
        )
