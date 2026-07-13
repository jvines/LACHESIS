"""Tests for Dartmouth grid parser and model grid."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.dartmouth import _parse_iso_file, DartmouthModelGrid


# ── Fixtures ─────────────────────────────────────────────────────


_SAMPLE_ISO = textwrap.dedent("""\
    #NUMBER OF AGES= 2 MAGS=10
    #----------------------------------------------------
    #MIX-LEN  Y      Z          Zeff        [Fe/H] [a/Fe]
    # 1.9380  0.2696 1.6115E-02 1.6115E-02   0.00   0.00
    #----------------------------------------------------
    #**PHOTOMETRIC SYSTEM**: Bessell UBV(RI)c + 2MASS JHKs + Kepler (Vega)
    #----------------------------------------------------
    #AGE= 5.000 EEPS=5
    #EEP   M/Mo    LogTeff  LogG   LogL/Lo U       B       V       R       I       J       H       Ks      Kp      D51
      10  0.200000  3.5100  5.1000 -2.5000  15.00  14.00  13.00  12.00  11.00  10.00   9.50   9.20  12.00  14.00
      50  0.500000  3.6000  4.8000 -1.2000  12.00  11.00  10.00   9.00   8.00   7.00   6.50   6.20   9.00  11.00
     100  0.800000  3.7000  4.5000 -0.5000  10.00   9.00   8.00   7.00   6.00   5.00   4.50   4.20   7.00   9.00
     200  1.000000  3.7500  4.3000  0.0000   8.00   7.00   6.00   5.00   4.00   3.00   2.50   2.20   5.00   7.00
     250  1.200000  3.7200  2.5000  1.5000   5.00   4.00   3.00   2.00   1.00   0.00  -0.50  -0.80   2.00   4.00
    #AGE=10.000 EEPS=4
    #EEP   M/Mo    LogTeff  LogG   LogL/Lo U       B       V       R       I       J       H       Ks      Kp      D51
      10  0.180000  3.5050  5.1500 -2.6000  15.50  14.50  13.50  12.50  11.50  10.50  10.00   9.70  12.50  14.50
      50  0.450000  3.5900  4.8500 -1.3000  12.50  11.50  10.50   9.50   8.50   7.50   7.00   6.70   9.50  11.50
     100  0.750000  3.6900  4.5500 -0.6000  10.50   9.50   8.50   7.50   6.50   5.50   5.00   4.70   7.50   9.50
     200  0.950000  3.7400  4.3500 -0.1000   8.50   7.50   6.50   5.50   4.50   3.50   3.00   2.70   5.50   7.50
""")


@pytest.fixture
def sample_iso(tmp_path):
    """Write a sample Dartmouth .iso file and return its path."""
    f = tmp_path / "dartmouth_feh+0.00.iso"
    f.write_text(_SAMPLE_ISO)
    return f


@pytest.fixture
def sample_grid_dir(tmp_path):
    """Directory with two fake Dartmouth .iso files at different [Fe/H]."""
    iso0 = tmp_path / "dartmouth_feh+0.00.iso"
    iso0.write_text(_SAMPLE_ISO)

    # Second file at [Fe/H] = -1.0
    iso1_text = _SAMPLE_ISO.replace(
        "0.00   0.00", "-1.00   0.00"
    ).replace(
        "1.6115E-02 1.6115E-02   0.00",
        "1.6115E-03 1.6115E-03  -1.00",
    )
    iso1 = tmp_path / "dartmouth_feh-1.00.iso"
    iso1.write_text(iso1_text)
    return tmp_path


# ── Parser tests ─────────────────────────────────────────────────


class TestParser:
    def test_extracts_feh(self, sample_iso):
        result = _parse_iso_file(sample_iso)
        assert result["feh"] == 0.0

    def test_extracts_afe(self, sample_iso):
        result = _parse_iso_file(sample_iso)
        assert result["afe"] == 0.0

    def test_two_ages(self, sample_iso):
        result = _parse_iso_file(sample_iso)
        assert len(result["isochrones"]) == 2
        ages = [a for a, _ in result["isochrones"]]
        assert ages == [5.0, 10.0]

    def test_eep_column(self, sample_iso):
        result = _parse_iso_file(sample_iso)
        age5_data = result["isochrones"][0][1]
        eeps = age5_data[:, 0].astype(int)
        assert list(eeps) == [10, 50, 100, 200, 250]

    def test_mass_column(self, sample_iso):
        result = _parse_iso_file(sample_iso)
        age5_data = result["isochrones"][0][1]
        assert age5_data[0, 1] == pytest.approx(0.2)
        assert age5_data[3, 1] == pytest.approx(1.0)

    def test_log_teff_column(self, sample_iso):
        result = _parse_iso_file(sample_iso)
        age5_data = result["isochrones"][0][1]
        assert age5_data[0, 2] == pytest.approx(3.51)

    def test_second_age_fewer_rows(self, sample_iso):
        result = _parse_iso_file(sample_iso)
        age10_data = result["isochrones"][1][1]
        assert len(age10_data) == 4  # 4 rows vs 5 for age=5


# ── Grid tests ───────────────────────────────────────────────────


class TestDartmouthModelGrid:
    def test_construction(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        assert grid.name == "Dartmouth"

    def test_feh_values(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        assert len(grid.feh_values) == 2
        assert grid.feh_values[0] == pytest.approx(-1.0)
        assert grid.feh_values[1] == pytest.approx(0.0)

    def test_age_values(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        assert len(grid.age_values) == 2
        # log10(5e9) ≈ 9.699, log10(10e9) = 10.0
        assert grid.age_values[0] == pytest.approx(np.log10(5e9), abs=0.01)
        assert grid.age_values[1] == pytest.approx(10.0, abs=0.01)

    def test_eep_range(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        lo, hi = grid.eep_range
        assert lo == 10
        assert hi == 250

    def test_4d_shape(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        assert grid._data.ndim == 4
        n_feh, n_age, n_eep, n_cols = grid._data.shape
        assert n_feh == 2
        assert n_age == 2
        assert n_eep == 241  # 10 to 250 inclusive
        assert n_cols == 14

    def test_columns(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        assert "log_Teff" in grid.columns
        assert "initial_mass" in grid.columns
        assert "dm_deep" in grid.columns
        assert "Teff" in grid.columns

    def test_log_r_computed(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # At [Fe/H]=0.0, age=5 Gyr, EEP=200 (idx=190)
        fi = 1  # feh=0.0
        ai = 0  # age=5 Gyr
        ei = 200 - 10  # eep_idx for EEP=200
        log_r = grid._data[fi, ai, ei, ci["log_R"]]
        log_l = grid._data[fi, ai, ei, ci["log_L"]]
        log_te = grid._data[fi, ai, ei, ci["log_Teff"]]
        expected = 0.5 * log_l + 2.0 * (np.log10(5772) - log_te)
        assert log_r == pytest.approx(expected, abs=1e-6)

    def test_teff_derived(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, ai, ei = 1, 0, 200 - 10
        teff = grid._data[fi, ai, ei, ci["Teff"]]
        log_teff = grid._data[fi, ai, ei, ci["log_Teff"]]
        assert teff == pytest.approx(10**log_teff, rel=1e-6)

    def test_nan_padding(self, sample_grid_dir):
        grid = DartmouthModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # EEP=15 (idx=5) should be NaN, not in the data
        fi, ai, ei = 0, 0, 5
        assert np.isnan(grid._data[fi, ai, ei, ci["log_Teff"]])


class TestHDF5Roundtrip:
    def test_roundtrip(self, sample_grid_dir, tmp_path):
        grid = DartmouthModelGrid(sample_grid_dir)
        h5_path = tmp_path / "test_dartmouth.h5"
        grid.to_hdf5(h5_path)

        loaded = DartmouthModelGrid.from_hdf5(h5_path)
        assert loaded.name == "Dartmouth"
        np.testing.assert_array_equal(loaded.feh_values, grid.feh_values)
        np.testing.assert_array_equal(loaded.age_values, grid.age_values)
        np.testing.assert_array_equal(loaded.eep_values, grid.eep_values)

        # Data matches (including NaNs)
        mask = np.isfinite(grid._data) & np.isfinite(loaded._data)
        np.testing.assert_array_almost_equal(
            grid._data[mask], loaded._data[mask]
        )
