"""Tests for STAREVOL grid parser and model grid."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.starevol import _parse_starevol_file, STAREVOLModelGrid


# -- Fixtures ---------------------------------------------------------

# Minimal synthetic STAREVOL isochrone with the real column header
# and 3 data rows (stripped to just the columns we use).
# Real files have ~100+ columns; we only need the first few + Mbol.
_HEADER = (
    "#M_ini Z_ini OmOc_ini logTeff logL logLgrav M R logg rho_phot "
    "logMdot logTc logPc logrhoc logTmax Mr_Tmax logrhomax "
    "eps_nucl eps_grav eps_nu Mr_b_CE normR_b_CE logT_b logrho_b "
    "Mr_t_CC normR_t_CC logT_t logrho_t "
    "tau_max Ro_max tau_g Ro_g tau_Hp2 Ro_Hp2 tau_Hp Ro_Hp "
    "tau_R2 Ro_R2 tau_M2 Ro_M2 tau_max_c Ro_max_c tau_g_c Ro_g_c "
    "tau_Hp2_c Ro_Hp2_c tau_Hp_c Ro_Hp_c tau_R2_c Ro_R2_c "
    "tau_M2_c Ro_M2_c k2_conv k2_rad Omega_s Omega_c Vsurf Prot "
    "J_act J_core OOc Vcrit torque B_equi D_nu D_nu_ech D_nu_err "
    "nu_max DPi_asym. R_acc_tot R_acc_BCE R_acc_He "
    "Mbol BC U-B B-V V-R V-I J-K H-K V-K G-V Gbp-V Grp-V "
    "M_U M_B M_V M_R M_I M_H M_J M_K M_G M_Gbp M_Grp "
    "H1s H2s He3s He4s Li6s Li7s Be7s Be9s B10s B11s "
    "C12s C13s C14s N14s N15s O16s O17s O18s F19s "
    "Ne20s Ne21s Ne22s Na23s Mg24s Mg25s Mg26s Al26s Al27s Si28s "
    "H1c H2c He3c He4c C12c C13c C14c N14c N15c O16c O17c O18c "
    "F19c Ne20c Ne21c Ne22c Na23c Mg24c Mg25c Mg26c Al26c Al27c Si28c"
)


def _header_col_count():
    """Count total columns in _HEADER."""
    return len(_HEADER.lstrip("#").split())


def _mbol_index():
    """Find the 0-based column index of Mbol in _HEADER."""
    cols = _HEADER.lstrip("#").split()
    return cols.index("Mbol")


def _make_row(m_ini, z, logte, logl, m_cur, r, logg, mbol):
    """Build a whitespace-separated data row matching the header."""
    n_total = _header_col_count()
    i_mbol = _mbol_index()

    # Start with all dummy values
    parts = ["0.0000E+00"] * n_total

    # Fill in the known columns at their correct positions
    parts[0] = f"{m_ini:.3f}"       # M_ini
    parts[1] = f"{z:.6f}"           # Z_ini
    parts[2] = "0.000"              # OmOc_ini
    parts[3] = f"{logte:.6f}"       # logTeff
    parts[4] = f"{logl:.6f}"        # logL
    parts[5] = "0.000000"           # logLgrav
    parts[6] = f"{m_cur:.8f}"       # M
    parts[7] = f"{r:.5f}"           # R
    parts[8] = f"{logg:.2f}"        # logg
    parts[9] = "0.100E-05"          # rho_phot
    parts[i_mbol] = f"{mbol:.4f}"   # Mbol

    return " ".join(parts)


def _make_file_text(z, log_age, rows_data):
    """Create synthetic STAREVOL isochrone file content."""
    lines = [_HEADER]
    for rd in rows_data:
        lines.append(_make_row(*rd))
    return "\n".join(lines) + "\n"


_ROWS_SOLAR_9 = [
    # m_ini, z,     logte,   logl,     m_cur,    r,      logg,   mbol
    (0.200, 0.0134, 3.4937, -2.3911, 0.20000, 0.2190, 5.06, 10.728),
    (0.500, 0.0134, 3.6500, -1.2000, 0.49500, 0.4500, 4.83,  7.750),
    (0.800, 0.0134, 3.7100, -0.4000, 0.79500, 0.7800, 4.55,  5.750),
    (1.000, 0.0134, 3.7500,  0.0000, 0.99500, 1.0000, 4.44,  4.740),
]

_ROWS_SOLAR_10 = [
    (0.200, 0.0134, 3.4930, -2.4000, 0.20000, 0.2180, 5.07, 10.750),
    (0.500, 0.0134, 3.6490, -1.2100, 0.49400, 0.4480, 4.84,  7.775),
    (0.800, 0.0134, 3.7080, -0.4200, 0.79000, 0.7700, 4.56,  5.800),
]

_ROWS_METAL_POOR_9 = [
    (0.200, 0.0020, 3.5200, -2.2000, 0.20000, 0.2050, 5.10, 10.250),
    (0.500, 0.0020, 3.6800, -0.9000, 0.49800, 0.4200, 4.89,  7.000),
    (0.800, 0.0020, 3.7400, -0.2000, 0.79800, 0.7500, 4.59,  5.250),
    (1.000, 0.0020, 3.7800,  0.2000, 0.99800, 1.0200, 4.42,  4.240),
    (1.200, 0.0020, 3.8000,  0.5000, 1.19500, 1.2500, 4.32,  3.490),
]


@pytest.fixture
def sample_file(tmp_path):
    """Single STAREVOL isochrone file at Z=0.0134, log_age=9.0."""
    f = tmp_path / "Isochr_Z0.0134_Vini0.00_t09.000.dat"
    f.write_text(_make_file_text(0.0134, 9.0, _ROWS_SOLAR_9))
    return f


@pytest.fixture
def sample_grid_dir(tmp_path):
    """Directory with STAREVOL files at two Z, two Vini, two ages."""
    for vini in [0.00, 0.40]:
        # Solar Z=0.0134
        f1 = tmp_path / f"Isochr_Z0.0134_Vini{vini:.2f}_t09.000.dat"
        f1.write_text(_make_file_text(0.0134, 9.0, _ROWS_SOLAR_9))

        f2 = tmp_path / f"Isochr_Z0.0134_Vini{vini:.2f}_t10.000.dat"
        f2.write_text(_make_file_text(0.0134, 10.0, _ROWS_SOLAR_10))

        # Metal-poor Z=0.0020
        f3 = tmp_path / f"Isochr_Z0.0020_Vini{vini:.2f}_t09.000.dat"
        f3.write_text(_make_file_text(0.0020, 9.0, _ROWS_METAL_POOR_9))

        f4 = tmp_path / f"Isochr_Z0.0020_Vini{vini:.2f}_t10.000.dat"
        f4.write_text(_make_file_text(0.0020, 10.0, _ROWS_SOLAR_10))

    return tmp_path


# -- Parser tests -----------------------------------------------------


class TestParser:
    def test_extracts_z_from_filename(self, sample_file):
        result = _parse_starevol_file(sample_file)
        assert result["z"] == pytest.approx(0.0134)

    def test_extracts_log_age_from_filename(self, sample_file):
        result = _parse_starevol_file(sample_file)
        assert result["log_age"] == pytest.approx(9.0)

    def test_row_count(self, sample_file):
        result = _parse_starevol_file(sample_file)
        assert len(result["data"]) == 4

    def test_mass_values(self, sample_file):
        result = _parse_starevol_file(sample_file)
        assert result["data"][0, 0] == pytest.approx(0.2)
        assert result["data"][3, 0] == pytest.approx(1.0)

    def test_log_teff_column(self, sample_file):
        result = _parse_starevol_file(sample_file)
        assert result["data"][0, 3] == pytest.approx(3.4937)

    def test_log_l_column(self, sample_file):
        result = _parse_starevol_file(sample_file)
        assert result["data"][0, 2] == pytest.approx(-2.3911)

    def test_logg_column(self, sample_file):
        result = _parse_starevol_file(sample_file)
        assert result["data"][0, 4] == pytest.approx(5.06)

    def test_radius_column(self, sample_file):
        result = _parse_starevol_file(sample_file)
        assert result["data"][0, 5] == pytest.approx(0.2190)


# -- Grid tests -------------------------------------------------------


class TestSTAREVOLModelGrid:
    def test_construction(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        assert grid.name == "STAREVOL"

    def test_feh_values(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        assert len(grid.feh_values) == 2
        # Z=0.0020 -> [Fe/H] = log10(0.002/0.0134) ~ -0.826
        # Z=0.0134 -> [Fe/H] = 0.0
        assert grid.feh_values[0] < 0
        assert grid.feh_values[1] == pytest.approx(0.0, abs=0.01)

    def test_age_values(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        assert len(grid.age_values) == 2
        assert grid.age_values[0] == pytest.approx(9.0, abs=0.01)
        assert grid.age_values[1] == pytest.approx(10.0, abs=0.01)

    def test_eep_range(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        lo, hi = grid.eep_range
        assert lo == 0
        assert hi == 4  # max 5 rows (metal-poor age=9) -> EEPs 0..4

    def test_fitting_eep_range_equals_eep_range(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        assert grid.fitting_eep_range == grid.eep_range

    def test_5d_shape(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        assert grid._data.ndim == 5
        n_feh, n_vini, n_age, n_eep, n_cols = grid._data.shape
        assert n_feh == 2
        assert n_vini == 2  # Vini=0.00 and 0.40
        assert n_age == 2
        assert n_eep == 5  # max rows across all files
        assert n_cols == 14

    def test_vini_values(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        assert len(grid.vini_values) == 2
        assert grid.vini_values[0] == pytest.approx(0.0)
        assert grid.vini_values[1] == pytest.approx(0.4)

    def test_columns(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        assert "log_Teff" in grid.columns
        assert "initial_mass" in grid.columns
        assert "dm_deep" in grid.columns
        assert "Teff" in grid.columns
        assert "log_g" in grid.columns

    def test_log_r_from_radius(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # feh=0.0 (Z=0.0134) index 1, age=9.0 index 0, EEP=3 (1 Msun)
        fi, vi, ai, ei = 1, 0, 0, 3
        log_r = grid._data[fi, vi, ai, ei, ci["log_R"]]
        expected = np.log10(1.0)  # R=1.0 Rsun
        assert log_r == pytest.approx(expected, abs=1e-4)

    def test_teff_derived(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, vi, ai, ei = 1, 0, 0, 0
        teff = grid._data[fi, vi, ai, ei, ci["Teff"]]
        log_teff = grid._data[fi, vi, ai, ei, ci["log_Teff"]]
        assert teff == pytest.approx(10 ** log_teff, rel=1e-6)

    def test_mbol_from_file(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, vi, ai, ei = 1, 0, 0, 0
        mbol = grid._data[fi, vi, ai, ei, ci["Mbol"]]
        assert mbol == pytest.approx(10.728, abs=0.01)

    def test_nan_padding(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # Solar age=10 has only 3 rows; EEP=3 and 4 should be NaN
        fi, vi, ai, ei = 1, 0, 1, 3
        assert np.isnan(grid._data[fi, vi, ai, ei, ci["log_Teff"]])

    def test_initial_mass_vs_star_mass(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, vi, ai, ei = 1, 0, 0, 1  # feh=0, vini=0, age=9, EEP=1
        m_ini = grid._data[fi, vi, ai, ei, ci["initial_mass"]]
        m_fin = grid._data[fi, vi, ai, ei, ci["star_mass"]]
        assert m_ini == pytest.approx(0.5)
        assert m_fin == pytest.approx(0.495)
        assert m_ini >= m_fin

    def test_density_positive(self, sample_grid_dir):
        grid = STAREVOLModelGrid(sample_grid_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, vi, ai, ei = 1, 0, 0, 3
        rho = grid._data[fi, vi, ai, ei, ci["density"]]
        assert np.isfinite(rho) and rho > 0


# -- HDF5 roundtrip ---------------------------------------------------


class TestHDF5Roundtrip:
    def test_roundtrip(self, sample_grid_dir, tmp_path):
        grid = STAREVOLModelGrid(sample_grid_dir)
        h5_path = tmp_path / "test_starevol.h5"
        grid.to_hdf5(h5_path)

        loaded = STAREVOLModelGrid.from_hdf5(h5_path)
        assert loaded.name == "STAREVOL"
        np.testing.assert_array_equal(loaded.feh_values, grid.feh_values)
        np.testing.assert_array_equal(loaded.vini_values, grid.vini_values)
        np.testing.assert_array_equal(loaded.age_values, grid.age_values)
        np.testing.assert_array_equal(loaded.eep_values, grid.eep_values)

        # Data matches (including NaNs)
        mask = np.isfinite(grid._data) & np.isfinite(loaded._data)
        np.testing.assert_array_almost_equal(
            grid._data[mask], loaded._data[mask]
        )
