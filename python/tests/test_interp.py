"""Tests for 3D grid interpolation — written BEFORE implementation (TDD)."""

import numpy as np
import pytest

from lachesis.grid.mist import MISTGrid, MISTModelGrid
from lachesis.interp import GridInterpolator


@pytest.fixture
def grid(sample_iso_path):
    return MISTGrid(sample_iso_path)  # single file, not directory


@pytest.fixture
def interp(grid):
    return GridInterpolator(grid)


@pytest.fixture
def model_grid(full_iso_path):
    return MISTModelGrid(full_iso_path.parent)


@pytest.fixture
def model_interp(model_grid):
    return GridInterpolator(model_grid)


class TestGridInterpolator:
    """Test 3D interpolation over (EEP, log_age, [Fe/H])."""

    def test_construct(self, interp):
        assert interp is not None

    def test_returns_dict(self, interp):
        """Should return dict of column_name → float."""
        result = interp(eep=300.0, log_age=9.0, feh=0.0)
        assert isinstance(result, dict)
        assert "log_Teff" in result
        assert "log_g" in result
        assert "log_L" in result
        assert "log_R" in result
        assert "initial_mass" in result

    def test_exact_grid_point(self, grid, interp):
        """At an exact grid point, interpolation should reproduce the value."""
        # Pick a known grid point: first [Fe/H], age=9.0, first valid EEP
        feh = grid.feh_values[0]
        age_idx = np.argmin(np.abs(grid.age_values - 9.0))
        log_age = grid.age_values[age_idx]
        # Find a valid EEP (non-NaN) at this (feh, age)
        col_idx = grid.columns.index("log_Teff")
        feh_idx = 0
        slice_at_age = grid._data[feh_idx, age_idx, :, col_idx]
        valid = np.where(~np.isnan(slice_at_age))[0]
        eep_idx = valid[len(valid) // 2]  # middle of valid range
        eep = float(grid.eep_values[eep_idx])
        expected_teff = grid._data[feh_idx, age_idx, eep_idx, col_idx]

        result = interp(eep=eep, log_age=log_age, feh=feh)
        assert result["log_Teff"] == pytest.approx(expected_teff, rel=1e-6)

    def test_interpolation_between_ages(self, grid, interp):
        """Between two age grid points, should return intermediate value."""
        feh = grid.feh_values[0]
        age1 = grid.age_values[50]
        age2 = grid.age_values[51]
        mid_age = (age1 + age2) / 2

        # Find a valid EEP at both ages
        col_idx = grid.columns.index("log_Teff")
        s1 = grid._data[0, 50, :, col_idx]
        s2 = grid._data[0, 51, :, col_idx]
        valid = np.where(~np.isnan(s1) & ~np.isnan(s2))[0]
        eep_idx = valid[len(valid) // 2]
        eep = float(grid.eep_values[eep_idx])

        r1 = interp(eep=eep, log_age=age1, feh=feh)
        r2 = interp(eep=eep, log_age=age2, feh=feh)
        rm = interp(eep=eep, log_age=mid_age, feh=feh)

        # Midpoint should be between the two endpoints
        lo = min(r1["log_Teff"], r2["log_Teff"])
        hi = max(r1["log_Teff"], r2["log_Teff"])
        assert lo <= rm["log_Teff"] <= hi

    def test_out_of_bounds_returns_nan(self, interp):
        """Out-of-bounds queries should return NaN."""
        result = interp(eep=9999.0, log_age=9.0, feh=0.0)
        assert np.isnan(result["log_Teff"])

    def test_vectorized_call(self, grid, interp):
        """Should accept arrays and return arrays."""
        n = 10
        feh = grid.feh_values[0]
        age = grid.age_values[50]
        # Valid EEPs — skip edges to avoid NaN from neighbor interpolation
        col_idx = grid.columns.index("log_Teff")
        valid = np.where(~np.isnan(grid._data[0, 50, :, col_idx]))[0]
        interior = valid[3:-3]  # skip edges
        eeps = grid.eep_values[interior[:n]].astype(float)
        ages = np.full(n, age)
        fehs = np.full(n, feh)

        result = interp(eep=eeps, log_age=ages, feh=fehs)
        assert isinstance(result, dict)
        assert len(result["log_Teff"]) == n
        assert not np.any(np.isnan(result["log_Teff"]))

    def test_physical_monotonicity_ms(self, grid, interp):
        """On the main sequence, Teff should increase with mass at fixed age."""
        feh = grid.feh_values[0]
        age = 9.0  # 1 Gyr — main sequence should be well-populated

        age_idx = np.argmin(np.abs(grid.age_values - age))
        col_idx_teff = grid.columns.index("log_Teff")
        col_idx_mass = grid.columns.index("initial_mass")

        # EEP 202-454 is main sequence (ZAMS to TAMS)
        ms_eeps = grid.eep_values[
            (grid.eep_values >= 202) & (grid.eep_values <= 400)
        ]
        # Sample a few points
        sample_eeps = ms_eeps[::10].astype(float)
        if len(sample_eeps) < 3:
            pytest.skip("Not enough MS EEPs")

        masses = []
        teffs = []
        for eep in sample_eeps:
            r = interp(eep=eep, log_age=age, feh=feh)
            if not np.isnan(r["log_Teff"]):
                masses.append(r["initial_mass"])
                teffs.append(r["log_Teff"])

        # Mass should increase with EEP on the MS
        assert len(masses) >= 3
        assert all(m2 >= m1 for m1, m2 in zip(masses, masses[1:]))
        # Teff should generally increase with mass on the MS
        assert teffs[-1] > teffs[0]


class TestModelGridInterpolator:
    """Test interpolation with the 16-column model grid."""

    def test_returns_all_16_columns(self, model_interp):
        result = model_interp(eep=300.0, log_age=9.0, feh=0.0)
        assert len(result) == 16
        assert "Teff" in result
        assert "Mbol" in result
        assert "density" in result
        assert "delta_nu" in result
        assert "nu_max" in result
        assert "dm_deep" in result

    def test_teff_consistent(self, model_interp):
        """Interpolated Teff should equal 10^(interpolated log_Teff)."""
        result = model_interp(eep=300.0, log_age=9.0, feh=0.0)
        if not np.isnan(result["log_Teff"]):
            expected = 10.0 ** result["log_Teff"]
            assert result["Teff"] == pytest.approx(expected, rel=0.01)

    def test_solar_like_values(self, model_grid, model_interp):
        """Interpolate near solar params — sanity check."""
        # Sun: ~1 Msun, age ~4.6 Gyr (log_age ~9.66), [Fe/H] ~0
        # EEP ~350 (mid main sequence)
        result = model_interp(eep=350.0, log_age=9.66, feh=0.0)
        if not np.isnan(result["Teff"]):
            # Solar-ish: Teff 4000-7000K range
            assert 4000 < result["Teff"] < 7000
            # logg 3.5-5.0
            assert 3.5 < result["log_g"] < 5.0

    def test_vectorized_with_model_grid(self, model_grid, model_interp):
        n = 5
        col_idx = model_grid.columns.index("log_Teff")
        valid = np.where(~np.isnan(model_grid._data[0, 50, :, col_idx]))[0]
        interior = valid[5:-5]
        eeps = model_grid.eep_values[interior[:n]].astype(float)
        ages = np.full(n, model_grid.age_values[50])
        fehs = np.full(n, model_grid.feh_values[0])

        result = model_interp(eep=eeps, log_age=ages, feh=fehs)
        assert len(result["Teff"]) == n
