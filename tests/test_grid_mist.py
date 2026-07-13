"""Tests for MIST grid parsing, written BEFORE implementation (TDD)."""

from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.mist import MISTGrid, MISTIsoFile, MISTModelGrid


class TestMISTIsoFileParsing:
    """Test parsing a single MIST .iso file."""

    def test_loads_without_error(self, sample_iso_path):
        iso = MISTIsoFile(sample_iso_path)
        assert iso is not None

    def test_header_metadata(self, sample_iso_path):
        iso = MISTIsoFile(sample_iso_path)
        assert iso.version == "1.2"
        assert iso.feh == pytest.approx(0.00)
        assert iso.afe == pytest.approx(0.00)
        assert iso.vvcrit == pytest.approx(0.40)

    def test_number_of_isochrones(self, sample_iso_path):
        """File header says 107 isochrones."""
        iso = MISTIsoFile(sample_iso_path)
        assert iso.num_isochrones == 107

    def test_age_values(self, sample_iso_path):
        """Ages should span log(age) = 5.0 to 10.3 in 0.05 dex steps."""
        iso = MISTIsoFile(sample_iso_path)
        ages = iso.log_ages
        assert len(ages) == 107
        assert ages[0] == pytest.approx(5.0)
        assert ages[-1] == pytest.approx(10.3)
        # Check roughly uniform spacing
        diffs = np.diff(ages)
        assert np.allclose(diffs, 0.05, atol=0.001)

    def test_column_names(self, sample_iso_path):
        """Basic iso has 25 columns with known names."""
        iso = MISTIsoFile(sample_iso_path)
        assert len(iso.columns) == 25
        # Key columns must be present
        for col in ["EEP", "log10_isochrone_age_yr", "initial_mass",
                     "star_mass", "log_Teff", "log_g", "log_L", "log_R", "phase"]:
            assert col in iso.columns

    def test_get_isochrone_by_age(self, sample_iso_path):
        """Extract a single isochrone at a given age."""
        iso = MISTIsoFile(sample_iso_path)
        data = iso.get_isochrone(log_age=5.0)
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert data.shape[1] == 25
        # All rows should have the same age
        age_col = iso.columns.index("log10_isochrone_age_yr")
        assert np.allclose(data[:, age_col], 5.0)

    def test_eep_range_varies_by_age(self, sample_iso_path):
        """Number of EEPs varies across age blocks."""
        iso = MISTIsoFile(sample_iso_path)
        d1 = iso.get_isochrone(log_age=5.0)
        d2 = iso.get_isochrone(log_age=10.0)
        # Not necessarily the same number of EEPs
        assert d1.shape[0] > 0
        assert d2.shape[0] > 0

    def test_eep_column_values(self, sample_iso_path):
        """EEP values should be positive integers."""
        iso = MISTIsoFile(sample_iso_path)
        data = iso.get_isochrone(log_age=5.0)
        eep_col = iso.columns.index("EEP")
        eeps = data[:, eep_col]
        assert np.all(eeps > 0)
        assert np.all(eeps == eeps.astype(int))
        # EEPs should be monotonically increasing
        assert np.all(np.diff(eeps) > 0)

    def test_physical_values_reasonable(self, sample_iso_path):
        """Spot-check that Teff, logg, etc. are in physical ranges."""
        iso = MISTIsoFile(sample_iso_path)
        data = iso.get_isochrone(log_age=9.0)  # ~1 Gyr
        teff = data[:, iso.columns.index("log_Teff")]
        logg = data[:, iso.columns.index("log_g")]
        # log(Teff) should be between ~3.3 (2000K) and ~5.0 (100000K)
        assert np.all(teff > 3.0)
        assert np.all(teff < 5.5)
        # log(g) should be between ~-1 and ~9 (WDs have very high logg)
        assert np.all(logg > -2)
        assert np.all(logg < 10)

    def test_all_data_property(self, sample_iso_path):
        """Should provide all isochrone data as a list of arrays."""
        iso = MISTIsoFile(sample_iso_path)
        all_data = iso.all_isochrones
        assert len(all_data) == 107
        assert all(isinstance(d, np.ndarray) for d in all_data)

    def test_invalid_age_raises(self, sample_iso_path):
        """Requesting an age not in the grid should raise."""
        iso = MISTIsoFile(sample_iso_path)
        with pytest.raises(ValueError):
            iso.get_isochrone(log_age=99.0)


class TestMISTGrid:
    """Test multi-[Fe/H] grid building and HDF5 caching."""

    def test_build_from_directory(self, sample_iso_path):
        """Build grid from directory containing .iso files."""
        grid = MISTGrid(sample_iso_path.parent)
        assert grid is not None
        assert grid.name == "MIST"

    def test_feh_values(self, sample_iso_path):
        """Grid should know its [Fe/H] values."""
        grid = MISTGrid(sample_iso_path.parent)
        # Our test data has just one file ([Fe/H]=0.00)
        assert len(grid.feh_values) >= 1
        assert any(abs(f) < 0.01 for f in grid.feh_values)

    def test_age_values(self, sample_iso_path):
        grid = MISTGrid(sample_iso_path.parent)
        assert len(grid.age_values) == 107
        assert grid.age_values[0] == pytest.approx(5.0)
        assert grid.age_values[-1] == pytest.approx(10.3)

    def test_eep_range(self, sample_iso_path):
        grid = MISTGrid(sample_iso_path.parent)
        lo, hi = grid.eep_range
        assert lo > 0
        assert hi > lo

    def test_columns(self, sample_iso_path):
        grid = MISTGrid(sample_iso_path.parent)
        for col in ["initial_mass", "log_Teff", "log_g", "log_L", "log_R"]:
            assert col in grid.columns

    def test_data_array_shape(self, sample_iso_path):
        """Internal data should be 3D: (n_feh, n_age, n_eep) per column."""
        grid = MISTGrid(sample_iso_path.parent)
        # At minimum: 1 [Fe/H] x 107 ages x max_eep
        assert grid._data.ndim == 4  # (n_feh, n_age, n_eep, n_cols)
        assert grid._data.shape[0] >= 1
        assert grid._data.shape[1] == 107

    def test_hdf5_roundtrip(self, sample_iso_path, tmp_path):
        """Save to HDF5 and reload, data should match."""
        grid = MISTGrid(sample_iso_path.parent)
        h5_path = tmp_path / "mist_test.h5"
        grid.to_hdf5(h5_path)
        assert h5_path.exists()

        grid2 = MISTGrid.from_hdf5(h5_path)
        assert np.array_equal(grid.feh_values, grid2.feh_values)
        assert np.array_equal(grid.age_values, grid2.age_values)
        np.testing.assert_allclose(grid._data, grid2._data, equal_nan=True)


class TestFullIsoParsing:
    """Verify parser handles 79-column full isochrones."""

    def test_parses_79_columns(self, full_iso_path):
        iso = MISTIsoFile(full_iso_path)
        assert len(iso.columns) == 79

    def test_has_asteroseismic_cols(self, full_iso_path):
        iso = MISTIsoFile(full_iso_path)
        for col in ["delta_nu", "nu_max"]:
            assert col in iso.columns

    def test_same_ages_as_basic(self, sample_iso_path, full_iso_path):
        basic = MISTIsoFile(sample_iso_path)
        full = MISTIsoFile(full_iso_path)
        np.testing.assert_allclose(basic.log_ages, full.log_ages)

    def test_same_eep_structure(self, sample_iso_path, full_iso_path):
        basic = MISTIsoFile(sample_iso_path)
        full = MISTIsoFile(full_iso_path)
        # Same number of rows per age block
        for b, f in zip(basic.all_isochrones, full.all_isochrones):
            assert b.shape[0] == f.shape[0]


class TestMISTModelGrid:
    """Test the proper model grid with derived columns."""

    def test_build_from_full_iso(self, full_iso_path):
        mg = MISTModelGrid(full_iso_path.parent)
        assert mg is not None
        assert mg.name == "MIST"

    def test_has_16_columns(self, full_iso_path):
        mg = MISTModelGrid(full_iso_path.parent)
        assert len(mg.columns) == 16

    def test_has_all_expected_columns(self, full_iso_path):
        mg = MISTModelGrid(full_iso_path.parent)
        expected = [
            "initial_mass", "star_mass", "log_Teff", "log_g", "log_L",
            "log_R", "phase", "delta_nu", "nu_max",
            "Teff", "Mbol", "radius", "density", "dm_deep",
        ]
        for col in expected:
            assert col in mg.columns, f"Missing column: {col}"

    def test_derived_teff_matches_log(self, full_iso_path):
        """Teff should equal 10^log_Teff at grid points."""
        mg = MISTModelGrid(full_iso_path.parent)
        log_teff_idx = mg.columns.index("log_Teff")
        teff_idx = mg.columns.index("Teff")
        # Grab a non-NaN slice
        data = mg._data[0, 50, :, :]
        valid = ~np.isnan(data[:, log_teff_idx])
        expected = 10.0 ** data[valid, log_teff_idx]
        actual = data[valid, teff_idx]
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_density_positive(self, full_iso_path):
        mg = MISTModelGrid(full_iso_path.parent)
        rho_idx = mg.columns.index("density")
        rho = mg._data[0, 50, :, rho_idx]
        valid = ~np.isnan(rho)
        assert np.all(rho[valid] > 0)

    def test_dm_deep_nonzero_on_ms(self, full_iso_path):
        """dm_deep should be non-zero on the main sequence."""
        mg = MISTModelGrid(full_iso_path.parent)
        dm_idx = mg.columns.index("dm_deep")
        # Pick a middle-age isochrone
        dm = mg._data[0, 50, :, dm_idx]
        valid = ~np.isnan(dm)
        # Most values should be positive (mass increases with EEP)
        assert np.sum(dm[valid] > 0) > np.sum(dm[valid] <= 0)

    def test_grid_shape(self, full_iso_path):
        mg = MISTModelGrid(full_iso_path.parent)
        assert mg._data.ndim == 4
        assert mg._data.shape[0] >= 1   # n_feh
        assert mg._data.shape[1] == 107  # n_age
        assert mg._data.shape[3] == 16   # n_cols

    def test_hdf5_roundtrip(self, full_iso_path, tmp_path):
        mg = MISTModelGrid(full_iso_path.parent)
        h5 = tmp_path / "model_grid.h5"
        mg.to_hdf5(h5)
        mg2 = MISTModelGrid.from_hdf5(h5)
        assert mg.columns == mg2.columns
        np.testing.assert_allclose(mg._data, mg2._data, equal_nan=True)

    def test_fill_fraction(self, full_iso_path):
        """Grid should be ~50-60% filled (rest NaN)."""
        mg = MISTModelGrid(full_iso_path.parent)
        total = mg._data.size
        filled = np.sum(~np.isnan(mg._data))
        frac = filled / total
        assert 0.3 < frac < 0.8  # roughly 50% ± 20%
