"""Tests for YAPSI grid parser and model grid."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from lachesis.grid.yapsi import _parse_yapsi_fits, YAPSIModelGrid


# ── Helpers ──────────────────────────────────────────────────────


def _make_fake_fits(path: Path, n_comp=2, n_mass=5, n_step=100):
    """Create a minimal YAPSI-like FITS file with 4D ImageHDU tracks.

    Structure mirrors the real YAPSI bundle:
      CAGE  (n_comp, n_mass, n_step, 8), age in Gyr
      CLOGT (n_comp, n_mass, n_step, 8), log(Teff)
      CLOGG (n_comp, n_mass, n_step, 8), log(g)
      CLOGL (n_comp, n_mass, n_step, 8), log(L/Lsun)
      CFEHS (n_comp, n_mass, n_step, 8), surface [Fe/H]
      CAMAX (n_comp, n_mass, 4), max age info
    """
    shape = (n_comp, n_mass, n_step, 8)
    feh_values = [0.0, -0.5]  # two compositions

    # Max age decreases with mass index (more massive = shorter life)
    max_ages_base = np.linspace(15.0, 0.6, n_mass)

    cage = np.zeros(shape, dtype=np.float32)
    clogt = np.zeros(shape, dtype=np.float32)
    clogg = np.zeros(shape, dtype=np.float32)
    clogl = np.zeros(shape, dtype=np.float32)
    crad = np.zeros(shape, dtype=np.float32)
    cfehs = np.zeros(shape, dtype=np.float32)
    camax = np.zeros((n_comp, n_mass, 4), dtype=np.float32)

    # Target masses: 0.6 to 2.0 Msun across n_mass bins
    target_masses = np.linspace(0.6, 2.0, n_mass)
    _LOGG_SUN = 4.4377

    for ci, feh in enumerate(feh_values):
        for mi in range(n_mass):
            max_age = max_ages_base[mi] * (1.0 + 0.1 * ci)
            ages = np.linspace(0.01, max_age, n_step)

            # Fake but physically plausible tracks
            mass_frac = mi / (n_mass - 1)  # 0 to 1
            log_teff = 3.5 + 0.35 * mass_frac + 0.01 * feh - 0.02 * ages / max_age
            log_l = -1.5 + 2.5 * mass_frac + 0.1 * feh + 0.5 * ages / max_age

            # Derive R from L and Teff (Stefan-Boltzmann)
            log_tsun = np.log10(5772.0)
            log_r = 0.5 * log_l + 2.0 * (log_tsun - log_teff)
            r_rsun = 10**log_r

            # Derive logg from target mass and R
            m = target_masses[mi]
            log_g = _LOGG_SUN + np.log10(m) - 2.0 * log_r

            # CRAD stores R/Rsun (linear, NOT log)
            cage[ci, mi, :, 0] = ages
            clogt[ci, mi, :, 0] = log_teff
            clogg[ci, mi, :, 0] = log_g
            clogl[ci, mi, :, 0] = log_l
            crad[ci, mi, :, 0] = r_rsun
            cfehs[ci, mi, :, 0] = feh

            camax[ci, mi, 0] = max_age

    hdu_list = [
        fits.PrimaryHDU(data=cage),
        fits.ImageHDU(data=clogt, name="CX"),       # placeholder (unused)
        fits.ImageHDU(data=crad, name="CRAD"),
        fits.ImageHDU(data=clogl, name="CLOGL"),
        fits.ImageHDU(data=clogt, name="CLOGT"),
        fits.ImageHDU(data=clogg, name="CLOGG"),
        fits.ImageHDU(data=cfehs, name="CFEHS"),
        fits.ImageHDU(data=clogt, name="CXC"),       # placeholder
        fits.ImageHDU(data=camax, name="CAMAX"),
    ]
    hdu_list[0].header["EXTNAME"] = "CAGE"

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(path, overwrite=True)
    return path


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def fake_fits(tmp_path):
    """Write a fake YAPSI FITS file and return its path."""
    return _make_fake_fits(tmp_path / "yapsi_test.fits")


@pytest.fixture
def fake_fits_dir(tmp_path):
    """Directory with a fake YAPSI FITS file."""
    _make_fake_fits(tmp_path / "yapsi_test.fits")
    return tmp_path


# ── Parser tests ─────────────────────────────────────────────────


class TestParser:
    def test_returns_list_of_dicts(self, fake_fits):
        result = _parse_yapsi_fits(fake_fits)
        assert isinstance(result, list)
        assert len(result) == 2  # two compositions

    def test_extracts_feh(self, fake_fits):
        result = _parse_yapsi_fits(fake_fits)
        fehs = sorted(r["feh"] for r in result)
        assert fehs[0] == pytest.approx(-0.5)
        assert fehs[1] == pytest.approx(0.0)

    def test_extracts_isochrones(self, fake_fits):
        result = _parse_yapsi_fits(fake_fits)
        for r in result:
            assert len(r["isochrones"]) > 0

    def test_isochrone_has_correct_cols(self, fake_fits):
        result = _parse_yapsi_fits(fake_fits)
        for r in result:
            for age_gyr, data in r["isochrones"]:
                assert data.shape[1] == 4  # mass_idx, log_Teff, log_g, log_L

    def test_isochrone_ages_are_positive(self, fake_fits):
        result = _parse_yapsi_fits(fake_fits)
        for r in result:
            for age_gyr, data in r["isochrones"]:
                assert age_gyr > 0

    def test_log_teff_sensible(self, fake_fits):
        result = _parse_yapsi_fits(fake_fits)
        for r in result:
            for age_gyr, data in r["isochrones"]:
                log_teffs = data[:, 1]
                assert np.all(log_teffs > 3.0)  # hotter than ~1000 K
                assert np.all(log_teffs < 5.0)  # cooler than 100,000 K


# ── Grid tests ───────────────────────────────────────────────────


class TestYAPSIModelGrid:
    def test_construction(self, fake_fits_dir):
        grid = YAPSIModelGrid(fake_fits_dir)
        assert grid.name == "YAPSI"

    def test_feh_values(self, fake_fits_dir):
        grid = YAPSIModelGrid(fake_fits_dir)
        assert len(grid.feh_values) == 2
        assert grid.feh_values[0] == pytest.approx(-0.5)
        assert grid.feh_values[1] == pytest.approx(0.0)

    def test_age_values_are_log(self, fake_fits_dir):
        grid = YAPSIModelGrid(fake_fits_dir)
        # Ages should be in log10(yr)
        assert np.all(grid.age_values > 8.0)  # > 100 Myr
        assert np.all(grid.age_values < 11.0)  # < 100 Gyr

    def test_fitting_eep_range_equals_eep_range(self, fake_fits_dir):
        grid = YAPSIModelGrid(fake_fits_dir)
        assert grid.fitting_eep_range == grid.eep_range

    def test_4d_shape(self, fake_fits_dir):
        grid = YAPSIModelGrid(fake_fits_dir)
        assert grid._data.ndim == 4
        n_feh, n_age, n_eep, n_cols = grid._data.shape
        assert n_feh == 2
        assert n_age > 0
        assert n_eep > 0
        assert n_cols == 14

    def test_columns(self, fake_fits_dir):
        grid = YAPSIModelGrid(fake_fits_dir)
        assert "log_Teff" in grid.columns
        assert "initial_mass" in grid.columns
        assert "dm_deep" in grid.columns
        assert "Teff" in grid.columns
        assert "density" in grid.columns
        assert len(grid.columns) == 14

    def test_log_r_computed(self, fake_fits_dir):
        grid = YAPSIModelGrid(fake_fits_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        # Find a populated cell
        fi, ai = 1, 0
        # Find first valid EEP
        for ei in range(grid._data.shape[2]):
            if np.isfinite(grid._data[fi, ai, ei, ci["log_Teff"]]):
                log_r = grid._data[fi, ai, ei, ci["log_R"]]
                log_l = grid._data[fi, ai, ei, ci["log_L"]]
                log_te = grid._data[fi, ai, ei, ci["log_Teff"]]
                expected = 0.5 * log_l + 2.0 * (np.log10(5772) - log_te)
                assert log_r == pytest.approx(expected, abs=1e-5)
                break

    def test_teff_derived(self, fake_fits_dir):
        grid = YAPSIModelGrid(fake_fits_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        fi, ai = 1, 0
        for ei in range(grid._data.shape[2]):
            if np.isfinite(grid._data[fi, ai, ei, ci["log_Teff"]]):
                teff = grid._data[fi, ai, ei, ci["Teff"]]
                log_teff = grid._data[fi, ai, ei, ci["log_Teff"]]
                assert teff == pytest.approx(10**log_teff, rel=1e-5)
                break

    def test_phase_is_nan(self, fake_fits_dir):
        """YAPSI has no phase labels, so phase should be NaN."""
        grid = YAPSIModelGrid(fake_fits_dir)
        ci = {c: i for i, c in enumerate(grid.columns)}
        assert np.all(np.isnan(grid._data[:, :, :, ci["phase"]]))


class TestHDF5Roundtrip:
    def test_roundtrip(self, fake_fits_dir, tmp_path):
        grid = YAPSIModelGrid(fake_fits_dir)
        h5_path = tmp_path / "test_yapsi.h5"
        grid.to_hdf5(h5_path)

        loaded = YAPSIModelGrid.from_hdf5(h5_path)
        assert loaded.name == "YAPSI"
        np.testing.assert_array_equal(loaded.feh_values, grid.feh_values)
        np.testing.assert_array_equal(loaded.age_values, grid.age_values)
        np.testing.assert_array_equal(loaded.eep_values, grid.eep_values)

        # Data matches (including NaNs)
        mask = np.isfinite(grid._data) & np.isfinite(loaded._data)
        np.testing.assert_array_almost_equal(
            grid._data[mask], loaded._data[mask]
        )
