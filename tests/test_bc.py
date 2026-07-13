"""Tests for bolometric correction tables, TDD."""

import numpy as np
import pytest

from lachesis.bc import BCTable


from lachesis.config import BC_DIR


@pytest.fixture(scope="module")
def bc():
    try:
        return BCTable(BC_DIR, system="UBVRIplus")
    except FileNotFoundError:
        pytest.skip("bc_tables.h5 not found")


class TestBCTable:

    def test_construct(self, bc):
        assert bc is not None

    def test_bands(self, bc):
        """Should list available photometric bands."""
        bands = bc.bands
        assert len(bands) == 24
        assert "Bessell_V" in bands
        assert "2MASS_J" in bands
        assert "Gaia_G_EDR3" in bands

    def test_grid_axes(self, bc):
        assert len(bc.teff_values) == 70
        assert len(bc.logg_values) == 26
        assert len(bc.feh_values) == 18
        assert len(bc.av_values) == 13

    def test_get_bc_solar(self, bc):
        """BC for Sun in V band should be small (≈ -0.07 to -0.12)."""
        bcs = bc.get_bc(teff=5772.0, logg=4.44, feh=0.0, av=0.0)
        bc_v = bcs["Bessell_V"]
        assert -0.5 < bc_v < 0.5  # should be close to zero for solar

    def test_get_bc_returns_all_bands(self, bc):
        bcs = bc.get_bc(teff=5772.0, logg=4.44, feh=0.0, av=0.0)
        assert len(bcs) == 24

    def test_extinction_makes_fainter(self, bc):
        """Higher Av should make BC more negative (band gets fainter)."""
        bc0 = bc.get_bc(teff=5772.0, logg=4.44, feh=0.0, av=0.0)
        bc3 = bc.get_bc(teff=5772.0, logg=4.44, feh=0.0, av=3.0)
        # V band should be more affected by extinction than K band
        assert bc3["Bessell_V"] < bc0["Bessell_V"]

    def test_hot_star_uv_bright(self, bc):
        """Hot star should have less negative BC in blue bands than cool star."""
        bc_hot = bc.get_bc(teff=20000.0, logg=4.0, feh=0.0, av=0.0)
        bc_cool = bc.get_bc(teff=4000.0, logg=4.0, feh=0.0, av=0.0)
        # Hot star: blue bands closer to bolometric
        assert bc_hot["Bessell_B"] > bc_cool["Bessell_B"]

    def test_absolute_mag(self, bc):
        """Compute absolute magnitude from Mbol and BC."""
        bcs = bc.get_bc(teff=5772.0, logg=4.44, feh=0.0, av=0.0)
        mbol_sun = 4.74
        m_v = mbol_sun - bcs["Bessell_V"]
        # Solar M_V should be around 4.8-4.9
        assert 4.5 < m_v < 5.2

    def test_apparent_mag(self, bc):
        """Apparent magnitude = absolute + distance modulus."""
        bcs = bc.get_bc(teff=5772.0, logg=4.44, feh=0.0, av=0.0)
        mbol_sun = 4.74
        m_v_abs = mbol_sun - bcs["Bessell_V"]
        distance_pc = 10.0  # 10 parsecs
        m_v_app = m_v_abs + 5.0 * np.log10(distance_pc / 10.0)
        # At 10 pc, apparent = absolute
        assert m_v_app == pytest.approx(m_v_abs)
