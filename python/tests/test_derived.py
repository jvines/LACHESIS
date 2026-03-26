"""Tests for derived stellar quantities."""

import numpy as np
import pytest

from lachesis.grid.derived import (
    M_SUN,
    MBOL_SUN,
    R_SUN,
    compute_density,
    compute_dm_deep,
    compute_mbol,
    compute_radius,
    compute_teff,
)


class TestDerivedQuantities:
    def test_teff_solar(self):
        """log(5772) ≈ 3.7613 → should give back 5772 K."""
        log_teff = np.log10(5772.0)
        assert compute_teff(log_teff) == pytest.approx(5772.0, rel=1e-6)

    def test_teff_array(self):
        log_teffs = np.array([3.5, 3.7, 4.0, 4.5])
        result = compute_teff(log_teffs)
        assert result.shape == (4,)
        assert result[0] == pytest.approx(10**3.5, rel=1e-6)

    def test_teff_nan_propagation(self):
        assert np.isnan(compute_teff(np.nan))

    def test_mbol_solar(self):
        """L = L_sun → log_L = 0 → Mbol = 4.74."""
        assert compute_mbol(0.0) == pytest.approx(MBOL_SUN)

    def test_mbol_brighter(self):
        """log_L = 1 (10x solar) → Mbol = 4.74 - 2.5 = 2.24."""
        assert compute_mbol(1.0) == pytest.approx(MBOL_SUN - 2.5)

    def test_radius_solar(self):
        """log_R = 0 → R = 1 R_sun."""
        assert compute_radius(0.0) == pytest.approx(1.0)

    def test_radius_giant(self):
        """log_R = 2 → R = 100 R_sun."""
        assert compute_radius(2.0) == pytest.approx(100.0)

    def test_density_solar(self):
        """Sun: M=1, R=1 → known density ~1.41 g/cm^3."""
        rho = compute_density(np.array(1.0), np.array(1.0))
        assert rho == pytest.approx(1.41, rel=0.02)

    def test_density_positive(self):
        masses = np.array([0.5, 1.0, 2.0])
        radii = np.array([0.5, 1.0, 1.5])
        rho = compute_density(masses, radii)
        assert np.all(rho > 0)

    def test_dm_deep_shape(self):
        """dm_deep should have same shape as input."""
        masses = np.random.rand(10, 20, 30)
        dm = compute_dm_deep(masses, eep_axis=2)
        assert dm.shape == masses.shape

    def test_dm_deep_monotonic(self):
        """For linearly increasing mass, dm_deep should be constant."""
        masses = np.linspace(0.1, 2.0, 100)
        dm = compute_dm_deep(masses)
        # Gradient of linear should be approximately constant
        assert np.allclose(dm, dm[0], rtol=0.05)
