"""Derived stellar quantities from raw isochrone columns."""


import numpy as np

# IAU 2015 nominal solar values
M_SUN = 1.98892e33   # g
R_SUN = 6.9566e10    # cm
MBOL_SUN = 4.74      # absolute bolometric magnitude


def compute_teff(log_teff: np.ndarray) -> np.ndarray:
    """Effective temperature in Kelvin."""
    return 10.0**log_teff


def compute_mbol(log_l: np.ndarray) -> np.ndarray:
    """Absolute bolometric magnitude."""
    return MBOL_SUN - 2.5 * log_l


def compute_radius(log_r: np.ndarray) -> np.ndarray:
    """Stellar radius in solar radii."""
    return 10.0**log_r


def compute_density(star_mass: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Mean density in g/cm^3."""
    r_cm = radius * R_SUN
    m_g = star_mass * M_SUN
    return m_g / (4.0 / 3.0 * np.pi * r_cm**3)


def compute_dm_deep(
    initial_mass: np.ndarray, eep_axis: int = -1
) -> np.ndarray:
    """Mass gradient d(initial_mass)/dEEP along the EEP axis."""
    return np.gradient(initial_mass, axis=eep_axis)
