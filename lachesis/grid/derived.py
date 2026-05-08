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
    """Mass gradient d(initial_mass)/dEEP along the EEP axis.

    Plain ``np.gradient`` returns 0 across constant-mass plateaus that arise
    when the (Fe/H, age, EEP) regrid samples the EEP axis more finely than
    the native mass-track spacing — so multiple EEP cells inherit the same
    mass. PARSEC's shipped grid has plateaus over ~37% of the MS volume,
    which the EEP prior ``log P(eep) = log IMF(M) + log |dM/dEEP|`` then
    rejects (``-inf`` from the zero gradient). MIST has a few such cells,
    Dartmouth/BaSTI/YAPSI essentially none.

    This implementation linearly interpolates the mass curve across plateaus
    on each (Fe/H, age) slice and takes ``np.gradient`` of the interpolated
    curve, which preserves the integral ``∫|dM/dEEP|dEEP`` between any two
    non-plateau anchor points. Cells outside the finite mass support of a
    track stay NaN (handled by the prior as out-of-grid).
    """
    arr = np.asarray(initial_mass)
    moved = np.moveaxis(arr, eep_axis, -1)
    *batch_shape, n_eep = moved.shape
    flat = moved.reshape(-1, n_eep)

    out = np.empty_like(flat, dtype=float)
    eep_idx = np.arange(n_eep, dtype=float)

    for b in range(flat.shape[0]):
        m = flat[b]
        finite = np.isfinite(m)
        n_fin = int(finite.sum())
        if n_fin < 2:
            out[b] = np.nan
            continue

        # Linearly interpolate across consecutive-equal-mass plateaus,
        # using only mass-change knots as anchors. The first finite cell is
        # always treated as a knot so we don't drop the leading plateau.
        idx = np.where(finite)[0]
        m_fin = m[idx]
        keep = np.empty(n_fin, dtype=bool)
        keep[0] = True
        keep[1:] = np.diff(m_fin) > 0  # mass strictly increased
        # Also keep the trailing point so a plateau at the end isn't extrapolated
        keep[-1] = True

        knot_idx = idx[keep].astype(float)
        knot_mass = m_fin[keep]

        smoothed = np.full(n_eep, np.nan)
        # np.interp clips outside [knot_idx[0], knot_idx[-1]]; that's fine
        # because we only fill at finite cells, all of which lie inside.
        smoothed[idx] = np.interp(eep_idx[idx], knot_idx, knot_mass)
        out[b] = np.gradient(smoothed)

    return np.moveaxis(out.reshape(*batch_shape, n_eep), -1, eep_axis)
