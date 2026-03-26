"""Binary star model — combined-flux photometry from two components.

Both components share age, [Fe/H], distance, Av.
Each has its own EEP (and thus mass, Teff, etc.).

Combined flux: F_combined = F_A + F_B in each band.
Combined magnitude: m_combined = -2.5 * log10(10^(-0.4*m_A) + 10^(-0.4*m_B))
"""


import numpy as np

from lachesis.interp import GridInterpolator


def binary_apparent_mags(
    interp: GridInterpolator,
    bc,
    eep_primary: float,
    eep_secondary: float | None,
    log_age: float,
    feh: float,
    distance: float,
    av: float,
) -> dict[str, float] | None:
    """Compute apparent magnitudes for a (possibly binary) system.

    If eep_secondary is None, returns single-star magnitudes.
    Returns None if any component is out of bounds.
    """
    pred_a = interp(eep=eep_primary, log_age=log_age, feh=feh)
    if np.isnan(pred_a.get("Mbol", np.nan)):
        return None

    mags_a = bc.get_apparent_mag(
        mbol=pred_a["Mbol"], teff=pred_a["Teff"],
        logg=pred_a["log_g"], feh=feh,
        av=av, distance_pc=distance,
    )

    if eep_secondary is None:
        return mags_a

    pred_b = interp(eep=eep_secondary, log_age=log_age, feh=feh)
    if np.isnan(pred_b.get("Mbol", np.nan)):
        return None

    mags_b = bc.get_apparent_mag(
        mbol=pred_b["Mbol"], teff=pred_b["Teff"],
        logg=pred_b["log_g"], feh=feh,
        av=av, distance_pc=distance,
    )

    # Combine fluxes: m_combined = -2.5 * log10(F_A + F_B)
    combined = {}
    for band in mags_a:
        m_a = mags_a[band]
        m_b = mags_b.get(band, np.nan)
        if np.isnan(m_a) or np.isnan(m_b):
            combined[band] = np.nan
        else:
            f_a = 10.0 ** (-0.4 * m_a)
            f_b = 10.0 ** (-0.4 * m_b)
            combined[band] = -2.5 * np.log10(f_a + f_b)
    return combined


def binary_log_likelihood(
    interp: GridInterpolator,
    bc,
    eep_primary: float,
    eep_secondary: float,
    log_age: float,
    feh: float,
    distance: float | None,
    av: float | None,
    observed: dict[str, float],
    uncertainties: dict[str, float],
) -> float:
    """Log-likelihood for a binary system.

    Photometric observables use combined-flux magnitudes.
    Spectroscopic observables use the primary's properties.
    """
    # Get primary predictions (for spectroscopic observables)
    pred_primary = interp(eep=eep_primary, log_age=log_age, feh=feh)
    if np.isnan(pred_primary.get("log_Teff", np.nan)):
        return -np.inf

    # Check secondary is valid
    pred_secondary = interp(eep=eep_secondary, log_age=log_age, feh=feh)
    if np.isnan(pred_secondary.get("log_Teff", np.nan)):
        return -np.inf

    # Determine photometric bands
    phot_bands = set()
    if bc is not None:
        phot_bands = set(bc.bands)

    # Compute combined magnitudes if needed
    pred_mags = {}
    if bc is not None and distance is not None and any(k in phot_bands for k in observed):
        av_val = av if av is not None else 0.0
        pred_mags_result = binary_apparent_mags(
            interp, bc=bc,
            eep_primary=eep_primary, eep_secondary=eep_secondary,
            log_age=log_age, feh=feh,
            distance=distance, av=av_val,
        )
        if pred_mags_result is None:
            return -np.inf
        pred_mags = pred_mags_result

    # Chi-squared
    lnl = 0.0
    for key in observed:
        obs_val = observed[key]
        sigma = uncertainties[key]

        if key == "feh":
            pred = feh
        elif key in phot_bands:
            pred = pred_mags.get(key, np.nan)
        else:
            # Spectroscopic: use primary
            pred = pred_primary.get(key, np.nan)

        if np.isnan(pred):
            return -np.inf

        lnl += -0.5 * ((obs_val - pred) / sigma) ** 2 - np.log(sigma)

    lnl += -0.5 * len(observed) * np.log(2 * np.pi)
    return lnl
