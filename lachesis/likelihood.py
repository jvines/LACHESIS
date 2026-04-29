"""Log-likelihood for isochrone fitting.

Supports spectroscopic observables (log_Teff, log_g, etc.) and
photometric magnitudes (via BC table integration).
"""


import numpy as np

from lachesis.interp import GridInterpolator

# Observables that come from free parameters, not the grid
_PARAM_OBSERVABLES = {"feh"}


def log_likelihood(
    interp: GridInterpolator,
    eep: float,
    log_age: float,
    feh: float,
    observed: dict[str, float],
    uncertainties: dict[str, float],
    bc_table=None,
    distance: float | None = None,
    av: float | None = None,
    predicted: dict | None = None,
) -> float:
    """Gaussian log-likelihood comparing predictions to observations.

    Handles both spectroscopic (grid columns) and photometric (BC table) observables.

    For photometric bands: m_predicted = Mbol - BC(Teff,logg,feh,Av) + 5*log10(d/10)

    If ``predicted`` is passed, skips the grid interpolation (avoids double work
    when the caller already interpolated for the prior).
    """
    missing_unc = set(observed) - set(uncertainties)
    if missing_unc:
        raise ValueError(
            f"observed keys without matching uncertainties: {sorted(missing_unc)}"
        )

    # Get predicted values from grid (skip if caller already did it)
    if predicted is None:
        predicted = interp(eep=eep, log_age=log_age, feh=feh)

    # Check for NaN / missing log_Teff (out-of-bounds or grid lacks the column)
    if not np.isfinite(predicted.get("log_Teff", np.nan)):
        return -np.inf

    # Determine which observables are photometric bands
    grid_cols = set(interp._columns)
    phot_bands = set()
    if bc_table is not None:
        phot_bands = set(bc_table.bands)

    # Filter to observables the grid can actually predict
    available = grid_cols | _PARAM_OBSERVABLES | phot_bands
    observed = {k: v for k, v in observed.items() if k in available}
    uncertainties = {k: v for k, v in uncertainties.items() if k in available}

    # Compute predicted magnitudes if needed
    pred_mags = {}
    if bc_table is not None and any(k in phot_bands for k in observed):
        if distance is None or distance <= 0:
            return -np.inf
        teff = predicted.get("Teff")
        if teff is None or not np.isfinite(teff):
            log_t = predicted.get("log_Teff", np.nan)
            teff = 10.0 ** log_t if np.isfinite(log_t) else np.nan
        logg = predicted.get("log_g", np.nan)
        mbol = predicted.get("Mbol", np.nan)
        if not (np.isfinite(teff) and np.isfinite(logg) and np.isfinite(mbol)):
            return -np.inf
        av_val = av if av is not None else 0.0
        pred_mags = bc_table.get_apparent_mag(
            mbol=mbol, teff=teff, logg=logg, feh=feh,
            av=av_val, distance_pc=distance,
        )

    # Chi-squared
    lnl = 0.0
    for key in observed:
        obs = observed[key]
        sigma = uncertainties[key]

        if key == "feh":
            pred = feh
        elif key in phot_bands:
            pred = pred_mags.get(key, np.nan)
        else:
            pred = predicted.get(key, np.nan)

        if np.isnan(pred):
            return -np.inf

        lnl += -0.5 * ((obs - pred) / sigma) ** 2 - np.log(sigma)

    lnl += -0.5 * len(observed) * np.log(2 * np.pi)
    return lnl
