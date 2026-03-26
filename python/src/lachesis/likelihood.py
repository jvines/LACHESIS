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
) -> float:
    """Gaussian log-likelihood comparing predictions to observations.

    Handles both spectroscopic (grid columns) and photometric (BC table) observables.

    For photometric bands: m_predicted = Mbol - BC(Teff,logg,feh,Av) + 5*log10(d/10)
    """
    if set(observed) != set(uncertainties):
        raise ValueError(
            f"observed and uncertainties must have same keys. "
            f"Got {set(observed)} vs {set(uncertainties)}"
        )

    # Get predicted values from grid
    predicted = interp(eep=eep, log_age=log_age, feh=feh)

    # Check for NaN (out-of-bounds)
    if np.isnan(predicted.get("log_Teff", 0.0)):
        return -np.inf

    # Determine which observables are photometric bands
    grid_cols = set(interp._columns)
    phot_bands = set()
    if bc_table is not None:
        phot_bands = set(bc_table.bands)

    # Validate observable names
    for key in observed:
        if key not in grid_cols and key not in _PARAM_OBSERVABLES and key not in phot_bands:
            raise ValueError(
                f"Unknown observable '{key}'. "
                f"Available: {sorted(grid_cols | _PARAM_OBSERVABLES | phot_bands)}"
            )

    # Compute predicted magnitudes if needed
    pred_mags = {}
    if bc_table is not None and any(k in phot_bands for k in observed):
        if distance is None or distance <= 0:
            return -np.inf
        teff = predicted.get("Teff", 10.0 ** predicted.get("log_Teff", np.nan))
        logg = predicted.get("log_g", np.nan)
        mbol = predicted.get("Mbol", np.nan)
        if np.isnan(teff) or np.isnan(logg) or np.isnan(mbol):
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
