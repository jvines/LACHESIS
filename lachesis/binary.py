"""Binary star model — combined-flux photometry from two components.

Both components share age, [Fe/H], distance, Av.
Each has its own EEP (and thus mass, Teff, etc.).

Combined flux: F_combined = F_A + F_B in each band.
Combined magnitude: m_combined = -2.5 * log10(10^(-0.4*m_A) + 10^(-0.4*m_B))
"""


import numpy as np

from lachesis.interp import GridInterpolator


def _combine_mags(mags_a: dict, mags_b: dict) -> dict:
    """Combine two component magnitude dicts via flux addition."""
    bands = list(mags_a.keys())
    ma = np.array([mags_a[b] for b in bands], dtype=float)
    mb = np.array([mags_b.get(b, np.nan) for b in bands], dtype=float)
    f_a = np.power(10.0, -0.4 * ma)
    f_b = np.power(10.0, -0.4 * mb)
    f_sum = f_a + f_b
    with np.errstate(invalid="ignore", divide="ignore"):
        m_comb = np.where(f_sum > 0, -2.5 * np.log10(f_sum), np.nan)
    return {b: float(m_comb[i]) for i, b in enumerate(bands)}


def _component_apparent_mags(bc, pred, feh, av, distance):
    """Apparent mags for one component; returns None if predictions are NaN."""
    if not np.isfinite(pred.get("Mbol", np.nan)):
        return None
    return bc.get_apparent_mag(
        mbol=pred["Mbol"], teff=pred["Teff"],
        logg=pred["log_g"], feh=feh,
        av=av, distance_pc=distance,
    )


def binary_apparent_mags(
    interp: GridInterpolator,
    bc,
    eep_primary: float,
    eep_secondary: float | None,
    log_age: float,
    feh: float,
    distance: float,
    av: float,
    pred_primary: dict | None = None,
    pred_secondary: dict | None = None,
) -> dict[str, float] | None:
    """Compute apparent magnitudes for a (possibly binary) system.

    If eep_secondary is None, returns single-star magnitudes.
    Returns None if any component is out of bounds.

    pred_primary / pred_secondary may be passed in to avoid duplicate
    interpolations when the caller has already evaluated the grid.
    """
    if pred_primary is None:
        pred_primary = interp(eep=eep_primary, log_age=log_age, feh=feh)
    mags_a = _component_apparent_mags(bc, pred_primary, feh, av, distance)
    if mags_a is None:
        return None
    if eep_secondary is None:
        return mags_a
    if pred_secondary is None:
        pred_secondary = interp(eep=eep_secondary, log_age=log_age, feh=feh)
    mags_b = _component_apparent_mags(bc, pred_secondary, feh, av, distance)
    if mags_b is None:
        return None
    return _combine_mags(mags_a, mags_b)


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
    jitter=None,
) -> float:
    """Log-likelihood for a binary system.

    Photometric observables use combined-flux magnitudes (flux-weighted blend).
    Spectroscopic observables use the primary's properties — note that real
    binaries produce flux-weighted blended spectra, so this is an approximation
    that treats spectroscopic priors as 'belongs to the primary'.
    """
    assert eep_secondary <= eep_primary, (
        "binary_log_likelihood expects eep_secondary <= eep_primary; the "
        "prior_transform enforces this — bypassing it produces nonsense."
    )

    pred_primary = interp(eep=eep_primary, log_age=log_age, feh=feh)
    if not np.isfinite(pred_primary.get("log_Teff", np.nan)):
        return -np.inf
    pred_secondary = interp(eep=eep_secondary, log_age=log_age, feh=feh)
    if not np.isfinite(pred_secondary.get("log_Teff", np.nan)):
        return -np.inf

    if bc is not None:
        phot_bands = set(bc.bands)
    else:
        phot_bands = set()

    needs_phot = bc is not None and distance is not None and any(
        k in phot_bands for k in observed
    )
    pred_mags = {}
    if needs_phot:
        av_val = av if av is not None else 0.0
        pred_mags_result = binary_apparent_mags(
            interp, bc=bc,
            eep_primary=eep_primary, eep_secondary=eep_secondary,
            log_age=log_age, feh=feh,
            distance=distance, av=av_val,
            pred_primary=pred_primary, pred_secondary=pred_secondary,
        )
        if pred_mags_result is None:
            return -np.inf
        pred_mags = pred_mags_result
    elif any(k in phot_bands for k in observed):
        # Photometric observables present but no BC table — refuse the model.
        return -np.inf

    lnl = 0.0
    phot_k = 0
    for key in observed:
        obs_val = observed[key]
        sigma = uncertainties[key]

        if key == "feh":
            pred = feh
        elif key in phot_bands:
            pred = pred_mags.get(key, np.nan)
        else:
            pred = pred_primary.get(key, np.nan)

        if not np.isfinite(pred):
            return -np.inf

        if key in phot_bands:
            # ONE jitter term per band, added in quadrature (combined-flux band).
            jk = jitter[phot_k] if jitter is not None else 0.0
            phot_k += 1
            sig2 = sigma * sigma + jk * jk
            lnl += -0.5 * (obs_val - pred) ** 2 / sig2 - 0.5 * np.log(sig2)
        else:
            lnl += -0.5 * ((obs_val - pred) / sigma) ** 2 - np.log(sigma)

    lnl += -0.5 * len(observed) * np.log(2 * np.pi)
    return lnl
