"""Log-likelihood for isochrone fitting.

Supports spectroscopic observables (log_Teff, log_g, etc.) and
photometric magnitudes (via BC table integration).

Hot path: ``log_likelihood`` is evaluated O(10^5–10^6) times per fit. All the
work that does NOT depend on the sampled parameters (which observables the grid
can predict, their σ, log σ, the normalisation constant) is invariant across a
fit, so it is precomputed once into a *likelihood plan* via
``build_likelihood_plan`` and reused by ``eval_likelihood_plan``. The
sampler builds the plan once; ``log_likelihood`` keeps the original one-shot
API by building a plan internally (identical result, used by non-hot callers).
"""


import math

import numpy as np

from lachesis.interp import GridInterpolator

# Observables that come from free parameters, not the grid
_PARAM_OBSERVABLES = {"feh"}

_LOG_2PI = math.log(2.0 * math.pi)

# observable "kind" tags (avoid string compares in the hot loop)
_KIND_GRID = 0
_KIND_FEH = 1
_KIND_PHOT = 2


def build_likelihood_plan(interp, observed, uncertainties, bc_table=None):
    """Precompute the parameter-independent part of the likelihood.

    Returns ``(plan, has_phot, const)`` where ``plan`` is a list of
    ``(key, obs, sigma, log_sigma, kind)`` tuples for every observable the grid
    / BC table can predict (original dict order), and ``const`` is the
    ``-0.5 * N * log(2π)`` normalisation.
    """
    missing_unc = set(observed) - set(uncertainties)
    if missing_unc:
        raise ValueError(
            f"observed keys without matching uncertainties: {sorted(missing_unc)}"
        )

    grid_cols = set(interp._columns)
    phot_bands = set(bc_table.bands) if bc_table is not None else set()
    available = grid_cols | _PARAM_OBSERVABLES | phot_bands

    plan = []
    has_phot = False
    for key, obs in observed.items():
        if key not in available:
            continue
        sigma = uncertainties[key]
        if key == "feh":
            kind = _KIND_FEH
        elif key in phot_bands:
            kind = _KIND_PHOT
            has_phot = True
        else:
            kind = _KIND_GRID
        plan.append((key, obs, sigma, math.log(sigma), kind))

    const = -0.5 * len(plan) * _LOG_2PI
    return plan, has_phot, const


def eval_likelihood_plan(plan, has_phot, const, predicted, feh,
                          bc_table=None, distance=None, av=None, jitter=0.0):
    """Evaluate the Gaussian log-likelihood from a precomputed plan.

    ``predicted`` must already be the grid interpolation at the sampled
    (eep, log_age, feh). Math matches the original per-key accumulation.

    ``jitter`` (mag) is the photometric excess-noise term, added in quadrature
    to every photometric band's catalogue uncertainty. ``0.0`` recovers the
    original fixed-error likelihood exactly.
    """
    pred_mags = None
    if has_phot:
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
        pred_mags = bc_table.get_apparent_mag(
            mbol=mbol, teff=teff, logg=logg, feh=feh,
            av=av if av is not None else 0.0, distance_pc=distance,
        )

    jit2 = jitter * jitter
    lnl = const
    for key, obs, sigma, log_sigma, kind in plan:
        if kind == _KIND_GRID:
            pred = predicted.get(key, np.nan)
        elif kind == _KIND_FEH:
            pred = feh
        else:
            pred = pred_mags.get(key, np.nan)
        if pred != pred:  # NaN, without np.isnan overhead
            return -np.inf
        if kind == _KIND_PHOT and jit2 > 0.0:
            # sigma_eff^2 = sigma_cat^2 + jitter^2; log-sigma is now per-band.
            sig2 = sigma * sigma + jit2
            d = obs - pred
            lnl += -0.5 * d * d / sig2 - 0.5 * math.log(sig2)
        else:
            d = (obs - pred) / sigma
            lnl += -0.5 * d * d - log_sigma
    return lnl


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

    One-shot API (builds a likelihood plan internally each call). The sampler's
    hot path instead builds the plan once via ``build_likelihood_plan`` and
    calls ``eval_likelihood_plan`` directly.

    For photometric bands: m_predicted = Mbol - BC(Teff,logg,feh,Av) + 5*log10(d/10)
    """
    if predicted is None:
        predicted = interp(eep=eep, log_age=log_age, feh=feh)

    if not np.isfinite(predicted.get("log_Teff", np.nan)):
        return -np.inf

    plan, has_phot, const = build_likelihood_plan(
        interp, observed, uncertainties, bc_table
    )
    return eval_likelihood_plan(
        plan, has_phot, const, predicted, feh,
        bc_table=bc_table, distance=distance, av=av,
    )
