"""Fully JIT-compiled log-likelihood kernel for the single-star hot path.

Fuses the grid interpolation, EEP (IMF × |dm/dEEP|) prior, and the Gaussian
data likelihood (spectroscopic grid columns + photometric BC bands) into one
``@njit`` function that operates on raw arrays, with no Python ``predicted`` dict,
no per-call dict building, no Python chi² loop.

The sampler uses this when the fit is eligible (single star, no external-KDE
priors, non-rotation grid, numba BC table); otherwise it falls back to the
pure-Python ``eval_likelihood_plan``. ``build_njit_args`` assembles the static
(parameter-independent) arrays once per fit.
"""
from __future__ import annotations

import numpy as np
import numba as nb

from lachesis.interp_numba import _trilinear
from lachesis.bc import _quadlinear as _bc_quadlinear

# IMF type codes
IMF_SALPETER = 0
IMF_CHABRIER = 1
IMF_KROUPA = 2


@nb.njit(cache=True, inline="always")
def _imf_njit(imf_type, mass):
    if mass <= 0.0:
        return 0.0
    if imf_type == IMF_SALPETER:
        return mass ** (-2.35)
    elif imf_type == IMF_CHABRIER:
        if mass < 1.0:
            return (0.158 / mass
                    * np.exp(-0.5 * ((np.log10(mass) - np.log10(0.08)) / 0.69) ** 2))
        return 0.0443 * mass ** (-2.3)
    else:  # kroupa
        if mass < 0.08:
            return mass ** (-0.3)
        if mass < 0.5:
            return 0.08 * mass ** (-1.3)
        return 0.04 * mass ** (-2.3)


@nb.njit(cache=True)
def loglike_kernel(
    grid_data, feh_ax, age_ax, eep_ax, n_cols,
    i_logTeff, i_Teff, i_logg, i_Mbol, i_initmass, i_dmdeep,
    gobs_col, gobs_val, gobs_sig, gobs_logsig,
    feh_present, feh_obs, feh_sig, feh_logsig,
    has_phot, bc_grid, bc_teff_ax, bc_logg_ax, bc_feh_ax, bc_av_ax, n_bands,
    phot_idx, phot_obs, phot_sig,
    imf_type, const,
    eep, log_age, feh, distance, av, jitter,
):
    vals = _trilinear(grid_data, feh_ax, age_ax, eep_ax, feh, log_age, eep, n_cols)
    log_teff = vals[i_logTeff]
    if not np.isfinite(log_teff):
        return -np.inf

    # EEP prior = log IMF(M_init) + log|dm/dEEP|
    initm = vals[i_initmass]
    dmdeep = vals[i_dmdeep]
    if np.isnan(initm) or np.isnan(dmdeep) or dmdeep <= 0.0 or initm <= 0.0:
        return -np.inf
    imf = _imf_njit(imf_type, initm)
    if imf <= 0.0:
        return -np.inf
    lnp_eep = np.log(imf) + np.log(dmdeep)

    lnl = const
    # Spectroscopic grid observables
    for k in range(gobs_col.shape[0]):
        pred = vals[gobs_col[k]]
        if np.isnan(pred):
            return -np.inf
        d = (gobs_val[k] - pred) / gobs_sig[k]
        lnl += -0.5 * d * d - gobs_logsig[k]
    # [Fe/H] observable (a sampled parameter, not a grid column)
    if feh_present:
        d = (feh_obs - feh) / feh_sig
        lnl += -0.5 * d * d - feh_logsig
    # Photometric bands
    if has_phot:
        if not np.isfinite(distance) or distance <= 0.0:
            return -np.inf
        teff = vals[i_Teff]
        if not np.isfinite(teff):
            teff = 10.0 ** log_teff
        logg = vals[i_logg]
        mbol = vals[i_Mbol]
        if not (np.isfinite(teff) and np.isfinite(logg) and np.isfinite(mbol)):
            return -np.inf
        bcvals = _bc_quadlinear(
            bc_grid, bc_teff_ax, bc_logg_ax, bc_feh_ax, bc_av_ax,
            teff, logg, feh, av, n_bands,
        )
        offset = mbol + 5.0 * np.log10(distance / 10.0)
        # Photometric excess noise, ONE jitter term PER band:
        # sigma_eff[k]^2 = sigma_cat[k]^2 + jitter[k]^2. `jitter` is the
        # per-band array aligned with phot_idx; the log-sigma normalisation is
        # per-band (depends on the sampled jitter) so cannot be precomputed.
        for k in range(phot_idx.shape[0]):
            m = offset - bcvals[phot_idx[k]]
            if np.isnan(m):
                return -np.inf
            jk = jitter[k]
            sig2 = phot_sig[k] * phot_sig[k] + jk * jk
            d = phot_obs[k] - m
            lnl += -0.5 * d * d / sig2 - 0.5 * np.log(sig2)

    return lnl + lnp_eep


def _imf_type(prior):
    from lachesis.prior import salpeter_imf, chabrier_imf, kroupa_imf
    fn = getattr(prior, "_imf", None)
    if fn is salpeter_imf:
        return IMF_SALPETER
    if fn is kroupa_imf:
        return IMF_KROUPA
    return IMF_CHABRIER


def build_njit_args(interp, bc_table, prior, plan, const):
    """Assemble the static njit-kernel arguments from a likelihood plan.

    Returns a tuple ready to splat before the (eep, log_age, feh, distance, av)
    params, or ``None`` if the fit isn't eligible for the njit path (rotation
    grid, scipy-fallback BC, missing columns).
    """
    # Eligibility: non-rotation grid + numba BC table.
    if getattr(interp, "_has_vini", False):
        return None
    if not hasattr(interp, "_data") or not hasattr(interp, "_eeps"):
        return None
    from lachesis.bc import _HAS_NUMBA
    if bc_table is not None and not _HAS_NUMBA:
        return None

    cols = list(interp._columns)
    try:
        i_logTeff = cols.index("log_Teff")
        i_Teff = cols.index("Teff")
        i_logg = cols.index("log_g")
        i_Mbol = cols.index("Mbol")
        i_initmass = cols.index("initial_mass")
        i_dmdeep = cols.index("dm_deep")
    except ValueError:
        return None

    band_order = list(bc_table.bands) if bc_table is not None else []
    band_pos = {b: i for i, b in enumerate(band_order)}

    gobs_col, gobs_val, gobs_sig, gobs_logsig = [], [], [], []
    feh_present, feh_obs, feh_sig, feh_logsig = False, 0.0, 1.0, 0.0
    phot_idx, phot_obs, phot_sig = [], [], []
    has_phot = False

    # plan entries: (key, obs, sigma, log_sigma, kind) with kind 0/1/2
    for key, obs, sigma, log_sigma, kind in plan:
        if kind == 2:  # phot
            has_phot = True
            phot_idx.append(band_pos[key])
            phot_obs.append(obs)
            phot_sig.append(sigma)
        elif kind == 1:  # feh
            feh_present = True
            feh_obs, feh_sig, feh_logsig = obs, sigma, log_sigma
        else:  # grid column
            if key not in cols:
                return None
            gobs_col.append(cols.index(key))
            gobs_val.append(obs)
            gobs_sig.append(sigma)
            gobs_logsig.append(log_sigma)

    bc_grid = (bc_table._active_grid if bc_table is not None
               and bc_table._active_grid is not None else
               (bc_table._grid if bc_table is not None else np.zeros((1, 1, 1, 1))))

    return (
        interp._data, interp._feh, interp._ages, interp._eeps, interp._n_cols,
        i_logTeff, i_Teff, i_logg, i_Mbol, i_initmass, i_dmdeep,
        np.asarray(gobs_col, dtype=np.int64), np.asarray(gobs_val, dtype=np.float64),
        np.asarray(gobs_sig, dtype=np.float64), np.asarray(gobs_logsig, dtype=np.float64),
        feh_present, float(feh_obs), float(feh_sig), float(feh_logsig),
        has_phot, bc_grid,
        bc_table._ax_teff if bc_table is not None else np.zeros(1),
        bc_table._ax_logg if bc_table is not None else np.zeros(1),
        bc_table._ax_feh if bc_table is not None else np.zeros(1),
        bc_table._ax_av if bc_table is not None else np.zeros(1),
        len(band_order),
        np.asarray(phot_idx, dtype=np.int64), np.asarray(phot_obs, dtype=np.float64),
        np.asarray(phot_sig, dtype=np.float64),
        _imf_type(prior), float(const),
    )
