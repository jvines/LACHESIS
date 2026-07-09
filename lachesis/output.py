"""Output: arviz InferenceData .nc, summary .dat, model weights."""

import numpy as np


def _lachesis_version() -> str:
    # Deferred import to avoid the lachesis package importing itself
    # during package load (output is imported by fitter, which is in __init__).
    from lachesis import __version__
    return __version__


def _samples_to_param_dict(samples: np.ndarray, param_names: list[str]) -> dict:
    """Map a (n, ndim) sample array to a dict keyed by prior.param_names.

    Each value is shaped (1, n) for arviz (chain, draw)."""
    out = {}
    for i, name in enumerate(param_names):
        out[name] = samples[:, i].reshape(1, -1)
    if "log_age" in param_names:
        i = param_names.index("log_age")
        out["age"] = (10.0 ** samples[:, i] / 1e9).reshape(1, -1)
    return out


def to_inference_data(
    fit_result: dict | None = None,
    bma_result=None,
    per_grid_results: dict | None = None,
    observed: dict | None = None,
    uncertainties: dict | None = None,
    grid_name: str = "unknown",
    star=None,
):
    """Convert fit results to arviz InferenceData.

    Single-grid: stores `posterior` + `observed_data` + `constant_data`.
    BMA: same plus per-sample `model` array in `posterior` (string labels)
    so per-grid posteriors can be reconstructed on load by filtering.
    Per-grid raw posteriors are also written to dedicated groups via
    `idata.add_groups(posterior_<grid>=...)` for arviz≥0.16 compatibility.
    """
    import arviz as az
    import xarray as xr

    if bma_result is not None:
        samples = bma_result.samples
        derived = bma_result.derived
        logz = float(bma_result.log_evidence)
        # Pull param_names from the first per-grid result; BMA-combined
        # samples share the same column order across grids.
        if per_grid_results:
            first_key = next(iter(per_grid_results))
            param_names = per_grid_results[first_key].get("param_names")
        else:
            param_names = None
    elif fit_result is not None:
        samples = fit_result["samples"]
        derived = fit_result["derived"]
        logz = float(fit_result["logz"])
        param_names = fit_result.get("param_names")
    else:
        raise ValueError("Provide fit_result or bma_result")

    if param_names is None:
        # Backward-compatible inference: assume the legacy column order.
        param_names = ["eep", "log_age", "feh"]
        ndim = samples.shape[1]
        if ndim >= 5:
            param_names += ["distance", "Av"]
        elif ndim == 4:
            param_names.append("distance")

    n = len(samples)

    posterior = _samples_to_param_dict(samples, param_names)

    # Derived quantities from grid + per-sample model labels (BMA only).
    for key, vals in derived.items():
        if key == "model":
            if isinstance(vals, np.ndarray) and len(vals) == n:
                # Store labels as a fixed-width unicode array; arviz/netCDF4
                # serialises this as a string variable.
                posterior["model"] = np.array(vals, dtype="U").reshape(1, -1)
            continue
        if isinstance(vals, np.ndarray) and len(vals) == n:
            posterior[_rename_key(key)] = vals.reshape(1, -1)

    data = {"posterior": posterior}

    if observed:
        obs_data = {}
        for k, v in observed.items():
            obs_data[f"{k}_obs"] = np.array([v])
        if uncertainties:
            for k, v in uncertainties.items():
                obs_data[f"{k}_err"] = np.array([v])
        data["observed_data"] = obs_data

    const = {"grid_name": np.array([grid_name])}
    if bma_result is not None:
        const["log_evidence"] = bma_result.log_evidences
        if getattr(bma_result, "log_evidence_errors", None) is not None:
            const["log_evidence_err"] = bma_result.log_evidence_errors
        const["model_weights"] = bma_result.weights
        const["model_names"] = np.array(bma_result.model_names)
    else:
        const["log_evidence"] = np.array([logz])

    if star is not None:
        if getattr(star, "teff", None) is not None:
            const["ariadne_teff"] = np.array([star.teff])
            if getattr(star, "teff_e", None) is not None:
                const["ariadne_teff_e"] = np.array([star.teff_e])
        if getattr(star, "radius", None) is not None:
            const["ariadne_radius"] = np.array([star.radius])
            if getattr(star, "radius_e", None) is not None:
                const["ariadne_radius_e"] = np.array([star.radius_e])

    data["constant_data"] = const

    idata = az.from_dict(data)

    # Per-grid posteriors as their own groups. arviz>=1.0 surfaces
    # InferenceData as an xarray DataTree, so we attach extras via
    # __setitem__ instead of the older add_groups API.
    if per_grid_results:
        for gname, gresult in per_grid_results.items():
            gs = gresult["samples"]
            gd = gresult["derived"]
            gnames = gresult.get("param_names", param_names)
            grid_post = _samples_to_param_dict(gs, gnames)
            for key, vals in gd.items():
                if key == "model":
                    continue
                if isinstance(vals, np.ndarray) and len(vals) == len(gs):
                    grid_post[_rename_key(key)] = vals.reshape(1, -1)
            ds = xr.Dataset(
                {k: (["chain", "draw"], v) for k, v in grid_post.items()}
            )
            try:
                idata[f"posterior_{gname}"] = ds
            except Exception:
                # Older arviz: use add_groups when __setitem__ is not supported.
                idata.add_groups({f"posterior_{gname}": ds})

    idata.attrs["log_evidence"] = logz
    idata.attrs["grid_name"] = grid_name
    idata.attrs["lachesis_version"] = _lachesis_version()

    return idata


def save_summary_dat(path: str, result: dict, param_names: list[str] | None = None):
    """Save parameter summary as text file (ARIADNE-compatible format).

    Columns: median, +1σ, -1σ, 1σ CI lo, 1σ CI hi, 3σ CI lo, 3σ CI hi.
    """
    samples = result["samples"]
    derived = result["derived"]
    if param_names is None:
        param_names = result.get("param_names")

    with open(path, "w") as f:
        f.write(
            "#Parameter\tmedian\t+1sig\t-1sig"
            "\t1sig_lo\t1sig_hi\t3sig_lo\t3sig_hi\n"
        )

        if "log_age" in (param_names or []):
            i = param_names.index("log_age")
            ages = 10.0 ** samples[:, i] / 1e9
            _write_param(f, "Age(Gyr)", ages)

        mass_arr = derived.get("star_mass", derived.get("mass"))
        if mass_arr is not None:
            _write_param(f, "Mass(Msun)", mass_arr)

        if "Teff" in derived:
            _write_param(f, "Teff(K)", derived["Teff"])
        logg_arr = derived.get("log_g", derived.get("logg"))
        if logg_arr is not None:
            _write_param(f, "logg", logg_arr)
        logl_arr = derived.get("log_L", derived.get("logL"))
        if logl_arr is not None:
            _write_param(f, "logL", logl_arr)
        if "radius" in derived:
            _write_param(f, "Radius(Rsun)", derived["radius"])
        if "density" in derived:
            _write_param(f, "Density(g/cc)", derived["density"])

        if param_names:
            label_for = {
                "feh": "[Fe/H]",
                "distance": "Distance(pc)",
                "Av": "Av(mag)",
                "eep": "EEP",
                "eep_secondary": "EEP_secondary",
                "vini": "Vini(km/s)",
            }
            for i, name in enumerate(param_names):
                if name == "log_age":
                    continue
                _write_param(f, label_for.get(name, name), samples[:, i])


def save_model_weights(path: str, bma_result):
    """Save model weights table.

    The log_evidence_err column is appended last so readers keyed on the
    original (grid, log_evidence, weight) columns keep working.
    """
    errs = getattr(bma_result, "log_evidence_errors", None)
    if errs is None:
        errs = np.full(len(bma_result.model_names), np.nan)
    with open(path, "w") as f:
        f.write("#Grid\tlog_evidence\tweight\tlog_evidence_err\n")
        for name, lz, w, lz_err in zip(
            bma_result.model_names, bma_result.log_evidences,
            bma_result.weights, errs
        ):
            f.write(f"{name}\t{lz:.4f}\t{w:.4f}\t{lz_err:.4f}\n")


def _write_param(f, name, arr):
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    med = np.median(arr)
    p1_lo = np.percentile(arr, 15.87)
    p1_hi = np.percentile(arr, 84.13)
    p3_lo = np.percentile(arr, 0.13)
    p3_hi = np.percentile(arr, 99.87)
    hi_sig = p1_hi - med
    lo_sig = med - p1_lo
    f.write(
        f"{name}\t{med:.4f}\t+{hi_sig:.4f}\t-{lo_sig:.4f}"
        f"\t{p1_lo:.4f}\t{p1_hi:.4f}\t{p3_lo:.4f}\t{p3_hi:.4f}\n"
    )


def _rename_key(key: str) -> str:
    """Rename internal keys to ecosystem-standard names."""
    mapping = {
        "log_g": "logg",
        "log_L": "logL",
        "log_R": "logR",
        "star_mass": "mass",
    }
    return mapping.get(key, key)
