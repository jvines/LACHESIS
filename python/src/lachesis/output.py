"""Output: arviz InferenceData .nc, summary .dat, model weights."""

import numpy as np


def to_inference_data(
    fit_result: dict | None = None,
    bma_result=None,
    per_grid_results: dict | None = None,
    observed: dict | None = None,
    uncertainties: dict | None = None,
    grid_name: str = "unknown",
):
    """Convert fit results to arviz InferenceData.

    Follows ARIADNE's .nc structure:
    - posterior: combined samples
    - observed_data: input observations
    - constant_data: evidence, weights, metadata
    - posterior_{grid}: per-grid posteriors (if BMA)
    """
    import arviz as az
    import xarray as xr

    # Determine source
    if bma_result is not None:
        samples = bma_result.samples
        derived = bma_result.derived
        logz = float(bma_result.log_evidences.max())
    elif fit_result is not None:
        samples = fit_result["samples"]
        derived = fit_result["derived"]
        logz = float(fit_result["logz"])
    else:
        raise ValueError("Provide fit_result or bma_result")

    n = len(samples)

    # Build posterior dict
    posterior = {}
    # Free parameters (columns of samples array)
    if samples.shape[1] >= 3:
        posterior["eep"] = samples[:, 0].reshape(1, -1)
        posterior["log_age"] = samples[:, 1].reshape(1, -1)
        posterior["feh"] = samples[:, 2].reshape(1, -1)
        # Derived: age in Gyr
        posterior["age"] = (10.0 ** samples[:, 1] / 1e9).reshape(1, -1)

    if samples.shape[1] >= 4:
        # Could be eep_secondary or distance depending on mode
        pass
    if samples.shape[1] >= 5:
        posterior["distance"] = samples[:, 3].reshape(1, -1)
        posterior["Av"] = samples[:, 4].reshape(1, -1)

    # Derived quantities from grid
    for key, vals in derived.items():
        if key == "model":
            continue  # string label, handle separately
        if isinstance(vals, np.ndarray) and len(vals) == n:
            # Rename for ecosystem consistency
            out_key = _rename_key(key)
            posterior[out_key] = vals.reshape(1, -1)

    data = {"posterior": posterior}

    # Observed data
    if observed:
        obs_data = {}
        for k, v in observed.items():
            obs_data[f"{k}_obs"] = np.array([v])
        if uncertainties:
            for k, v in uncertainties.items():
                obs_data[f"{k}_err"] = np.array([v])
        data["observed_data"] = obs_data

    # Constant data
    const = {"grid_name": np.array([grid_name])}
    if bma_result is not None:
        const["log_evidence"] = bma_result.log_evidences
        const["model_weights"] = bma_result.weights
        const["model_names"] = np.array(bma_result.model_names)
    else:
        const["log_evidence"] = np.array([logz])
    data["constant_data"] = const

    # Per-grid posteriors (BMA)
    if per_grid_results:
        for gname, gresult in per_grid_results.items():
            gs = gresult["samples"]
            gd = gresult["derived"]
            grid_post = {
                "eep": gs[:, 0].reshape(1, -1),
                "log_age": gs[:, 1].reshape(1, -1),
                "feh": gs[:, 2].reshape(1, -1),
                "age": (10.0 ** gs[:, 1] / 1e9).reshape(1, -1),
            }
            for key, vals in gd.items():
                if isinstance(vals, np.ndarray) and len(vals) == len(gs):
                    grid_post[_rename_key(key)] = vals.reshape(1, -1)
            data[f"posterior_{gname}"] = grid_post

    idata = az.from_dict(data)

    # Store evidence in attrs
    idata.attrs["log_evidence"] = logz
    idata.attrs["grid_name"] = grid_name
    idata.attrs["lachesis_version"] = "0.1.0"

    return idata


def save_summary_dat(path: str, result: dict):
    """Save parameter summary as text file (ARIADNE-compatible format)."""
    samples = result["samples"]
    derived = result["derived"]

    with open(path, "w") as f:
        f.write("#Parameter\tmedian\tupper\tlower\t1sig_lo\t1sig_hi\n")

        # Age
        if samples.shape[1] >= 3:
            ages = 10.0 ** samples[:, 1] / 1e9
            _write_param(f, "Age(Gyr)", ages)

        # Derived
        for key in ["initial_mass", "Teff", "log_g", "log_L", "radius"]:
            if key in derived:
                arr = derived[key]
                if isinstance(arr, np.ndarray):
                    _write_param(f, key, arr)


def save_model_weights(path: str, bma_result):
    """Save model weights table."""
    with open(path, "w") as f:
        f.write("#Grid\tlog_evidence\tweight\n")
        for name, lz, w in zip(
            bma_result.model_names, bma_result.log_evidences, bma_result.weights
        ):
            f.write(f"{name}\t{lz:.4f}\t{w:.4f}\n")


def _write_param(f, name, arr):
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    med = np.median(arr)
    lo = med - np.percentile(arr, 15.87)
    hi = np.percentile(arr, 84.13) - med
    sig_lo = np.percentile(arr, 15.87)
    sig_hi = np.percentile(arr, 84.13)
    f.write(f"{name}\t{med:.4f}\t+{hi:.4f}\t-{lo:.4f}\t{sig_lo:.4f}\t{sig_hi:.4f}\n")


def _rename_key(key: str) -> str:
    """Rename internal keys to ecosystem-standard names."""
    mapping = {
        "log_g": "logg",
        "log_L": "logL",
        "log_R": "logR",
        "star_mass": "mass",
    }
    return mapping.get(key, key)
