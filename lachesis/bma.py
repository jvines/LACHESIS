"""Bayesian Model Averaging across isochrone grids.

Given fit results from multiple grids (each with nested sampling evidence),
combine posteriors weighted by evidence:

    w_k = Z_k / Σ_j Z_j

    P(θ|D) = Σ_k w_k * P_k(θ|D)

where Z_k = exp(logz_k) is the evidence for grid k.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BMAResult:
    """Combined BMA posterior."""
    weights: np.ndarray          # (n_models,) evidence weights
    samples: np.ndarray          # (n_combined, n_params) combined posterior
    derived: dict                # combined derived quantities + "model" labels
    model_names: list[str]       # names of each model
    log_evidences: np.ndarray    # (n_models,) log-evidence per model
    # Optional per-grid raw nested-sampling posteriors, keyed by model name.
    # These are the unweighted per-grid outputs — used by the plotter for
    # per-model histograms, HR tracks, etc. `samples`/`derived` above are
    # BMA-weighted; these are not.
    per_grid_samples: dict | None = None  # {name: (n, n_params) array}
    per_grid_derived: dict | None = None  # {name: derived dict}


def bayesian_model_average(
    results: list[dict],
    names: list[str] | None = None,
) -> BMAResult:
    """Combine nested sampling results via Bayesian Model Averaging.

    Parameters
    ----------
    results : list of fit result dicts (from IsochroneFitter.fit())
        Each must have: "samples", "logz", "logzerr", "derived"
    names : optional model names (e.g., ["MIST", "PARSEC"])

    Returns
    -------
    BMAResult with evidence-weighted combined posterior.
    """
    n_models = len(results)
    if names is None:
        names = [f"model_{i}" for i in range(n_models)]

    # Evidence weights
    log_z = np.array([r["logz"] for r in results])
    # Normalize in log-space for numerical stability
    log_z_max = log_z.max()
    weights = np.exp(log_z - log_z_max)
    weights /= weights.sum()

    # Resample from each model proportional to its weight
    total_samples = sum(len(r["samples"]) for r in results)
    n_per_model = np.round(weights * total_samples).astype(int)
    # Ensure at least 1 sample per model with nonzero weight
    n_per_model = np.maximum(n_per_model, (weights > 0).astype(int))

    all_samples = []
    all_derived = {}
    all_model_labels = []

    # Collect derived keys common to ALL results
    derived_keys = set(results[0]["derived"].keys())
    for r in results[1:]:
        derived_keys &= set(r["derived"].keys())
    derived_keys = sorted(derived_keys)

    for ki, (result, name, n_draw) in enumerate(
        zip(results, names, n_per_model)
    ):
        samples = result["samples"]
        derived = result["derived"]

        if n_draw >= len(samples):
            idx = np.arange(len(samples))
        else:
            idx = np.random.choice(len(samples), size=n_draw, replace=False)

        all_samples.append(samples[idx])
        all_model_labels.extend([name] * len(idx))

        for key in derived_keys:
            if key not in all_derived:
                all_derived[key] = []
            vals = derived[key]
            if isinstance(vals, np.ndarray):
                all_derived[key].append(vals[idx])
            else:
                all_derived[key].append(np.full(len(idx), vals))

    # Concatenate
    combined_samples = np.concatenate(all_samples, axis=0)

    combined_derived = {}
    for key in derived_keys:
        combined_derived[key] = np.concatenate(all_derived[key])
    combined_derived["model"] = np.array(all_model_labels)

    return BMAResult(
        weights=weights,
        samples=combined_samples,
        derived=combined_derived,
        model_names=names,
        log_evidences=log_z,
    )
