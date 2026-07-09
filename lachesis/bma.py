"""Bayesian Model Averaging across isochrone grids.

Given fit results from multiple grids (each with nested sampling evidence),
combine posteriors weighted by evidence:

    w_k = Z_k / Σ_j Z_j

    P(θ|D) = Σ_k w_k * P_k(θ|D)

where Z_k = exp(logz_k) is the evidence for grid k.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class BMAResult:
    """Combined BMA posterior."""
    weights: np.ndarray          # (n_models,) evidence weights
    samples: np.ndarray          # (n_combined, n_params) combined posterior
    derived: dict                # combined derived quantities + "model" labels
    model_names: list[str]       # names of each model
    log_evidences: np.ndarray    # (n_models,) log-evidence per model
    log_evidence: float = 0.0    # combined BMA log-evidence: logsumexp(log_z) - log(K)
    log_evidence_errors: np.ndarray | None = None  # (n_models,) per-grid nested-sampling log-evidence uncertainty
    # Optional per-grid raw nested-sampling posteriors, keyed by model name.
    # These are the unweighted per-grid outputs — used by the plotter for
    # per-model histograms, HR tracks, etc. `samples`/`derived` above are
    # BMA-weighted; these are not.
    per_grid_samples: dict | None = None  # {name: (n, n_params) array}
    per_grid_derived: dict | None = None  # {name: derived dict}


def bayesian_model_average(
    results: list[dict],
    names: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> BMAResult:
    """Combine nested sampling results via Bayesian Model Averaging.

    Parameters
    ----------
    results : list of fit result dicts (from IsochroneFitter.fit())
        Each must have: "samples", "logz", "logzerr", "derived"
    names : optional model names (e.g., ["MIST", "PARSEC"])
    rng : optional numpy Generator. Pass to make BMA reproducible.

    Returns
    -------
    BMAResult with evidence-weighted combined posterior.
    """
    n_models = len(results)
    if names is None:
        names = [f"model_{i}" for i in range(n_models)]
    if rng is None:
        rng = np.random.default_rng()

    # Evidence weights
    log_z = np.array([r["logz"] for r in results])
    # Per-grid nested-sampling uncertainty on each log-evidence (persisted so the
    # weights carry their evidence error; NaN if a result predates it).
    log_z_err = np.array([r.get("logzerr", np.nan) for r in results])
    # Normalize in log-space for numerical stability
    log_z_max = log_z.max()
    weights = np.exp(log_z - log_z_max)
    weights /= weights.sum()

    # Combined BMA log-evidence assuming equal model priors:
    #   log Z_BMA = logsumexp(log Z_k) - log K
    log_evidence = float(logsumexp(log_z) - np.log(n_models))

    if weights.max() > 0.99 and n_models > 1:
        dominant = names[int(weights.argmax())]
        warnings.warn(
            f"BMA weights collapsed to a one-hot on '{dominant}' "
            f"(max weight {weights.max():.4f}); BMA degenerates to model "
            f"selection here. Inspect log-evidence spread before trusting "
            f"the combined posterior.",
            stacklevel=2,
        )

    # Resample from each model proportional to its weight
    total_samples = sum(len(r["samples"]) for r in results)
    n_per_model = np.round(weights * total_samples).astype(int)
    # Ensure at least 1 sample per model with nonzero weight
    n_per_model = np.maximum(n_per_model, (weights > 0).astype(int))

    all_samples = []
    all_derived = {}
    all_model_labels = []

    # Collect derived keys common to ALL results, plus warn on dropped keys.
    per_grid_keys = [set(r["derived"].keys()) for r in results]
    derived_keys = set.intersection(*per_grid_keys)
    dropped = set.union(*per_grid_keys) - derived_keys
    if dropped:
        warnings.warn(
            f"BMA dropping derived keys present in only some grids: "
            f"{sorted(dropped)}",
            stacklevel=2,
        )
    derived_keys = sorted(derived_keys)

    for result, name, n_draw in zip(results, names, n_per_model):
        samples = result["samples"]
        derived = result["derived"]

        if n_draw >= len(samples):
            idx = np.arange(len(samples))
        else:
            idx = rng.choice(len(samples), size=n_draw, replace=False)

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

    per_grid_samples = {n: r["samples"] for n, r in zip(names, results)}
    per_grid_derived = {n: r["derived"] for n, r in zip(names, results)}

    return BMAResult(
        weights=weights,
        samples=combined_samples,
        derived=combined_derived,
        model_names=names,
        log_evidences=log_z,
        log_evidence_errors=log_z_err,
        log_evidence=log_evidence,
        per_grid_samples=per_grid_samples,
        per_grid_derived=per_grid_derived,
    )
