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
    # These are the unweighted per-grid outputs, used by the plotter for
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
    if not np.all(np.isfinite(log_z)):
        # A single non-finite log-evidence poisons everything downstream and
        # does it quietly: log_z.max() becomes NaN, every weight becomes NaN,
        # np.round(NaN * N) is 0 and the (weights > 0) floor does not rescue it
        # because NaN > 0 is False. Every model then draws zero samples, so
        # this returns a (0, ndim) posterior with no exception and the run
        # reports success with an empty result.
        bad = [
            f"{nm}={z!r}" for nm, z in zip(names, log_z) if not np.isfinite(z)
        ]
        raise ValueError(
            "Cannot model-average on a non-finite log-evidence: "
            + ", ".join(bad)
            + ". Drop these grids, or check for a zero-width prior dimension."
        )
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

    # Each per-grid posterior already contains equally weighted samples.
    n_samples = np.array(
        [len(r["samples"]) for r in results], dtype=int
    )
    if np.any(n_samples == 0):
        raise ValueError(
            "There are empty model posteriors. "
            f"n_samples={n_samples}. Cannot perform model averaging."
        )

    total_samples = int(n_samples.sum())

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

    for result, name in zip(results, names):
        samples = result["samples"]
        derived = result["derived"]
        n = len(samples)

        all_samples.append(samples)
        all_model_labels.extend([name] * n)

        for key in derived_keys:
            if key not in all_derived:
                all_derived[key] = []
            vals = derived[key]
            if isinstance(vals, np.ndarray):
                all_derived[key].append(vals)
            else:
                all_derived[key].append(np.full(n, vals))

    # Concatenate
    combined_samples = np.concatenate(all_samples, axis=0)

    combined_derived = {}
    for key in derived_keys:
        combined_derived[key] = np.concatenate(all_derived[key])
    combined_derived["model"] = np.array(all_model_labels)

    # Grid k has total probability weights[k], which is shared equally
    # among its n_samples[k] posterior draws.
    sample_weights = np.repeat(weights / n_samples, n_samples)

    cdf = np.cumsum(sample_weights)
    cdf /= cdf[-1]
    cdf[-1] = 1.0

    # Systematic resampling at evenly spaced positions with random offset.
    positions = (rng.random() + np.arange(total_samples)) / total_samples
    resample_idx = np.searchsorted(cdf, positions, side="right")
    rng.shuffle(resample_idx)

    # Preserve alignment between parameters, derived quantities, and labels.
    combined_samples = combined_samples[resample_idx]
    combined_derived = {
        key: values[resample_idx]
        for key, values in combined_derived.items()
    }

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
