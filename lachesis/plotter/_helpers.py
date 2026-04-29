"""Pure helper functions used by the plotter API.

Extracted from the original monolithic plotter.py so the ISOPlotter class
in api.py is not the only home for utility logic. Importable directly for
unit tests.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.stats import gaussian_kde

from lachesis.bma import BMAResult

from ._constants import _PARAM_LABELS


def _is_bma(result):
    """Check whether *result* is a BMAResult."""
    return isinstance(result, BMAResult)


def _extract_from(samples, derived, name):
    """Extract a named parameter from a (samples, derived) pair."""
    if name == "age_gyr":
        return 10.0 ** samples[:, 1] / 1e9
    if name == "[Fe/H]":
        return samples[:, 2]
    if name == "eep":
        return samples[:, 0]
    if name == "log_age":
        return samples[:, 1]
    if name == "distance" and samples.shape[1] >= 4:
        return samples[:, 3]
    if name == "Av" and samples.shape[1] >= 5:
        return samples[:, 4]
    if name in derived:
        return derived[name]
    _aliases = {
        "log_L": "logL", "logL": "log_L",
        "log_g": "logg", "logg": "log_g",
        "log_R": "logR", "logR": "log_R",
        "mass": "star_mass", "star_mass": "mass",
    }
    alt = _aliases.get(name)
    if alt and alt in derived:
        return derived[alt]
    raise KeyError(f"Parameter '{name}' not found in result")


def _extract_param(result, name):
    """Extract a named parameter array from a result dict or BMAResult.

    Operates on the combined (BMA-weighted) samples/derived when given a
    BMAResult. For per-grid raw samples use ``_extract_from`` directly.
    """
    if _is_bma(result):
        samples = result.samples
        derived = result.derived
    else:
        samples = result["samples"]
        derived = result["derived"]
    return _extract_from(samples, derived, name)


def _get_model_labels(result):
    """Return per-sample model label array if BMA, else None."""
    if _is_bma(result):
        return result.derived.get("model")
    return None


def _label_for(name):
    """Return LaTeX label for a parameter name."""
    return _PARAM_LABELS.get(name, name)


def _percentile_str(arr):
    """Return (median, lo_err, hi_err) from 15.87/50/84.13 percentiles."""
    lo, med, hi = np.percentile(arr, [15.87, 50, 84.13])
    return med, med - lo, hi - med


def _kde_on_hist(ax, data, color, bins=40, alpha=0.3, density=True,
                 label=None, histtype="stepfilled", kde_lw=2):
    """Histogram + gaussian_kde overlay.  Returns (n, bins, kde_line)."""
    finite = data[np.isfinite(data)]
    if len(finite) < 2:
        return None, None, None
    n, bin_edges, _ = ax.hist(
        finite, bins=bins, color=color, alpha=alpha,
        density=density, histtype=histtype, label=label,
        edgecolor="black", linewidth=0.3,
    )
    try:
        kde = gaussian_kde(finite)
        xx = np.linspace(bin_edges[0], bin_edges[-1], 300)
        line, = ax.plot(xx, kde(xx), color=color, lw=kde_lw, alpha=1)
    except np.linalg.LinAlgError:
        line = None
    return n, bin_edges, line


def _apply_style(ax, fontname, fontsize, tick_labelsize):
    """Apply ARIADNE-style font and tick configuration to an Axes."""
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.tick_params(axis="both", which="minor", labelsize=tick_labelsize)
    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)


def _make_transparent_cmap(hex_color):
    """Colormap from transparent to *hex_color*."""
    rgba = to_rgba(hex_color)
    colors = [
        (rgba[0], rgba[1], rgba[2], 0.0),
        (rgba[0], rgba[1], rgba[2], 0.7),
    ]
    return LinearSegmentedColormap.from_list("custom", colors, N=128)


def _finalize(fig, filename, dpi=150):
    """Save (PDF or PNG) or return; closes the figure when saving.

    Without the explicit close, batch runs over many stars accumulate
    matplotlib figures and trip the 20-figure RuntimeWarning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except (ValueError, RuntimeError):
            pass
    if filename is not None:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
