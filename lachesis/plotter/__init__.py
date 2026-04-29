"""Plotting for LACHESIS isochrone fitting results.

ARIADNE-style plots: corner, HR diagram, BMA histograms, mass-age,
model weights, summary panels, and LaTeX table output.

The package is split into:
- ``_constants``: module-level constants (palettes, default param sets,
  LaTeX labels, settings-file path).
- ``_helpers``: pure helper functions (extraction, label lookup,
  percentile summaries, KDE-on-histogram, figure finalisation).
- ``_api``: the ``ISOPlotter`` class.

Public surface stays ``from lachesis.plotter import ISOPlotter``; the
helpers and constants are also re-exported for unit-test access.
"""

__all__ = ["ISOPlotter"]

from ._api import ISOPlotter
from ._helpers import (
    _apply_style,
    _extract_from,
    _extract_param,
    _finalize,
    _get_model_labels,
    _is_bma,
    _kde_on_hist,
    _label_for,
    _make_transparent_cmap,
    _percentile_str,
)
from ._constants import (
    _DEFAULT_CORNER_PARAMS,
    _DEFAULT_HIST_PARAMS,
    _DEFAULT_LATEX_PARAMS,
    _MODEL_COLORS,
    _PARAM_LABELS,
    _SETTINGS_FILE,
)
