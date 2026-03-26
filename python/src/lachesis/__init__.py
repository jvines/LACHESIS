"""LACHESIS — Isochrone fitting with Bayesian Model Averaging."""

__version__ = "0.1.0"

from lachesis.star import Star
from lachesis.fitter import Fitter

__all__ = ["Star", "Fitter", "__version__"]
