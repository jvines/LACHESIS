"""Paths and constants."""

import os
from pathlib import Path

# Package data directories
_PACKAGE_DIR = Path(__file__).resolve().parent
DATAFILES_DIR = _PACKAGE_DIR / "Datafiles"
BC_DIR = Path(os.environ.get("LACHESIS_BC_DIR", DATAFILES_DIR / "BC_tables"))


def _resolve_grid_dir() -> Path:
    """LACHESIS_GRID_DIR takes precedence over the lachesis_grids package."""
    env_dir = os.environ.get("LACHESIS_GRID_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        from lachesis_grids import grid_path as _grid_path
    except ImportError as exc:
        raise ImportError(
            "Isochrone grids not found: install the 'lachesis-grids' package "
            "(a dependency of astroLACHESIS) or set LACHESIS_GRID_DIR to a grid "
            "directory."
        ) from exc
    return _grid_path("mist_v1.2_vvcrit0.4.h5").parent


GRID_DIR = _resolve_grid_dir()

# MIST EEP phase boundaries
EEP_PHASES = {
    "PreMS": 1,
    "ZAMS": 202,
    "IAMS": 353,
    "TAMS": 454,
    "RGBTip": 605,
    "ZACHeB": 631,
    "TAHeB": 707,
    "TPAGB": 808,
}
