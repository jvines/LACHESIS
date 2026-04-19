"""Paths and constants."""

from pathlib import Path

# Package data directories
_PACKAGE_DIR = Path(__file__).resolve().parent
DATAFILES_DIR = _PACKAGE_DIR / "Datafiles"
BC_DIR = DATAFILES_DIR / "BC_tables"

# Grids come from the lachesis-grids package
try:
    from lachesis_grids import grid_path as _grid_path
    GRID_DIR = _grid_path("mist_v1.2_vvcrit0.4.h5").parent
except ImportError:
    # Fallback to local Datafiles for development
    GRID_DIR = DATAFILES_DIR / "grids"

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
