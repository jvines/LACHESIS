"""Paths and constants."""

from pathlib import Path

# Package data directory: lachesis/Datafiles/
_PACKAGE_DIR = Path(__file__).resolve().parent
DATAFILES_DIR = _PACKAGE_DIR / "Datafiles"
GRID_DIR = DATAFILES_DIR / "grids"
BC_DIR = DATAFILES_DIR / "BC_tables"

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
