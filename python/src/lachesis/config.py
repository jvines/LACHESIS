"""Paths and constants."""

import os
from pathlib import Path

# Default data directory: LACHESIS/data/ (overridable via env var)
_default_data = Path(__file__).resolve().parents[3] / "data"
DATA_DIR = Path(os.environ.get("LACHESIS_DATA", _default_data))
MIST_DIR = DATA_DIR / "mist"
MIST_RAW_DIR = MIST_DIR / "raw"
MIST_GRID_DIR = MIST_DIR / "grids"
PARSEC_DIR = DATA_DIR / "parsec" / "raw"
PARSEC_GRID_DIR = DATA_DIR / "parsec" / "grids"

MIST_BASE_URL = "https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2"
MIST_BC_URL = "https://waps.cfa.harvard.edu/MIST/BC_tables"

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
