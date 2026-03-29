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
DARTMOUTH_DIR = DATA_DIR / "dartmouth"
DARTMOUTH_RAW_DIR = DARTMOUTH_DIR / "raw"
DARTMOUTH_GRID_DIR = DARTMOUTH_DIR / "grids"
BASTI_DIR = DATA_DIR / "basti"
BASTI_RAW_DIR = BASTI_DIR / "raw"
BASTI_GRID_DIR = BASTI_DIR / "grids"
YAPSI_DIR = DATA_DIR / "yapsi"
YAPSI_RAW_DIR = YAPSI_DIR / "raw"
YAPSI_GRID_DIR = YAPSI_DIR / "grids"
GENEVA_DIR = DATA_DIR / "geneva"
GENEVA_RAW_DIR = GENEVA_DIR / "raw"
GENEVA_GRID_DIR = GENEVA_DIR / "grids"
BHAC15_DIR = DATA_DIR / "bhac15"
BHAC15_RAW_DIR = BHAC15_DIR / "raw"
BHAC15_GRID_DIR = BHAC15_DIR / "grids"
STAREVOL_DIR = DATA_DIR / "starevol"
STAREVOL_RAW_DIR = STAREVOL_DIR / "raw"
STAREVOL_GRID_DIR = STAREVOL_DIR / "grids"

MIST_BASE_URL = "https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2"
MIST_BC_URL = "https://waps.cfa.harvard.edu/MIST/BC_tables"
DARTMOUTH_BASE_URL = "https://rcweb.dartmouth.edu/stellar"

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
