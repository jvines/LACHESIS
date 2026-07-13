"""Test fixtures for LACHESIS.

Test data lives in `tests/data/` (not shipped). Tests requiring it are
skipped cleanly when the files are absent, failing the assert at fixture
setup would silently break unrelated tests in some pytest configs.
The location can be overridden with the LACHESIS_TEST_DATA env var.
"""

import os
from pathlib import Path

import pytest

_DEFAULT_TEST_DATA = Path(__file__).parent / "data"
TEST_DATA = Path(os.environ.get("LACHESIS_TEST_DATA", _DEFAULT_TEST_DATA))
SAMPLE_ISO = TEST_DATA / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso"
FULL_ISO = TEST_DATA / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_full.iso"


@pytest.fixture
def sample_iso_path():
    """Path to a real MIST .iso file (basic 25-col)."""
    if not SAMPLE_ISO.exists():
        pytest.skip(f"Test data not found: {SAMPLE_ISO}")
    return SAMPLE_ISO


@pytest.fixture(scope="session")
def full_iso_path():
    """Path to a real MIST .iso file (full 79-col)."""
    if not FULL_ISO.exists():
        pytest.skip(f"Test data not found: {FULL_ISO}")
    return FULL_ISO


def mist_h5_path() -> Path | None:
    """Locate the MIST HDF5 grid via lachesis-grids or LACHESIS_GRID_DIR.

    Returns None when neither source provides the grid so tests can skip
    cleanly. Replaces the repo-relative `parents[2]/data/...` shortcuts
    that only resolved on the original author's machine.
    """
    try:
        from lachesis_grids import grid_path
        p = grid_path("mist_v1.2_vvcrit0.4.h5")
        if p.exists():
            return p
    except Exception:
        pass
    grid_dir = os.environ.get("LACHESIS_GRID_DIR")
    if grid_dir:
        p = Path(grid_dir) / "mist_v1.2_vvcrit0.4.h5"
        if p.exists():
            return p
    return None


def parsec_raw_dir() -> Path | None:
    """Locate raw PARSEC tables via LACHESIS_PARSEC_DIR env var."""
    raw = os.environ.get("LACHESIS_PARSEC_DIR")
    if raw:
        p = Path(raw)
        if p.exists() and list(p.glob("*.csv")):
            return p
    return None
