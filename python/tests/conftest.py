"""Test fixtures for LACHESIS."""

from pathlib import Path

import pytest

TEST_DATA = Path(__file__).parent / "data"
SAMPLE_ISO = TEST_DATA / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso"
FULL_ISO = TEST_DATA / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_full.iso"


@pytest.fixture
def sample_iso_path():
    """Path to a real MIST .iso file ([Fe/H]=0.00, v/vcrit=0.4, basic 25-col)."""
    assert SAMPLE_ISO.exists(), f"Test data not found: {SAMPLE_ISO}"
    return SAMPLE_ISO


@pytest.fixture(scope="session")
def full_iso_path():
    """Path to a real MIST .iso file ([Fe/H]=0.00, v/vcrit=0.4, full 79-col)."""
    assert FULL_ISO.exists(), f"Test data not found: {FULL_ISO}"
    return FULL_ISO
