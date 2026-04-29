"""Tests for photometric fitting with distance and extinction — TDD."""

from pathlib import Path

import numpy as np
import pytest

from lachesis.bc import BCTable
from lachesis.grid.mist import MISTModelGrid
from lachesis.interp import GridInterpolator
from lachesis.sampler import IsochroneFitter

import os

from tests.conftest import mist_h5_path

FULL_GRID_H5 = mist_h5_path()
BC_DIR = os.environ.get("LACHESIS_BC_DIR", "/tmp")


@pytest.fixture(scope="module")
def phot_fitter():
    if FULL_GRID_H5 is None:
        pytest.skip("MIST grid not available")
    mg = MISTModelGrid.from_hdf5(FULL_GRID_H5)
    interp = GridInterpolator(mg)
    try:
        bc = BCTable(BC_DIR, system="UBVRIplus")
    except FileNotFoundError:
        pytest.skip("BC tables not extracted")
    return IsochroneFitter(
        interp=interp,
        eep_range=(200, 808),
        age_range=(8.0, 10.3),
        feh_range=(-1.0, 0.5),
        bc_table=bc,
        distance_range=(1.0, 1000.0),
        av_range=(0.0, 1.0),
    )


class TestPhotometricFitting:

    def test_fitter_has_5_params(self, phot_fitter):
        """With distance and Av, we have 5 free parameters."""
        assert phot_fitter.prior.ndim == 5
        assert phot_fitter.prior.param_names == [
            "eep", "log_age", "feh", "distance", "av"
        ]

    def test_fit_with_magnitudes(self, phot_fitter):
        """Fit using photometric magnitudes as observables."""
        # Solar-ish star at ~100 pc
        observed = {
            "Gaia_G_EDR3": 10.0,
            "Gaia_BP_EDR3": 10.5,
            "Gaia_RP_EDR3": 9.3,
        }
        uncertainties = {
            "Gaia_G_EDR3": 0.02,
            "Gaia_BP_EDR3": 0.02,
            "Gaia_RP_EDR3": 0.02,
        }
        result = phot_fitter.fit(
            observed=observed,
            uncertainties=uncertainties,
            nlive=50,
            dlogz=1.0,
        )
        assert "samples" in result
        assert result["samples"].shape[1] == 5  # eep, age, feh, distance, av
        assert np.isfinite(result["logz"])

    def test_fit_mixed_spectro_and_phot(self, phot_fitter):
        """Can mix spectroscopic and photometric observables."""
        observed = {
            "log_Teff": 3.76,
            "Gaia_G_EDR3": 10.0,
        }
        uncertainties = {
            "log_Teff": 0.01,
            "Gaia_G_EDR3": 0.02,
        }
        result = phot_fitter.fit(
            observed=observed,
            uncertainties=uncertainties,
            nlive=50,
            dlogz=1.0,
        )
        assert np.isfinite(result["logz"])

    def test_distance_posterior_reasonable(self, phot_fitter):
        """Distance posterior should be positive."""
        observed = {
            "Gaia_G_EDR3": 10.0,
            "Gaia_BP_EDR3": 10.5,
            "Gaia_RP_EDR3": 9.3,
        }
        uncertainties = {k: 0.02 for k in observed}
        result = phot_fitter.fit(
            observed=observed,
            uncertainties=uncertainties,
            nlive=50,
            dlogz=1.0,
        )
        distances = result["samples"][:, 3]
        assert np.all(distances > 0)
