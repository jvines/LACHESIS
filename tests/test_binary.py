"""Tests for binary star model, TDD."""

from pathlib import Path

import numpy as np
import pytest

from lachesis.bc import BCTable
from lachesis.binary import binary_log_likelihood, binary_apparent_mags
from lachesis.grid.mist import MISTModelGrid
from lachesis.interp import GridInterpolator
from lachesis.sampler import IsochroneFitter

import os

from tests.conftest import mist_h5_path

FULL_GRID_H5 = mist_h5_path()
BC_DIR = os.environ.get("LACHESIS_BC_DIR", "/tmp")


@pytest.fixture(scope="module")
def interp():
    if FULL_GRID_H5 is None:
        pytest.skip("MIST grid not available")
    mg = MISTModelGrid.from_hdf5(FULL_GRID_H5)
    return GridInterpolator(mg)


@pytest.fixture(scope="module")
def bc():
    try:
        return BCTable(BC_DIR, system="UBVRIplus")
    except FileNotFoundError:
        pytest.skip("BC tables not extracted")


class TestBinaryLikelihood:

    def test_returns_finite(self, interp, bc):
        """Valid binary point should give finite lnL."""
        observed = {"Gaia_G_EDR3": 10.0, "Gaia_BP_EDR3": 10.5}
        uncertainties = {"Gaia_G_EDR3": 0.02, "Gaia_BP_EDR3": 0.02}
        lnl = binary_log_likelihood(
            interp, bc=bc,
            eep_primary=350.0, eep_secondary=300.0,
            log_age=9.5, feh=0.0,
            distance=100.0, av=0.1,
            observed=observed, uncertainties=uncertainties,
        )
        assert np.isfinite(lnl)

    def test_oob_returns_neginf(self, interp, bc):
        """Out-of-bounds secondary should give -inf."""
        observed = {"Gaia_G_EDR3": 10.0}
        uncertainties = {"Gaia_G_EDR3": 0.02}
        lnl = binary_log_likelihood(
            interp, bc=bc,
            eep_primary=350.0, eep_secondary=9999.0,
            log_age=9.5, feh=0.0,
            distance=100.0, av=0.1,
            observed=observed, uncertainties=uncertainties,
        )
        assert lnl == -np.inf

    def test_secondary_makes_brighter(self, interp, bc):
        """Adding a companion should make the system brighter (lower mag)."""
        mag_single = binary_apparent_mags(
            interp, bc=bc,
            eep_primary=350.0, eep_secondary=None,
            log_age=9.5, feh=0.0,
            distance=100.0, av=0.1,
        )
        mag_binary = binary_apparent_mags(
            interp, bc=bc,
            eep_primary=350.0, eep_secondary=300.0,
            log_age=9.5, feh=0.0,
            distance=100.0, av=0.1,
        )
        if mag_single is not None and mag_binary is not None:
            # Binary should be brighter (lower magnitude) in every band
            for band in mag_single:
                if np.isfinite(mag_single[band]) and np.isfinite(mag_binary[band]):
                    assert mag_binary[band] < mag_single[band]

    def test_equal_components_0_75_mag_brighter(self, interp, bc):
        """Equal-mass binary should be ~0.75 mag brighter than single."""
        mag_single = binary_apparent_mags(
            interp, bc=bc,
            eep_primary=350.0, eep_secondary=None,
            log_age=9.5, feh=0.0,
            distance=100.0, av=0.0,
        )
        mag_binary = binary_apparent_mags(
            interp, bc=bc,
            eep_primary=350.0, eep_secondary=350.0,
            log_age=9.5, feh=0.0,
            distance=100.0, av=0.0,
        )
        if mag_single is not None and mag_binary is not None:
            for band in ["Bessell_V", "2MASS_J", "Gaia_G_EDR3"]:
                if np.isfinite(mag_single[band]) and np.isfinite(mag_binary[band]):
                    diff = mag_single[band] - mag_binary[band]
                    # 2x flux -> -2.5*log10(2) = 0.7526 mag brighter
                    assert diff == pytest.approx(0.7526, abs=0.01)

    def test_spectro_binary_works(self, interp):
        """Binary likelihood should work with spectroscopic observables too."""
        observed = {"log_Teff": 3.76, "log_g": 4.44}
        uncertainties = {"log_Teff": 0.01, "log_g": 0.1}
        # For spectroscopic, we just use the primary's properties
        lnl = binary_log_likelihood(
            interp, bc=None,
            eep_primary=350.0, eep_secondary=300.0,
            log_age=9.5, feh=0.0,
            distance=None, av=None,
            observed=observed, uncertainties=uncertainties,
        )
        assert np.isfinite(lnl)


class TestBinaryFitter:

    @pytest.fixture
    def binary_fitter(self, interp, bc):
        return IsochroneFitter(
            interp=interp,
            eep_range=(200, 808),
            age_range=(8.0, 10.3),
            feh_range=(-1.0, 0.5),
            bc_table=bc,
            distance_range=(1.0, 1000.0),
            av_range=(0.0, 1.0),
            binary=True,
        )

    def test_binary_has_6_params(self, binary_fitter):
        assert binary_fitter.prior.ndim == 6
        assert "eep_secondary" in binary_fitter.prior.param_names

    def test_binary_fit_runs(self, binary_fitter):
        observed = {
            "Gaia_G_EDR3": 10.0,
            "Gaia_BP_EDR3": 10.5,
            "Gaia_RP_EDR3": 9.3,
        }
        uncertainties = {k: 0.05 for k in observed}
        result = binary_fitter.fit(
            observed=observed,
            uncertainties=uncertainties,
            nlive=50,
            dlogz=1.0,
        )
        assert result["samples"].shape[1] == 6
        assert np.isfinite(result["logz"])
