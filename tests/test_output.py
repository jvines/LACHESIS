"""Tests for arviz InferenceData output — TDD."""

from pathlib import Path

import numpy as np
import pytest

from lachesis.grid.mist import MISTModelGrid
from lachesis.interp import GridInterpolator
from lachesis.sampler import IsochroneFitter

from tests.conftest import mist_h5_path

FULL_GRID_H5 = mist_h5_path()


@pytest.fixture(scope="module")
def fit_result():
    if FULL_GRID_H5 is None:
        pytest.skip("MIST grid not available; install lachesis-grids or set LACHESIS_GRID_DIR")
    mg = MISTModelGrid.from_hdf5(FULL_GRID_H5)
    interp = GridInterpolator(mg)
    fitter = IsochroneFitter(
        interp=interp,
        eep_range=(200, 808),
        age_range=(8.0, 10.3),
        feh_range=(-1.0, 0.5),
    )
    return fitter.fit(
        observed={"log_Teff": 3.76, "log_g": 4.44},
        uncertainties={"log_Teff": 0.01, "log_g": 0.1},
        nlive=50,
        dlogz=1.0,
    )


class TestArvizOutput:

    def test_to_inference_data(self, fit_result):
        from lachesis.output import to_inference_data
        idata = to_inference_data(fit_result)
        assert idata is not None

    def test_has_posterior_group(self, fit_result):
        from lachesis.output import to_inference_data
        idata = to_inference_data(fit_result)
        assert hasattr(idata, "posterior")
        # Free parameters
        assert "eep" in idata.posterior
        assert "log_age" in idata.posterior
        assert "feh" in idata.posterior

    def test_has_derived_in_posterior(self, fit_result):
        from lachesis.output import to_inference_data
        idata = to_inference_data(fit_result)
        # Note: output renames log_g -> logg / log_L -> logL on disk to
        # match the ARIADNE / arviz ecosystem convention.
        for key in ["initial_mass", "Teff", "logg", "radius"]:
            assert key in idata.posterior, f"Missing {key} in posterior"

    def test_save_and_load_nc(self, fit_result, tmp_path):
        from lachesis.output import to_inference_data
        import arviz as az

        idata = to_inference_data(fit_result)
        nc_path = tmp_path / "test_result.nc"
        idata.to_netcdf(str(nc_path))
        assert nc_path.exists()

        # Reload
        idata2 = az.from_netcdf(str(nc_path))
        assert "eep" in idata2.posterior
        assert "initial_mass" in idata2.posterior

    def test_log_evidence_in_attrs(self, fit_result):
        from lachesis.output import to_inference_data
        idata = to_inference_data(fit_result)
        # log_evidence is stored on the InferenceData attrs (and per-grid
        # in constant_data for BMA), not on a sample_stats group.
        assert "log_evidence" in idata.attrs
