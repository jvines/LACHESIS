"""Tests for arviz InferenceData output, TDD."""

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


class TestSaveSummaryDat:
    """The v0.0.5 _BMA.dat files were missing Age, [Fe/H], Distance, Av,
    EEP rows because Fitter._save_bma called save_summary_dat without
    param_names; output.py's param-list block is gated on `if param_names:`
    and silently skipped them. These tests guard both paths.
    """

    def test_writes_sampled_param_rows_when_given_param_names(self, fit_result, tmp_path):
        from lachesis.output import save_summary_dat
        out = tmp_path / "summary.dat"
        param_names = fit_result["param_names"]
        save_summary_dat(str(out), fit_result, param_names=param_names)

        text = out.read_text()
        # Derived rows always present
        assert "Mass(Msun)" in text
        assert "Teff(K)" in text
        assert "Radius(Rsun)" in text
        # Sampled-param rows that the v0.0.5 .dat lost
        assert "Age(Gyr)" in text, "Age row dropped, log_age->Age conversion broken"
        assert "[Fe/H]" in text, "[Fe/H] row missing"
        # Distance / Av only present if those params were sampled in the
        # particular fit. The basic fit_result fixture only fits eep,
        # log_age, feh, so just confirm EEP shows up.
        assert "EEP" in text, "EEP row missing"

    def test_no_param_rows_when_param_names_unavailable(self, fit_result, tmp_path):
        # Prior to the v0.0.5 fix, fitter.py constructed a result dict for
        # save_summary_dat that lacked the param_names key, AND passed no
        # param_names kwarg, so output.py's `param_names = result.get(...)`
        # fallback resolved to None and silently dropped the sampled rows.
        # Reproduce that exact failure shape.
        from lachesis.output import save_summary_dat
        result_no_pn = {k: v for k, v in fit_result.items() if k != "param_names"}
        out = tmp_path / "summary_no_pn.dat"
        save_summary_dat(str(out), result_no_pn, param_names=None)
        text = out.read_text()
        assert "Mass(Msun)" in text  # derived rows still written
        assert "[Fe/H]" not in text  # sampled rows skipped, this is the
                                     # v0.0.5 _BMA.dat bug we're guarding
        assert "EEP" not in text


class TestFitterSaveBMAPassesParamNames:
    """End-to-end: Fitter._save_bma must pass param_names to save_summary_dat
    so the human-readable .dat retains all rows the .nc has. Source-inspection
    test rather than a full BMA run because that costs ~30 s.
    """

    def test_save_bma_threads_param_names(self):
        import inspect
        from lachesis.fitter import Fitter
        src = inspect.getsource(Fitter._save_bma)
        assert "param_names=" in src, (
            "Fitter._save_bma must explicitly pass param_names= to "
            "save_summary_dat or the .dat loses sampled-parameter rows"
        )
