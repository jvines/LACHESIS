"""Tests for ISOPlotter — headless matplotlib rendering.

The plotter loads results from a saved .nc file at construction time
and exposes plot_corner / plot_histograms / plot_hr / plot_mass_age /
plot_model_weights / summary / to_latex methods that take no result
argument. These tests build a synthetic .nc fixture via
lachesis.output.to_inference_data and exercise the public API.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lachesis.bma import BMAResult, bayesian_model_average
from lachesis.output import to_inference_data
from lachesis.plotter import ISOPlotter, _extract_param, _percentile_str


# ── Fixtures ────────────────────────────────────────────────────────


def _fake_single_result(seed: int = 42, n: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    samples = np.column_stack([
        rng.uniform(200, 500, n),       # eep
        rng.uniform(9.0, 10.1, n),      # log_age
        rng.normal(-0.1, 0.2, n),       # feh
        rng.uniform(10, 100, n),        # distance
        rng.uniform(0.0, 0.5, n),       # Av
    ])
    derived = {
        "initial_mass": rng.uniform(0.8, 1.3, n),
        "star_mass": rng.uniform(0.8, 1.3, n),
        "Teff": rng.normal(5800, 100, n),
        "log_g": rng.normal(4.4, 0.1, n),
        "log_L": rng.normal(0.0, 0.05, n),
        "log_R": rng.normal(0.0, 0.03, n),
        "radius": rng.normal(1.0, 0.05, n),
        "density": rng.normal(1.4, 0.1, n),
        "log_Teff": np.log10(rng.normal(5800, 100, n)),
        "Mbol": rng.normal(4.7, 0.05, n),
    }
    return {
        "samples": samples,
        "derived": derived,
        "logz": -50.0,
        "logzerr": 0.1,
        "param_names": ["eep", "log_age", "feh", "distance", "Av"],
    }


def _fake_bma_result(seed: int = 99, n_per: int = 200):
    """Return a BMAResult plus the per_grid_results dict for to_inference_data."""
    rng = np.random.default_rng(seed)
    grids = ["MIST", "PARSEC"]
    per_grid = {}
    fit_results = []
    for i, name in enumerate(grids):
        samples = np.column_stack([
            rng.uniform(200 + 50 * i, 500, n_per),
            rng.uniform(9.0 + 0.1 * i, 10.1, n_per),
            rng.normal(-0.1 + 0.05 * i, 0.2, n_per),
            rng.uniform(10, 100, n_per),
            rng.uniform(0.0, 0.5, n_per),
        ])
        derived = {
            "initial_mass": rng.uniform(0.8, 1.3, n_per),
            "star_mass": rng.uniform(0.8, 1.3, n_per),
            "Teff": rng.normal(5700 + 50 * i, 100, n_per),
            "log_g": rng.normal(4.3, 0.15, n_per),
            "log_L": rng.normal(0.0, 0.07, n_per),
            "log_R": rng.normal(0.0, 0.04, n_per),
            "radius": rng.normal(1.0, 0.07, n_per),
            "density": rng.normal(1.3, 0.15, n_per),
            "log_Teff": np.log10(rng.normal(5700 + 50 * i, 100, n_per)),
            "Mbol": rng.normal(4.7, 0.05, n_per),
        }
        result = {
            "samples": samples,
            "derived": derived,
            "logz": -50.0 + i * 0.5,
            "logzerr": 0.1,
            "param_names": ["eep", "log_age", "feh", "distance", "Av"],
        }
        fit_results.append(result)
        per_grid[name] = result
    bma = bayesian_model_average(
        fit_results, names=grids, rng=np.random.default_rng(seed),
    )
    return bma, per_grid


@pytest.fixture
def single_nc(tmp_path):
    fit = _fake_single_result()
    idata = to_inference_data(
        fit_result=fit, grid_name="mist",
        observed={"Gaia_G_EDR3": 10.0},
        uncertainties={"Gaia_G_EDR3": 0.02},
    )
    p = tmp_path / "single.nc"
    idata.to_netcdf(str(p))
    return p


@pytest.fixture
def bma_nc(tmp_path):
    bma, per_grid = _fake_bma_result()
    idata = to_inference_data(
        bma_result=bma, per_grid_results=per_grid, grid_name="BMA",
        observed={"Gaia_G_EDR3": 10.0},
        uncertainties={"Gaia_G_EDR3": 0.02},
    )
    p = tmp_path / "bma.nc"
    idata.to_netcdf(str(p))
    return p


@pytest.fixture
def single_plotter(single_nc, tmp_path):
    return ISOPlotter(single_nc, tmp_path / "out_single")


@pytest.fixture
def bma_plotter(bma_nc, tmp_path):
    return ISOPlotter(bma_nc, tmp_path / "out_bma")


# ── Init / settings ─────────────────────────────────────────────────


class TestInit:
    def test_loads_single_nc(self, single_plotter):
        assert single_plotter.bma is False
        assert single_plotter.result is not None

    def test_loads_bma_nc(self, bma_plotter):
        assert bma_plotter.bma is True
        assert isinstance(bma_plotter.result, BMAResult)
        assert bma_plotter.result.model_names == ["MIST", "PARSEC"]

    def test_creates_out_folder(self, single_nc, tmp_path):
        out = tmp_path / "fresh_out"
        ISOPlotter(single_nc, out)
        assert out.exists()

    def test_default_style(self, single_plotter):
        assert single_plotter.fontname == "serif"
        assert single_plotter.fontsize == 26

    def test_pdf_extension(self, single_nc, tmp_path):
        p = ISOPlotter(single_nc, tmp_path / "pdf_out", pdf=True)
        assert p._ext == ".pdf"


# ── Helpers ─────────────────────────────────────────────────────────


class TestHelpers:
    def test_extract_age_gyr_from_dict(self):
        fit = _fake_single_result()
        age = _extract_param(fit, "age_gyr")
        assert len(age) == len(fit["samples"])
        assert np.all(age > 0)

    def test_extract_feh_from_dict(self):
        fit = _fake_single_result()
        feh = _extract_param(fit, "[Fe/H]")
        assert len(feh) == len(fit["samples"])

    def test_extract_missing_raises(self):
        fit = _fake_single_result()
        with pytest.raises(KeyError, match="totally_missing"):
            _extract_param(fit, "totally_missing")

    def test_extract_alias_logg(self):
        fit = _fake_single_result()
        a = _extract_param(fit, "log_g")
        d2 = {**fit["derived"], "logg": fit["derived"]["log_g"]}
        d2.pop("log_g")
        b = _extract_param({**fit, "derived": d2}, "log_g")
        assert len(a) == len(b)

    def test_percentile_str(self):
        arr = np.tile(np.arange(101.0), 10)
        med, lo, hi = _percentile_str(arr)
        assert med == pytest.approx(50.0, abs=1.0)
        assert lo > 0
        assert hi > 0


# ── Plotting smoke tests ────────────────────────────────────────────


class TestSinglePlots:
    def test_corner_single(self, single_plotter):
        single_plotter.plot_corner()
        plt.close("all")

    def test_corner_bma(self, bma_plotter):
        bma_plotter.plot_corner()
        plt.close("all")

    def test_histograms_single(self, single_plotter):
        single_plotter.plot_histograms()
        plt.close("all")

    def test_summary_single(self, single_plotter):
        single_plotter.summary()
        plt.close("all")

    def test_summary_bma(self, bma_plotter):
        bma_plotter.summary()
        plt.close("all")

    def test_to_latex_single(self, single_plotter):
        out = single_plotter.to_latex()
        # to_latex returns either a formatted string or a per-param dict.
        assert out is None or isinstance(out, (str, dict))
        plt.close("all")


# ── BMA-specific ────────────────────────────────────────────────────


class TestBMAOnly:
    def test_model_weights(self, bma_plotter):
        bma_plotter.plot_model_weights()
        plt.close("all")

    def test_per_grid_labels_round_trip(self, bma_plotter):
        # The combined BMA posterior carries per-sample model labels
        # written by output.to_inference_data; the loader pulls them
        # straight from the posterior group rather than resampling.
        labels = bma_plotter.result.derived.get("model")
        assert labels is not None
        assert set(labels.tolist()) == {"MIST", "PARSEC"}


# ── Figure-leak guard ───────────────────────────────────────────────


class TestFigureLeak:
    def test_corner_closes_on_save(self, single_plotter):
        before = len(plt.get_fignums())
        single_plotter.plot_corner()
        after = len(plt.get_fignums())
        assert after <= before + 1
        plt.close("all")
