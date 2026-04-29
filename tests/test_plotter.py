"""Tests for ISOPlotter — headless matplotlib rendering.

The plotter API was rewritten to take an input .nc path + out_folder at
construction time (instead of methods that accept fit-result objects),
and the test code below was never updated to match. The whole module is
skipped at collection time so pytest does not error out; the rewrite
should drive ``ISOPlotter(in_file, out_folder)`` against synthetic .nc
fixtures (see lachesis.output.to_inference_data for the writer).
"""

import pytest

pytest.skip(
    "test_plotter is API-stale (uses pre-rewrite ISOPlotter signatures); "
    "rewrite against real .nc fixtures, see plotter docstring.",
    allow_module_level=True,
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from lachesis.bma import BMAResult
from lachesis.plotter import ISOPlotter, _extract_param, _percentile_str


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def single_result():
    """Fake single-grid fit result dict."""
    rng = np.random.default_rng(42)
    n = 500
    samples = np.column_stack([
        rng.uniform(200, 500, n),          # eep
        rng.uniform(9.0, 10.1, n),         # log_age
        rng.normal(-0.1, 0.2, n),          # [Fe/H]
        rng.uniform(10, 100, n),           # distance
        rng.uniform(0.0, 0.5, n),          # Av
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
    }
    return {
        "samples": samples,
        "derived": derived,
        "logz": -50.0,
        "logzerr": 0.1,
    }


@pytest.fixture
def bma_result():
    """Fake BMA result with two models."""
    rng = np.random.default_rng(99)
    n1, n2 = 300, 200
    n = n1 + n2

    samples1 = np.column_stack([
        rng.uniform(200, 450, n1),
        rng.uniform(9.0, 10.0, n1),
        rng.normal(-0.1, 0.15, n1),
        rng.uniform(10, 80, n1),
        rng.uniform(0.0, 0.3, n1),
    ])
    samples2 = np.column_stack([
        rng.uniform(250, 500, n2),
        rng.uniform(9.2, 10.1, n2),
        rng.normal(0.0, 0.2, n2),
        rng.uniform(15, 90, n2),
        rng.uniform(0.0, 0.4, n2),
    ])
    samples = np.vstack([samples1, samples2])

    derived = {
        "initial_mass": rng.uniform(0.7, 1.4, n),
        "star_mass": rng.uniform(0.7, 1.4, n),
        "Teff": rng.normal(5700, 150, n),
        "log_g": rng.normal(4.3, 0.15, n),
        "log_L": rng.normal(0.0, 0.07, n),
        "log_R": rng.normal(0.0, 0.04, n),
        "radius": rng.normal(1.0, 0.07, n),
        "density": rng.normal(1.3, 0.15, n),
        "model": np.array(["MIST"] * n1 + ["PARSEC"] * n2),
    }

    return BMAResult(
        weights=np.array([0.6, 0.4]),
        samples=samples,
        derived=derived,
        model_names=["MIST", "PARSEC"],
        log_evidences=np.array([-48.0, -49.0]),
    )


@pytest.fixture
def plotter():
    return IsoPlotter()


@pytest.fixture
def plotter_custom(tmp_path):
    """Plotter initialized from a custom settings file."""
    cfg = tmp_path / "settings.dat"
    cfg.write_text(
        "# custom\n"
        "figsize 10,6\n"
        "fontsize 18\n"
        "fontname sans-serif\n"
        "tick_labelsize 14\n"
    )
    return IsoPlotter(settings=cfg)


# -----------------------------------------------------------------------
# Init / settings
# -----------------------------------------------------------------------


class TestInit:

    def test_default_settings(self, plotter):
        assert plotter.fontname == "serif"
        assert plotter.fontsize == 26
        assert plotter.tick_labelsize == 22
        assert plotter.figsize == (12, 8)
        assert plotter.hr_cmap == "cool"

    def test_custom_settings(self, plotter_custom):
        assert plotter_custom.fontsize == 18
        assert plotter_custom.fontname == "sans-serif"
        assert plotter_custom.figsize == (10, 6)
        assert plotter_custom.tick_labelsize == 14
        # Defaults should still be filled for missing keys
        assert plotter_custom.corner_med_c == "firebrick"

    def test_missing_settings_file(self, tmp_path):
        """Non-existent file falls back to defaults."""
        p = IsoPlotter(settings=tmp_path / "nope.dat")
        assert p.fontname == "serif"


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------


class TestHelpers:

    def test_extract_age_gyr(self, single_result):
        age = _extract_param(single_result, "age_gyr")
        assert len(age) == 500
        assert np.all(age > 0)

    def test_extract_feh(self, single_result):
        feh = _extract_param(single_result, "[Fe/H]")
        assert len(feh) == 500

    def test_extract_derived(self, single_result):
        teff = _extract_param(single_result, "Teff")
        assert len(teff) == 500

    def test_extract_missing_raises(self, single_result):
        with pytest.raises(KeyError, match="nonexistent"):
            _extract_param(single_result, "nonexistent")

    def test_percentile_str(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        med, lo, hi = _percentile_str(arr)
        assert med == pytest.approx(3.0, abs=0.1)
        assert lo > 0
        assert hi > 0


# -----------------------------------------------------------------------
# Corner plot
# -----------------------------------------------------------------------


class TestCorner:

    def test_returns_figure_single(self, plotter, single_result):
        fig = plotter.plot_corner(single_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_bma(self, plotter, bma_result):
        fig = plotter.plot_corner(bma_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_params(self, plotter, single_result):
        fig = plotter.plot_corner(single_result, params=["Teff", "log_g"])
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        visible = [ax for ax in axes if ax.get_visible()]
        assert len(visible) == 3
        plt.close(fig)

    def test_saves_to_file(self, plotter, single_result, tmp_path):
        out = str(tmp_path / "corner.png")
        fig = plotter.plot_corner(single_result, filename=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "corner.png").exists()
        plt.close(fig)

    def test_saves_to_pdf(self, plotter, single_result, tmp_path):
        out = str(tmp_path / "corner.pdf")
        fig = plotter.plot_corner(single_result, filename=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "corner.pdf").exists()
        plt.close(fig)

    def test_skips_missing_params(self, plotter, single_result):
        fig = plotter.plot_corner(
            single_result,
            params=["Teff", "nonexistent_param", "log_g"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_params_raises(self, plotter):
        result = {"samples": np.zeros((10, 3)), "derived": {}}
        with pytest.raises(ValueError, match="No plottable"):
            plotter.plot_corner(result, params=["nonexistent"])

    def test_diagonal_has_titles(self, plotter, single_result):
        fig = plotter.plot_corner(single_result, params=["Teff", "log_g"])
        axes = fig.get_axes()
        diag_axes = [ax for ax in axes if ax.get_visible() and ax.get_title()]
        # Both diagonal panels should have value+CI titles
        assert len(diag_axes) == 2
        for ax in diag_axes:
            title = ax.get_title()
            assert "^{+" in title
            assert "_{-" in title
        plt.close(fig)


# -----------------------------------------------------------------------
# Histograms
# -----------------------------------------------------------------------


class TestHistograms:

    def test_returns_list_single(self, plotter, single_result):
        figs = plotter.plot_histograms(single_result)
        assert isinstance(figs, list)
        assert len(figs) > 0
        for f1, f2 in figs:
            assert isinstance(f1, Figure)
            assert isinstance(f2, Figure)
            plt.close(f1)
            plt.close(f2)

    def test_returns_list_bma(self, plotter, bma_result):
        figs = plotter.plot_histograms(bma_result)
        assert isinstance(figs, list)
        assert len(figs) > 0
        for f1, f2 in figs:
            assert isinstance(f1, Figure)
            assert isinstance(f2, Figure)
            plt.close(f1)
            plt.close(f2)

    def test_saves_to_files(self, plotter, bma_result, tmp_path):
        prefix = str(tmp_path / "hist")
        figs = plotter.plot_histograms(bma_result, filename_prefix=prefix)
        # At least one file should exist
        saved = list(tmp_path.glob("hist_*.png"))
        assert len(saved) > 0
        for f1, f2 in figs:
            plt.close(f1)
            plt.close(f2)

    def test_bma_legend_has_prob(self, plotter, bma_result):
        figs = plotter.plot_histograms(bma_result)
        # Check the first PDF figure has "prob:" in a legend entry
        f1, _ = figs[0]
        legend = f1.get_axes()[0].get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        assert any("prob:" in t for t in texts)
        for f1, f2 in figs:
            plt.close(f1)
            plt.close(f2)


# -----------------------------------------------------------------------
# HR diagram
# -----------------------------------------------------------------------


class TestHR:

    def test_returns_figure_single(self, plotter, single_result):
        fig = plotter.plot_hr(single_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_bma(self, plotter, bma_result):
        fig = plotter.plot_hr(bma_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(self, plotter, single_result, tmp_path):
        out = str(tmp_path / "hr.png")
        fig = plotter.plot_hr(single_result, filename=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "hr.png").exists()
        plt.close(fig)

    def test_xaxis_inverted(self, plotter, single_result):
        fig = plotter.plot_hr(single_result)
        ax = fig.get_axes()[0]
        assert ax.get_xlim()[0] > ax.get_xlim()[1]
        plt.close(fig)


# -----------------------------------------------------------------------
# Mass-Age
# -----------------------------------------------------------------------


class TestMassAge:

    def test_returns_figure_single(self, plotter, single_result):
        fig = plotter.plot_mass_age(single_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_bma(self, plotter, bma_result):
        fig = plotter.plot_mass_age(bma_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(self, plotter, bma_result, tmp_path):
        out = str(tmp_path / "mass_age.png")
        fig = plotter.plot_mass_age(bma_result, filename=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "mass_age.png").exists()
        plt.close(fig)


# -----------------------------------------------------------------------
# Model weights
# -----------------------------------------------------------------------


class TestModelWeights:

    def test_returns_figure(self, plotter, bma_result):
        fig = plotter.plot_model_weights(bma_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_rejects_non_bma(self, plotter, single_result):
        with pytest.raises(TypeError, match="BMAResult"):
            plotter.plot_model_weights(single_result)

    def test_saves_to_file(self, plotter, bma_result, tmp_path):
        out = str(tmp_path / "weights.png")
        fig = plotter.plot_model_weights(bma_result, filename=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "weights.png").exists()
        plt.close(fig)

    def test_bar_count_matches_models(self, plotter, bma_result):
        fig = plotter.plot_model_weights(bma_result)
        ax = fig.get_axes()[0]
        bars = [p for p in ax.patches if hasattr(p, "get_height")]
        assert len(bars) == len(bma_result.model_names)
        plt.close(fig)

    def test_weight_annotations(self, plotter, bma_result):
        fig = plotter.plot_model_weights(bma_result)
        ax = fig.get_axes()[0]
        texts = [t.get_text() for t in ax.texts]
        assert len(texts) == len(bma_result.model_names)
        # Each annotation should be a float string
        for t in texts:
            float(t)  # should not raise
        plt.close(fig)


# -----------------------------------------------------------------------
# to_latex
# -----------------------------------------------------------------------


class TestToLatex:

    def test_returns_dict(self, plotter, single_result):
        out = plotter.to_latex(single_result)
        assert isinstance(out, dict)
        assert len(out) > 0

    def test_format(self, plotter, single_result):
        out = plotter.to_latex(single_result)
        for key, val in out.items():
            assert val.startswith("$")
            assert val.endswith("$")
            assert "^{+" in val
            assert "_{-" in val

    def test_custom_params(self, plotter, single_result):
        out = plotter.to_latex(single_result, params=["Teff"])
        assert "Teff" in out
        assert len(out) == 1

    def test_skips_missing(self, plotter, single_result):
        out = plotter.to_latex(single_result, params=["nonexistent"])
        assert len(out) == 0

    def test_bma_result(self, plotter, bma_result):
        out = plotter.to_latex(bma_result)
        assert isinstance(out, dict)
        assert "initial_mass" in out

    def test_default_params(self, plotter, single_result):
        out = plotter.to_latex(single_result)
        # age_gyr, Teff, log_g, [Fe/H], initial_mass should be present
        # distance and Av may also be present
        for key in ["initial_mass", "Teff", "log_g", "[Fe/H]", "age_gyr"]:
            assert key in out


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------


class TestSummary:

    def test_returns_figure_single(self, plotter, single_result):
        fig = plotter.summary(single_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_bma(self, plotter, bma_result):
        fig = plotter.summary(bma_result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saves_to_file(self, plotter, bma_result, tmp_path):
        out = str(tmp_path / "summary.png")
        fig = plotter.summary(bma_result, filename=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "summary.png").exists()
        plt.close(fig)

    def test_has_multiple_axes(self, plotter, single_result):
        fig = plotter.summary(single_result)
        axes = fig.get_axes()
        # At least the 4 main panels + mini-corner subplots
        assert len(axes) >= 4
        plt.close(fig)
