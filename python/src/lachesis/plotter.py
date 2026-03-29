"""Plotting for LACHESIS isochrone fitting results.

ARIADNE-style plots: corner, HR diagram, BMA histograms, mass-age,
model weights, summary panels, and LaTeX table output.

All plots use matplotlib only (no seaborn, no corner package).
Style matches astroARIADNE conventions (fontname=serif, fontsize=26,
tick_labelsize=22, gaussian_kde overlays, LineCollection HR tracks).
"""

__all__ = ["IsoPlotter"]

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from lachesis.bma import BMAResult

# -----------------------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------------------

# Per-model colors (same as ARIADNE BMA palette)
_MODEL_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown",
]

# Default parameters for corner / to_latex
_DEFAULT_CORNER_PARAMS = [
    "initial_mass", "Teff", "log_g", "[Fe/H]", "age_gyr",
]

_DEFAULT_LATEX_PARAMS = [
    "initial_mass", "Teff", "log_g", "[Fe/H]", "age_gyr",
    "distance", "Av",
]

# LaTeX-formatted axis labels
_PARAM_LABELS = {
    "initial_mass": r"$M_{\mathrm{init}}$ [$M_\odot$]",
    "Teff": r"$T_{\mathrm{eff}}$ [K]",
    "log_g": r"$\log g$ [dex]",
    "[Fe/H]": r"[Fe/H] [dex]",
    "age_gyr": r"Age [Gyr]",
    "star_mass": r"$M_\star$ [$M_\odot$]",
    "log_L": r"$\log L/L_\odot$",
    "log_R": r"$\log R/R_\odot$",
    "log_Teff": r"$\log T_{\mathrm{eff}}$",
    "radius": r"$R_\star$ [$R_\odot$]",
    "density": r"$\rho$ [g cm$^{-3}$]",
    "distance": r"Distance [pc]",
    "Av": r"$A_V$ [mag]",
    "eep": "EEP",
}

# Default settings file shipped with the package
_SETTINGS_FILE = Path(__file__).parent / "plot_settings.dat"

# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------


def _is_bma(result):
    """Check whether *result* is a BMAResult."""
    return isinstance(result, BMAResult)


def _extract_param(result, name):
    """Extract a named parameter array from a result dict or BMAResult.

    Handles special cases:
      age_gyr  -> 10**(log_age) / 1e9
      [Fe/H]   -> samples column 2
      eep      -> samples column 0
      log_age  -> samples column 1
      distance -> samples column 3 (if 5-column)
      Av       -> samples column 4 (if 5-column)
      others   -> derived dict
    """
    if _is_bma(result):
        samples = result.samples
        derived = result.derived
    else:
        samples = result["samples"]
        derived = result["derived"]

    if name == "age_gyr":
        return 10.0 ** samples[:, 1] / 1e9
    if name == "[Fe/H]":
        return samples[:, 2]
    if name == "eep":
        return samples[:, 0]
    if name == "log_age":
        return samples[:, 1]
    if name == "distance" and samples.shape[1] >= 4:
        return samples[:, 3]
    if name == "Av" and samples.shape[1] >= 5:
        return samples[:, 4]
    if name in derived:
        return derived[name]
    raise KeyError(f"Parameter '{name}' not found in result")


def _get_model_labels(result):
    """Return per-sample model label array if BMA, else None."""
    if _is_bma(result):
        return result.derived.get("model")
    return None


def _label_for(name):
    """Return LaTeX label for a parameter name."""
    return _PARAM_LABELS.get(name, name)


def _percentile_str(arr):
    """Return (median, lo_err, hi_err) from 15.87/50/84.13 percentiles."""
    lo, med, hi = np.percentile(arr, [15.87, 50, 84.13])
    return med, med - lo, hi - med


def _kde_on_hist(ax, data, color, bins=40, alpha=0.3, density=True,
                 label=None, histtype="stepfilled", kde_lw=2):
    """Histogram + gaussian_kde overlay.  Returns (n, bins, kde_line)."""
    finite = data[np.isfinite(data)]
    if len(finite) < 2:
        return None, None, None
    n, bin_edges, _ = ax.hist(
        finite, bins=bins, color=color, alpha=alpha,
        density=density, histtype=histtype, label=label,
        edgecolor="black", linewidth=0.3,
    )
    try:
        kde = gaussian_kde(finite)
        xx = np.linspace(bin_edges[0], bin_edges[-1], 300)
        line, = ax.plot(xx, kde(xx), color=color, lw=kde_lw, alpha=1)
    except np.linalg.LinAlgError:
        line = None
    return n, bin_edges, line


def _apply_style(ax, fontname, fontsize, tick_labelsize):
    """Apply ARIADNE-style font and tick configuration to an Axes."""
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.tick_params(axis="both", which="minor", labelsize=tick_labelsize)
    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)


def _make_transparent_cmap(hex_color):
    """Colormap from transparent to *hex_color*."""
    rgba = to_rgba(hex_color)
    colors = [
        (rgba[0], rgba[1], rgba[2], 0.0),
        (rgba[0], rgba[1], rgba[2], 0.7),
    ]
    return LinearSegmentedColormap.from_list("custom", colors, N=128)


def _finalize(fig, filename, dpi=150):
    """Save (PDF or PNG) or show; return the figure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass
    if filename is not None:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    return fig


# -----------------------------------------------------------------------
# IsoPlotter
# -----------------------------------------------------------------------


class IsoPlotter:
    """Plotting interface for LACHESIS isochrone fitting results.

    Parameters
    ----------
    settings : str or Path, optional
        Path to a plot_settings.dat file.  Falls back to the bundled
        default that mirrors ARIADNE's style.
    """

    def __init__(self, settings=None):
        self._read_config(settings)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _read_config(self, settings_path):
        """Parse a key-value settings file and set instance attributes.

        Follows ARIADNE's ``__read_config`` convention exactly.
        """
        path = Path(settings_path) if settings_path else _SETTINGS_FILE
        if not path.exists():
            # Fall back to hard-coded defaults
            self._set_defaults()
            return
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                attr, raw = parts[0], parts[1].strip()
                if attr in ("figsize", "hr_figsize"):
                    vals = raw.split(",")
                    val = (int(vals[0]), int(vals[1]))
                elif "alpha" in attr:
                    val = float(raw)
                else:
                    try:
                        val = int(raw)
                    except ValueError:
                        val = raw
                setattr(self, attr, val)
        # Ensure all defaults exist even if the file is sparse
        self._fill_defaults()

    def _set_defaults(self):
        """Hard-coded ARIADNE-compatible defaults."""
        self.figsize = (12, 8)
        self.fontsize = 26
        self.fontname = "serif"
        self.tick_labelsize = 22
        self.corner_fontsize = 20
        self.corner_tick_fontsize = 15
        self.corner_labelpad = 15
        self.corner_med_c = "firebrick"
        self.corner_med_style = "--"
        self.corner_v_c = "lightcoral"
        self.corner_v_style = "-."
        self.hr_figsize = (12, 8)
        self.hr_cmap = "cool"
        self.hr_marker = "*"
        self.hr_color = "greenyellow"

    def _fill_defaults(self):
        """Set any missing attributes to ARIADNE defaults."""
        defaults = dict(
            figsize=(12, 8),
            fontsize=26,
            fontname="serif",
            tick_labelsize=22,
            corner_fontsize=20,
            corner_tick_fontsize=15,
            corner_labelpad=15,
            corner_med_c="firebrick",
            corner_med_style="--",
            corner_v_c="lightcoral",
            corner_v_style="-.",
            hr_figsize=(12, 8),
            hr_cmap="cool",
            hr_marker="*",
            hr_color="greenyellow",
        )
        for k, v in defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    # ------------------------------------------------------------------
    # 1.  Corner plot (from scratch — no `corner` package)
    # ------------------------------------------------------------------

    def plot_corner(self, result, params=None, filename=None):
        """Corner plot of the posterior (implemented from scratch).

        1D marginals on diagonal: histogram + KDE + median (firebrick
        dashed) + 68 % CI lines (lightcoral dash-dot).
        2D panels (lower triangle): 2D histogram + contour overlay.
        For BMA results, per-model color-coding in both 1D and 2D.

        Parameters
        ----------
        result : dict or BMAResult
        params : list of str, optional
        filename : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        if params is None:
            params = list(_DEFAULT_CORNER_PARAMS)

        # Filter to available params
        available = []
        for p in params:
            try:
                _extract_param(result, p)
                available.append(p)
            except (KeyError, IndexError):
                pass
        params = available
        n = len(params)
        if n == 0:
            raise ValueError("No plottable parameters found in result")

        data = np.column_stack([_extract_param(result, p) for p in params])
        model_labels = _get_model_labels(result)

        fig, axes = plt.subplots(
            n, n,
            figsize=(3 * n, 3 * n),
        )
        if n == 1:
            axes = np.array([[axes]])

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]

                if j > i:
                    ax.set_visible(False)
                    continue

                if i == j:
                    self._corner_1d(ax, data[:, i], params[i], model_labels)
                else:
                    self._corner_2d(ax, data[:, j], data[:, i], model_labels)

                # Axis labels — bottom row and left column only
                if i == n - 1:
                    ax.set_xlabel(
                        _label_for(params[j]),
                        fontsize=self.corner_fontsize,
                        fontname=self.fontname,
                        labelpad=self.corner_labelpad,
                    )
                else:
                    ax.set_xticklabels([])

                if j == 0 and i > 0:
                    ax.set_ylabel(
                        _label_for(params[i]),
                        fontsize=self.corner_fontsize,
                        fontname=self.fontname,
                        labelpad=self.corner_labelpad,
                    )
                elif j > 0 and i != j:
                    ax.set_yticklabels([])

                # Tick styling
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(self.corner_tick_fontsize)
                    tick.label1.set_fontname(self.fontname)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(self.corner_tick_fontsize)
                    tick.label1.set_fontname(self.fontname)

        # Bottom-right label (same as ARIADNE's explicit handling)
        if n > 1:
            ax_last = axes[-1, -1]
            ax_last.set_xlabel(
                _label_for(params[-1]),
                fontsize=self.corner_fontsize,
                fontname=self.fontname,
                labelpad=self.corner_labelpad,
            )

        return _finalize(fig, filename)

    # ------------------------------------------------------------------
    # 2.  BMA histograms
    # ------------------------------------------------------------------

    def plot_histograms(self, result, filename_prefix=None):
        """Per-parameter ARIADNE-style BMA histograms.

        For each parameter: per-model histograms with KDE overlay +
        weighted combined posterior.  Both normalized (PDF) and weighted
        (N) versions.

        Returns a list of (fig_pdf, fig_weighted) tuples.

        Parameters
        ----------
        result : dict or BMAResult
        filename_prefix : str, optional
            If given, saves ``{prefix}_{param}.png`` and
            ``{prefix}_weighted_{param}.png``.

        Returns
        -------
        list of (Figure, Figure)
        """
        params = list(_DEFAULT_CORNER_PARAMS)
        available = []
        for p in params:
            try:
                _extract_param(result, p)
                available.append(p)
            except (KeyError, IndexError):
                pass
        params = available

        model_labels = _get_model_labels(result)
        is_bma = model_labels is not None
        if is_bma:
            unique_models = list(dict.fromkeys(model_labels))
            weights_dict = dict(
                zip(result.model_names, result.weights)
            )
        else:
            unique_models = None

        figures = []
        for param in params:
            arr = _extract_param(result, param)
            label = _label_for(param)

            f1, ax1 = plt.subplots(figsize=(12, 6))
            f2, ax2 = plt.subplots(figsize=(12, 6))

            if is_bma:
                for k, model in enumerate(unique_models):
                    mask = model_labels == model
                    samp = arr[mask]
                    samp = samp[np.isfinite(samp)]
                    if len(samp) < 2:
                        continue
                    color = _MODEL_COLORS[k % len(_MODEL_COLORS)]
                    w = weights_dict.get(model, 0)
                    mlabel = f"{model} prob: {w:.3f}"

                    # Normalized (PDF)
                    _kde_on_hist(
                        ax1, samp, color, bins=20, alpha=0.3,
                        label=mlabel,
                    )
                    # Weighted (N)
                    n_w, bins_w, _ = ax2.hist(
                        samp, bins=20, alpha=0.3, label=mlabel,
                        color=color,
                        weights=np.full(len(samp), w),
                    )
                    # KDE scaled to weighted histogram amplitude
                    try:
                        kde = gaussian_kde(samp)
                        xx = np.linspace(bins_w[0], bins_w[-1], 300)
                        scale = n_w.max() / max(kde(xx).max(), 1e-30)
                        ax2.plot(xx, kde(xx) * scale, color=color,
                                 lw=2, alpha=1)
                    except np.linalg.LinAlgError:
                        pass

                # Combined weighted posterior on the PDF panel
                finite = arr[np.isfinite(arr)]
                if len(finite) >= 2:
                    _kde_on_hist(
                        ax1, finite, "tab:cyan", bins=20, alpha=0.3,
                        label="Combined", kde_lw=2,
                    )

            else:
                # Single-grid histogram
                _kde_on_hist(ax1, arr, "tab:blue", bins=20, alpha=0.3)
                _kde_on_hist(ax2, arr, "tab:blue", bins=20, alpha=0.3,
                             density=False)

            # Style both axes
            for ax, ylabel in [(ax1, "PDF"), (ax2, "N")]:
                ax.set_xlabel(label,
                              fontsize=self.fontsize,
                              fontname=self.fontname)
                ax.set_ylabel(ylabel,
                              fontsize=self.fontsize,
                              fontname=self.fontname)
                _apply_style(ax, self.fontname, self.fontsize,
                             self.tick_labelsize)
                ax.legend(loc=0, prop={"size": 16})

            # Save
            safe_name = param.replace("[", "").replace("]", "").replace("/", "_")
            if filename_prefix:
                _finalize(f1, f"{filename_prefix}_{safe_name}.png")
                _finalize(f2, f"{filename_prefix}_weighted_{safe_name}.png")
            else:
                _finalize(f1, None)
                _finalize(f2, None)
            figures.append((f1, f2))

        return figures

    # ------------------------------------------------------------------
    # 3.  HR diagram with LineCollection tracks
    # ------------------------------------------------------------------

    def plot_hr(self, result, grids=None, filename=None, n_samples=50):
        """Isochrone-track HR diagram (log Teff vs log L, inverted x-axis).

        Best-fit track as mass-colored LineCollection.  Random posterior
        draws as gray lines for uncertainty.  Star position with error
        bars.

        Parameters
        ----------
        result : dict or BMAResult
        grids : dict of {name: grid_object}, optional
            Keyed by model name.  Grid objects must expose ``_data``,
            ``_feh_values``, ``_age_values``, ``_eep_values``, and
            ``columns``.  If None, plots posterior density instead of
            isochrone tracks.
        filename : str, optional
        n_samples : int
            Number of random posterior isochrone draws (gray lines).

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=self.hr_figsize)

        # --- Extract posterior for star position ---
        log_teff_arr = _extract_param(result, "log_Teff") if "log_Teff" in (
            result.derived if _is_bma(result) else result.get("derived", {})
        ) else np.log10(_extract_param(result, "Teff"))
        log_l_arr = _extract_param(result, "log_L")

        med_lt, lo_lt, hi_lt = _percentile_str(log_teff_arr)
        med_ll, lo_ll, hi_ll = _percentile_str(log_l_arr)

        if grids is not None and len(grids) > 0:
            # Pick best grid (highest BMA weight or first)
            if _is_bma(result):
                best_idx = int(np.argmax(result.weights))
                best_name = result.model_names[best_idx]
                grid = grids.get(best_name, next(iter(grids.values())))
            else:
                grid = next(iter(grids.values()))

            # Best-fit isochrone track
            age_arr = _extract_param(result, "age_gyr")
            feh_arr = _extract_param(result, "[Fe/H]")
            med_age = np.median(age_arr)
            med_feh = np.median(feh_arr)
            log_age_bf = np.log10(med_age * 1e9)

            track = self._get_track(grid, log_age_bf, med_feh)
            if track is not None:
                logteff_t, logl_t, mass_t = track
                # Mass-colored LineCollection
                points = np.array([logteff_t, logl_t]).T.reshape(-1, 1, 2)
                segments = np.concatenate(
                    [points[:-1], points[1:]], axis=1
                )
                norm = plt.Normalize(mass_t.min(), mass_t.max())
                lc = LineCollection(
                    segments, cmap=self.hr_cmap, norm=norm, linewidths=5,
                )
                lc.set_array(mass_t)
                line = ax.add_collection(lc)
                line.zorder = 1000
                cbar = fig.colorbar(line, ax=ax, pad=0.01)
                cbar.set_label(
                    r"$M_\odot$", rotation=270,
                    fontsize=self.fontsize, fontname=self.fontname,
                    labelpad=20,
                )
                for ll in cbar.ax.yaxis.get_ticklabels():
                    ll.set_fontsize(self.tick_labelsize)

            # Random posterior draws
            n_draw = min(n_samples, len(age_arr))
            rng = np.random.default_rng()
            idx = rng.choice(len(age_arr), size=n_draw, replace=False)
            for ii in idx:
                la = np.log10(age_arr[ii] * 1e9)
                feh_i = feh_arr[ii]
                t = self._get_track(grid, la, feh_i)
                if t is not None:
                    ax.plot(t[0], t[1], color="gray", alpha=0.15,
                            lw=0.5, zorder=1)

        else:
            # No grids — plot posterior density
            ax.hist2d(
                log_teff_arr, log_l_arr, bins=60, cmap="Blues",
                density=True,
            )
            model_labels = _get_model_labels(result)
            if model_labels is not None:
                unique_models = list(dict.fromkeys(model_labels))
                for k, model in enumerate(unique_models):
                    mask = model_labels == model
                    color = _MODEL_COLORS[k % len(_MODEL_COLORS)]
                    self._plot_contour(
                        ax, log_teff_arr[mask], log_l_arr[mask],
                        color=color, label=model,
                    )
                ax.legend(fontsize=16, loc="upper left")

        # Star position
        ax.errorbar(
            med_lt, med_ll,
            xerr=[[lo_lt], [hi_lt]], yerr=[[lo_ll], [hi_ll]],
            color=self.hr_color, zorder=1001,
        )
        ax.scatter(
            med_lt, med_ll, s=350, color=self.hr_color,
            zorder=1002, edgecolors="k", marker=self.hr_marker,
        )

        ax.invert_xaxis()
        ax.set_xlabel(
            r"$\log T_{\mathrm{eff}}$",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        ax.set_ylabel(
            r"$\log L/L_\odot$",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        _apply_style(ax, self.fontname, self.fontsize, self.tick_labelsize)

        return _finalize(fig, filename)

    # ------------------------------------------------------------------
    # 4.  Mass vs Age posterior
    # ------------------------------------------------------------------

    def plot_mass_age(self, result, filename=None):
        """Mass vs Age 2D density with contours.

        For BMA results, per-model color-coding.

        Parameters
        ----------
        result : dict or BMAResult
        filename : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        mass = _extract_param(result, "initial_mass")
        age = _extract_param(result, "age_gyr")
        model_labels = _get_model_labels(result)

        fig, ax = plt.subplots(figsize=self.figsize)

        if model_labels is not None:
            unique_models = list(dict.fromkeys(model_labels))
            ax.hist2d(
                mass, age, bins=60, cmap="Greys", alpha=0.4, density=True,
            )
            for k, model in enumerate(unique_models):
                mask = model_labels == model
                color = _MODEL_COLORS[k % len(_MODEL_COLORS)]
                self._plot_contour(
                    ax, mass[mask], age[mask], color=color, label=model,
                )
            ax.legend(fontsize=16)
        else:
            ax.hist2d(mass, age, bins=60, cmap="Blues", density=True)
            self._plot_contour(ax, mass, age, color="navy")

        ax.set_xlabel(
            r"$M_{\mathrm{init}}$ [$M_\odot$]",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        ax.set_ylabel(
            r"Age [Gyr]",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        _apply_style(ax, self.fontname, self.fontsize, self.tick_labelsize)

        return _finalize(fig, filename)

    # ------------------------------------------------------------------
    # 5.  Model weights bar chart
    # ------------------------------------------------------------------

    def plot_model_weights(self, bma_result, filename=None):
        """Bar chart of BMA model weights with weight annotations.

        Parameters
        ----------
        bma_result : BMAResult
        filename : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _is_bma(bma_result):
            raise TypeError("plot_model_weights requires a BMAResult")

        names = bma_result.model_names
        weights = bma_result.weights
        n = len(names)

        fig, ax = plt.subplots(figsize=(max(6, 2 * n), 6))
        colors = [_MODEL_COLORS[k % len(_MODEL_COLORS)] for k in range(n)]
        bars = ax.bar(
            names, weights, color=colors,
            edgecolor="black", linewidth=0.5,
        )

        for bar, w in zip(bars, weights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{w:.3f}",
                ha="center", va="bottom",
                fontsize=self.corner_fontsize,
                fontname=self.fontname,
            )

        ax.set_ylabel(
            "Weight",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        ax.set_title(
            "BMA Model Weights",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        ax.set_ylim(0, min(1.0, weights.max() * 1.3))
        _apply_style(ax, self.fontname, self.fontsize, self.tick_labelsize)

        return _finalize(fig, filename)

    # ------------------------------------------------------------------
    # 6.  LaTeX table output
    # ------------------------------------------------------------------

    def to_latex(self, result, params=None):
        r"""Return a dict of LaTeX-formatted credibility intervals.

        Format: ``$value^{+hi}_{-lo}$`` using 68 % intervals.

        Parameters
        ----------
        result : dict or BMAResult
        params : list of str, optional

        Returns
        -------
        dict of {param_name: latex_string}
        """
        if params is None:
            params = list(_DEFAULT_LATEX_PARAMS)

        out = {}
        for p in params:
            try:
                arr = _extract_param(result, p)
            except (KeyError, IndexError):
                continue
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            med, lo, hi = _percentile_str(finite)
            out[p] = f"${med:.4g}^{{+{hi:.3g}}}_{{-{lo:.3g}}}$"
        return out

    # ------------------------------------------------------------------
    # 7.  Multi-panel summary
    # ------------------------------------------------------------------

    def summary(self, result, filename=None):
        """Multi-panel summary figure.

        Layout (2x2 GridSpec):
          [0,0] corner (mass, Teff, age) — mini 3x3
          [0,1] HR diagram
          [1,0] Mass-Age
          [1,1] Model weights (BMA) or [Fe/H] marginal

        Parameters
        ----------
        result : dict or BMAResult
        filename : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

        model_labels = _get_model_labels(result)

        # ------ [0,0]: Mini corner ------
        corner_params = ["initial_mass", "Teff", "age_gyr"]
        available = []
        for p in corner_params:
            try:
                _extract_param(result, p)
                available.append(p)
            except (KeyError, IndexError):
                pass
        corner_params = available
        n_cp = len(corner_params)

        if n_cp > 0:
            gs_corner = gs[0, 0].subgridspec(
                n_cp, n_cp, hspace=0.08, wspace=0.08,
            )
            corner_data = np.column_stack(
                [_extract_param(result, p) for p in corner_params]
            )
            for i in range(n_cp):
                for j in range(n_cp):
                    ax = fig.add_subplot(gs_corner[i, j])
                    if j > i:
                        ax.set_visible(False)
                        continue
                    if i == j:
                        self._corner_1d(
                            ax, corner_data[:, i], corner_params[i],
                            model_labels,
                        )
                    else:
                        self._corner_2d(
                            ax, corner_data[:, j], corner_data[:, i],
                            model_labels,
                        )
                    if i == n_cp - 1:
                        ax.set_xlabel(
                            _label_for(corner_params[j]),
                            fontsize=self.corner_tick_fontsize,
                            fontname=self.fontname,
                        )
                    else:
                        ax.set_xticklabels([])
                    if j == 0 and i > 0:
                        ax.set_ylabel(
                            _label_for(corner_params[i]),
                            fontsize=self.corner_tick_fontsize,
                            fontname=self.fontname,
                        )
                    elif j > 0 and i != j:
                        ax.set_yticklabels([])
                    ax.tick_params(labelsize=8)

        # ------ [0,1]: HR diagram ------
        ax_hr = fig.add_subplot(gs[0, 1])
        log_teff = np.log10(_extract_param(result, "Teff"))
        log_g = _extract_param(result, "log_g")

        if model_labels is not None:
            unique_models = list(dict.fromkeys(model_labels))
            ax_hr.hist2d(
                log_teff, log_g, bins=40, cmap="Greys", alpha=0.4,
                density=True,
            )
            for k, model in enumerate(unique_models):
                mask = model_labels == model
                color = _MODEL_COLORS[k % len(_MODEL_COLORS)]
                self._plot_contour(
                    ax_hr, log_teff[mask], log_g[mask],
                    color=color, label=model,
                )
            ax_hr.legend(fontsize=12, loc="upper left")
        else:
            ax_hr.hist2d(
                log_teff, log_g, bins=40, cmap="Blues", density=True,
            )

        ax_hr.set_xlabel(
            r"$\log T_{\mathrm{eff}}$",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        ax_hr.set_ylabel(
            r"$\log g$ [dex]",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        ax_hr.invert_xaxis()
        ax_hr.invert_yaxis()
        _apply_style(
            ax_hr, self.fontname, self.fontsize, self.tick_labelsize,
        )

        # ------ [1,0]: Mass-Age ------
        ax_ma = fig.add_subplot(gs[1, 0])
        mass = _extract_param(result, "initial_mass")
        age = _extract_param(result, "age_gyr")

        if model_labels is not None:
            ax_ma.hist2d(
                mass, age, bins=40, cmap="Greys", alpha=0.4, density=True,
            )
            for k, model in enumerate(unique_models):
                mask = model_labels == model
                color = _MODEL_COLORS[k % len(_MODEL_COLORS)]
                self._plot_contour(
                    ax_ma, mass[mask], age[mask], color=color, label=model,
                )
        else:
            ax_ma.hist2d(mass, age, bins=40, cmap="Blues", density=True)

        ax_ma.set_xlabel(
            r"$M_{\mathrm{init}}$ [$M_\odot$]",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        ax_ma.set_ylabel(
            r"Age [Gyr]",
            fontsize=self.fontsize, fontname=self.fontname,
        )
        _apply_style(
            ax_ma, self.fontname, self.fontsize, self.tick_labelsize,
        )

        # ------ [1,1]: Model weights or [Fe/H] ------
        ax_bw = fig.add_subplot(gs[1, 1])
        if _is_bma(result):
            names = result.model_names
            weights = result.weights
            n = len(names)
            colors = [_MODEL_COLORS[k % len(_MODEL_COLORS)] for k in range(n)]
            bars = ax_bw.bar(
                names, weights, color=colors,
                edgecolor="black", linewidth=0.5,
            )
            for bar, w in zip(bars, weights):
                ax_bw.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{w:.3f}",
                    ha="center", va="bottom",
                    fontsize=self.corner_fontsize,
                    fontname=self.fontname,
                )
            ax_bw.set_ylabel(
                "Weight",
                fontsize=self.fontsize, fontname=self.fontname,
            )
            ax_bw.set_title(
                "Model Weights",
                fontsize=self.fontsize, fontname=self.fontname,
            )
            ax_bw.set_ylim(0, min(1.0, weights.max() * 1.3))
        else:
            try:
                feh = _extract_param(result, "[Fe/H]")
                _kde_on_hist(
                    ax_bw, feh, "tab:blue", bins=40, alpha=0.3,
                )
                med, lo, hi = _percentile_str(feh)
                ax_bw.axvline(
                    med, color=self.corner_med_c,
                    ls=self.corner_med_style, lw=1.5,
                )
                ax_bw.axvline(
                    med - lo, color=self.corner_v_c,
                    ls=self.corner_v_style, lw=1,
                )
                ax_bw.axvline(
                    med + hi, color=self.corner_v_c,
                    ls=self.corner_v_style, lw=1,
                )
                ax_bw.set_xlabel(
                    r"[Fe/H] [dex]",
                    fontsize=self.fontsize, fontname=self.fontname,
                )
                ax_bw.set_ylabel(
                    "Density",
                    fontsize=self.fontsize, fontname=self.fontname,
                )
                ax_bw.set_title(
                    f"[Fe/H] = {med:.3f} $^{{+{hi:.3f}}}_{{-{lo:.3f}}}$",
                    fontsize=self.fontsize, fontname=self.fontname,
                )
            except (KeyError, IndexError):
                ax_bw.set_visible(False)
        _apply_style(
            ax_bw, self.fontname, self.fontsize, self.tick_labelsize,
        )

        fig.suptitle(
            "LACHESIS Summary",
            fontsize=self.fontsize + 2, fontname=self.fontname, y=1.01,
        )

        return _finalize(fig, filename)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _corner_1d(self, ax, data, param_name, model_labels=None):
        """1D marginal on diagonal: histogram + KDE + median + 68% CI."""
        finite = data[np.isfinite(data)]
        if len(finite) < 2:
            return

        if model_labels is not None:
            unique_models = list(dict.fromkeys(model_labels))
            bins = np.histogram_bin_edges(finite, bins=40)
            for k, model in enumerate(unique_models):
                mask = model_labels == model
                subset = data[mask]
                subset = subset[np.isfinite(subset)]
                if len(subset) < 2:
                    continue
                color = _MODEL_COLORS[k % len(_MODEL_COLORS)]
                _kde_on_hist(
                    ax, subset, color, bins=bins, alpha=0.3,
                    histtype="stepfilled",
                )
        else:
            _kde_on_hist(
                ax, finite, "tab:blue", bins=40, alpha=0.5,
                histtype="stepfilled",
            )

        # Median + 68% CI lines (ARIADNE style)
        med, lo, hi = _percentile_str(finite)
        ax.axvline(
            med, color=self.corner_med_c,
            linestyle=self.corner_med_style, lw=1.5,
        )
        ax.axvline(
            med - lo, color=self.corner_v_c,
            linestyle=self.corner_v_style, lw=1.0,
        )
        ax.axvline(
            med + hi, color=self.corner_v_c,
            linestyle=self.corner_v_style, lw=1.0,
        )

        # Title: value +hi -lo
        ax.set_title(
            f"{med:.3g}$^{{+{hi:.2g}}}_{{-{lo:.2g}}}$",
            fontsize=self.corner_fontsize,
            fontname=self.fontname,
        )
        ax.set_yticks([])

    def _corner_2d(self, ax, x, y, model_labels=None):
        """2D histogram + contour overlay in off-diagonal panels."""
        finite = np.isfinite(x) & np.isfinite(y)
        xf, yf = x[finite], y[finite]
        if len(xf) < 10:
            return

        if model_labels is not None:
            unique_models = list(dict.fromkeys(model_labels))
            for k, model in enumerate(unique_models):
                mask = model_labels[finite] == model
                if mask.sum() < 10:
                    continue
                color = _MODEL_COLORS[k % len(_MODEL_COLORS)]
                ax.hist2d(
                    xf[mask], yf[mask], bins=30,
                    cmap=_make_transparent_cmap(color), density=True,
                )
                self._plot_contour(ax, xf[mask], yf[mask], color=color)
        else:
            ax.hist2d(xf, yf, bins=30, cmap="Blues", density=True)
            self._plot_contour(ax, xf, yf, color="navy")

        # Median crosshairs
        med_x = np.median(xf)
        med_y = np.median(yf)
        ax.axvline(
            med_x, color=self.corner_med_c,
            linestyle=self.corner_med_style, lw=0.7, alpha=0.5,
        )
        ax.axhline(
            med_y, color=self.corner_med_c,
            linestyle=self.corner_med_style, lw=0.7, alpha=0.5,
        )

    @staticmethod
    def _plot_contour(ax, x, y, color="navy", label=None):
        """Overlay 1-sigma and 2-sigma density contours."""
        finite = np.isfinite(x) & np.isfinite(y)
        xf, yf = x[finite], y[finite]
        if len(xf) < 20:
            return

        n_bins = 40
        H, xedges, yedges = np.histogram2d(xf, yf, bins=n_bins, density=True)
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])

        H_sorted = np.sort(H.ravel())[::-1]
        H_cumsum = np.cumsum(H_sorted)
        if H_cumsum[-1] == 0:
            if label is not None:
                ax.plot([], [], color=color, lw=1.5, label=label)
            return
        H_cumsum /= H_cumsum[-1]

        level_68 = H_sorted[np.searchsorted(H_cumsum, 0.68)]
        level_95 = H_sorted[np.searchsorted(H_cumsum, 0.95)]

        levels = sorted(set([level_95, level_68]))
        levels = [lv for lv in levels if lv > 0]
        if not levels:
            if label is not None:
                ax.plot([], [], color=color, lw=1.5, label=label)
            return

        widths = [0.7] if len(levels) == 1 else [0.7, 1.2]
        ax.contour(
            xc, yc, H.T, levels=levels,
            colors=color, linewidths=widths, alpha=0.8,
        )
        if label is not None:
            ax.plot([], [], color=color, lw=1.5, label=label)

    @staticmethod
    def _get_track(grid, log_age, feh):
        """Extract isochrone track (logTeff, logL, mass) from a grid.

        Returns None if the requested (age, feh) falls outside the grid
        or if the track has insufficient valid points.
        """
        try:
            cols = grid.columns
            ci_lt = cols.index("log_Teff")
            ci_ll = cols.index("log_L")
            ci_m = cols.index("initial_mass")

            # Nearest feh
            fi = int(np.argmin(np.abs(grid._feh_values - feh)))
            # Nearest age
            ai = int(np.argmin(np.abs(grid._age_values - log_age)))

            track_data = grid._data[fi, ai, :, :]
            # Remove NaN rows
            valid = np.all(np.isfinite(track_data[:, [ci_lt, ci_ll, ci_m]]),
                           axis=1)
            if valid.sum() < 3:
                return None
            return (
                track_data[valid, ci_lt],
                track_data[valid, ci_ll],
                track_data[valid, ci_m],
            )
        except (AttributeError, ValueError, IndexError):
            return None
