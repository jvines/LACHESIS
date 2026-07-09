"""ISOPlotter API class.

The bulk of the rendering implementation lives here; pure helpers and
constants live in sibling modules ._helpers and ._constants.
"""

__all__ = ["ISOPlotter"]

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from lachesis.bma import BMAResult

from ._constants import (
    _DEFAULT_CORNER_PARAMS,
    _DEFAULT_HIST_PARAMS,
    _DEFAULT_LATEX_PARAMS,
    _MODEL_COLORS,
    _PARAM_LABELS,
    _SETTINGS_FILE,
)
from ._helpers import (
    _apply_style,
    _extract_from,
    _extract_param,
    _finalize,
    _get_model_labels,
    _is_bma,
    _kde_on_hist,
    _label_for,
    _make_transparent_cmap,
    _percentile_str,
)


# -----------------------------------------------------------------------
# IsoPlotter
# -----------------------------------------------------------------------


class ISOPlotter:
    """Plotting interface for LACHESIS isochrone fitting results.

    Matches ARIADNE's SEDPlotter interface: load results from disk,
    plot methods take no arguments.

    Parameters
    ----------
    input_file : str or Path
        Path to a LACHESIS .nc result file.
    out_folder : str or Path
        Directory for saving plot files.
    pdf : bool
        Save as PDF instead of PNG (default False).
    settings : str or Path, optional
        Path to a plot_settings.dat file. Falls back to the bundled
        default that mirrors ARIADNE's style.
    """

    def __init__(self, input_file, out_folder, pdf=False, settings=None):
        import os
        self.out_folder = str(out_folder)
        self.pdf = pdf
        self._ext = ".pdf" if pdf else ".png"
        os.makedirs(self.out_folder, exist_ok=True)

        self.result, self._grid_names, self._ariadne = self._load_nc(input_file)
        self.bma = _is_bma(self.result)
        self._read_config(settings)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_nc(path):
        """Load a .nc file and reconstruct a (result, grid_names) pair.

        For BMA results:
        - The combined BMA-weighted posterior is loaded from the
          ``posterior`` group. This is what `result.samples`/`result.derived`
          reflect — properly evidence-weighted draws as produced by
          ``bayesian_model_average``.
        - The per-grid RAW nested-sampling posteriors are loaded from the
          ``posterior_{gname}`` groups and stored on
          ``result.per_grid_samples`` / ``result.per_grid_derived``. These
          are unweighted — used by the plotter for per-model histograms,
          HR tracks, etc.

        Returns
        -------
        result : BMAResult or dict
            Combined posterior and derived quantities.
        grid_names : list of str
            Names of the grids used by the fit, in order. For BMA these
            match ``result.model_names``; for single-grid fits it's a
            one-element list from ``constant_data/grid_name``.
        """
        import arviz as az

        idata = az.from_netcdf(str(path))

        _SKIP_DERIVED = {"eep", "log_age", "feh", "distance", "Av", "age", "jitter"}

        def _group_ds(group):
            """Return the xarray Dataset for a DataTree group."""
            return group.ds if hasattr(group, "ds") else group

        def _ds_to_samples_and_derived(ds):
            cols = []
            for k in ("eep", "log_age", "feh", "eep_secondary", "distance", "Av", "vini", "jitter"):
                if k in ds:
                    cols.append(ds[k].values.flatten())
            samples = np.column_stack(cols) if cols else np.empty((0, 0))
            derived = {}
            model_arr = None
            for var in ds.data_vars:
                if var == "model":
                    model_arr = ds[var].values.flatten()
                    continue
                if var not in _SKIP_DERIVED and var != "eep_secondary" and var != "vini":
                    derived[var] = ds[var].values.flatten()
            if "age" in ds:
                derived["age_gyr"] = ds["age"].values.flatten()
            if model_arr is not None:
                derived["model"] = np.array([str(m) for m in model_arr])
            return samples, derived

        post = _group_ds(idata.posterior)
        samples, derived = _ds_to_samples_and_derived(post)

        # Extract ARIADNE-derived params (Teff, radius) if stored
        const_group = getattr(idata, "constant_data", None)
        const_ds = _group_ds(const_group) if const_group is not None else None
        ariadne = {}
        if const_ds is not None:
            for key in ("ariadne_teff", "ariadne_teff_e",
                        "ariadne_radius", "ariadne_radius_e"):
                if key in const_ds:
                    ariadne[key] = float(const_ds[key].values.flat[0])

        # Single-grid fit
        is_bma = const_ds is not None and "model_weights" in const_ds
        if not is_bma:
            grid_names = []
            if const_ds is not None and "grid_name" in const_ds:
                gn = const_ds["grid_name"].values
                if gn.ndim == 0:
                    grid_names = [str(gn)]
                else:
                    grid_names = [str(gn[0])]
            return {"samples": samples, "derived": derived}, grid_names, ariadne

        # BMA: load combined posterior + per-grid groups
        weights = const_ds["model_weights"].values
        log_ev = const_ds["log_evidence"].values
        names = [str(n) for n in const_ds["model_names"].values]

        per_grid_samples = {}
        per_grid_derived = {}
        for name in names:
            group_name = f"posterior_{name}"
            if not hasattr(idata, group_name):
                continue
            ds = _group_ds(getattr(idata, group_name))
            s, d = _ds_to_samples_and_derived(ds)
            if s.size == 0:
                continue
            per_grid_samples[name] = s
            per_grid_derived[name] = d

        # The combined `posterior` group is the source of truth — it carries
        # the BMA-weighted draws and per-sample `model` labels exactly as
        # `bayesian_model_average` produced them. Don't resample here.
        log_evidence = float(getattr(idata, "attrs", {}).get(
            "log_evidence", float(np.max(log_ev))
        ))

        result = BMAResult(
            weights=weights,
            samples=samples,
            derived=derived,
            model_names=names,
            log_evidences=log_ev,
            log_evidence=log_evidence,
            per_grid_samples=per_grid_samples,
            per_grid_derived=per_grid_derived,
        )
        return result, names, ariadne

    def clean(self):
        """Close all open figures (useful when iterating through many stars)."""
        plt.close("all")

    def _auto_load_grids(self):
        """Load the grids used by the fit from the shipped HDF5 cache.

        Called by plot_hr when the caller doesn't supply ``grids``.  Uses
        the grid names stored on the plotter during ``_load_nc`` (from the
        .nc file's ``constant_data/model_names`` or ``grid_name``).
        """
        from lachesis.fitter import _load_grid
        out = {}
        for name in self._grid_names:
            try:
                out[name] = _load_grid(name)
            except Exception:
                pass
        return out

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

    def plot_corner(self, params=None):
        """Corner plot of the posterior."""
        result = self.result
        filename = f"{self.out_folder}/CORNER{self._ext}"
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

    def plot_histograms(self):
        """Per-parameter BMA histograms (mirrors ARIADNE's plot_bma_hist).

        Writes two files per parameter into ``{out_folder}/histograms/``:
        - ``{param}.png``: normalized (PDF) panel
        - ``weighted_{param}.png``: weighted counts (N) panel

        In BMA mode, each model gets its own colored histogram + KDE.  The
        PDF panel additionally shows TWO combined posteriors:
        - Weighted sampling (cyan, dashed): BMA-weighted draws
        - Weighted average (pink, dash-dot): sum_i w_i * kde_i(x)
        """
        import os

        result = self.result
        hist_dir = os.path.join(self.out_folder, "histograms")
        os.makedirs(hist_dir, exist_ok=True)

        is_bma = _is_bma(result)
        has_per_grid = (
            is_bma
            and getattr(result, "per_grid_samples", None)
            and getattr(result, "per_grid_derived", None)
        )

        # Parameters to plot — must exist in the combined posterior
        params = []
        for p in _DEFAULT_HIST_PARAMS:
            try:
                _extract_param(result, p)
                params.append(p)
            except (KeyError, IndexError):
                pass

        if is_bma:
            weights_dict = dict(zip(result.model_names, result.weights))
        else:
            weights_dict = None

        figures = []
        for param in params:
            label = _label_for(param)
            arr_combined = _extract_param(result, param)

            f1, ax1 = plt.subplots(figsize=(12, 6))
            f2, ax2 = plt.subplots(figsize=(12, 6))

            if has_per_grid:
                # Use the RAW per-grid posteriors for per-model histograms.
                # These are nested-sampling outputs without BMA resampling,
                # so each model's true posterior shape is preserved
                # regardless of its evidence weight.
                per_model_samples = {}
                per_model_kde = {}
                for name in result.model_names:
                    if name not in result.per_grid_samples:
                        continue
                    try:
                        samp = _extract_from(
                            result.per_grid_samples[name],
                            result.per_grid_derived[name],
                            param,
                        )
                    except KeyError:
                        continue
                    samp = samp[np.isfinite(samp)]
                    if len(samp) < 2:
                        continue
                    per_model_samples[name] = samp
                    try:
                        per_model_kde[name] = gaussian_kde(samp)
                    except np.linalg.LinAlgError:
                        pass

                for k, name in enumerate(result.model_names):
                    if name not in per_model_samples:
                        continue
                    samp = per_model_samples[name]
                    color = _MODEL_COLORS[k % len(_MODEL_COLORS)]
                    w = weights_dict.get(name, 0)
                    mlabel = f"{name} prob: {w:.3f}"

                    # PDF panel — each model's own posterior shape
                    _kde_on_hist(
                        ax1, samp, color, bins=20, alpha=0.3,
                        label=mlabel,
                    )

                    # Weighted (N) panel — bars scaled by BMA weight
                    n_w, bins_w, _ = ax2.hist(
                        samp, bins=20, alpha=0.3, label=mlabel,
                        color=color,
                        weights=np.full(len(samp), w),
                    )
                    if name in per_model_kde and n_w.max() > 0:
                        kde = per_model_kde[name]
                        xx = np.linspace(bins_w[0], bins_w[-1], 300)
                        scale = n_w.max() / max(kde(xx).max(), 1e-30)
                        ax2.plot(xx, kde(xx) * scale, color=color,
                                 lw=2, alpha=1)

                # Combined overlays on the PDF panel
                # (1) Weighted sampling: the BMA-weighted resampled combined
                #     posterior (what result.samples actually contains)
                finite = arr_combined[np.isfinite(arr_combined)]
                if len(finite) >= 2:
                    try:
                        kde_ws = gaussian_kde(finite)
                        xx = np.linspace(finite.min(), finite.max(), 400)
                        ax1.plot(xx, kde_ws(xx), color="tab:cyan", lw=2,
                                 alpha=1, ls="--", label="Weighted sampling")
                    except np.linalg.LinAlgError:
                        pass

                # (2) Weighted average: sum_i w_i * kde_i(x) on a common grid
                if per_model_kde:
                    lo_wa = min(s.min() for s in per_model_samples.values())
                    hi_wa = max(s.max() for s in per_model_samples.values())
                    xx_wa = np.linspace(lo_wa, hi_wa, 400)
                    pdf_wa = np.zeros_like(xx_wa)
                    for name, kde_m in per_model_kde.items():
                        w = weights_dict.get(name, 0)
                        pdf_wa += w * kde_m(xx_wa)
                    ax1.plot(xx_wa, pdf_wa, color="tab:pink", lw=2,
                             alpha=1, ls="-.", label="Weighted average")

            else:
                # No per-grid data available — single-grid histogram
                _kde_on_hist(ax1, arr_combined, "tab:blue", bins=20, alpha=0.3)
                _kde_on_hist(ax2, arr_combined, "tab:blue", bins=20, alpha=0.3,
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

            # Save with ARIADNE's filename convention
            safe_name = (
                param.replace("[", "").replace("]", "").replace("/", "_")
            )
            _finalize(f1, os.path.join(hist_dir, f"{safe_name}{self._ext}"))
            _finalize(f2, os.path.join(hist_dir, f"weighted_{safe_name}{self._ext}"))
            figures.append((f1, f2))

        return figures

    # ------------------------------------------------------------------
    # 3.  HR diagram with LineCollection tracks
    # ------------------------------------------------------------------

    def plot_hr(self, grids=None, n_samples=50):
        """Isochrone-track HR diagram (log Teff vs log L, inverted x-axis).

        Best-fit track as mass-colored LineCollection.  Random BMA-weighted
        posterior draws as gray lines for uncertainty.  Star position with
        error bars.  Grids are loaded automatically from the shipped HDF5
        cache based on the .nc's model names; no need to pass them in.
        """
        result = self.result
        filename = f"{self.out_folder}/HR_diagram{self._ext}"
        fig, ax = plt.subplots(figsize=self.hr_figsize)

        # --- Extract posterior for star position ---
        # Prefer ARIADNE-derived Teff when available (better constrained
        # by the SED than by the isochrone grid interpolation).
        if "ariadne_teff" in self._ariadne:
            med_lt = np.log10(self._ariadne["ariadne_teff"])
            teff_e = self._ariadne.get("ariadne_teff_e", 0.0)
            # Propagate linear error to log: σ_log = σ / (x * ln10)
            log_err = teff_e / (self._ariadne["ariadne_teff"] * np.log(10))
            lo_lt = log_err
            hi_lt = log_err
        else:
            log_teff_arr = _extract_param(result, "log_Teff") if "log_Teff" in (
                result.derived if _is_bma(result) else result.get("derived", {})
            ) else np.log10(_extract_param(result, "Teff"))
            med_lt, lo_lt, hi_lt = _percentile_str(log_teff_arr)

        log_l_arr = _extract_param(result, "log_L")
        med_ll, lo_ll, hi_ll = _percentile_str(log_l_arr)

        # Auto-load grids if the caller didn't supply them
        if grids is None:
            grids = self._auto_load_grids()

        if grids:
            if _is_bma(result) and "model" in result.derived:
                self._draw_bma_tracks(
                    ax, fig, result, grids, n_samples,
                )
            else:
                # Single-grid: pick the one available grid and use combined
                # samples (which are from that single grid)
                grid = next(iter(grids.values()))
                self._draw_single_grid_tracks(
                    ax, fig, result, grid, n_samples,
                )

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

        # Error cross in the lower-left corner showing the 1σ scale.
        # Drawn after axis limits are finalized so the corner position
        # maps correctly to data coordinates.
        fig.canvas.draw()
        xlim = ax.get_xlim()  # inverted: xlim[0] > xlim[1]
        ylim = ax.get_ylim()
        cx = xlim[0] - 0.10 * (xlim[0] - xlim[1])
        cy = ylim[0] + 0.10 * (ylim[1] - ylim[0])
        xerr = 0.5 * (lo_lt + hi_lt)
        yerr = 0.5 * (lo_ll + hi_ll)
        ax.errorbar(
            cx, cy, xerr=xerr, yerr=yerr,
            color="k", fmt="none", capsize=4, lw=1.5,
            zorder=1003,
        )

        return _finalize(fig, filename)

    # ------------------------------------------------------------------
    # 4.  Mass vs Age posterior
    # ------------------------------------------------------------------

    def plot_mass_age(self):
        """Mass vs Age 2D density with contours."""
        result = self.result
        filename = f"{self.out_folder}/mass_age{self._ext}"
        # Current stellar mass, not ZAMS — matches the rest of the plotter
        mass = _extract_param(result, "mass")
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
            r"$M_\star$ [$M_\odot$]",
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

    def plot_model_weights(self):
        """Bar chart of BMA model weights with weight annotations."""
        bma_result = self.result
        filename = f"{self.out_folder}/model_weights{self._ext}"
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

    def to_latex(self, params=None):
        r"""Return a dict of LaTeX-formatted credibility intervals.

        Format: ``$value^{+hi}_{-lo}$`` using 68 % intervals.
        """
        result = self.result
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

    def summary(self):
        """Multi-panel summary figure.

        Layout (2x2 GridSpec):
          [0,0] corner (mass, Teff, age): mini 3x3
          [0,1] HR diagram
          [1,0] Mass-Age
          [1,1] Model weights (BMA) or [Fe/H] marginal
        """
        result = self.result
        filename = f"{self.out_folder}/summary{self._ext}"
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

    def _draw_bma_tracks(self, ax, fig, result, grids, n_samples):
        """Draw gray BMA-weighted posterior tracks + the best-fit track.

        The gray tracks are drawn by sampling ``n_samples`` random rows from
        ``result.samples``, which is already BMA-weighted — so tracks are
        drawn from each grid in proportion to its evidence weight. Each
        sample's model label (``result.derived["model"]``) picks the grid
        to use, and its ``(log_age, feh)`` define where on the isochrone
        manifold to draw from.

        The best-fit track is drawn as a thick mass-colored LineCollection
        using the best-weighted grid's per-model median ``(log_age, feh)``.
        """
        model_labels = result.derived["model"]
        samples = result.samples

        # --- Gray posterior draws ---
        n_total = len(samples)
        n_draw = min(n_samples, n_total)
        rng = np.random.default_rng()
        idx = rng.choice(n_total, size=n_draw, replace=False)
        for i in idx:
            name = str(model_labels[i])
            grid = grids.get(name)
            if grid is None:
                continue
            log_age = samples[i, 1]
            feh = samples[i, 2]
            t = self._get_track(grid, log_age, feh)
            if t is not None:
                ax.plot(t[0], t[1], color="gray", alpha=0.2, lw=0.6,
                        zorder=1)

        # --- Best-fit colored track (best-weighted model's own median) ---
        best_idx = int(np.argmax(result.weights))
        best_name = result.model_names[best_idx]
        best_grid = grids.get(best_name)
        if best_grid is None:
            return

        mask = model_labels == best_name
        if mask.sum() < 3:
            return
        log_age_bf = float(np.median(samples[mask, 1]))
        feh_bf = float(np.median(samples[mask, 2]))

        track = self._get_track(best_grid, log_age_bf, feh_bf)
        if track is None:
            return
        self._draw_best_fit_track(ax, fig, *track)

    def _draw_single_grid_tracks(self, ax, fig, result, grid, n_samples):
        """Draw gray posterior tracks + best-fit track for a single-grid fit."""
        samples = result.samples if _is_bma(result) else result["samples"]

        # --- Gray posterior draws ---
        n_total = len(samples)
        n_draw = min(n_samples, n_total)
        rng = np.random.default_rng()
        idx = rng.choice(n_total, size=n_draw, replace=False)
        for i in idx:
            log_age = samples[i, 1]
            feh = samples[i, 2]
            t = self._get_track(grid, log_age, feh)
            if t is not None:
                ax.plot(t[0], t[1], color="gray", alpha=0.2, lw=0.6,
                        zorder=1)

        # --- Best-fit colored track ---
        log_age_bf = float(np.median(samples[:, 1]))
        feh_bf = float(np.median(samples[:, 2]))
        track = self._get_track(grid, log_age_bf, feh_bf)
        if track is None:
            return
        self._draw_best_fit_track(ax, fig, *track)

    def _draw_best_fit_track(self, ax, fig, logteff_t, logl_t, mass_t):
        """Render a mass-colored LineCollection for the best-fit track."""
        points = np.array([logteff_t, logl_t]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
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

    @staticmethod
    def _get_track(grid, log_age, feh):
        """Extract isochrone track (logTeff, logL, current mass) from a grid.

        Returns None if the requested (age, feh) falls outside the grid
        or if the track has insufficient valid points. Uses ``star_mass``
        (current mass after mass loss) for the mass axis, matching the
        rest of the plotter's "current mass" convention.
        """
        try:
            cols = grid.columns
            ci_lt = cols.index("log_Teff")
            ci_ll = cols.index("log_L")
            # Prefer current mass; fall back to initial_mass if a grid
            # only exposes one of them.
            if "star_mass" in cols:
                ci_m = cols.index("star_mass")
            else:
                ci_m = cols.index("initial_mass")

            # Nearest feh
            fi = int(np.argmin(np.abs(grid._feh_values - feh)))
            # Nearest age
            ai = int(np.argmin(np.abs(grid._age_values - log_age)))

            # 4D grids: (feh, age, eep, col)
            # 5D grids (STAREVOL): (feh, vini, age, eep, col) — pick middle vini
            if grid._data.ndim == 5:
                vi = grid._data.shape[1] // 2
                track_data = grid._data[fi, vi, ai, :, :]
            else:
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
