"""Fitter class — ARIADNE-compatible property-based isochrone fitter."""

import os
import time

import numpy as np

from lachesis.bma import bayesian_model_average
from lachesis.display import (
    display_elapsed,
    display_fitting_grid,
    display_model_weights,
    display_routine,
    display_summary,
)
from lachesis.error import FittingError, GridError, InputError
from lachesis.interp import make_interpolator
from lachesis.output import save_model_weights, save_summary_dat, to_inference_data
from lachesis.sampler import IsochroneFitter
from lachesis.star import Star


_GRID_REGISTRY = {
    "mist": ("lachesis.grid.mist", "MISTModelGrid", "mist_v1.2_vvcrit0.4.h5"),
    "parsec": ("lachesis.grid.parsec", "PARSECModelGrid", "parsec_v1.2S.h5"),
    "dartmouth": ("lachesis.grid.dartmouth", "DartmouthModelGrid", "dartmouth_dsep.h5"),
    "basti": ("lachesis.grid.basti", "BaSTIModelGrid", "basti.h5"),
    "yapsi": ("lachesis.grid.yapsi", "YAPSIModelGrid", "yapsi.h5"),
    "geneva": ("lachesis.grid.geneva", "GenevaModelGrid", "geneva.h5"),
    "bhac15": ("lachesis.grid.bhac15", "BHAC15ModelGrid", "bhac15.h5"),
    "starevol": ("lachesis.grid.starevol", "STAREVOLModelGrid", "starevol.h5"),
}


def _load_grid(name: str):
    """Load a named grid from shipped HDF5 cache."""
    from importlib import import_module
    from lachesis.config import GRID_DIR

    name = name.lower()
    if name not in _GRID_REGISTRY:
        raise GridError(
            f"Unknown grid '{name}'. Available: {sorted(_GRID_REGISTRY)}"
        )

    module_path, class_name, h5_name = _GRID_REGISTRY[name]
    h5 = GRID_DIR / h5_name
    if not h5.exists():
        raise GridError(f"{name} grid not found at {h5}")

    mod = import_module(module_path)
    cls = getattr(mod, class_name)
    return cls.from_hdf5(h5)


class Fitter:
    """Isochrone fitter with Bayesian Model Averaging.

    Property-based configuration matching ARIADNE's pattern::

        f = Fitter()
        f.star = s
        f.grids = ['mist', 'parsec']
        f.bma = True
        f.setup = ['dynesty', 500, 0.01, 'multi', 'rwalk', 4, False]
        f.out_folder = 'output/'
        f.initialize()
        f.fit_bma()
    """

    def __init__(self):
        self._star = None
        self._grids = ["mist", "parsec", "dartmouth", "basti", "yapsi"]
        self._bma = False
        self._binary = False
        self._verbose = True
        self._out_folder = None
        self._setup = ["dynesty"]
        self._prior_setup = None
        self._av_law = "fitzpatrick"
        self._imf = "chabrier"
        self._eep_range = (200, 808)
        self._age_range = (8.0, 10.3)
        self._feh_range = (-2.0, 0.5)
        self._bc_system = None
        self._parsed_setup = None
        self._initialized = False
        self._grid_objects = {}
        self._interpolators = {}
        self._fitters = {}
        self._bc_table = None

    # --- Properties with validation ---

    @property
    def star(self) -> Star:
        return self._star

    @star.setter
    def star(self, s):
        if not isinstance(s, Star):
            raise InputError("star must be a lachesis.Star instance")
        self._star = s
        self._initialized = False

    @property
    def grids(self) -> list[str]:
        return self._grids

    @grids.setter
    def grids(self, g):
        # starevol_vXX variants expand to base name for validation
        valid_base = {"mist", "parsec", "dartmouth", "basti", "yapsi", "geneva", "bhac15", "starevol"}
        for name in g:
            base = name.lower().split("_v")[0] if name.lower().startswith("starevol_v") else name.lower()
            if base not in valid_base:
                raise InputError(f"Unknown grid '{name}'. Available: {sorted(valid_base)}")
        self._grids = [name.lower() for name in g]
        self._initialized = False

    @property
    def bma(self) -> bool:
        return self._bma

    @bma.setter
    def bma(self, val: bool):
        self._bma = bool(val)

    @property
    def binary(self) -> bool:
        return self._binary

    @binary.setter
    def binary(self, val: bool):
        self._binary = bool(val)

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, val: bool):
        self._verbose = bool(val)

    @property
    def out_folder(self) -> str | None:
        return self._out_folder

    @out_folder.setter
    def out_folder(self, path: str):
        self._out_folder = path

    @property
    def setup(self) -> list:
        return self._setup

    @setup.setter
    def setup(self, s: list):
        self._setup = list(s)
        self._initialized = False

    @property
    def prior_setup(self) -> dict | None:
        return self._prior_setup

    @prior_setup.setter
    def prior_setup(self, ps: dict):
        self._prior_setup = dict(ps)

    @property
    def eep_range(self) -> tuple[float, float]:
        return self._eep_range

    @eep_range.setter
    def eep_range(self, r):
        self._eep_range = tuple(r)

    @property
    def age_range(self) -> tuple[float, float]:
        return self._age_range

    @age_range.setter
    def age_range(self, r):
        self._age_range = tuple(r)

    @property
    def feh_range(self) -> tuple[float, float]:
        return self._feh_range

    @feh_range.setter
    def feh_range(self, r):
        self._feh_range = tuple(r)

    @property
    def bc_system(self) -> str | None:
        return self._bc_system

    @bc_system.setter
    def bc_system(self, system: str | None):
        self._bc_system = system

    @property
    def av_law(self) -> str:
        return self._av_law

    @av_law.setter
    def av_law(self, law: str):
        valid = {"fitzpatrick", "cardelli", "odonnell", "calzetti"}
        if law.lower() not in valid:
            raise InputError(f"Unknown Av law '{law}'. Available: {sorted(valid)}")
        self._av_law = law.lower()

    @property
    def imf(self) -> str:
        return self._imf

    @imf.setter
    def imf(self, name: str):
        valid = {"chabrier", "salpeter"}
        if name.lower() not in valid:
            raise InputError(f"Unknown IMF '{name}'. Available: {sorted(valid)}")
        self._imf = name.lower()

    # --- Methods ---

    def initialize(self):
        """Validate config, load grids, build fitters, create output dir."""
        if self._star is None:
            raise InputError("No star set. Assign f.star = Star(...) first.")

        # BMA guardrails: BHAC15 and STAREVOL cannot participate in BMA
        _BMA_FORBIDDEN = {"bhac15", "starevol"}
        if self._bma:
            forbidden = _BMA_FORBIDDEN & set(self._grids)
            if forbidden:
                raise InputError(
                    f"Cannot use {forbidden} in BMA mode. "
                    f"BHAC15 (single metallicity) and STAREVOL (rotation parameter) "
                    f"are single-grid fit only."
                )

        # Parse setup list
        self._parsed_setup = self._parse_setup(self._setup)

        # Create output directory
        if self._out_folder:
            os.makedirs(self._out_folder, exist_ok=True)

        # Reset per-run state so a second initialize() doesn't accumulate
        # entries from a previous run with a different grid set.
        self._grid_objects = {}
        self._interpolators = {}
        self._fitters = {}
        self._bc_table = None

        # Load grids
        for name in self._grids:
            self._grid_objects[name] = _load_grid(name)
            self._interpolators[name] = make_interpolator(self._grid_objects[name])

        # Load BC tables
        from lachesis.bc import BCTable
        from lachesis.config import BC_DIR
        if self._bc_system:
            self._bc_table = BCTable(BC_DIR, system=self._bc_system)
        else:
            self._bc_table = BCTable.multi_system(BC_DIR)
        # Restrict BC to only the bands the star actually uses
        used_bands = [
            k for k in self._star.observed
            if k in self._bc_table._band_indices
        ]
        if used_bands:
            self._bc_table.set_active_bands(used_bands)

        # Parse priors (matching ARIADNE's prior_setup dict pattern)
        ps = self._prior_setup or {}

        # [Fe/H] prior priority:
        #   1. User override in prior_setup (normal, morton, rave)
        #   2. Full ARIADNE posterior samples (KDE) — preserves shape
        #   3. Spectroscopic prior from star.feh/feh_e (gaussian)
        #   4. Uniform (default)
        feh_prior = None
        feh_cfg = ps.get("feh") or ps.get("z")
        if feh_cfg and isinstance(feh_cfg, tuple):
            if feh_cfg[0] == "normal":
                feh_prior = ("gaussian", feh_cfg[1], feh_cfg[2])
            elif feh_cfg[0] == "morton":
                feh_prior = ("morton",)
            elif feh_cfg[0] == "rave":
                feh_prior = ("rave",)
        elif feh_cfg and isinstance(feh_cfg, str):
            if feh_cfg == "morton":
                feh_prior = ("morton",)
            elif feh_cfg == "rave":
                feh_prior = ("rave",)
        if feh_prior is None:
            if getattr(self._star, "feh_posterior", None) is not None:
                feh_prior = ("kde", self._star.feh_posterior)
            elif self._star.feh is not None and self._star.feh_e is not None:
                feh_prior = ("gaussian", self._star.feh, self._star.feh_e)

        # Age prior: "log_uniform" (flat in log age, default) or
        # "uniform" (flat in linear age, a la Morton/isochrones)
        age_cfg = ps.get("log_age")
        age_prior = "log_uniform"
        if age_cfg and isinstance(age_cfg, tuple) and age_cfg[0] == "uniform":
            age_prior = "uniform"
        elif age_cfg and isinstance(age_cfg, str) and age_cfg == "uniform":
            age_prior = "uniform"

        # Distance prior: truncated normal from BJ distance (like ARIADNE)
        dist_cfg = ps.get("dist")
        if dist_cfg and isinstance(dist_cfg, tuple) and dist_cfg[0] == "normal":
            distance_prior = ("normal", dist_cfg[1], dist_cfg[2])
        elif self._star.distance is not None and self._star.distance_e is not None:
            distance_prior = (
                "normal", self._star.distance, self._star.distance_e
            )
        else:
            distance_prior = ("normal", 500.0, 5000.0)

        # Av prior
        av_cfg = ps.get("Av")
        if av_cfg and isinstance(av_cfg, tuple) and av_cfg[0] == "fixed":
            av_range = (av_cfg[1], av_cfg[1])
        elif av_cfg and isinstance(av_cfg, tuple) and av_cfg[0] == "uniform":
            av_range = (av_cfg[1], av_cfg[2])
        else:
            av_hi = 1.0 if self._star.Av is None else max(self._star.Av, 1e-3)
            av_range = (0.0, av_hi)

        # Build KDE-based external priors from ARIADNE posteriors.
        # Pre-tabulate log(pdf) on a fine grid and interpolate with np.interp
        # at evaluation time — O(1) per call instead of O(N_samples).
        external_kdes = {}
        if self._star.external_posteriors:
            from scipy.stats import gaussian_kde
            from numpy.linalg import LinAlgError
            for param, samples in self._star.external_posteriors.items():
                if len(samples) < 10:
                    continue
                try:
                    kde = gaussian_kde(samples)
                except LinAlgError:
                    # Degenerate samples (e.g. zero variance); skip silently.
                    continue
                lo, hi = float(samples.min()), float(samples.max())
                std = float(np.std(samples))
                # Pad ≥5σ so the KDE tails aren't clipped to a hard wall.
                pad = max(0.1 * (hi - lo), 5.0 * std)
                grid = np.linspace(lo - pad, hi + pad, 2048)
                log_pdf = np.log(np.maximum(kde(grid), 1e-300))
                external_kdes[param] = (grid, log_pdf)
            if self._verbose and external_kdes:
                print(f"\t\t\t External priors (KDE): {', '.join(external_kdes)}")

        # Build IsochroneFitter per grid
        for name in self._grids:
            grid = self._grid_objects[name]
            if hasattr(grid, "fitting_eep_range"):
                eep_lo, eep_hi = grid.fitting_eep_range
            else:
                eep_lo = max(self._eep_range[0], grid.eep_range[0])
                eep_hi = min(self._eep_range[1], grid.eep_range[1])
            # Clamp age and [Fe/H] to each grid's actual coverage
            grid_feh_lo = max(self._feh_range[0], float(grid.feh_values[0]))
            grid_feh_hi = min(self._feh_range[1], float(grid.feh_values[-1]))
            grid_age_lo = max(self._age_range[0], float(grid.age_values[0]))
            grid_age_hi = min(self._age_range[1], float(grid.age_values[-1]))
            # Rotation range for grids with a Vini axis (e.g. STAREVOL)
            vini_range = None
            if hasattr(grid, "vini_values") and grid.vini_values is not None:
                vini_range = (float(grid.vini_values[0]), float(grid.vini_values[-1]))

            self._fitters[name] = IsochroneFitter(
                interp=self._interpolators[name],
                eep_range=(eep_lo, eep_hi),
                age_range=(grid_age_lo, grid_age_hi),
                feh_range=(grid_feh_lo, grid_feh_hi),
                feh_prior=feh_prior,
                age_prior=age_prior,
                bc_table=self._bc_table,
                distance_prior=distance_prior,
                av_range=av_range,
                vini_range=vini_range,
                binary=self._binary,
                imf=self._imf,
                external_kdes=external_kdes,
            )

        self._initialized = True

        if self._verbose:
            display_routine(self)

    def show_priors(self):
        """Display prior configuration."""
        import random
        from termcolor import colored
        if not self._initialized:
            raise InputError("Call initialize() first.")
        c = random.choice(['red', 'green', 'blue', 'yellow', 'grey', 'magenta', 'cyan', 'white'])
        t3 = "\t\t\t"
        fitter = next(iter(self._fitters.values()))
        p = fitter.prior
        print(colored(f"{t3}{'Parameter':12s}  Prior", c))
        print(colored(f"{t3}{'-' * 40}", c))
        for name in p.param_names:
            if name == "eep":
                print(colored(f"{t3}{'eep':12s}  U({p.eep_lo:.0f}, {p.eep_hi:.0f})", c))
            elif name == "log_age":
                if p._age_type == "uniform":
                    print(colored(f"{t3}{'log_age':12s}  Flat in linear age ({p.age_lo:.2f}, {p.age_hi:.2f})", c))
                else:
                    print(colored(f"{t3}{'log_age':12s}  U({p.age_lo:.2f}, {p.age_hi:.2f})", c))
            elif name == "feh":
                if p._feh_type == "gaussian":
                    print(colored(f"{t3}{'feh':12s}  N({p._feh_mean:.3f}, {p._feh_sigma:.3f})", c))
                elif p._feh_type == "kde":
                    n = len(p._feh_cdf_x)
                    print(colored(f"{t3}{'feh':12s}  KDE (ARIADNE posterior, {n} grid pts)", c))
                elif p._feh_type == "morton":
                    print(colored(f"{t3}{'feh':12s}  Morton (2-Gaussian disk + halo)", c))
                elif p._feh_type == "rave":
                    print(colored(f"{t3}{'feh':12s}  RAVE DR5 N(-0.125, 0.234)", c))
                else:
                    print(colored(f"{t3}{'feh':12s}  U({p.feh_lo:.2f}, {p.feh_hi:.2f})", c))
            elif name == "distance":
                print(colored(f"{t3}{'distance':12s}  N({p._dist_mean:.3f}, {p._dist_sigma:.3f})", c))
            elif name == "av":
                print(colored(f"{t3}{'av':12s}  U({p.av_lo:.2f}, {p.av_hi:.2f})", c))
            elif name == "eep_secondary":
                print(colored(f"{t3}{'eep_2nd':12s}  U({p.eep_lo:.0f}, eep_primary)", c))
            elif name == "vini":
                print(colored(f"{t3}{'vini':12s}  U({p.vini_lo:.2f}, {p.vini_hi:.2f})", c))
        print()

    def fit(self):
        """Single-grid fit (uses first grid in self.grids)."""
        if not self._initialized:
            raise InputError("Call initialize() first.")

        t0 = time.time()
        name = self._grids[0]
        setup = self._parsed_setup

        if self._verbose:
            display_fitting_grid(name)

        result = self._fitters[name].fit(
            observed=self._star.observed,
            uncertainties=self._star.uncertainties,
            nlive=setup["nlive"],
            dlogz=setup["dlogz"],
            sample=setup["sample"],
        )

        if self._verbose:
            display_summary(
                result["samples"], result["derived"],
                self._fitters[name].prior.param_names,
            )
            display_elapsed(t0)

        self._save(result, name)
        return result

    def fit_bma(self):
        """Fit all grids and combine via Bayesian Model Averaging."""
        if not self._initialized:
            raise InputError("Call initialize() first.")
        if len(self._grids) < 2 and self._bma:
            raise InputError("BMA requires at least 2 grids.")

        t0 = time.time()
        setup = self._parsed_setup
        results = {}

        for name in self._grids:
            if self._verbose:
                display_fitting_grid(name)
            try:
                results[name] = self._fitters[name].fit(
                    observed=self._star.observed,
                    uncertainties=self._star.uncertainties,
                    nlive=setup["nlive"],
                    dlogz=setup["dlogz"],
                    sample=setup["sample"],
                )
            except Exception as e:
                raise FittingError(
                    f"Fitting failed on grid {name}: {e}",
                    grid_name=name,
                    partial_results=results,
                )

        # BMA
        bma_result = bayesian_model_average(
            list(results.values()), names=list(results.keys()),
        )

        if self._verbose:
            # Use first grid's param names for display
            param_names = self._fitters[self._grids[0]].prior.param_names
            display_summary(bma_result.samples, bma_result.derived, param_names)
            display_model_weights(bma_result)
            display_elapsed(t0)

        self._save_bma(results, bma_result)
        return bma_result

    def _save(self, result, grid_name):
        """Save single-grid results."""
        if not self._out_folder:
            return
        stem = f"lachesis_{self._star.starname.replace(' ', '_')}"
        nc_path = os.path.join(self._out_folder, f"{stem}.nc")
        dat_path = os.path.join(self._out_folder, f"{stem}.dat")

        idata = to_inference_data(
            result,
            observed=self._star.observed,
            uncertainties=self._star.uncertainties,
            grid_name=grid_name,
            star=self._star,
        )
        idata.to_netcdf(nc_path)
        save_summary_dat(dat_path, result)

    def _save_bma(self, results, bma_result):
        """Save BMA results."""
        if not self._out_folder:
            return
        stem = f"lachesis_{self._star.starname.replace(' ', '_')}"
        nc_path = os.path.join(self._out_folder, f"{stem}_BMA.nc")
        dat_path = os.path.join(self._out_folder, f"{stem}_BMA.dat")
        weights_path = os.path.join(self._out_folder, "model_weights.dat")

        idata = to_inference_data(
            bma_result=bma_result,
            per_grid_results=results,
            observed=self._star.observed,
            uncertainties=self._star.uncertainties,
            grid_name="BMA",
            star=self._star,
        )
        idata.to_netcdf(nc_path)
        # Stack-level logzerr: variance of evidence-weighted log_z across grids.
        log_z = bma_result.log_evidences
        weights = bma_result.weights
        mean_lz = float(np.sum(weights * log_z))
        logzerr_bma = float(np.sqrt(max(np.sum(weights * (log_z - mean_lz) ** 2), 0.0)))
        save_summary_dat(dat_path, {
            "samples": bma_result.samples,
            "derived": bma_result.derived,
            "logz": bma_result.log_evidence,
            "logzerr": logzerr_bma,
        })
        save_model_weights(weights_path, bma_result)

    @staticmethod
    def _parse_setup(setup: list) -> dict:
        """Parse ARIADNE-format setup list into dict."""
        defaults = {
            "engine": "dynesty",
            "nlive": 500,
            "dlogz": 0.01,
            "bound": "multi",
            "sample": "rwalk",
            "threads": 1,
            "dynamic": False,
        }
        keys = list(defaults.keys())
        for i, val in enumerate(setup):
            if i < len(keys):
                defaults[keys[i]] = val
        return defaults
