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


def _load_grid(name: str):
    """Load a named grid from HDF5 cache or raw data."""
    from lachesis.config import MIST_GRID_DIR, PARSEC_DIR

    name = name.lower()
    if name == "mist":
        from lachesis.grid.mist import MISTModelGrid
        h5 = MIST_GRID_DIR / "mist_v1.2_vvcrit0.4.h5"
        if h5.exists():
            return MISTModelGrid.from_hdf5(h5)
        raise GridError(f"MIST grid not found at {h5}. Build it first.")
    elif name == "parsec":
        from lachesis.grid.parsec import PARSECModelGrid
        h5 = MIST_GRID_DIR.parent.parent / "parsec" / "grids" / "parsec_v1.2S.h5"
        if h5.exists():
            return PARSECModelGrid.from_hdf5(h5)
        raw = PARSEC_DIR
        if raw.exists() and list(raw.glob("*.csv")):
            return PARSECModelGrid(raw)
        raise GridError(f"PARSEC grid not found. Download via ezpadova first.")
    elif name == "dartmouth":
        from lachesis.config import DARTMOUTH_GRID_DIR, DARTMOUTH_RAW_DIR
        from lachesis.grid.dartmouth import DartmouthModelGrid
        h5 = DARTMOUTH_GRID_DIR / "dartmouth_dsep.h5"
        if h5.exists():
            return DartmouthModelGrid.from_hdf5(h5)
        raw = DARTMOUTH_RAW_DIR
        if raw.exists() and list(raw.glob("*.iso")):
            return DartmouthModelGrid(raw)
        raise GridError(
            f"Dartmouth grid not found. Run: "
            f"python -m lachesis.grid.dartmouth_download"
        )
    elif name == "basti":
        from lachesis.config import BASTI_GRID_DIR, BASTI_RAW_DIR
        from lachesis.grid.basti import BaSTIModelGrid
        h5 = BASTI_GRID_DIR / "basti.h5"
        if h5.exists():
            return BaSTIModelGrid.from_hdf5(h5)
        raw = BASTI_RAW_DIR
        if raw.exists() and list(raw.glob("*.dat")):
            return BaSTIModelGrid(raw)
        raise GridError(
            f"BaSTI grid not found. Run: "
            f"python -m lachesis.grid.basti_download"
        )
    elif name == "yapsi":
        from lachesis.config import YAPSI_GRID_DIR, YAPSI_RAW_DIR
        from lachesis.grid.yapsi import YAPSIModelGrid
        h5 = YAPSI_GRID_DIR / "yapsi.h5"
        if h5.exists():
            return YAPSIModelGrid.from_hdf5(h5)
        raw = YAPSI_RAW_DIR
        fits_files = list(raw.glob("*.fits"))
        if raw.exists() and fits_files:
            return YAPSIModelGrid(raw)
        raise GridError(
            f"YAPSI grid not found. Run: "
            f"python -m lachesis.grid.yapsi_download"
        )
    elif name == "geneva":
        from lachesis.config import GENEVA_GRID_DIR, GENEVA_RAW_DIR
        from lachesis.grid.geneva import GenevaModelGrid
        h5 = GENEVA_GRID_DIR / "geneva.h5"
        if h5.exists():
            return GenevaModelGrid.from_hdf5(h5)
        raw = GENEVA_RAW_DIR
        if raw.exists() and list(raw.glob("Isochr_*.dat")):
            return GenevaModelGrid(raw)
        raise GridError(
            f"Geneva grid not found. Run: "
            f"python -m lachesis.grid.geneva_download"
        )
    elif name == "bhac15":
        from lachesis.config import BHAC15_GRID_DIR, BHAC15_RAW_DIR
        from lachesis.grid.bhac15 import BHAC15ModelGrid
        h5 = BHAC15_GRID_DIR / "bhac15.h5"
        if h5.exists():
            return BHAC15ModelGrid.from_hdf5(h5)
        raw = BHAC15_RAW_DIR
        if raw.exists() and list(raw.glob("BHAC15_iso.*")):
            return BHAC15ModelGrid(raw)
        raise GridError(
            f"BHAC15 grid not found. Run: "
            f"python -m lachesis.grid.bhac15_download"
        )
    elif name == "starevol":
        from lachesis.config import STAREVOL_GRID_DIR, STAREVOL_RAW_DIR
        from lachesis.grid.starevol import STAREVOLModelGrid
        h5 = STAREVOL_GRID_DIR / "starevol.h5"
        if h5.exists():
            return STAREVOLModelGrid.from_hdf5(h5)
        raw = STAREVOL_RAW_DIR
        if raw.exists() and list(raw.glob("Isochr_*.dat")):
            return STAREVOLModelGrid(raw)
        raise GridError(
            f"STAREVOL grid not found. Run: "
            f"python -m lachesis.grid.starevol_download"
        )
    else:
        raise GridError(
            f"Unknown grid '{name}'. Available: "
            f"mist, parsec, dartmouth, basti, yapsi, geneva, bhac15, starevol"
        )


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

        # Load grids
        for name in self._grids:
            if self._verbose:
                print(f"\t\t\t\tLoading grid : {name.upper()}")
            self._grid_objects[name] = _load_grid(name)
            self._interpolators[name] = make_interpolator(self._grid_objects[name])

        # Load BC table if photometric mode
        if self._bc_system or self._star.mode == "photometric":
            from lachesis.bc import BCTable
            from lachesis.config import MIST_RAW_DIR
            bc_dir = MIST_RAW_DIR / "BC_tables"
            if not bc_dir.exists():
                bc_dir = MIST_RAW_DIR.parent  # fallback
            if self._bc_system:
                # Explicit single system requested
                self._bc_table = BCTable(bc_dir, system=self._bc_system)
            else:
                # Load all available systems (WISE, SDSS, PanSTARRS, etc.)
                self._bc_table = BCTable.multi_system(bc_dir)
            # Restrict BC to only the bands the star actually uses
            used_bands = [
                k for k in self._star.observed
                if k in self._bc_table._band_indices
            ]
            if used_bands:
                self._bc_table.set_active_bands(used_bands)

        # Parse priors
        feh_prior = None
        distance_range = None
        av_range = None
        if self._prior_setup:
            feh_cfg = self._prior_setup.get("feh")
            if feh_cfg and feh_cfg[0] == "normal":
                feh_prior = feh_cfg
        if self._star.mode == "photometric" or self._bc_system:
            # Use parallax-derived distance as tight range if available
            if self._star.distance is not None and self._star.distance_e is not None:
                d = self._star.distance
                d_e = self._star.distance_e
                distance_range = (max(0.1, d - 5 * d_e), d + 5 * d_e)
            else:
                distance_range = (1.0, 10000.0)
            av_range = (0.0, self._star.Av or 1.0)

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
                bc_table=self._bc_table,
                distance_range=distance_range,
                av_range=av_range,
                vini_range=vini_range,
                binary=self._binary,
            )

        self._initialized = True

        if self._verbose:
            display_routine(self)

    def show_priors(self):
        """Display prior configuration."""
        if not self._initialized:
            raise InputError("Call initialize() first.")
        fitter = next(iter(self._fitters.values()))
        p = fitter.prior
        print(f"\t\t\t\t{'Parameter':12s}  Prior")
        print(f"\t\t\t\t{'-' * 40}")
        for name in p.param_names:
            if name == "eep":
                print(f"\t\t\t\t{'eep':12s}  U({p.eep_lo:.0f}, {p.eep_hi:.0f})")
            elif name == "log_age":
                print(f"\t\t\t\t{'log_age':12s}  U({p.age_lo:.2f}, {p.age_hi:.2f})")
            elif name == "feh":
                if p._feh_type == "gaussian":
                    print(f"\t\t\t\t{'feh':12s}  N({p._feh_mean:.3f}, {p._feh_sigma:.3f})")
                else:
                    print(f"\t\t\t\t{'feh':12s}  U({p.feh_lo:.2f}, {p.feh_hi:.2f})")
            elif name == "distance":
                print(f"\t\t\t\t{'distance':12s}  U({p.dist_lo:.0f}, {p.dist_hi:.0f})")
            elif name == "av":
                print(f"\t\t\t\t{'av':12s}  U({p.av_lo:.2f}, {p.av_hi:.2f})")
            elif name == "eep_secondary":
                print(f"\t\t\t\t{'eep_2nd':12s}  U({p.eep_lo:.0f}, eep_primary)")
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
        )
        idata.to_netcdf(nc_path)
        save_summary_dat(dat_path, {
            "samples": bma_result.samples,
            "derived": bma_result.derived,
            "logz": bma_result.log_evidences.max(),
            "logzerr": 0.0,
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
