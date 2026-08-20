"""Fitter class, ARIADNE-compatible property-based isochrone fitter."""

import multiprocessing as mp
import os
import time
import warnings

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

# Set by Fitter._fit_grids_parallel before forking the grid Pool; read by the
# worker processes (which inherit it via fork).
_BMA_FITTER = None


def _bma_grid_worker(name):
    """Fit one BMA grid single-threaded in a forked worker.

    Inherits the initialised Fitter (loaded grids, interpolators, numba
    kernels) from the parent via fork, so only the small result dict is pickled
    back. Returns ``(name, result)`` on success or ``(name, exception)`` on
    failure (the parent re-raises). Reseeds numpy's RNG so each forked grid
    draws its nested-sampling proposals independently.
    """
    fitter = _BMA_FITTER
    np.random.seed()
    try:
        result = fitter._fit_one_grid(name, fitter._parsed_setup)
        return name, result
    except Exception as e:  # noqa: BLE001 - delivered to parent for re-raise
        return name, e


def _log_box(lo, hi):
    """log of a uniform prior's width, treating a zero width as a delta.

    A dimension of zero width carries no prior volume, so it contributes 0 to
    the common-scale evidence correction and is identical across grids. Taking
    log(0) instead put -inf into that grid's log-evidence, which propagates
    into the BMA as NaN weights and an empty combined posterior that the run
    reports as a success. Single-metallicity grids (geneva, bhac15) clamp to a
    zero-width [Fe/H] box, and a feh_range narrowed past a grid's axis inverts
    the interval outright.
    """
    w = hi - lo
    if not (w > 0) or not np.isfinite(w):
        return 0.0
    return float(np.log(w))


def _is_zero_support_error(exc) -> bool:
    """True when a grid fit died because no point in the prior volume has a
    finite likelihood, i.e. the star lies outside the grid's coverage (e.g.
    an evolved star beyond YaPSI's EEP/age extent). Matches dynesty's
    initial-point failures ("could not find a single point that have a valid
    log-likelihood" / "Not a single provided live point has a valid
    log-likelihood"); NaN/+inf likelihood bugs raise a different message
    ("is invalid") and do not match.
    """
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if "valid log-likelihood" in str(e):
            return True
        e = e.__cause__ or e.__context__
    return False


def _grid_min_logg(grid) -> float | None:
    """Smallest finite log g anywhere in a loaded grid cube, or None if the
    grid has no log_g column. Measures how far down the giant branch the grid
    reaches (YaPSI bottoms out at 3.69; MIST/PARSEC/BaSTI go below 0)."""
    cols = list(grid.columns)
    if "log_g" not in cols:
        return None
    lg = grid._data[..., cols.index("log_g")]
    lg = lg[np.isfinite(lg)]
    return float(lg.min()) if lg.size else None


_GRID_REGISTRY = {
    "mist": ("lachesis.grid.mist", "MISTModelGrid", "mist_v1.2_vvcrit0.4.h5"),
    "parsec": ("lachesis.grid.parsec", "PARSECModelGrid", "parsec_v1.2S_eeprebuild.h5"),
    "dartmouth": ("lachesis.grid.dartmouth", "DartmouthModelGrid", "dartmouth_dsep.h5"),
    "basti": ("lachesis.grid.basti", "BaSTIModelGrid", "basti.h5"),
    # yapsi.h5 has real central-H-anchored EEPs on the Dotter scale, built from
    # the YaPSI tracks by scripts/rebuild_yapsi_eep.py. (It replaced an earlier
    # mass-index parameterization that mis-interpolated the turnoff/subgiant and
    # was punted by the BMA.)
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
        self._grids = ["mist", "parsec", "dartmouth", "basti", "yapsi", "geneva"]
        self._bma = False
        self._binary = False
        self._verbose = True
        self._out_folder = None
        self._setup = ["dynesty"]
        self._prior_setup = None
        self._av_law = "fitzpatrick"
        self._imf = "chabrier"
        self._eep_range = (200, 808)
        # Upper bound: age of the universe (13.8 Gyr, Planck 2018). Also
        # equalizes the age support across grids with different native
        # ceilings (MIST 10.30, PARSEC 10.25, DSEP/BaSTI/YaPSI 10.176), so
        # the common-measure evidence correction cannot reward a grid for
        # carrying isochrones into the unphysical old regime. Override via
        # the age_range setter for e.g. grid-systematics experiments.
        self._age_range = (8.0, 10.1399)
        self._feh_range = (-2.0, 0.5)
        self._bc_system = None
        self._parsed_setup = None
        self._initialized = False
        # Grids dropped at fit time for having no finite-likelihood support
        # for this star (list of (grid_name, reason)); audit trail for the
        # graceful BMA degradation in fit_bma().
        self.dropped_grids = []
        self._grid_objects = {}
        self._interpolators = {}
        self._fitters = {}
        self._bc_table = None
        # Number of BMA model grids to fit concurrently, each single-threaded.
        # 1 (default) = sequential (best for batch runs, where parallelism is
        # at the star/process level). Set to len(grids) for an interactive
        # single-star run to fit every grid at once. Mirrors ARIADNE's
        # n_grid_jobs. Each job is a forked process holding its own grid, so
        # pick a value the machine's cores + memory can handle.
        self._n_grid_jobs = 1
        # Fit an ARIADNE-style photometric excess-noise (jitter) term, in mag,
        # added in quadrature to every band's catalogue uncertainty. Catalogue
        # photometric errors are routinely optimistic; at the main-sequence
        # turnoff that optimism collapses the age posterior to an unphysically
        # tight (and biased-old) value. Marginalising over the jitter restores
        # honest uncertainties. Log-uniform prior over (1e-3, 0.5) mag.
        self._fit_jitter = True
        self._jitter_range = (1e-3, 0.5)

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
    def n_grid_jobs(self) -> int:
        """Number of BMA grids to fit concurrently (1 = sequential)."""
        return self._n_grid_jobs

    @n_grid_jobs.setter
    def n_grid_jobs(self, n: int):
        n = int(n)
        if n < 1:
            raise InputError("n_grid_jobs must be >= 1")
        self._n_grid_jobs = n

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
    def fit_jitter(self) -> bool:
        """Whether to fit a photometric excess-noise (jitter) term."""
        return self._fit_jitter

    @fit_jitter.setter
    def fit_jitter(self, value: bool):
        self._fit_jitter = bool(value)

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
        self.dropped_grids = []

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

        unusable = self._star.unusable_observables()
        if unusable:
            verb = "carries" if len(unusable) == 1 else "carry"
            warnings.warn(
                f"{self._star.starname}: "
                + ", ".join(unusable)
                + f" {verb} no usable uncertainty, so it is NOT constraining "
                "this fit. A quantity fixed in the upstream fit enters through "
                "its prior instead; otherwise supply an uncertainty.",
                RuntimeWarning,
            )

        # Parse priors (matching ARIADNE's prior_setup dict pattern)
        ps = self._prior_setup or {}

        # [Fe/H] prior priority:
        #   1. User override in prior_setup (normal, morton, rave)
        #   2. Full ARIADNE posterior samples (KDE), preserves shape
        #   3. Spectroscopic prior from star.feh/feh_e (gaussian)
        #   4. Uniform (default)
        feh_prior = None
        feh_cfg = ps.get("feh") or ps.get("z")
        if feh_cfg and isinstance(feh_cfg, tuple):
            if feh_cfg[0] == "normal":
                feh_prior = ("gaussian", feh_cfg[1], feh_cfg[2])
            elif feh_cfg[0] == "fixed":
                feh_prior = ("fixed", float(feh_cfg[1]))
            elif feh_cfg[0] == "uniform":
                feh_prior = None
                self._feh_range = (float(feh_cfg[1]), float(feh_cfg[2]))
            elif feh_cfg[0] == "morton":
                feh_prior = ("morton",)
            elif feh_cfg[0] == "rave":
                feh_prior = ("rave",)
            elif feh_cfg[0] == "default":
                # Explicit "no override": fall through to the ARIADNE
                # posterior / spectroscopic / uniform chain below.
                feh_prior = None
            else:
                # Silently ignoring an unrecognised head sent the user's
                # explicit prior straight past every branch and on to the
                # ARIADNE KDE default, i.e. it accepted the instruction and
                # did nothing with it. ARIADNE's own key here is 'z' and it
                # accepts 'fixed', so the spelling carries over.
                raise InputError(
                    f"Unrecognised [Fe/H] prior {feh_cfg[0]!r}. Use "
                    f"('normal', mu, sigma), ('fixed', value), "
                    f"('uniform', lo, hi), ('morton',), ('rave',) or ('default',)."
                )
        elif feh_cfg and isinstance(feh_cfg, str):
            if feh_cfg == "morton":
                feh_prior = ("morton",)
            elif feh_cfg == "rave":
                feh_prior = ("rave",)
            elif feh_cfg != "default":
                raise InputError(
                    f"Unrecognised [Fe/H] prior {feh_cfg!r}. Use 'morton', "
                    f"'rave', 'default', or a tuple form."
                )
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
        elif self._star.distance is not None and self._star.distance <= 70.0:
            # Inside the Local Bubble (~70 pc) interstellar extinction is
            # negligible. Fix Av = 0 by dropping it from the sampled vector
            # (av_range=None -> the likelihood uses av=0) rather than letting a
            # noisy dustmap value or the sampler introduce spurious reddening.
            # An explicit user override in prior_setup still takes precedence.
            av_range = None
        else:
            av_hi = 1.0 if self._star.Av is None else max(self._star.Av, 1e-3)
            av_range = (0.0, av_hi)

        # Photometric jitter: only meaningful when the star carries photometry
        # (it enters the likelihood through the BC bands). Without it the term
        # would be an unconstrained nuisance dimension sampled straight from the
        # prior, so skip it for spectroscopy-only fits.
        jitter_range = (
            self._jitter_range if (self._fit_jitter and used_bands) else None
        )

        # Build KDE-based external priors from ARIADNE posteriors.
        # Pre-tabulate log(pdf) on a fine grid and interpolate with np.interp
        # at evaluation time, O(1) per call instead of O(N_samples).
        external_kdes = {}
        if self._star.external_posteriors:
            from scipy.stats import gaussian_kde
            for param, samples in self._star.external_posteriors.items():
                if param == "Av" and av_range is None:
                    # An Av prior with no sampled Av is unsatisfiable. Inside
                    # the Local Bubble av_range is None, so 'Av' is not in the
                    # parameter vector and the sampler reads it back as None,
                    # which returns -inf for EVERY proposal. Verified end to
                    # end: the fit dies in dynesty's live-point initialisation
                    # and is misreported as a grid-coverage failure. The Av = 0
                    # assumption supersedes the upstream prior. Reconcile this
                    # BEFORE the spread test below, or a fixed upstream Av
                    # aborts the run on a prior we were going to delete anyway.
                    warnings.warn(
                        f"{self._star.starname} is inside the Local Bubble "
                        f"(d <= 70 pc), where Av is fixed at 0 and not "
                        f"sampled. Dropping the upstream Av prior, which "
                        f"cannot be applied to a parameter that is not being "
                        f"fitted.",
                        RuntimeWarning,
                    )
                    continue
                samples = np.asarray(samples, dtype=float).ravel()
                finite = samples[np.isfinite(samples)]
                if float(np.ptp(finite)) == 0.0 and finite.size:
                    # A constant column means the parameter was FIXED upstream.
                    # Skipping it silently deleted a whole constraint from the
                    # likelihood without a word, and the surviving-priors line
                    # below lists only what was kept, so nothing showed it.
                    # Tabulating it anyway is worse: lo == hi makes pad zero,
                    # the 2048-node grid collapses to a single point, and the
                    # sampler's range check then rejects every proposal, which
                    # surfaces as dynesty finding no valid log-likelihood and
                    # is reported as the star lying outside every grid.
                    raise InputError(
                        f"The external prior for '{param}' is constant at "
                        f"{float(finite[0]):.6g} over {finite.size} sample(s); "
                        f"it was FIXED in the upstream fit. A delta constraint "
                        f"has zero measure and makes every likelihood "
                        f"evaluation -inf. Re-run the upstream fit with "
                        f"'{param}' free, or remove it with "
                        f"star.external_posteriors.pop({param!r})."
                    )
                if finite.size < 10:
                    # A different failure with a different remedy, so do not
                    # share the message above. Too few samples to estimate a
                    # density from; skip, but not in silence.
                    warnings.warn(
                        f"The external prior for '{param}' has only "
                        f"{finite.size} finite sample(s), too few to build a "
                        f"KDE from. It is NOT constraining this fit.",
                        RuntimeWarning,
                    )
                    continue
                kde = gaussian_kde(finite)
                lo, hi = float(finite.min()), float(finite.max())
                std = float(np.std(finite))
                # Pad ≥5σ so the KDE tails aren't clipped to a hard wall.
                pad = max(0.1 * (hi - lo), 5.0 * std)
                grid = np.linspace(lo - pad, hi + pad, 2048)
                log_pdf = np.log(np.maximum(kde(grid), 1e-300))
                external_kdes[param] = (grid, log_pdf)

            if self._verbose and external_kdes:
                print(f"\t\t\t External priors (KDE): {', '.join(external_kdes)}")

        # Drop grids whose [Fe/H] coverage doesn't overlap the star's prior.
        # Without this, dynesty wastes 1000 init attempts before raising
        # "could not find a single point that have a valid log-likelihood"
        #, e.g. YAPSI ([-0.75, +0.55]) asked to fit a halo star with
        # feh_prior = N(-1.47, 0.07). The clamp on grid_feh_lo/hi below
        # narrows the prior interval but cannot rescue a Gaussian prior
        # whose ±3σ window is entirely outside the grid axis.
        # The test needs a central value, not a prior type. Gating it on
        # "gaussian" meant it never ran on the default ARIADNE path, which
        # produces ("kde", samples), so the protection it was written to give
        # was absent exactly where it is needed most. A fixed [Fe/H] outside a
        # grid's axis is the same failure with none of the slack.
        feh_centre = None
        if feh_prior is not None:
            if feh_prior[0] == "gaussian":
                feh_centre = float(feh_prior[1])
                feh_desc = f"N({feh_centre:.2f}, {float(feh_prior[2]):.2f})"
            elif feh_prior[0] == "fixed":
                feh_centre = float(feh_prior[1])
                feh_desc = f"fixed at {feh_centre:+.2f}"
            elif feh_prior[0] == "kde":
                _fs = np.asarray(feh_prior[1], dtype=float).ravel()
                _fs = _fs[np.isfinite(_fs)]
                if _fs.size:
                    feh_centre = float(np.median(_fs))
                    feh_desc = f"posterior median {feh_centre:+.2f}"
        if feh_centre is not None:
            mu = feh_centre
            dropped = []
            kept = []
            for name in self._grids:
                grid = self._grid_objects[name]
                g_lo = float(grid.feh_values[0])
                g_hi = float(grid.feh_values[-1])
                # Drop a grid whose [Fe/H] axis does not contain the prior's
                # central value. Otherwise the sampler pins [Fe/H] at the grid
                # boundary and returns a metallicity-biased (boundary) solution
                # that can still score high evidence and corrupt the BMA. The
                # previous ±3σ-overlap test kept such grids whenever a wide
                # prior merely grazed the grid edge, e.g. a narrow grid like
                # Geneva ([-0.33, +0.54]) on a star with [Fe/H] < -0.33.
                if mu < g_lo or mu > g_hi:
                    dropped.append((name, g_lo, g_hi))
                else:
                    kept.append(name)
            if dropped:
                # Not gated on verbose, and recorded. This test used to run
                # only for a Gaussian prior, so it almost never fired; now
                # that it covers the ARIADNE path it fires on ordinary
                # pipeline stars, and a batch run must not lose grids from
                # the BMA without a warning or a provenance record.
                self.dropped_grids.extend(
                    (n, "[Fe/H] axis does not cover the prior centre")
                    for n, _lo, _hi in dropped
                )
                warnings.warn(
                    f"Auto-dropped {len(dropped)} grid(s) from BMA whose "
                    f"[Fe/H] axis does not contain the star's [Fe/H] prior "
                    f"({feh_desc}): "
                    + ", ".join(f"{n} ([{lo:.2f}, {hi:.2f}])"
                                for n, lo, hi in dropped),
                    RuntimeWarning,
                )
            if not kept:
                raise InputError(
                    f"All grids have [Fe/H] coverage incompatible with the "
                    f"star's [Fe/H] prior ({feh_desc}). Verify the prior is "
                    f"plausible for the star."
                )
            if self._bma and len(kept) < 2:
                # BMA needs ≥2 grids; if only one survives the feh filter,
                # downgrade to single-grid fit silently.
                self._bma = False
            self._grids = kept
            self._grid_objects = {k: v for k, v in self._grid_objects.items() if k in kept}
            self._interpolators = {k: v for k, v in self._interpolators.items() if k in kept}

        # Drop grids that cannot represent an evolved star. YaPSI bottoms out
        # at log g 3.69 (BHAC15 at 3.23, STAREVOL at 3.81), so a star
        # pre-classified as a giant (spectroscopic log g, FLAME radius, or
        # dereddened CMD position; see Star.evolutionary_state) has zero
        # likelihood support there and dynesty would burn its init attempts
        # before failing. Mirrors the [Fe/H]-coverage drop above; the
        # fit-time zero-support drop in fit_bma() remains the backstop for
        # misclassified stars. Acts only on the unambiguous 'giant' state, # subgiants stay with every grid (partial coverage is enough for
        # the sampler to find support).
        if getattr(self._star, "evolutionary_state", "unknown") == "giant":
            dropped_g, kept_g = [], []
            for name in self._grids:
                min_lg = _grid_min_logg(self._grid_objects[name])
                if min_lg is not None and min_lg > 3.2:
                    dropped_g.append((name, min_lg))
                else:
                    kept_g.append(name)
            if dropped_g:
                warnings.warn(
                    f"{self._star.starname} is pre-classified as a giant "
                    f"(Star.evolutionary_state); dropped grid(s) without "
                    f"giant coverage: "
                    + ", ".join(f"{n} (min log g {lg:.2f})"
                                for n, lg in dropped_g),
                    RuntimeWarning,
                )
                self.dropped_grids.extend(
                    (n, "no giant coverage") for n, _ in dropped_g)
            if not kept_g:
                raise InputError(
                    f"{self._star.starname} is pre-classified as a giant but "
                    f"no requested grid reaches log g < 3.2. Fit with grids "
                    f"that cover the giant branch (mist, parsec, dartmouth, "
                    f"basti)."
                )
            if self._bma and len(kept_g) < 2:
                self._bma = False
            self._grids = kept_g
            self._grid_objects = {k: v for k, v in self._grid_objects.items() if k in kept_g}
            self._interpolators = {k: v for k, v in self._interpolators.items() if k in kept_g}

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
                jitter_range=jitter_range,
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
                elif p._feh_type == "fixed":
                    print(colored(f"{t3}{'feh':12s}  FIXED at {p._feh_value:+.4f}", c))
                elif p.feh_hi <= p.feh_lo:
                    # Single-metallicity grid: a zero-width box is a delta,
                    # not the wide uniform the else branch would print.
                    print(colored(f"{t3}{'feh':12s}  FIXED at {p.feh_lo:+.4f} "
                                  f"(single-metallicity grid)", c))
                else:
                    print(colored(f"{t3}{'feh':12s}  U({p.feh_lo:.2f}, {p.feh_hi:.2f})", c))
            elif name == "distance":
                print(colored(f"{t3}{'distance':12s}  N({p._dist_mean:.3f}, {p._dist_sigma:.3f})", c))
            elif name == "Av":
                print(colored(f"{t3}{'Av':12s}  U({p.av_lo:.2f}, {p.av_hi:.2f})", c))
            elif name == "eep_secondary":
                print(colored(f"{t3}{'eep_2nd':12s}  U({p.eep_lo:.0f}, eep_primary)", c))
            elif name == "vini":
                print(colored(f"{t3}{'vini':12s}  U({p.vini_lo:.2f}, {p.vini_hi:.2f})", c))
            elif name.endswith("_noise"):
                lbl = name[:-len("_noise")]
                print(colored(f"{t3}{lbl:12s}  log-U({p.jit_lo:.3f}, {p.jit_hi:.2f}) mag", c))
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

        pool, threads = self._build_pool(setup)
        try:
            result = self._fitters[name].fit(
                observed=self._star.observed,
                uncertainties=self._star.uncertainties,
                nlive=setup["nlive"],
                dlogz=setup["dlogz"],
                sample=setup["sample"],
                **self._pool_kwargs(pool, threads),
            )
        finally:
            if pool is not None:
                pool.close(); pool.join()

        if self._verbose:
            display_summary(
                result["samples"], result["derived"],
                self._fitters[name].prior.param_names,
            )
            display_elapsed(t0)

        self._save(result, name)
        return result

    @staticmethod
    def _build_pool(setup: dict):
        """No inner likelihood pool: the per-grid fit runs single-threaded.

        A ``ThreadPool`` over the likelihood was measured to be pure OVERHEAD,
        not parallelism. dynesty calls a Python ``loglike`` closure (which holds
        the GIL for theta unpacking + dispatch) wrapping a fast (~20 us) njit
        kernel, so the pool's per-proposal dispatch cost dominates and makes the
        fit ~1.6x SLOWER, and worse the more likelihood calls there are, i.e.
        the more dimensions/photometric bands. (Marking the kernel ``nogil`` did
        not help: the GIL-free window is too small next to the wrapper + queue
        overhead.) BMA parallelism comes from the grid-level PROCESS pool
        (``n_grid_jobs``) instead; ``setup['threads']`` is kept for API
        compatibility but no longer builds an inner pool.
        """
        return None, 1

    @staticmethod
    def _pool_kwargs(pool, threads: int) -> dict:
        """Build the dynesty kwargs to forward when a pool exists.

        dynesty requires ``queue_size`` ≥ 2 to use the pool, otherwise
        proposals run serially even with a pool attached.
        """
        if pool is None:
            return {}
        return {"pool": pool, "queue_size": max(2, threads)}

    def _fit_one_grid(self, name, setup, pool=None, threads=1):
        """Run the nested-sampling fit for a single grid."""
        return self._fitters[name].fit(
            observed=self._star.observed,
            uncertainties=self._star.uncertainties,
            nlive=setup["nlive"],
            dlogz=setup["dlogz"],
            sample=setup["sample"],
            **self._pool_kwargs(pool, threads),
        )

    def _fit_grids_sequential(self, setup) -> dict:
        """Fit grids one at a time (optionally with an inner likelihood pool)."""
        results = {}
        pool, threads = self._build_pool(setup)
        try:
            for name in self._grids:
                if self._verbose:
                    display_fitting_grid(name)
                try:
                    results[name] = self._fit_one_grid(name, setup, pool, threads)
                except Exception as e:
                    if _is_zero_support_error(e):
                        # The star is outside this grid's coverage (e.g. an
                        # evolved star beyond YaPSI's extent): degrade
                        # gracefully, average the surviving grids.
                        warnings.warn(
                            f"Grid '{name}' has no finite-likelihood support "
                            f"for {self._star.starname} (star outside grid "
                            f"coverage); dropping it from the BMA.",
                            RuntimeWarning,
                        )
                        self.dropped_grids.append((name, "no likelihood support"))
                        continue
                    raise FittingError(
                        f"Fitting failed on grid {name}: {e}",
                        grid_name=name, partial_results=results,
                    )
        finally:
            if pool is not None:
                pool.close(); pool.join()
        return results

    def _fit_grids_parallel(self, setup, n_jobs: int) -> dict:
        """Fit grids concurrently, one forked process per grid (single-thread).

        Mirrors ARIADNE's n_grid_jobs: the grids are independent fits, so a
        forked Pool runs several at once. Workers inherit the fully-initialised
        Fitter (loaded grids, interpolators, numba-compiled kernels) via fork, only the per-grid result dict is pickled back. Each grid runs
        single-threaded; grid-level parallelism replaces the inner pool.
        """
        global _BMA_FITTER
        ncpu = os.cpu_count() or 1
        if n_jobs > ncpu:
            warnings.warn(
                f"n_grid_jobs={n_jobs} exceeds the {ncpu} available CPU cores; "
                "this will oversubscribe the machine.", RuntimeWarning,
            )
        if self._verbose:
            print(f"Fitting {len(self._grids)} grids, {n_jobs} concurrently")
        _BMA_FITTER = self
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(n_jobs) as pool:
                pairs = pool.map(_bma_grid_worker, self._grids)
        finally:
            _BMA_FITTER = None
        results = {}
        for name, res in pairs:
            if isinstance(res, BaseException):
                if _is_zero_support_error(res):
                    # Star outside this grid's coverage: degrade gracefully,
                    # average the surviving grids.
                    warnings.warn(
                        f"Grid '{name}' has no finite-likelihood support for "
                        f"{self._star.starname} (star outside grid coverage); "
                        f"dropping it from the BMA.",
                        RuntimeWarning,
                    )
                    self.dropped_grids.append((name, "no likelihood support"))
                    continue
                raise FittingError(
                    f"Fitting failed on grid {name}: {res}",
                    grid_name=name, partial_results=results,
                )
            results[name] = res
        # Preserve grid order
        return {name: results[name] for name in self._grids if name in results}

    def fit_bma(self):
        """Fit all grids and combine via Bayesian Model Averaging."""
        if not self._initialized:
            raise InputError("Call initialize() first.")
        if len(self._grids) < 2 and self._bma:
            raise InputError("BMA requires at least 2 grids.")

        t0 = time.time()
        setup = self._parsed_setup

        n_jobs = min(max(int(self._n_grid_jobs), 1), len(self._grids))
        if n_jobs > 1:
            results = self._fit_grids_parallel(setup, n_jobs)
        else:
            results = self._fit_grids_sequential(setup)

        if not results:
            raise FittingError(
                f"No grid has finite-likelihood support for "
                f"{self._star.starname}: the star lies outside the coverage "
                f"of every grid in the BMA "
                f"({', '.join(n for n, _ in self.dropped_grids)}). Verify the "
                f"observables, or fit with grids that cover this regime."
            )

        # A-posteriori [Fe/H] edge-railing drop. When a grid's metallicity is
        # bounded only by its own axis (no informative [Fe/H] prior pinned it
        # pre-emptively in initialize()), a star whose true [Fe/H] lies outside
        # the grid piles the posterior against the grid edge and returns a
        # boundary-biased solution that can still score high evidence, e.g. a
        # sub-(-0.33) star on the Geneva grid. Exclude any grid whose [Fe/H]
        # posterior rails against its axis boundary, but never drop them all.
        if len(results) > 1:
            rail_kept, railed = {}, []
            for name, res in results.items():
                grid = self._grid_objects[name]
                g_lo = float(grid.feh_values[0])
                g_hi = float(grid.feh_values[-1])
                pn = res.get("param_names") or self._fitters[name].prior.param_names
                fe = np.asarray(res["samples"])[:, pn.index("feh")]
                feh_type = getattr(
                    self._fitters[name].prior, "_feh_type", "uniform")
                if feh_type == "fixed":
                    # [Fe/H] was fixed, so there is no posterior to rail. The
                    # test is a step function on a constant column, and a fixed
                    # value that happens to sit within tol of an axis edge would
                    # drop an otherwise fine grid (tol is 0.09 dex on MIST).
                    #
                    # Key this on the prior TYPE, not on np.ptp(fe) == 0. A
                    # single-metallicity grid (geneva, bhac15: feh_values is
                    # the single point [0.0]) also yields a constant column,
                    # but via a zero-width UNIFORM box, and such a grid is
                    # railed by construction. Bypassing it on spread alone let
                    # geneva into the default BMA and put a delta atom at
                    # exactly 0.0 into the combined [Fe/H] posterior.
                    rail_kept[name] = res
                    continue
                tol = max(0.03, 0.02 * (g_hi - g_lo))
                frac_edge = float(np.mean((fe <= g_lo + tol) | (fe >= g_hi - tol)))
                if frac_edge >= 0.5:
                    railed.append((name, frac_edge, g_lo, g_hi))
                else:
                    rail_kept[name] = res
            if railed and not rail_kept:
                # Every grid railed. Dropping them all would leave nothing, so
                # the guard below declines to act, but staying silent about it
                # disarms the one diagnostic that catches a broken [Fe/H] prior
                # precisely when it is broken everywhere.
                warnings.warn(
                    f"The [Fe/H] posterior railed against the axis edge on ALL "
                    f"{len(railed)} grid(s) for {self._star.starname} "
                    + ", ".join(f"{n} ({f:.0%})" for n, f, _lo, _hi in railed)
                    + ". That points at the [Fe/H] prior rather than at the "
                    "grids; no grid was dropped. Check the metallicity prior "
                    "before using these results.",
                    RuntimeWarning,
                )
            if railed and rail_kept:
                for name, frac, lo, hi in railed:
                    self.dropped_grids.append(
                        (name, "[Fe/H] posterior railed against grid edge"))
                    if self._verbose:
                        warnings.warn(
                            f"Grid '{name}' [Fe/H] posterior railed against its "
                            f"axis edge for {self._star.starname} ({frac:.0%} of "
                            f"samples at [{lo:.2f}, {hi:.2f}]); excluded from BMA.",
                            RuntimeWarning,
                        )
                results = rail_kept
                self._grids = [n for n in self._grids if n in results]
                if self._bma and len(results) < 2:
                    self._bma = False

        if self.dropped_grids and len(results) == 1:
            warnings.warn(
                f"Only one grid ({next(iter(results))}) survived for "
                f"{self._star.starname}; the 'BMA' result is a single-grid "
                f"posterior with no model-averaging.",
                RuntimeWarning,
            )

        # Common-scale evidence normalisation for the BMA weights. Each grid is
        # sampled uniformly in EEP, log-age, and [Fe/H] over its OWN coverage
        # box (bounds clamped in initialize()), so dynesty's evidence carries a
        # per-grid prior-volume factor 1/box that is unrelated to fit quality:
        # a grid with narrower coverage or a finer EEP index would otherwise get
        # an arbitrary boost or penalty in the weights (e.g. YaPSI's narrow mass
        # range, BaSTI's ~2000-point mass index). We add back ln(box volume) of
        # the uniformly sampled dimensions, which puts every grid on a common
        # prior scale (universal IMF, flat log-age and [Fe/H]) so the weights
        # compare marginal likelihoods rather than coverage. Non-uniform priors
        # (Gaussian/KDE [Fe/H]) are already normalised, so they are left as-is.
        # This shifts each log-evidence by a constant and does not touch the
        # per-grid posteriors.
        for _name, _res in results.items():
            _p = self._fitters[_name].prior
            _off = _log_box(_p.eep_lo, _p.eep_hi)
            if getattr(_p, "_age_type", "log_uniform") == "uniform":
                _off += _log_box(10.0 ** _p.age_lo, 10.0 ** _p.age_hi)
            else:
                _off += _log_box(_p.age_lo, _p.age_hi)
            if getattr(_p, "_feh_type", "uniform") == "uniform":
                _off += _log_box(_p.feh_lo, _p.feh_hi)
            _res["logz"] = float(_res["logz"] + _off)

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
        # param_names from the first grid's prior, same across all BMA grids
        # by construction. Pass them through so save_summary_dat can write
        # the sampled-parameter rows (log_age, feh, distance, Av, eep, ...).
        param_names = self._fitters[self._grids[0]].prior.param_names
        save_summary_dat(dat_path, {
            "samples": bma_result.samples,
            "derived": bma_result.derived,
            "logz": bma_result.log_evidence,
            "logzerr": logzerr_bma,
        }, param_names=param_names)
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
