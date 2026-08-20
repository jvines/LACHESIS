"""Prior transforms and log-prior for isochrone fitting.

The EEP prior P(eep) = IMF(mass) * |dm/dEEP| transforms the initial
mass function into EEP space via the Jacobian dm_deep (Dotter 2016).
Without this, the prior is flat in EEP which over-weights high-mass
evolutionary states.
"""

import math
import warnings

import numpy as np
from scipy.special import ndtr, ndtri

from lachesis.error import PriorError

# Nodes in the tabulated inverse CDF used to sample a KDE-based [Fe/H] prior.
_FEH_CDF_NODES = 1024
# Below this many surviving nodes the tabulation cannot represent the
# distribution at all: 1 node pins every draw at a single value and 2 nodes
# degrade to a uniform ramp across the whole prior range. Both used to happen
# silently, so treat anything this coarse as a fixed [Fe/H] instead.
_FEH_CDF_MIN_NODES = 32


def _log_width(hi, lo):
    """-log of a uniform prior's width, treating a zero width as a delta.

    A fixed parameter is expressed as a zero-width range (Fitter does this for
    prior_setup {'Av': ('fixed', v)}, and single-metallicity grids clamp to it),
    for which -log(0) is +inf. A delta carries no prior volume, so it
    contributes 0 and stays comparable across grids.
    """
    w = hi - lo
    if not (w > 0):
        return 0.0
    return -float(np.log(w))


def _truncated_normal_ppf(u, mean, sigma, lo, hi):
    """Truncated-normal inverse CDF.

    Maps u in [0, 1] to a sample of N(mean, sigma) restricted to [lo, hi].
    Avoids the np.clip-on-Gaussian artefact (delta spikes at boundaries).

    A non-positive sigma collapses the normal to a delta at `mean`. That is
    reachable from an upstream-fixed parameter, whose posterior standard
    deviation is 0 (or a ~1e-17 rounding residue), and the unguarded division
    below raises ZeroDivisionError for a Python float and silently returns NaN
    for a np.float64. Return the delta instead.
    """
    if not (sigma > 0) or not np.isfinite(sigma):
        return float(np.clip(mean, lo, hi))
    a = (lo - mean) / sigma
    b = (hi - mean) / sigma
    Fa = ndtr(a)
    Fb = ndtr(b)
    val = mean + sigma * ndtri(Fa + u * (Fb - Fa))
    if not math.isfinite(val):
        # For a prior narrow enough that Fa underflows to 0 (or Fb to 1),
        # ndtri saturates at -inf/+inf at the ends of the unit cube. dynesty
        # does propose u == 0.0. Return the truncated mean rather than an
        # infinite parameter value.
        return float(np.clip(mean, lo, hi))
    return val


# ---------------------------------------------------------------------------
# Initial Mass Functions
# ---------------------------------------------------------------------------

def salpeter_imf(mass: float) -> float:
    """Salpeter (1955) IMF: dN/dM ∝ M^{-2.35}."""
    if mass <= 0:
        return 0.0
    return mass ** (-2.35)


def chabrier_imf(mass: float) -> float:
    """Chabrier (2003) IMF: lognormal below 1 Msun, power-law above."""
    if mass <= 0:
        return 0.0
    if mass < 1.0:
        return (
            0.158 / mass
            * np.exp(-0.5 * ((np.log10(mass) - np.log10(0.08)) / 0.69) ** 2)
        )
    return 0.0443 * mass ** (-2.3)


def kroupa_imf(mass: float) -> float:
    """Kroupa (2001) IMF: broken power-law with continuity at breakpoints.

    ξ(M) ∝ M^{-α_i} with α = 0.3, 1.3, 2.3 for the three mass segments.
    Normalisation constants enforce continuity at 0.08 and 0.5 Msun.
    """
    if mass <= 0:
        return 0.0
    if mass < 0.08:
        return mass ** (-0.3)
    if mass < 0.5:
        # continuity at 0.08: k1 * 0.08^-1.3 = 0.08^-0.3  =>  k1 = 0.08
        return 0.08 * mass ** (-1.3)
    # continuity at 0.5: k2 * 0.5^-2.3 = 0.08 * 0.5^-1.3  =>  k2 = 0.08 * 0.5
    return 0.04 * mass ** (-2.3)


_IMF_FUNCTIONS = {
    "salpeter": salpeter_imf,
    "chabrier": chabrier_imf,
    "kroupa": kroupa_imf,
}


# ---------------------------------------------------------------------------
# [Fe/H] population priors
# ---------------------------------------------------------------------------

def _feh_prior_morton(feh: float, halo_fraction: float = 0.001) -> float:
    """Morton/Bovy [Fe/H] prior: 2-Gaussian local disk + Gaussian halo.

    Based on a fit to the local SDSS metallicity distribution
    (Casagrande et al. 2011).

    Disk (local):
      - Component 1: weight=0.8, mu=+0.016, sigma=0.15
      - Component 2: weight=0.2, mu=-0.15,  sigma=0.22
    Halo:
      - mu=-1.5, sigma=0.4
    """
    inv_sqrt2pi = 0.3989422804014327
    # Disk: mixture of two Gaussians
    g1 = 0.8 / 0.15 * inv_sqrt2pi * np.exp(
        -0.5 * ((feh - 0.016) / 0.15) ** 2
    )
    g2 = 0.2 / 0.22 * inv_sqrt2pi * np.exp(
        -0.5 * ((feh + 0.15) / 0.22) ** 2
    )
    disk = g1 + g2
    # Halo
    halo = inv_sqrt2pi / 0.4 * np.exp(
        -0.5 * ((feh + 1.5) / 0.4) ** 2
    )
    return (1.0 - halo_fraction) * disk + halo_fraction * halo


def _feh_prior_rave(feh: float) -> float:
    """ARIADNE's default [Fe/H] prior: N(-0.125, 0.234).

    Gaussian summary of the RAVE DR5 metallicity distribution
    (Kunder et al. 2017).
    """
    mu, sigma = -0.125, 0.234
    return (
        0.3989422804014327 / sigma
        * np.exp(-0.5 * ((feh - mu) / sigma) ** 2)
    )


class IsochronePrior:
    """Prior for the isochrone fitting parameter space.

    The EEP prior is P(eep) = IMF(mass) * |dm/dEEP|, not uniform.
    This requires the interpolator to provide initial_mass and dm_deep
    at each proposed (eep, age, feh) point.
    """

    def __init__(
        self,
        eep_range: tuple[float, float],
        age_range: tuple[float, float],
        feh_range: tuple[float, float],
        feh_prior: tuple[str, ...] | None = None,
        age_prior: str = "log_uniform",
        distance_prior: tuple[str, ...] | None = None,
        av_range: tuple[float, float] | None = None,
        vini_range: tuple[float, float] | None = None,
        jitter_range: tuple[float, float] | None = None,
        binary: bool = False,
        imf: str = "chabrier",
    ):
        self.eep_lo, self.eep_hi = eep_range
        self.age_lo, self.age_hi = age_range
        self.feh_lo, self.feh_hi = feh_range
        self._binary = binary
        self._imf = _IMF_FUNCTIONS.get(imf, chabrier_imf)

        # Age prior type: "log_uniform" (flat in log_age, default) or
        # "uniform" (flat in linear age, P(log_age) ∝ 10^log_age).
        self._age_type = age_prior

        # [Fe/H] prior
        if feh_prior is None:
            self._feh_type = "uniform"
        else:
            self._feh_type = feh_prior[0]
            if self._feh_type == "fixed":
                # feh_prior = ("fixed", value)
                self._feh_value = float(feh_prior[1])
            elif self._feh_type == "gaussian":
                self._feh_mean = float(feh_prior[1])
                self._feh_sigma = float(feh_prior[2])
                if not (self._feh_sigma > 0) or not np.isfinite(self._feh_sigma):
                    # N(mu, 0) is a delta, not a normal. Reachable from a
                    # parameter fixed upstream, whose posterior sigma is 0.
                    self._set_fixed_feh(self._feh_mean, reason="degenerate")
            elif self._feh_type == "kde":
                # feh_prior = ("kde", samples_array)
                samples = np.asarray(feh_prior[1])
                self._build_feh_kde(samples)
            elif self._feh_type in ("morton", "rave"):
                # Named population priors, build inverse CDF for sampling
                self._build_named_feh_prior(self._feh_type)

        self._has_distance = distance_prior is not None
        self._has_av = av_range is not None
        self._has_vini = vini_range is not None
        if self._has_distance:
            # ("normal", mean, sigma), truncated normal at 0, like ARIADNE
            self._dist_type = distance_prior[0]
            self._dist_mean = float(distance_prior[1])
            self._dist_sigma = float(distance_prior[2])
            # A zero or non-finite sigma collapses dist_lo onto dist_hi and the
            # ndtr calls below divide by it: ZeroDivisionError for a Python
            # float, a silent NaN in every parameter vector for a np.float64.
            # It reaches here from a parallax or distance fixed upstream.
            if not (self._dist_sigma > 0) or not np.isfinite(self._dist_sigma):
                raise PriorError(
                    f"The distance prior has sigma={distance_prior[2]!r}, which "
                    f"is not a positive width. A distance that was fixed in the "
                    f"upstream fit cannot be used as a distance PRIOR; pass it "
                    f"explicitly, e.g. prior_setup={{'dist': ('normal', "
                    f"{self._dist_mean:.4g}, <uncertainty>)}}."
                )
            if not np.isfinite(self._dist_mean):
                raise PriorError(
                    f"The distance prior has a non-finite mean "
                    f"({distance_prior[1]!r})."
                )
            # Sampling range: mean ± 10σ, truncated at 0
            self.dist_lo = max(0.1, self._dist_mean - 10 * self._dist_sigma)
            self.dist_hi = self._dist_mean + 10 * self._dist_sigma
            # Precompute the truncation CDF bounds ONCE (they depend only on the
            # fixed prior, not the sampled u), so prior_transform runs a single
            # ndtri per call instead of two ndtr + one ndtri each proposal.
            self._dist_Fa = float(ndtr((self.dist_lo - self._dist_mean) / self._dist_sigma))
            self._dist_Fb = float(ndtr((self.dist_hi - self._dist_mean) / self._dist_sigma))
        if self._has_av:
            self.av_lo, self.av_hi = av_range
        if self._has_vini:
            self.vini_lo, self.vini_hi = vini_range
        # Photometric excess-noise (jitter) term, in magnitudes, added in
        # quadrature to every photometric band's uncertainty. Sampled in
        # log-uniform (Jeffreys) space so the floor is effectively "off".
        # ONE excess-noise term PER photometric band (ARIADNE-style): the
        # parameter vector carries n_band jitter terms, each added in quadrature
        # to its own band. The ordered band list is supplied by the sampler once
        # the observed photometry is known, via ``set_jitter_bands``.
        self._fit_jitter = jitter_range is not None
        self._jitter_bands: list[str] = []
        self._has_jitter = False
        if self._fit_jitter:
            self.jit_lo, self.jit_hi = jitter_range
            self._log_jit_lo = float(np.log10(self.jit_lo))
            self._log_jit_hi = float(np.log10(self.jit_hi))

    def set_jitter_bands(self, bands) -> None:
        """Configure one jitter term per photometric band (ARIADNE-style).

        ``bands`` must be in the SAME order the likelihood consumes photometry
        (observed-dict order), so the trailing ``jitter_<band>`` entries in
        ``param_names`` align positionally with the per-band variance in the
        kernel. A no-op when jitter fitting is disabled.
        """
        self._jitter_bands = list(bands) if self._fit_jitter else []
        self._has_jitter = len(self._jitter_bands) > 0

    def _build_feh_kde(self, samples):
        """Build KDE + inverse CDF for sampling from an arbitrary distribution.

        The samples are normally an ARIADNE [Fe/H] posterior. When [Fe/H] was
        FIXED in the ARIADNE run the column is constant, and handing that to
        scipy is a coin flip rather than a clean failure: the singular-covariance
        guard fires only when the covariance evaluates to exactly zero, which
        depends on the bit pattern of the fixed value and on the sample count.
        A constant -0.5 raises LinAlgError; a constant -0.13 builds a KDE of
        bandwidth ~1e-18 whose entire mass falls between two nodes of the
        tabulated CDF, so every node underflows to zero, the normalisation is
        0/0, the monotonicity mask keeps a single node and every draw comes back
        as feh_lo. That silently pins the fit at the grid's metallicity floor.

        So measure the spread first and treat anything the tabulation cannot
        resolve as a fixed [Fe/H]. np.ptp is the right test: it is exactly zero
        for a constant array whatever the value, whereas np.std both misses
        (2e-16 for a constant -1.47) and over-reports (exactly 0.0 for a
        constant 3.0, which scipy nonetheless accepts).
        """
        from scipy.stats import gaussian_kde

        samples = np.asarray(samples, dtype=float).ravel()
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            raise PriorError(
                "The [Fe/H] prior samples are all non-finite. Check the "
                "metallicity column of the source posterior."
            )
        if finite.size < samples.size:
            warnings.warn(
                f"Dropped {samples.size - finite.size} non-finite sample(s) "
                f"from the [Fe/H] prior.",
                RuntimeWarning, stacklevel=2,
            )
        samples = finite

        # A posterior narrower than one cell of the inverse-CDF grid cannot be
        # tabulated, and that includes the degenerate (upstream-fixed) case.
        cell = (self.feh_hi - self.feh_lo) / (_FEH_CDF_NODES - 1)
        if float(np.ptp(samples)) <= cell:
            self._set_fixed_feh(float(np.median(samples)), reason="degenerate")
            return

        # Clip samples to the feh prior range so the inverse CDF is bounded
        in_range = samples[(samples >= self.feh_lo) & (samples <= self.feh_hi)]
        if len(in_range) < 10:
            # Outside this grid's metallicity coverage. Falling back to uniform
            # silently replaces a measured [Fe/H] with an uninformative one AND
            # changes the evidence normalisation (Fitter._bma_offset adds the
            # prior box only for uniform [Fe/H]), so say so out loud. The
            # coverage drop in Fitter.initialize is what should catch this
            # first; this is the backstop.
            warnings.warn(
                f"Only {len(in_range)} of {len(samples)} [Fe/H] prior samples "
                f"fall inside this grid's coverage "
                f"[{self.feh_lo:.2f}, {self.feh_hi:.2f}]; falling back to a "
                f"UNIFORM [Fe/H] prior for this grid. The measured metallicity "
                f"is not being used here, and this grid's evidence is not "
                f"comparable with grids that did use it.",
                RuntimeWarning, stacklevel=2,
            )
            self._feh_type = "uniform"
            return
        samples = in_range

        self._feh_kde = gaussian_kde(samples)
        # Build the inverse CDF on the sample support rather than the full prior
        # range. Spanning the whole range wastes the tabulation on a posterior
        # much narrower than it: a sigma of 2e-4 dex over [-2.0, 0.5] resolves
        # to two nodes, which np.interp turns into a uniform ramp across the
        # lower half of the axis.
        lo, hi = float(samples.min()), float(samples.max())
        pad = max(0.1 * (hi - lo), 5.0 * float(np.std(samples)))
        grid = np.linspace(
            max(self.feh_lo, lo - pad), min(self.feh_hi, hi + pad),
            _FEH_CDF_NODES,
        )
        pdf = self._feh_kde(grid)
        cdf = np.cumsum(pdf)
        total = float(cdf[-1])
        if not np.isfinite(total) or total <= 0:
            raise PriorError(
                "The [Fe/H] KDE integrates to zero over "
                f"[{grid[0]:.3f}, {grid[-1]:.3f}]. The posterior is too narrow "
                "to tabulate; supply it as a Gaussian prior instead."
            )
        cdf = cdf / total
        # Remove duplicates for strict monotonicity
        mask = np.concatenate([[True], np.diff(cdf) > 0])
        self._feh_cdf_x = grid[mask]
        self._feh_cdf_y = cdf[mask]
        if self._feh_cdf_x.size < _FEH_CDF_MIN_NODES:
            self._set_fixed_feh(float(np.median(samples)), reason="unresolved")

    def _set_fixed_feh(self, value: float, reason: str = "degenerate") -> None:
        """Switch the [Fe/H] prior to a delta at `value`."""
        self._feh_type = "fixed"
        self._feh_value = float(value)
        detail = {
            "degenerate": "has zero spread",
            "unresolved": "is too narrow to tabulate",
        }.get(reason, reason)
        warnings.warn(
            f"The [Fe/H] prior sample {detail}; treating [Fe/H] as FIXED at "
            f"{self._feh_value:+.4f}. This is what an upstream fit with a fixed "
            f"metallicity looks like. Stellar age and mass uncertainties from "
            f"this fit will NOT include any metallicity uncertainty.",
            RuntimeWarning, stacklevel=3,
        )

    def _build_named_feh_prior(self, name: str):
        """Build inverse CDF for a named [Fe/H] population prior."""
        if name == "morton":
            pdf_func = _feh_prior_morton
        elif name == "rave":
            pdf_func = _feh_prior_rave
        else:
            self._feh_type = "uniform"
            return
        grid = np.linspace(self.feh_lo, self.feh_hi, 1024)
        pdf = np.array([pdf_func(x) for x in grid])
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        mask = np.concatenate([[True], np.diff(cdf) > 0])
        self._feh_cdf_x = grid[mask]
        self._feh_cdf_y = cdf[mask]
        # Store the pdf function for log_prior evaluation
        self._feh_named_func = pdf_func

    @property
    def param_names(self) -> list[str]:
        names = ["eep", "log_age", "feh"]
        if self._binary:
            names.append("eep_secondary")
        if self._has_distance:
            names.append("distance")
        if self._has_av:
            names.append("Av")
        if self._has_vini:
            names.append("vini")
        # ARIADNE-style per-band white-noise naming: "<band>_noise".
        for b in self._jitter_bands:
            names.append(f"{b}_noise")
        return names

    @property
    def ndim(self) -> int:
        return len(self.param_names)

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Map unit cube [0,1]^N -> physical parameter space."""
        theta = np.empty_like(u)
        theta[0] = self.eep_lo + u[0] * (self.eep_hi - self.eep_lo)

        # Age prior
        if self._age_type == "uniform":
            # Flat in LINEAR age: P(tau) = const -> P(log_tau) ∝ 10^log_tau
            # Inverse CDF: log_tau = log10(10^lo + u * (10^hi - 10^lo))
            tau_lo = 10.0 ** self.age_lo
            tau_hi = 10.0 ** self.age_hi
            theta[1] = np.log10(tau_lo + u[1] * (tau_hi - tau_lo))
        else:
            # Flat in log age (default)
            theta[1] = self.age_lo + u[1] * (self.age_hi - self.age_lo)

        # [Fe/H] prior
        if self._feh_type == "gaussian":
            theta[2] = _truncated_normal_ppf(
                u[2], self._feh_mean, self._feh_sigma,
                self.feh_lo, self.feh_hi,
            )
        elif self._feh_type == "fixed":
            # Delta prior: the dimension stays in the vector (dropping it would
            # desynchronise every positional theta index downstream) but every
            # draw returns the same value, so it integrates to exactly 1 and
            # leaves the evidence comparable across grids.
            theta[2] = self._feh_value
        elif self._feh_type in ("kde", "morton", "rave"):
            # Inverse CDF sampling (KDE, Morton, or RAVE)
            theta[2] = np.interp(u[2], self._feh_cdf_y, self._feh_cdf_x)
        else:
            theta[2] = self.feh_lo + u[2] * (self.feh_hi - self.feh_lo)

        idx = 3
        if self._binary:
            theta[idx] = self.eep_lo + u[idx] * (theta[0] - self.eep_lo)
            idx += 1
        if self._has_distance:
            # Inlined truncated-normal PPF with precomputed Fa/Fb (see __init__).
            theta[idx] = self._dist_mean + self._dist_sigma * ndtri(
                self._dist_Fa + u[idx] * (self._dist_Fb - self._dist_Fa))
            idx += 1
        if self._has_av:
            theta[idx] = self.av_lo + u[idx] * (self.av_hi - self.av_lo)
            idx += 1
        if self._has_vini:
            theta[idx] = self.vini_lo + u[idx] * (self.vini_hi - self.vini_lo)
            idx += 1
        if self._has_jitter:
            # One log-uniform photometric excess-noise term per band (mag),
            # vectorised over bands (the per-band Python loop was ~1.4 us/call).
            n = len(self._jitter_bands)
            theta[idx:idx + n] = 10.0 ** (
                self._log_jit_lo
                + u[idx:idx + n] * (self._log_jit_hi - self._log_jit_lo)
            )
            idx += n

        return theta

    def log_eep_prior(
        self,
        initial_mass: float | None,
        dm_deep: float | None,
    ) -> float:
        """Log of IMF(mass) * |dm/dEEP|, the EEP prior weight.

        This is separated out because it's folded into the loglikelihood
        (dynesty's prior_transform can't encode grid-dependent priors).
        """
        if initial_mass is None or dm_deep is None:
            return -np.inf
        if np.isnan(initial_mass) or np.isnan(dm_deep):
            return -np.inf
        if dm_deep <= 0 or initial_mass <= 0:
            return -np.inf
        imf_val = self._imf(initial_mass)
        if imf_val <= 0:
            return -np.inf
        return np.log(imf_val) + np.log(dm_deep)

    def log_prior(
        self,
        eep: float,
        log_age: float,
        feh: float,
        initial_mass: float | None = None,
        dm_deep: float | None = None,
        distance: float | None = None,
        av: float | None = None,
    ) -> float:
        """Log-prior density.

        Parameters
        ----------
        initial_mass : from grid interpolation at (eep, age, feh)
        dm_deep : |d(initial_mass)/dEEP| from grid interpolation
        """
        # Bounds check
        if not (self.eep_lo <= eep <= self.eep_hi):
            return -np.inf
        if not (self.age_lo <= log_age <= self.age_hi):
            return -np.inf
        if not (self.feh_lo <= feh <= self.feh_hi):
            return -np.inf
        if self._has_distance and distance is not None:
            if not (self.dist_lo <= distance <= self.dist_hi):
                return -np.inf
        if self._has_av and av is not None:
            if not (self.av_lo <= av <= self.av_hi):
                return -np.inf

        lnp = 0.0

        # EEP prior: IMF(mass) * |dm/dEEP|
        # This is the key fix, not flat in EEP
        if initial_mass is not None and dm_deep is not None and dm_deep > 0:
            imf_val = self._imf(initial_mass)
            if imf_val <= 0:
                return -np.inf
            lnp += np.log(imf_val) + np.log(dm_deep)
        else:
            # Fallback: flat in EEP (wrong but won't crash)
            lnp += _log_width(self.eep_hi, self.eep_lo)

        # Age prior
        if self._age_type == "uniform":
            # Flat in linear age: P(log_tau) = ln(10) * 10^log_tau / (10^hi - 10^lo)
            ln10 = np.log(10.0)
            lnp += (
                np.log(ln10)
                + log_age * ln10
                - np.log(10.0 ** self.age_hi - 10.0 ** self.age_lo)
            )
        else:
            # Flat in log_age
            lnp += _log_width(self.age_hi, self.age_lo)

        # [Fe/H] prior
        if self._feh_type == "gaussian":
            a = (self.feh_lo - self._feh_mean) / self._feh_sigma
            b = (self.feh_hi - self._feh_mean) / self._feh_sigma
            log_norm = np.log(ndtr(b) - ndtr(a))
            lnp += (
                -0.5 * ((feh - self._feh_mean) / self._feh_sigma) ** 2
                - np.log(self._feh_sigma)
                - 0.5 * np.log(2 * np.pi)
                - log_norm
            )
        elif self._feh_type == "kde":
            pdf = float(self._feh_kde(feh)[0])
            if pdf <= 0:
                return -np.inf
            lnp += np.log(pdf)
        elif self._feh_type in ("morton", "rave"):
            pdf = self._feh_named_func(feh)
            if pdf <= 0:
                return -np.inf
            lnp += np.log(pdf)
        elif self._feh_type == "fixed":
            # A delta carries no prior volume and is identical across grids.
            lnp += 0.0
        else:
            lnp += _log_width(self.feh_hi, self.feh_lo)

        # Distance prior (truncated normal, like ARIADNE)
        if self._has_distance:
            a = (self.dist_lo - self._dist_mean) / self._dist_sigma
            b = (self.dist_hi - self._dist_mean) / self._dist_sigma
            log_norm = np.log(ndtr(b) - ndtr(a))
            lnp += (
                -0.5 * ((distance - self._dist_mean) / self._dist_sigma) ** 2
                - np.log(self._dist_sigma)
                - 0.5 * np.log(2 * np.pi)
                - log_norm
            )
        # Av prior (uniform, or a delta when Av was fixed)
        if self._has_av:
            lnp += _log_width(self.av_hi, self.av_lo)

        return lnp


def __getattr__(name):
    # Deprecated alias for the old misspelled class name.
    if name == "IsochonePrior":
        import warnings
        warnings.warn(
            "IsochonePrior is deprecated; use IsochronePrior instead.",
            DeprecationWarning, stacklevel=2,
        )
        return IsochronePrior
    raise AttributeError(name)
