"""Prior transforms and log-prior for isochrone fitting.

The EEP prior follows Morton's isochrones:
    P(eep | age, feh) = IMF(mass(eep, age, feh)) * |dm/dEEP|

This transforms a Salpeter/Chabrier IMF on mass into EEP space via
the Jacobian dm_deep. Without this, the prior is flat in EEP which
over-weights high-mass evolutionary states.
"""

import numpy as np


def salpeter_imf(mass: float) -> float:
    """Salpeter IMF: dN/dM ∝ M^{-2.35}."""
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


_IMF_FUNCTIONS = {
    "salpeter": salpeter_imf,
    "chabrier": chabrier_imf,
}


class IsochonePrior:
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
        distance_range: tuple[float, float] | None = None,
        av_range: tuple[float, float] | None = None,
        binary: bool = False,
        imf: str = "chabrier",
    ):
        self.eep_lo, self.eep_hi = eep_range
        self.age_lo, self.age_hi = age_range
        self.feh_lo, self.feh_hi = feh_range
        self._binary = binary
        self._imf = _IMF_FUNCTIONS.get(imf, chabrier_imf)

        if feh_prior is None:
            self._feh_type = "uniform"
        else:
            self._feh_type = feh_prior[0]
            if self._feh_type == "gaussian":
                self._feh_mean = feh_prior[1]
                self._feh_sigma = feh_prior[2]

        self._has_distance = distance_range is not None
        self._has_av = av_range is not None
        if self._has_distance:
            self.dist_lo, self.dist_hi = distance_range
        if self._has_av:
            self.av_lo, self.av_hi = av_range

    @property
    def param_names(self) -> list[str]:
        names = ["eep", "log_age", "feh"]
        if self._binary:
            names.append("eep_secondary")
        if self._has_distance:
            names.append("distance")
        if self._has_av:
            names.append("av")
        return names

    @property
    def ndim(self) -> int:
        return len(self.param_names)

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Map unit cube [0,1]^N → physical parameter space."""
        theta = np.empty_like(u)
        theta[0] = self.eep_lo + u[0] * (self.eep_hi - self.eep_lo)
        theta[1] = self.age_lo + u[1] * (self.age_hi - self.age_lo)

        if self._feh_type == "gaussian":
            from scipy.special import ndtri
            theta[2] = self._feh_mean + self._feh_sigma * ndtri(u[2])
            theta[2] = np.clip(theta[2], self.feh_lo, self.feh_hi)
        else:
            theta[2] = self.feh_lo + u[2] * (self.feh_hi - self.feh_lo)

        idx = 3
        if self._binary:
            theta[idx] = self.eep_lo + u[idx] * (theta[0] - self.eep_lo)
            idx += 1
        if self._has_distance:
            theta[idx] = self.dist_lo + u[idx] * (self.dist_hi - self.dist_lo)
            idx += 1
        if self._has_av:
            theta[idx] = self.av_lo + u[idx] * (self.av_hi - self.av_lo)

        return theta

    def log_eep_prior(
        self,
        initial_mass: float | None,
        dm_deep: float | None,
    ) -> float:
        """Log of IMF(mass) * |dm/dEEP| — the EEP prior weight.

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
        # This is the key fix — not flat in EEP
        if initial_mass is not None and dm_deep is not None and dm_deep > 0:
            imf_val = self._imf(initial_mass)
            if imf_val <= 0:
                return -np.inf
            lnp += np.log(imf_val) + np.log(dm_deep)
        else:
            # Fallback: flat in EEP (wrong but won't crash)
            lnp += -np.log(self.eep_hi - self.eep_lo)

        # Uniform in log_age
        lnp += -np.log(self.age_hi - self.age_lo)

        # [Fe/H] prior
        if self._feh_type == "gaussian":
            lnp += (
                -0.5 * ((feh - self._feh_mean) / self._feh_sigma) ** 2
                - np.log(self._feh_sigma)
                - 0.5 * np.log(2 * np.pi)
            )
        else:
            lnp += -np.log(self.feh_hi - self.feh_lo)

        # Distance prior (uniform for now)
        if self._has_distance:
            lnp += -np.log(self.dist_hi - self.dist_lo)
        # Av prior (uniform)
        if self._has_av:
            lnp += -np.log(self.av_hi - self.av_lo)

        return lnp
