"""Nested sampling for isochrone fitting with dynesty.

The EEP prior P(eep) = IMF(mass) * |dm/dEEP| is data-dependent (requires
grid interpolation). Since dynesty's prior_transform can only encode
analytic priors, the IMF*dm_deep weight is folded into the log-likelihood.
"""

import numpy as np

from lachesis.interp import GridInterpolator
from lachesis.likelihood import log_likelihood
from lachesis.prior import IsochonePrior


class IsochroneFitter:
    """Isochrone fitter using nested sampling.

    Modes:
    - 3D: (EEP, log_age, [Fe/H])
    - 5D: + (distance, Av) for photometry
    - 6D: + (eep_secondary) for binary
    """

    def __init__(
        self,
        interp: GridInterpolator,
        eep_range: tuple[float, float],
        age_range: tuple[float, float],
        feh_range: tuple[float, float],
        feh_prior: tuple[str, ...] | None = None,
        bc_table=None,
        distance_range: tuple[float, float] | None = None,
        av_range: tuple[float, float] | None = None,
        binary: bool = False,
        imf: str = "chabrier",
    ):
        self.interp = interp
        self.bc_table = bc_table
        self._binary = binary
        self.prior = IsochonePrior(
            eep_range=eep_range,
            age_range=age_range,
            feh_range=feh_range,
            feh_prior=feh_prior,
            distance_range=distance_range,
            av_range=av_range,
            binary=binary,
            imf=imf,
        )

    def fit(
        self,
        observed: dict[str, float],
        uncertainties: dict[str, float],
        nlive: int = 500,
        dlogz: float = 0.01,
        **dynesty_kwargs,
    ) -> dict:
        """Run nested sampling."""
        import dynesty
        from dynesty.utils import resample_equal

        interp = self.interp
        prior = self.prior
        bc = self.bc_table
        obs = observed
        unc = uncertainties
        is_binary = self._binary
        param_names = prior.param_names

        def loglike(theta):
            params = dict(zip(param_names, theta))
            eep = params["eep"]
            log_age = params["log_age"]
            feh = params["feh"]
            distance = params.get("distance")
            av = params.get("av")

            # Get grid predictions (needed for both likelihood and IMF prior)
            predicted = interp(eep=eep, log_age=log_age, feh=feh)
            if np.isnan(predicted.get("log_Teff", np.nan)):
                return -np.inf

            # IMF * dm_deep prior weight (folded into loglike for dynesty)
            initial_mass = predicted.get("initial_mass")
            dm_deep = predicted.get("dm_deep")
            lnp_eep = prior.log_eep_prior(initial_mass, dm_deep)
            if not np.isfinite(lnp_eep):
                return -np.inf

            # Data likelihood
            if is_binary:
                from lachesis.binary import binary_log_likelihood
                lnl = binary_log_likelihood(
                    interp, bc=bc,
                    eep_primary=eep,
                    eep_secondary=params["eep_secondary"],
                    log_age=log_age, feh=feh,
                    distance=distance, av=av,
                    observed=obs, uncertainties=unc,
                )
            else:
                lnl = log_likelihood(
                    interp, eep=eep, log_age=log_age, feh=feh,
                    observed=obs, uncertainties=unc,
                    bc_table=bc, distance=distance, av=av,
                )

            if not np.isfinite(lnl):
                return -np.inf

            return lnl + lnp_eep

        sampler = dynesty.NestedSampler(
            loglike,
            prior.prior_transform,
            ndim=prior.ndim,
            nlive=nlive,
            **dynesty_kwargs,
        )
        sampler.run_nested(dlogz=dlogz, print_progress=False)
        results = sampler.results

        weights = np.exp(results.logwt - results.logz[-1])
        weights /= weights.sum()
        samples = resample_equal(results.samples, weights)

        derived = self._compute_derived(samples)

        return {
            "samples": samples,
            "weights": weights,
            "logz": results.logz[-1],
            "logzerr": results.logzerr[-1],
            "derived": derived,
            "dynesty_results": results,
        }

    def _compute_derived(self, samples: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate grid at each posterior sample (primary component)."""
        eeps = samples[:, 0]
        ages = samples[:, 1]
        fehs = samples[:, 2]
        return self.interp(eep=eeps, log_age=ages, feh=fehs)
