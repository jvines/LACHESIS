"""Nested sampling for isochrone fitting with dynesty.

The EEP prior P(eep) = IMF(mass) * |dm/dEEP| is data-dependent (requires
grid interpolation). Since dynesty's prior_transform can only encode
analytic priors, the IMF*dm_deep weight is folded into the log-likelihood.
"""

import numpy as np

from lachesis.interp import GridInterpolator
from lachesis.likelihood import log_likelihood
from lachesis.prior import IsochronePrior


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
        age_prior: str = "log_uniform",
        bc_table=None,
        distance_prior: tuple[float, float] | None = None,
        av_range: tuple[float, float] | None = None,
        vini_range: tuple[float, float] | None = None,
        binary: bool = False,
        imf: str = "chabrier",
        external_kdes: dict | None = None,
    ):
        self.interp = interp
        self.bc_table = bc_table
        self._binary = binary
        self._has_vini = vini_range is not None
        self._external_kdes = external_kdes or {}
        self.prior = IsochronePrior(
            eep_range=eep_range,
            age_range=age_range,
            feh_range=feh_range,
            feh_prior=feh_prior,
            age_prior=age_prior,
            distance_prior=distance_prior,
            av_range=av_range,
            vini_range=vini_range,
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
        ext_kdes = self._external_kdes

        def loglike(theta):
            params = dict(zip(param_names, theta))
            eep = params["eep"]
            log_age = params["log_age"]
            feh = params["feh"]
            distance = params.get("distance")
            av = params.get("av")
            vini = params.get("vini")

            # Get grid predictions (needed for both likelihood and IMF prior)
            predicted = interp(eep=eep, log_age=log_age, feh=feh, vini=vini)
            if np.isnan(predicted.get("log_Teff", np.nan)):
                return -np.inf

            # IMF * dm_deep prior weight (folded into loglike for dynesty)
            initial_mass = predicted.get("initial_mass")
            dm_deep = predicted.get("dm_deep")
            lnp_eep = prior.log_eep_prior(initial_mass, dm_deep)
            if not np.isfinite(lnp_eep):
                return -np.inf

            # External priors from ARIADNE posteriors.
            # ext_kdes stores (grid, log_pdf) tables for O(1) interp lookup.
            lnp_ext = 0.0
            if ext_kdes:
                for param, (grid_x, log_pdf) in ext_kdes.items():
                    if param == "Teff":
                        val = predicted.get("Teff")
                    elif param == "log_g":
                        val = predicted.get("log_g")
                    elif param == "radius":
                        val = predicted.get("radius")
                    elif param == "luminosity":
                        log_l = predicted.get("log_L")
                        val = 10.0 ** log_l if log_l is not None and not np.isnan(log_l) else None
                    elif param == "distance":
                        val = distance
                    elif param == "Av":
                        val = av
                    else:
                        continue
                    if val is None or np.isnan(val):
                        return -np.inf
                    if val < grid_x[0] or val > grid_x[-1]:
                        return -np.inf
                    lnp_ext += float(np.interp(val, grid_x, log_pdf))

            # Data likelihood (pass predicted to avoid double interpolation)
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
                    predicted=predicted,
                )

            if not np.isfinite(lnl):
                return -np.inf

            return lnl + lnp_eep + lnp_ext

        dynesty_kwargs.setdefault("sample", "rwalk")
        sampler = dynesty.NestedSampler(
            loglike,
            prior.prior_transform,
            ndim=prior.ndim,
            nlive=nlive,
            **dynesty_kwargs,
        )
        sampler.run_nested(dlogz=dlogz, print_progress=True)
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
        vini = None
        if self._has_vini:
            # Vini is the last parameter
            vini_idx = self.prior.param_names.index("vini")
            vini = samples[:, vini_idx]
        return self.interp(eep=eeps, log_age=ages, feh=fehs, vini=vini)
