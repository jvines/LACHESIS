"""Star class — holds observed stellar properties for isochrone fitting."""

import numpy as np

from lachesis.display import display_banner, display_star_info
from lachesis.error import InputError


class Star:
    """Observed stellar properties for isochrone fitting.

    Mimics ARIADNE's Star but for the reverse problem: given stellar params,
    infer age/mass/evolutionary state via isochrone fitting.
    """

    def __init__(
        self,
        starname: str,
        *,
        teff: float | None = None,
        teff_e: float | None = None,
        logg: float | None = None,
        logg_e: float | None = None,
        feh: float | None = None,
        feh_e: float | None = None,
        luminosity: float | None = None,
        luminosity_e: float | None = None,
        radius: float | None = None,
        radius_e: float | None = None,
        parallax: float | None = None,
        parallax_e: float | None = None,
        distance: float | None = None,
        distance_e: float | None = None,
        magnitudes: dict | None = None,
        Av: float | None = None,
        verbose: bool = True,
    ):
        self.starname = starname
        self.teff = teff
        self.teff_e = teff_e
        self.logg = logg
        self.logg_e = logg_e
        self.feh = feh
        self.feh_e = feh_e
        self.luminosity = luminosity
        self.luminosity_e = luminosity_e
        self.radius = radius
        self.radius_e = radius_e
        self.parallax = parallax
        self.parallax_e = parallax_e
        self.distance = distance
        self.distance_e = distance_e
        self.magnitudes = magnitudes or {}
        self.Av = Av

        # Derive distance from parallax if not given
        if self.distance is None and self.parallax is not None and self.parallax > 0:
            self.distance = 1000.0 / self.parallax
            if self.parallax_e is not None:
                self.distance_e = self.distance * (self.parallax_e / self.parallax)

        if verbose:
            display_banner(starname)
            display_star_info(self)

    @classmethod
    def from_ariadne(cls, nc_path: str, starname: str | None = None, verbose: bool = True):
        """Load stellar properties from an ARIADNE InferenceData .nc file."""
        import arviz as az

        idata = az.from_netcdf(nc_path)
        post = idata.posterior

        if starname is None:
            from pathlib import Path
            starname = Path(nc_path).stem

        def _extract(name):
            if name in post:
                arr = post[name].values.flatten()
                return float(np.median(arr)), float(np.std(arr))
            return None, None

        # ARIADNE name mapping
        teff, teff_e = _extract("Teff")
        if teff is None:
            teff, teff_e = _extract("teff")
        logg, logg_e = _extract("logg")
        feh, feh_e = _extract("feh")
        if feh is None:
            feh, feh_e = _extract("z")
        lum, lum_e = _extract("luminosity")
        if lum is None:
            lum, lum_e = _extract("lum")
        rad, rad_e = _extract("radius")
        if rad is None:
            rad, rad_e = _extract("rad")
        dist, dist_e = _extract("distance")
        if dist is None:
            dist, dist_e = _extract("dist")

        return cls(
            starname,
            teff=teff, teff_e=teff_e,
            logg=logg, logg_e=logg_e,
            feh=feh, feh_e=feh_e,
            luminosity=lum, luminosity_e=lum_e,
            radius=rad, radius_e=rad_e,
            distance=dist, distance_e=dist_e,
            verbose=verbose,
        )

    @property
    def observed(self) -> dict[str, float]:
        """Observable dict for the likelihood, translated to grid column names."""
        obs = {}
        if self.teff is not None:
            obs["log_Teff"] = np.log10(self.teff)
        if self.logg is not None:
            obs["log_g"] = self.logg
        if self.feh is not None:
            obs["feh"] = self.feh
        if self.luminosity is not None:
            obs["log_L"] = np.log10(self.luminosity)
        if self.radius is not None:
            obs["log_R"] = np.log10(self.radius)
        # Photometric bands
        for band, (mag, _err) in self.magnitudes.items():
            obs[band] = mag
        return obs

    @property
    def uncertainties(self) -> dict[str, float]:
        """Uncertainty dict matching observed keys."""
        unc = {}
        if self.teff is not None and self.teff_e is not None:
            unc["log_Teff"] = self.teff_e / (self.teff * np.log(10))
        if self.logg is not None and self.logg_e is not None:
            unc["log_g"] = self.logg_e
        if self.feh is not None and self.feh_e is not None:
            unc["feh"] = self.feh_e
        if self.luminosity is not None and self.luminosity_e is not None:
            unc["log_L"] = self.luminosity_e / (self.luminosity * np.log(10))
        if self.radius is not None and self.radius_e is not None:
            unc["log_R"] = self.radius_e / (self.radius * np.log(10))
        for band, (_mag, err) in self.magnitudes.items():
            unc[band] = err
        return unc

    @property
    def mode(self) -> str:
        if self.magnitudes and self.distance is not None:
            return "photometric"
        return "spectroscopic"
