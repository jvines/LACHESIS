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
    def from_ariadne_star(cls, ariadne_star, verbose: bool = True):
        """Create a LACHESIS Star from an astroARIADNE Star object.

        Extracts magnitudes, parallax, distance, and spectroscopic params.
        Works with any ARIADNE Star — just pass the object directly.

        Example
        -------
        >>> from astroARIADNE.star import Star as ARIADNEStar
        >>> a = ARIADNEStar('HD 209458', 330.795, 18.884)
        >>> s = Star.from_ariadne_star(a)
        """
        s = ariadne_star

        # Extract magnitudes from ARIADNE's array-based storage
        magnitudes = {}
        if hasattr(s, 'mags') and hasattr(s, 'mag_errs') and hasattr(s, 'used_filters'):
            for i, filt in enumerate(s.filter_names):
                if s.used_filters[i] >= 1 and s.mag_errs[i] > 0:
                    magnitudes[filt] = (float(s.mags[i]), float(s.mag_errs[i]))

        # Extract parallax
        plx = getattr(s, 'plx', None)
        plx_e = getattr(s, 'plx_e', None)
        if plx is not None and plx <= 0:
            plx = None

        # Extract distance
        dist = getattr(s, 'dist', None)
        dist_e = getattr(s, 'dist_e', None)
        if dist is not None and dist <= 0:
            dist = None

        # Spectroscopic params (if available from RAVE or user input)
        logg = getattr(s, 'logg', None)
        logg_e = getattr(s, 'logg_e', None)
        feh = None
        feh_e = None
        rave = getattr(s, 'rave_params', None)
        if rave is not None:
            feh = rave.get('feh')
            feh_e = rave.get('feh_err')
            if logg is None:
                logg = rave.get('logg')
                logg_e = rave.get('logg_err')

        return cls(
            s.starname,
            logg=logg, logg_e=logg_e,
            feh=feh, feh_e=feh_e,
            parallax=plx, parallax_e=plx_e,
            distance=dist, distance_e=dist_e,
            magnitudes=magnitudes,
            verbose=verbose,
        )

    @classmethod
    def from_name(cls, name: str, verbose: bool = True, **kwargs):
        """Create a Star by resolving a name via Simbad + Librarian.

        Handles high proper motion stars by resolving the Gaia DR3 ID
        through Simbad rather than relying on coordinate cone search.

        Example
        -------
        >>> s = Star.from_name('HD 209458')
        """
        from astroquery.simbad import Simbad
        from lachesis.librarian import Librarian

        # Resolve coordinates
        from astropy.coordinates import SkyCoord
        coord = SkyCoord.from_name(name)
        ra, dec = coord.ra.deg, coord.dec.deg

        # Try to get Gaia DR3 ID from Simbad
        gaia_id = None
        try:
            ids = Simbad.query_objectids(name)
            if ids is not None:
                for row in ids:
                    rid = str(row[0]) if hasattr(row, '__getitem__') else str(row)
                    if 'Gaia DR3' in rid:
                        gaia_id = int(rid.split()[-1])
                        break
        except Exception:
            pass

        lib = Librarian(ra, dec, gaia_id=gaia_id, verbose=verbose, **kwargs)
        return lib.to_star(name, verbose=verbose)

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
        # Photometric bands — translate pyphot names to BC table names
        from lachesis.filters import PYPHOT_TO_BC
        for band, (mag, err) in self.magnitudes.items():
            bc_name = PYPHOT_TO_BC.get(band)
            if bc_name is not None and err > 0:
                obs[bc_name] = mag
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
        from lachesis.filters import PYPHOT_TO_BC
        for band, (_mag, err) in self.magnitudes.items():
            bc_name = PYPHOT_TO_BC.get(band)
            if bc_name is not None and err > 0:
                unc[bc_name] = err
        return unc

    @property
    def mode(self) -> str:
        if self.magnitudes and self.distance is not None:
            return "photometric"
        return "spectroscopic"
