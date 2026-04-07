"""Star class — holds observed stellar properties for isochrone fitting."""

import numpy as np
from scipy.constants import h as _h, c as _c, k as _k

from lachesis.display import display_banner
from lachesis.error import InputError


def _planck_mag(lam_m, T):
    """Planck function in magnitude space (arbitrary zero-point)."""
    x = _h * _c / (lam_m * _k * np.maximum(T, 1.0))
    flux = 1.0 / (lam_m ** 3 * (np.exp(np.clip(x, 0, 500)) - 1.0))
    return -2.5 * np.log10(np.maximum(flux, 1e-300))


class Star:
    """Observed stellar properties for isochrone fitting.

    Matches ARIADNE's Star interface: provide a name, RA, DEC, and optionally
    a Gaia DR3 ID. The Librarian runs internally to retrieve photometry,
    parallax, and spectroscopic priors automatically.

    Parameters
    ----------
    starname : str
        Star name (for identification).
    ra, dec : float
        Right ascension and declination in degrees.
    g_id : int, optional
        Gaia DR3 source_id. Bypasses cone search for high-PM stars.
    magnitudes : dict, optional
        Pre-computed magnitudes dict {filter: (mag, err)}. If provided,
        skips the automatic photometry retrieval.
    plx, plx_e : float, optional
        Parallax and error in mas. Overrides Gaia lookup.
    dist, dist_e : float, optional
        Distance and error in pc. Overrides parallax-derived distance.
    logg, logg_e : float, optional
        Surface gravity and error. Overrides spectroscopic lookup.
    feh, feh_e : float, optional
        Metallicity and error. Overrides spectroscopic lookup.
    Av : float, optional
        Maximum extinction in V band.
    verbose : bool
        Print retrieval progress (default True).

    Examples
    --------
    >>> from lachesis.star import Star
    >>> s = Star('HD 209458', 330.795, 18.884)
    >>> s = Star('HD 103095', 178.0, 6.6, g_id=4034171629042489088)
    """

    def __init__(
        self,
        starname: str,
        ra: float,
        dec: float,
        g_id: int | None = None,
        magnitudes: dict | None = None,
        plx: float | None = None,
        plx_e: float | None = None,
        dist: float | None = None,
        dist_e: float | None = None,
        logg: float | None = None,
        logg_e: float | None = None,
        feh: float | None = None,
        feh_e: float | None = None,
        Av: float | None = None,
        verbose: bool = True,
        dustmap: str = 'SFD',
    ):
        self.starname = starname
        self.ra = ra
        self.dec = dec
        self.g_id = g_id
        self.external_posteriors = {}  # populated by from_ariadne()
        self.feh_posterior = None       # populated by from_ariadne()

        if magnitudes is None:
            if verbose:
                from termcolor import colored
                c = display_banner(starname)
                print(colored(f'\t\t*** LOOKING UP ARCHIVAL INFORMATION ***', c))
            from lachesis.librarian import Librarian
            lib = Librarian(ra, dec, gaia_id=g_id, verbose=verbose)

            # Bright star warning
            gmag = lib.magnitudes.get("Gaia_G", (None,))[0]
            vmag = lib.magnitudes.get("GROUND_JOHNSON_V", (None,))[0]
            bright_mag = gmag if gmag is not None else vmag
            if bright_mag is not None and bright_mag < 2.0:
                import logging
                logging.getLogger("lachesis").warning(
                    "%s is very bright (mag=%.1f). Survey photometry is likely "
                    "saturated — fit results may be unreliable.",
                    starname, bright_mag,
                )

            # Copy retrieved data, allow user overrides
            self.magnitudes = lib.magnitudes
            self.parallax = plx if plx is not None else lib._parallax
            self.parallax_e = plx_e if plx_e is not None else lib._parallax_e
            self.distance = dist if dist is not None else lib._distance
            self.distance_e = dist_e if dist_e is not None else lib._distance_e
            self.teff = lib._teff
            self.teff_e = lib._teff_e
            self.Av = Av if Av is not None else lib.Av
            self.luminosity = None
            self.luminosity_e = None
            self.radius = None
            self.radius_e = None

            # Spectroscopic priors: user override > Librarian lookup
            sp = lib._spectroscopic_params
            if sp:
                self.logg = logg if logg is not None else sp.get("logg")
                self.logg_e = logg_e if logg_e is not None else sp.get("logg_err")
                self.feh = feh if feh is not None else sp.get("feh")
                self.feh_e = feh_e if feh_e is not None else sp.get("feh_err")
            else:
                self.logg = logg
                self.logg_e = logg_e
                self.feh = feh
                self.feh_e = feh_e
        else:
            # User provided magnitudes — skip Librarian
            if verbose:
                display_banner(starname)
            self.magnitudes = magnitudes
            self.parallax = plx
            self.parallax_e = plx_e
            self.distance = dist
            self.distance_e = dist_e
            self.logg = logg
            self.logg_e = logg_e
            self.feh = feh
            self.feh_e = feh_e
            self.Av = Av
            self.teff = None
            self.teff_e = None
            self.luminosity = None
            self.luminosity_e = None
            self.radius = None
            self.radius_e = None

        # Derive distance from parallax if not given
        if self.distance is None and self.parallax is not None and self.parallax > 0:
            self.distance = 1000.0 / self.parallax
            if self.parallax_e is not None:
                self.distance_e = self.distance * (self.parallax_e / self.parallax)

        # Max line-of-sight extinction from dustmaps (like ARIADNE)
        if self.Av is None:
            self._query_extinction(dustmap)

        # Blackbody QC: flag photometry outliers (warn only, no removal)
        if self.magnitudes:
            self._blackbody_check()

    _DUSTMAPS = {
        'SFD': ('dustmaps.sfd', 'SFDQuery'),
        'Lenz': ('dustmaps.lenz2017', 'Lenz2017Query'),
        'Planck13': ('dustmaps.planck', 'PlanckQuery'),
        'Planck16': ('dustmaps.planck', 'PlanckGNILCQuery'),
        'Bayestar': ('dustmaps.bayestar', 'BayestarQuery'),
    }

    def _query_extinction(self, dustmap='SFD'):
        """Query max line-of-sight Av from dustmaps (matching ARIADNE)."""
        from importlib import import_module
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if dustmap not in self._DUSTMAPS:
            self.Av = 0.1
            return

        mod_path, cls_name = self._DUSTMAPS[dustmap]
        try:
            mod = import_module(mod_path)
            dmap = getattr(mod, cls_name)()
        except Exception:
            self.Av = 0.1
            return

        d = self.distance if self.distance is not None else 1000.0
        coords = SkyCoord(self.ra, self.dec, distance=d,
                          unit=(u.deg, u.deg, u.pc), frame='icrs')

        try:
            if dustmap in ('SFD', 'Lenz'):
                ebv = dmap(coords)
                self.Av = float(ebv) * 2.742
            elif dustmap == 'Bayestar':
                ebvs = dmap(coords, mode='percentile', pct=[15, 50, 84])
                if np.any(np.isnan(ebvs)):
                    # Fallback to SFD
                    sfd_mod = import_module('dustmaps.sfd')
                    ebv = sfd_mod.SFDQuery()(coords)
                    self.Av = float(ebv) * 2.742
                else:
                    mags = ebvs * 2.742 * 0.884
                    self.Av = float(mags[1])
            elif dustmap in ('Planck13', 'Planck16'):
                ebv = dmap(coords)
                self.Av = float(ebv) * 3.1
        except Exception:
            self.Av = 0.1

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

        magnitudes = {}
        if hasattr(s, 'mags') and hasattr(s, 'mag_errs') and hasattr(s, 'used_filters'):
            for i, filt in enumerate(s.filter_names):
                if s.used_filters[i] >= 1 and s.mag_errs[i] > 0:
                    magnitudes[filt] = (float(s.mags[i]), float(s.mag_errs[i]))

        plx = getattr(s, 'plx', None)
        plx_e = getattr(s, 'plx_e', None)
        if plx is not None and plx <= 0:
            plx = None

        dist = getattr(s, 'dist', None)
        dist_e = getattr(s, 'dist_e', None)
        if dist is not None and dist <= 0:
            dist = None

        logg = getattr(s, 'logg', None)
        logg_e = getattr(s, 'logg_e', None)
        feh = None
        feh_e = None
        sp = getattr(s, 'spectroscopic_params', None) or getattr(s, 'rave_params', None)
        if sp is not None:
            feh = sp.get('feh')
            feh_e = sp.get('feh_err')
            if logg is None:
                logg = sp.get('logg')
                logg_e = sp.get('logg_err')

        return cls(
            s.starname, s.ra, s.dec,
            magnitudes=magnitudes,
            logg=logg, logg_e=logg_e,
            feh=feh, feh_e=feh_e,
            plx=plx, plx_e=plx_e,
            dist=dist, dist_e=dist_e,
            verbose=verbose,
        )

    @classmethod
    def from_ariadne(cls, nc_path: str, starname: str | None = None,
                     ra: float | None = None, dec: float | None = None,
                     g_id: int | None = None, verbose: bool = True):
        """Load stellar properties from an ARIADNE InferenceData .nc file.

        Stores full posterior sample arrays as external priors for the
        isochrone fit. The Fitter builds KDEs from these automatically.

        If ra/dec are provided, photometry is retrieved via the Librarian
        so that the isochrone fit uses both the ARIADNE priors AND the
        photometric data (no double-counting — see docs).
        """
        import arviz as az

        idata = az.from_netcdf(nc_path)
        post = idata.posterior

        if starname is None:
            from pathlib import Path
            starname = Path(nc_path).stem

        def _samples(*names):
            for n in names:
                if n in post:
                    return post[n].values.flatten()
            return None

        def _summary(arr):
            if arr is not None:
                return float(np.median(arr)), float(np.std(arr))
            return None, None

        # Extract full posterior arrays (ARIADNE name mapping)
        teff_arr = _samples("Teff", "teff")
        logg_arr = _samples("logg")
        feh_arr = _samples("feh", "z")
        rad_arr = _samples("radius", "rad")
        dist_arr = _samples("distance", "dist")
        av_arr = _samples("Av", "av")
        lum_arr = _samples("luminosity", "lum", "L")

        teff, teff_e = _summary(teff_arr)
        logg, logg_e = _summary(logg_arr)
        feh, feh_e = _summary(feh_arr)
        rad, rad_e = _summary(rad_arr)
        dist, dist_e = _summary(dist_arr)

        if ra is not None and dec is not None:
            # Retrieve photometry via Librarian, use ARIADNE priors on top
            star = cls(
                starname, ra, dec, g_id=g_id,
                logg=logg, logg_e=logg_e,
                feh=feh, feh_e=feh_e,
                verbose=verbose,
            )
        else:
            # No coordinates — spectroscopic-only using ARIADNE params directly
            star = cls(
                starname, 0.0, 0.0,
                magnitudes={},
                logg=logg, logg_e=logg_e,
                feh=feh, feh_e=feh_e,
                dist=dist, dist_e=dist_e,
                verbose=verbose,
            )
        star.teff = teff
        star.teff_e = teff_e
        star.radius = rad
        star.radius_e = rad_e

        # Store posterior samples:
        # - feh_posterior → sampled parameter, fed through prior transform
        #   as a KDE-based inverse CDF, preserving full distribution shape.
        # - external_posteriors → KDE penalties in the likelihood for all
        #   other available ARIADNE posteriors (grid-derived and sampled).
        star.feh_posterior = feh_arr  # full samples, not just mean/std
        star.external_posteriors = {}
        if teff_arr is not None:
            star.external_posteriors["Teff"] = teff_arr
        if logg_arr is not None:
            star.external_posteriors["log_g"] = logg_arr
        if rad_arr is not None:
            star.external_posteriors["radius"] = rad_arr
        if dist_arr is not None:
            star.external_posteriors["distance"] = dist_arr
        if av_arr is not None:
            star.external_posteriors["Av"] = av_arr
        if lum_arr is not None:
            star.external_posteriors["luminosity"] = lum_arr

        return star

    # --- Magnitude management ---

    def add_mag(self, mag: float, err: float, band: str):
        """Add or update a magnitude."""
        self.magnitudes[band] = (mag, err)

    def remove_mag(self, band: str):
        """Remove a magnitude by pyphot band name."""
        self.magnitudes.pop(band, None)

    def print_mags(self):
        """Print magnitudes sorted by wavelength."""
        from lachesis.filters import FILTER_WAVELENGTH
        bands = sorted(
            self.magnitudes.keys(),
            key=lambda b: FILTER_WAVELENGTH.get(b, 99.0),
        )
        for b in bands:
            mag, err = self.magnitudes[b]
            print(f"  {b:25s}  {mag:7.3f} +/- {err:.3f}")

    # --- Blackbody photometry QC ---

    def _blackbody_check(self, sigma=5.0, err_floor=0.05):
        """Flag photometry outliers via a blackbody consensus fit.

        Fits a Planck function in magnitude space across all bands.
        Bands with normalized residuals > sigma are flagged with a
        warning but NOT removed.
        """
        from lachesis.filters import FILTER_WAVELENGTH

        bands, wavelengths, mags, errs = [], [], [], []
        for b, (m, e) in self.magnitudes.items():
            lam = FILTER_WAVELENGTH.get(b)
            if lam is not None and e > 0:
                bands.append(b)
                wavelengths.append(lam * 1e-6)  # microns → metres
                mags.append(m)
                errs.append(max(e, err_floor))

        if len(bands) < 3:
            self._bb_teff = None
            self._bb_residuals = {}
            return

        wavelengths = np.array(wavelengths)
        mags = np.array(mags)
        errs = np.array(errs)

        # Grid search for best-fit temperature
        temps = np.linspace(2000, 60000, 300)
        best_rss = np.inf
        best_T = 5000.0
        best_offset = 0.0

        for T in temps:
            pred = _planck_mag(wavelengths, T)
            offset = np.mean(mags - pred)
            resid = (mags - (pred + offset)) / errs
            rss = np.sum(resid ** 2)
            if rss < best_rss:
                best_rss = rss
                best_T = T
                best_offset = offset

        # Compute per-band residuals at best T
        pred = _planck_mag(wavelengths, best_T) + best_offset
        self._bb_teff = best_T
        self._bb_offset = best_offset
        self._bb_residuals = {}
        flagged = []

        for i, b in enumerate(bands):
            z = abs(mags[i] - pred[i]) / errs[i]
            self._bb_residuals[b] = (
                wavelengths[i] * 1e6,  # back to microns for plotting
                mags[i], pred[i], z,
            )
            if z > sigma:
                flagged.append((b, z))

        if flagged:
            from termcolor import colored
            for b, z in flagged:
                print(colored(
                    f"\t\t\tWARNING: {b} is potentially problematic "
                    f"({z:.1f}sigma from blackbody fit) -- not removed",
                    "yellow",
                ))

    def plot_blackbody(self, outfile=None):
        """Diagnostic plot: blackbody fit with color-coded photometry.

        Parameters
        ----------
        outfile : str, optional
            Save path. If None, calls plt.show().
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if not getattr(self, "_bb_residuals", None):
            print("No blackbody check results — call _blackbody_check() first.")
            return

        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=(8, 6), height_ratios=[3, 1],
            sharex=True, gridspec_kw={"hspace": 0.05},
        )

        # Collect data
        bands = list(self._bb_residuals.keys())
        lams = np.array([self._bb_residuals[b][0] for b in bands])
        obs = np.array([self._bb_residuals[b][1] for b in bands])
        pred = np.array([self._bb_residuals[b][2] for b in bands])
        zvals = np.array([self._bb_residuals[b][3] for b in bands])
        errs_raw = np.array([
            max(self.magnitudes[b][1], 0.05) for b in bands
        ])

        # Wavelength colormap
        cmap = plt.cm.Spectral_r
        norm = LogNorm(vmin=max(lams.min(), 0.1), vmax=lams.max())

        # Smooth blackbody curve
        lam_fine = np.geomspace(lams.min() * 0.8, lams.max() * 1.2, 500)
        bb_fine = _planck_mag(lam_fine * 1e-6, self._bb_teff) + self._bb_offset
        ax_main.plot(lam_fine, bb_fine, "k-", lw=1, alpha=0.5, zorder=1)

        # Plot each band
        for i, b in enumerate(bands):
            color = cmap(norm(lams[i]))
            marker = "o"
            edgecolor = color
            if zvals[i] > 5.0:
                marker = "s"
                edgecolor = "red"
            ax_main.errorbar(
                lams[i], obs[i], yerr=errs_raw[i],
                fmt=marker, color=color, mec=edgecolor, mew=1.5,
                ms=7, capsize=2, zorder=3, label=b,
            )
            # Residuals
            ax_res.scatter(
                lams[i], (obs[i] - pred[i]) / errs_raw[i],
                c=[color], marker=marker, edgecolors=edgecolor,
                linewidths=1.5, s=50, zorder=3,
            )

        ax_main.invert_yaxis()
        ax_main.set_ylabel("Magnitude")
        ax_main.set_title(
            f"{self.starname}  —  $T_{{\\rm bb}}$ = {self._bb_teff:.0f} K"
        )
        ax_main.legend(
            fontsize=6, ncol=3, loc="upper right",
            framealpha=0.7,
        )

        ax_res.axhline(0, color="k", lw=0.5)
        ax_res.axhline(5, color="r", lw=0.5, ls="--", alpha=0.5)
        ax_res.axhline(-5, color="r", lw=0.5, ls="--", alpha=0.5)
        ax_res.set_xlabel("Wavelength ($\\mu$m)")
        ax_res.set_ylabel("Residual ($\\sigma$)")
        ax_res.set_xscale("log")

        fig.tight_layout()
        if outfile:
            fig.savefig(outfile, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

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

