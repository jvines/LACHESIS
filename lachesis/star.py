"""Star class — holds observed stellar properties for isochrone fitting."""

import logging

import numpy as np

from lachesis.display import display_banner
from lachesis.error import InputError

logger = logging.getLogger(__name__)

# Av/E(B-V) coefficients (Schlafly+11 V-band).
_AV_PER_EBV_SFD = 2.742  # for SFD/Lenz
_AV_PER_EBV_PLANCK = 3.1
_AV_BAYESTAR_FACTOR = 2.742 * 0.884  # Bayestar correction (Schlafly+11)
_AV_FALLBACK = 0.1  # used when dustmap query fails

# Gaia DR3 GSP_Phot Teff is bias-calibrated against APOGEE/GALAH but the
# reported errors (b_Teff/B_Teff) are unrealistically tight for bright
# stars — sometimes only a few Kelvin. Using them raw pins the fit at a
# possibly wrong Teff. We apply a floor based on Andrae+2023's external
# RMS (~100 K typical, 200 K conservative).
_GAIA_TEFF_ERR_FLOOR = 100.0


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
        offline: bool = False,
    ):
        self.starname = starname
        self.ra = ra
        self.dec = dec
        self.g_id = g_id
        self.external_posteriors = {}  # populated by from_ariadne()
        self.feh_posterior = None       # populated by from_ariadne()

        if magnitudes is None and not offline:
            if verbose:
                from lachesis.display import colored
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
            # Teff is never auto-injected as a likelihood prior. The
            # photometric SED constrains Teff through the BC tables; an
            # external Teff observable would double-count and overconstrain
            # (see librarian/_api.py for the GSP-Phot extraction note).
            self.teff = None
            self.teff_e = None
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
                self.spectroscopic_source = sp.get("source")
                self.spectroscopic_params = dict(sp)
            else:
                self.logg = logg
                self.logg_e = logg_e
                self.feh = feh
                self.feh_e = feh_e
                self.spectroscopic_source = None
                self.spectroscopic_params = None
        else:
            # User provided magnitudes (or offline=True): skip Librarian
            if verbose:
                display_banner(starname)
            self.magnitudes = magnitudes if magnitudes is not None else {}
            self.parallax = plx
            self.parallax_e = plx_e if plx is not None else None
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
            self.spectroscopic_source = (
                "user_supplied" if (logg is not None or feh is not None) else None
            )
            self.spectroscopic_params = None

        # Derive distance from parallax if not given
        if self.distance is None and self.parallax is not None and self.parallax > 0:
            self.distance = 1000.0 / self.parallax
            if self.parallax_e is not None:
                self.distance_e = self.distance * (self.parallax_e / self.parallax)

        # Max line-of-sight extinction from dustmaps (like ARIADNE)
        if self.Av is None:
            self._query_extinction(dustmap)

        # Photometry QC: flag outliers (warn only, no removal)
        if self.magnitudes:
            self._photometry_check()

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
            logger.warning("Unknown dustmap '%s'; using fallback Av=%.2f",
                           dustmap, _AV_FALLBACK)
            self.Av = _AV_FALLBACK
            return

        mod_path, cls_name = self._DUSTMAPS[dustmap]
        try:
            mod = import_module(mod_path)
            dmap = getattr(mod, cls_name)()
        except Exception as e:
            logger.warning("Dustmap '%s' init failed (%s); using fallback Av=%.2f",
                           dustmap, e, _AV_FALLBACK)
            self.Av = _AV_FALLBACK
            return

        d = self.distance if self.distance is not None else 1000.0
        # SFD / Planck queries are 2D and ignore distance, but Bayestar uses
        # it; pass it through for everyone since it does not change the SFD
        # result.
        coords = SkyCoord(self.ra, self.dec, distance=d,
                          unit=(u.deg, u.deg, u.pc), frame='icrs')

        try:
            if dustmap in ('SFD', 'Lenz'):
                ebv = dmap(coords)
                self.Av = float(ebv) * _AV_PER_EBV_SFD
            elif dustmap == 'Bayestar':
                ebvs = dmap(coords, mode='percentile', pct=[15, 50, 84])
                if np.any(np.isnan(ebvs)):
                    sfd_mod = import_module('dustmaps.sfd')
                    ebv = sfd_mod.SFDQuery()(coords)
                    self.Av = float(ebv) * _AV_PER_EBV_SFD
                else:
                    mags = ebvs * _AV_BAYESTAR_FACTOR
                    self.Av = float(mags[1])
            elif dustmap in ('Planck13', 'Planck16'):
                ebv = dmap(coords)
                self.Av = float(ebv) * _AV_PER_EBV_PLANCK
        except Exception as e:
            logger.warning("Dustmap '%s' query failed (%s); using fallback Av=%.2f",
                           dustmap, e, _AV_FALLBACK)
            self.Av = _AV_FALLBACK

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
                     g_id: int | None = None, magnitudes: dict | None = None,
                     verbose: bool = True):
        """Load stellar properties from an ARIADNE InferenceData .nc file.

        Stores full posterior sample arrays as external priors for the
        isochrone fit. The Fitter builds KDEs from these automatically.

        Parameters
        ----------
        magnitudes : dict, optional
            Pre-fetched photometry ``{band: (mag, err)}``. If provided,
            skips the Librarian VizieR lookup (avoids redundant query when
            photometry is already available from the ARIADNE preview).
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
            # Use provided magnitudes or retrieve via Librarian
            star = cls(
                starname, ra, dec, g_id=g_id,
                magnitudes=magnitudes,
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

    # --- Photometry QC ---

    _GAIA_PREFIXES = ("Gaia_", "GaiaDR")

    def _photometry_check(self, sigma=5.0, err_floor=0.05):
        """Iterative photometric outlier detection via BC table model.

        Uses bolometric correction tables (proper synthetic photometry
        with correct per-filter zero points), eliminating false positives
        from system zero-point differences.

        Seeds the trusted set from Gaia anchors (BP/G/RP), then
        iteratively promotes the best-fitting untrusted band if its
        residual is within *sigma*. Bands never promoted are flagged.

        Falls back to iterative worst-outlier removal when no Gaia
        bands are present.
        """
        from lachesis.filters import FILTER_WAVELENGTH, PYPHOT_TO_BC

        # Load BC table for synthetic photometry
        try:
            from lachesis.bc import BCTable
            from lachesis.config import BC_DIR
            bc = BCTable.multi_system(BC_DIR)
        except (FileNotFoundError, ImportError):
            self._phot_teff = None
            self._phot_residuals = {}
            return

        # Collect bands with valid errors AND BC table coverage
        bands, bc_names, wavelengths, mags, errs = [], [], [], [], []
        for b, (m, e) in self.magnitudes.items():
            bc_name = PYPHOT_TO_BC.get(b)
            lam = FILTER_WAVELENGTH.get(b)
            if bc_name is not None and bc_name in bc._band_indices and lam is not None and e > 0:
                bands.append(b)
                bc_names.append(bc_name)
                wavelengths.append(lam)
                mags.append(m)
                errs.append(max(e, err_floor))

        if len(bands) < 3:
            self._phot_teff = None
            self._phot_residuals = {}
            return

        bands = np.array(bands)
        wavelengths = np.array(wavelengths)
        mags = np.array(mags)
        errs = np.array(errs)

        bc.set_active_bands(bc_names)

        # Build model magnitude grid: pred = -BC(Teff) + offset
        # Mbol cancels in the offset, so relative mags are just -BC
        teff_grid = bc.teff_values
        # Pre-compute BC vectors for each temperature
        bc_vectors = np.full((len(teff_grid), len(bands)), np.nan)
        for ti, T in enumerate(teff_grid):
            bcs = bc.get_bc(T, 4.5, 0.0, 0.0)
            for bi, bn in enumerate(bc_names):
                v = bcs.get(bn)
                if v is not None:
                    bc_vectors[ti, bi] = -v

        def _fit_residuals(fit_idx):
            """Best-fit BC model residuals using only fit_idx bands."""
            if len(fit_idx) < 2:
                return np.full(len(mags), np.inf), 5000.0, 0.0
            best_resid = np.full(len(mags), np.inf)
            best_rss = np.inf
            best_T, best_off = 5000.0, 0.0
            for ti in range(len(teff_grid)):
                model = bc_vectors[ti]
                if np.any(np.isnan(model[fit_idx])):
                    continue
                offset = np.nanmean(mags[fit_idx] - model[fit_idx])
                resid = np.abs(mags - (model + offset))
                rss = np.nansum(resid[fit_idx] ** 2)
                if rss < best_rss:
                    best_rss = rss
                    best_resid = resid
                    best_T = teff_grid[ti]
                    best_off = offset
            return best_resid, best_T, best_off

        # Iterative acceptance seeded from Gaia anchors
        gaia_mask = np.array([
            any(b.startswith(p) for p in self._GAIA_PREFIXES) for b in bands
        ])

        if np.any(gaia_mask):
            trusted = gaia_mask.copy()
            while True:
                untrusted_idx = np.where(~trusted)[0]
                if len(untrusted_idx) == 0:
                    break
                resid, _, _ = _fit_residuals(np.where(trusted)[0])
                z = resid / errs
                best = untrusted_idx[np.argmin(z[untrusted_idx])]
                if z[best] < sigma:
                    trusted[best] = True
                else:
                    break
            outlier_mask = ~trusted
        else:
            # No Gaia: iterative worst-outlier removal
            active = np.ones(len(bands), dtype=bool)
            while True:
                active_idx = np.where(active)[0]
                if len(active_idx) < 4:
                    break
                resid, _, _ = _fit_residuals(active_idx)
                z = resid / errs
                worst = active_idx[np.argmax(z[active_idx])]
                if z[worst] >= sigma:
                    active[worst] = False
                else:
                    break
            outlier_mask = ~active

        # Final fit using all trusted bands
        trusted_idx = np.where(~outlier_mask)[0]
        _, best_T, best_offset = _fit_residuals(trusted_idx)

        # Find the best-fit BC vector for the final plot
        ti_best = np.argmin(np.abs(teff_grid - best_T))
        pred = bc_vectors[ti_best] + best_offset

        self._phot_teff = best_T
        self._phot_offset = best_offset
        self._phot_residuals = {}
        flagged = []

        for i, b in enumerate(bands):
            z = abs(mags[i] - pred[i]) / errs[i] if not np.isnan(pred[i]) else 0.0
            self._phot_residuals[b] = (
                wavelengths[i],  # microns
                mags[i], pred[i], z,
            )
            if outlier_mask[i]:
                flagged.append((b, z))

        if flagged:
            from termcolor import colored
            for b, z in flagged:
                print(colored(
                    f"\t\t\tWARNING: {b} is potentially problematic "
                    f"({z:.1f}sigma from photometry consensus fit) -- not removed",
                    "yellow",
                ))

    def plot_photometry(self, outfile=None):
        """Diagnostic plot: BC-model fit with color-coded photometry.

        Parameters
        ----------
        outfile : str, optional
            Save path. If None, calls plt.show().
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from matplotlib.colors import LogNorm

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        if not getattr(self, "_phot_residuals", None):
            print("No photometry QC results available.")
            return

        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=(8, 6), height_ratios=[3, 1],
            sharex=True, gridspec_kw={"hspace": 0.05},
        )

        # Collect data
        bands = list(self._phot_residuals.keys())
        lams = np.array([self._phot_residuals[b][0] for b in bands])
        obs = np.array([self._phot_residuals[b][1] for b in bands])
        pred = np.array([self._phot_residuals[b][2] for b in bands])
        zvals = np.array([self._phot_residuals[b][3] for b in bands])
        errs_raw = np.array([
            max(self.magnitudes[b][1], 0.05) for b in bands
        ])

        # Wavelength colormap
        cmap = plt.cm.Spectral_r
        norm = LogNorm(vmin=max(lams.min(), 0.1), vmax=lams.max())

        # Model curve: connect predicted magnitudes sorted by wavelength
        order = np.argsort(lams)
        valid = ~np.isnan(pred[order])
        ax_main.plot(
            lams[order][valid], pred[order][valid],
            "k-", lw=1, alpha=0.4, zorder=1,
        )

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
            f"{self.starname}  —  $T_{{\\rm fit}}$ = {self._phot_teff:.0f} K"
        )
        ax_main.legend(
            fontsize=6, ncol=3, loc=0,
            framealpha=0.7,
        )

        ax_res.axhline(0, color="k", lw=0.5)
        ax_res.axhline(5, color="r", lw=0.5, ls="--", alpha=0.5)
        ax_res.axhline(-5, color="r", lw=0.5, ls="--", alpha=0.5)
        ax_res.set_xlabel("Wavelength ($\\mu$m)")
        ax_res.set_ylabel("Residual ($\\sigma$)")
        ax_res.set_xscale("log")
        ax_res.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax_res.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax_res.ticklabel_format(axis="x", style="plain")

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

