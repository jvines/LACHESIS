"""Star class, holds observed stellar properties for isochrone fitting."""

import logging

import numpy as np

from lachesis.display import display_banner
from lachesis.error import InputError, LachesisError

logger = logging.getLogger(__name__)


class ExtinctionError(LachesisError):
    """Raised when the line-of-sight Av cannot be obtained from any source.

    Use ``offline=True`` and pass ``Av=...`` explicitly to bypass dustmap
    queries entirely, or install the SFD/Bayestar/Planck data files and
    configure ``dustmaps.config['data_dir']`` before constructing a Star.
    """


# Av/E(B-V) coefficients (Schlafly+11 V-band).
_AV_PER_EBV_SFD = 2.742  # for SFD/Lenz
_AV_PER_EBV_PLANCK = 3.1
_AV_BAYESTAR_FACTOR = 2.742 * 0.884  # Bayestar correction (Schlafly+11)

# Gaia DR3 GSP_Phot Teff is bias-calibrated against APOGEE/GALAH but the
# reported errors (b_Teff/B_Teff) are unrealistically tight for bright
# stars, sometimes only a few Kelvin. Using them raw pins the fit at a
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
                    "saturated, fit results may be unreliable.",
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
            # Gaia FLAME radius (already fetched by the Librarian): not a
            # likelihood observable, kept only for pre-fit evolutionary
            # triage (see evolutionary_state).
            self.radius_flame = lib._radius_flame
            self.radius_flame_e = lib._radius_flame_e

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
            self.radius_flame = None
            self.radius_flame_e = None
            self.spectroscopic_source = (
                "user_supplied" if (logg is not None or feh is not None) else None
            )
            self.spectroscopic_params = None

        # Derive distance from parallax if not given
        if self.distance is None and self.parallax is not None and self.parallax > 0:
            self.distance = 1000.0 / self.parallax
            if self.parallax_e is not None:
                self.distance_e = self.distance * (self.parallax_e / self.parallax)

        # Local Bubble shortcut: for stars within ~70 pc the SFD/Planck
        # all-the-way-to-infinity dust column drastically over-estimates the
        # actual foreground extinction (the LB is essentially dust-free out
        # to ≳ 80 pc; cf. Lallement+19, Vergely+22). Fix Av = 0 for nearby
        # stars unless the user explicitly supplied an Av.
        if self.Av is None and self.distance is not None and self.distance < 70.0:
            self.Av = 0.0
        # Otherwise, query the dustmap for the line-of-sight upper bound.
        if self.Av is None:
            self._query_extinction(dustmap)

        # Photometry QC, three independent checks (warn-only):
        #   1. BC-table consensus fit + per-catalogue coherent offset
        #   2. ARIADNE-equivalent blackbody fit
        # All flagged bands accumulate on the Star instance under
        # `_qc_flagged_bands`, `_qc_catalogue_flags`,
        # `_qc_bb_flagged_bands`, the user decides removal.
        if self.magnitudes:
            self._photometry_check()
            self._photometry_check_blackbody()

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
            raise ExtinctionError(
                f"Unknown dustmap '{dustmap}'. Valid options: "
                f"{sorted(self._DUSTMAPS)}. Pass Av=... explicitly to bypass."
            )

        mod_path, cls_name = self._DUSTMAPS[dustmap]
        try:
            mod = import_module(mod_path)
            dmap = getattr(mod, cls_name)()
        except Exception as e:
            raise ExtinctionError(
                f"Dustmap '{dustmap}' could not be initialised ({type(e).__name__}: {e}). "
                f"Either install the dust data (e.g. `python -c \"import dustmaps.sfd; "
                f"dustmaps.sfd.fetch()\"`) and configure dustmaps.config['data_dir'], "
                f"or pass Av=... explicitly when constructing the Star."
            ) from e

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
            raise ExtinctionError(
                f"Dustmap '{dustmap}' query failed at "
                f"(ra={self.ra}, dec={self.dec}, d={d:.1f} pc): "
                f"{type(e).__name__}: {e}. Pass Av=... explicitly to bypass."
            ) from e

    @classmethod
    def from_ariadne_star(cls, ariadne_star, verbose: bool = True):
        """Create a LACHESIS Star from an astroARIADNE Star object.

        Extracts magnitudes, parallax, distance, and spectroscopic params.
        Works with any ARIADNE Star; just pass the object directly.

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
            # No coordinates, spectroscopic-only using ARIADNE params directly
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
        # - feh_posterior -> sampled parameter, fed through prior transform
        #   as a KDE-based inverse CDF, preserving full distribution shape.
        # - external_posteriors -> KDE penalties in the likelihood for all
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

    # Catalogue -> band-name prefix mapping used for catalogue-coherent
    # offset detection. Bands with the same source catalogue should share
    # a single per-source cross-match; a coherent offset of the whole group
    # is a tell-tale sign that catalogue was matched to the wrong sky source.
    _CATALOGUE_PREFIXES = {
        "Gaia":       ("Gaia_", "GaiaDR"),
        "2MASS":      ("2MASS_",),
        "WISE":       ("WISE_",),
        "PanSTARRS":  ("PS1_",),
        "SDSS":       ("SDSS_",),
        "Tycho":      ("TYCHO2_", "TYCHO_"),
        "APASS":      ("APASS_",),
        "SkyMapper":  ("SkyMapper_",),
        "GALEX":      ("GALEX_",),
        "Johnson":    ("Bessell_", "GROUND_JOHNSON_"),
        "TESS":       ("TESS",),
    }

    def _band_catalogue(self, band: str) -> str | None:
        for cat, prefixes in self._CATALOGUE_PREFIXES.items():
            if any(band.startswith(p) for p in prefixes):
                return cat
        return None

    # ── ARIADNE-style blackbody QC ─────────────────────────────────────
    # Zero-point fluxes (erg / cm² / s / Å at effective wavelength).
    # Vega convention for most catalogues, AB monochromatic flux density
    # f_ν = 3631 Jy = 3.631e-20 erg/s/cm²/Hz converted to f_λ for AB ones.
    _AB_FNU = 3.631e-20  # erg/s/cm²/Hz, AB zero point
    _AB_FAMILIES = ("SDSS_", "PS1_", "GALEX_", "SkyMapper_")

    def _photometry_check_blackbody(self, fit_tol=0.5, flag_tol=0.75,
                                     flag_tol_model=1.5, err_floor=0.05,
                                     T_min=2000.0, T_max=60000.0,
                                     n_T=300):
        """Iterative SED outlier rejection using a pure-Planck synthetic SED.

        Fits a blackbody across [T_min, T_max], compares observed magnitudes
        to synthetic BB magnitudes via per-filter zero-point fluxes, seeds the
        trusted set from Gaia anchors and iteratively promotes untrusted bands
        within *fit_tol* magnitudes so the fit stays anchored to well-behaved
        photometry. Outlier strength is the magnitude deviation from the model
        SED, deliberately independent of the photometric errorbar, so a wildly
        discrepant point carrying a large error cannot hide below threshold.

        A band is flagged when its deviation exceeds the flag tolerance. That
        tolerance is *flag_tol* in the optical, where a blackbody is a good
        model, but *flag_tol_model* in the IR (Rayleigh-Jeans, molecular bands)
        and the blue/UV (Balmer jump, line blanketing), where a single-Planck
        SED deviates by up to ~1.4 mag for real stars; flagging those as bad
        photometry would be wrong. Contamination (saturation, cross-match) sits
        at >=5 mag, far above either tolerance. Warn-only; the user decides.
        """
        from lachesis.filters import FILTER_WAVELENGTH

        # Physical constants (SI for the Planck-law form below)
        h = 6.62607015e-34
        c = 2.99792458e8
        k = 1.380649e-23

        bands, mags, errs, lam_um = [], [], [], []
        for b, (m, e) in self.magnitudes.items():
            lam = FILTER_WAVELENGTH.get(b)
            if lam is None or e <= 0:
                continue
            bands.append(b); mags.append(m); errs.append(max(e, err_floor))
            lam_um.append(lam)
        if len(bands) < 3:
            self._qc_bb_flagged_bands = []
            return
        bands = np.array(bands)
        mags  = np.array(mags)
        errs  = np.array(errs)
        lam_m = np.array(lam_um) * 1e-6  # μm -> m

        # Per-filter Vega/AB zero-point flux density (erg/s/cm²/Å).
        f0 = np.empty(len(bands))
        for i, b in enumerate(bands):
            if any(b.startswith(p) for p in self._AB_FAMILIES):
                # AB: f_ν = 3631 Jy; convert to f_λ at λ_eff
                lam_A = lam_um[i] * 1e4  # Å
                f0[i] = self._AB_FNU * c * 1e10 / lam_A**2  # erg/s/cm²/Å
            else:
                # Vega: pyphot Vega_zero_flux on best-matching filter name.
                f0[i] = self._vega_zero_flux(b, lam_um[i])

        def _synth_mags(T):
            x = h * c / (lam_m * k * max(T, 1.0))
            B = 1.0 / (lam_m**5 * (np.exp(np.clip(x, 0, 500)) - 1.0))
            return -2.5 * np.log10(np.maximum(B / f0, 1e-300))

        def _residuals(fit_idx):
            if len(fit_idx) < 2:
                return np.full(len(mags), np.inf), 5000.0, 0.0
            best_resid = np.full(len(mags), np.inf)
            best_rss = np.inf
            best_T, best_off = 5000.0, 0.0
            for T in np.linspace(T_min, T_max, n_T):
                model = _synth_mags(T)
                offset = np.mean(mags[fit_idx] - model[fit_idx])
                resid = np.abs(mags - (model + offset))
                rss = np.sum(resid[fit_idx]**2)
                if rss < best_rss:
                    best_rss = rss
                    best_resid = resid
                    best_T = T
                    best_off = offset
            return best_resid, best_T, best_off

        gaia_mask = np.array([any(b.startswith(p) for p in self._GAIA_PREFIXES)
                              for b in bands])
        if np.any(gaia_mask):
            trusted = gaia_mask.copy()
            while True:
                untrusted_idx = np.where(~trusted)[0]
                if len(untrusted_idx) == 0:
                    break
                resid, _, _ = _residuals(np.where(trusted)[0])
                z = resid  # |Δmag| from the model; errorbar deliberately ignored
                best = untrusted_idx[np.argmin(z[untrusted_idx])]
                if z[best] < fit_tol:
                    trusted[best] = True
                else:
                    break
            outlier_idx = np.where(~trusted)[0]
        else:
            active = np.ones(len(bands), dtype=bool)
            while True:
                active_idx = np.where(active)[0]
                if len(active_idx) < 4:
                    break
                resid, _, _ = _residuals(active_idx)
                z = resid  # |Δmag| from the model; errorbar deliberately ignored
                worst = active_idx[np.argmax(z[active_idx])]
                if z[worst] >= fit_tol:
                    active[worst] = False
                else:
                    break
            outlier_idx = np.where(~active)[0]

        # Final fit on the trusted set, capturing T and offset for plotting.
        trusted_idx = np.setdiff1d(np.arange(len(bands)), outlier_idx)
        resid_final, best_T, best_offset = _residuals(trusted_idx)
        self._qc_bb_teff = float(best_T)
        self._qc_bb_offset = float(best_offset)
        # Persist f0 so the plotting routine can reconstruct the BB curve in
        # F_λ units without recomputing zero points.
        self._qc_bb_f0_per_band = dict(zip(bands.tolist(), f0.tolist()))

        from termcolor import colored
        # Flag on the deviation from the final BB, but loosen the tolerance
        # where the blackbody itself is an unreliable model (IR and blue/UV),
        # so genuine model deviation is not mistaken for bad photometry. The
        # trusted set above still used the tight fit_tol, so the fit is robust.
        def _bb_unreliable(b):
            return (b.startswith(("2MASS_", "WISE_", "SPITZER_", "GALEX_",
                                  "STROMGREN_"))
                    or b in ("GROUND_JOHNSON_U", "GROUND_JOHNSON_B"))
        flagged = []
        for i in range(len(bands)):
            tol = flag_tol_model if _bb_unreliable(bands[i]) else flag_tol
            if resid_final[i] > tol:
                z = resid_final[i]  # |Δmag| from the BB model
                flagged.append((bands[i], float(z)))
                print(colored(
                    f"\t\t\tWARNING: {bands[i]} flagged "
                    f"(blackbody QC, {z:.2f} mag from BB fit at T={best_T:.0f} K) -- not removed",
                    "magenta",
                ))
        self._qc_bb_flagged_bands = [b for b, _ in flagged]

    @staticmethod
    def _vega_zero_flux(pyphot_name: str, lam_um: float) -> float:
        """Vega zero-point f_λ in erg/s/cm²/Å, from pyphot's filter library.

        pyphot is a hard dependency (see pyproject). Raises if the filter
        cannot be resolved rather than substituting an approximate value: a
        missing zero point must surface loudly, not silently miscalibrate the
        SED QC. (The old blackbody fallback was ~5.4 mag off and made every
        AB-system band look like a saturated outlier.)
        """
        import pyphot
        lib = pyphot.get_library()
        for candidate in (pyphot_name, pyphot_name.replace("Gaia_", "GaiaDR3_"),
                          pyphot_name.replace("Gaia_", "GaiaDR2_"),
                          pyphot_name.replace("WISE_RSR_", "WISE_"),
                          pyphot_name.replace("TYCHO_", "Tycho_"),
                          pyphot_name.replace("_MvB", "")):
            if candidate in lib.content:
                return float(lib[candidate].Vega_zero_flux.to_value(
                    "erg / (Angstrom s cm2)"))
        raise ValueError(
            f"pyphot has no Vega zero point for filter {pyphot_name!r} "
            f"(tried name variants). Add a name mapping or zero point; do not "
            f"approximate."
        )

    def _photometry_check(self, mag_tol=0.75, err_floor=0.05,
                          catalogue_mag_tol=0.5):
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
                z = resid  # |Δmag| from the model; errorbar deliberately ignored
                best = untrusted_idx[np.argmin(z[untrusted_idx])]
                if z[best] < mag_tol:
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
                z = resid  # |Δmag| from the model; errorbar deliberately ignored
                worst = active_idx[np.argmax(z[active_idx])]
                if z[worst] >= mag_tol:
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
            z = abs(mags[i] - pred[i]) if not np.isnan(pred[i]) else 0.0
            self._phot_residuals[b] = (
                wavelengths[i],  # microns
                mags[i], pred[i], z,
            )
            if outlier_mask[i]:
                flagged.append((b, z))

        # ── Catalogue-coherent offset check ───────────────────────────────
        # Group residuals by source catalogue. A whole catalogue whose *mean
        # signed* magnitude offset exceeds catalogue_mag_tol is almost certainly
        # mis-cross-matched (the catalogue picked up a different sky source). The
        # mean cancels for non-coherent scatter, so it isolates genuine bulk
        # offsets. Errorbar-independent and magnitude-scaled: a 0.2 mag zero-
        # point difference between systems is left alone, while a real cross-
        # match (>=0.5 mag bulk shift) flags the whole group at once.
        cat_flags: dict[str, dict] = {}
        signed_dev = np.where(np.isnan(pred), 0.0, mags - pred)  # magnitudes
        for cat in self._CATALOGUE_PREFIXES:
            idx = np.array([i for i, b in enumerate(bands)
                            if self._band_catalogue(b) == cat])
            if len(idx) < 2:
                continue
            mean_dev = float(np.mean(signed_dev[idx]))
            if abs(mean_dev) > catalogue_mag_tol:
                cat_flags[cat] = {
                    "mean_dev": mean_dev,
                    "n_bands": len(idx),
                    "bands": [bands[i] for i in idx],
                }
                # If they survived the per-band trusted set, still mark them
                # as outliers so callers / `auto_remove` can act.
                outlier_mask[idx] = True
                for i in idx:
                    dev = abs(mags[i] - pred[i]) if not np.isnan(pred[i]) else 0.0
                    if (bands[i], dev) not in flagged:
                        flagged.append((bands[i], dev))

        from termcolor import colored
        if cat_flags:
            for cat, info in cat_flags.items():
                msg = (
                    f"\t\t\tWARNING: catalogue '{cat}' has a coherent "
                    f"{info['mean_dev']:+.2f} mag offset across {info['n_bands']} "
                    f"bands; likely cross-matched to a different source"
                )
                print(colored(msg, "red"))

        # Per-band warnings (still distinguish "flagged via per-band sigma"
        # from "flagged via catalogue-coherent offset" in the message).
        cat_member_bands = {b for info in cat_flags.values() for b in info["bands"]}
        if flagged:
            for b, z in flagged:
                via_cat = b in cat_member_bands
                tag = ("catalogue-coherent offset" if via_cat
                       else "per-band outlier")
                print(colored(
                    f"\t\t\tWARNING: {b} flagged ({tag}, {z:.2f} mag from "
                    f"photometry consensus fit) -- not removed",
                    "yellow",
                ))

        # Persist QC artefacts so plotters / callers can act on them.
        self._qc_catalogue_flags = cat_flags
        self._qc_flagged_bands = [b for b, _ in flagged]

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
            f"{self.starname}, $T_{{\\rm fit}}$ = {self._phot_teff:.0f} K"
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

    def plot_SED_no_model(self, outfile: str | None = None):
        """ARIADNE-style raw SED plot: λF_λ vs λ in log-log, no model overlay.

        Each band is drawn as a coloured point with its own marker, with
        photometric error bars and the filter bandpass shown as horizontal
        x-error bars. QC-flagged bands (from either the BC-table or the
        blackbody check) are outlined in red and labelled with the flag
        reason. Designed for user inspection: eyeballing whether a given
        catalogue's points "belong" to the same SED as the rest.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from lachesis.filters import FILTER_WAVELENGTH

        # Collect bands with wavelengths
        bands, mags, errs, lam_um = [], [], [], []
        for b, (m, e) in self.magnitudes.items():
            lam = FILTER_WAVELENGTH.get(b)
            if lam is None or e <= 0:
                continue
            bands.append(b); mags.append(m); errs.append(e); lam_um.append(lam)
        if not bands:
            return
        bands = np.array(bands)
        mags  = np.array(mags)
        errs  = np.array(errs)
        lam_um = np.array(lam_um)

        # Per-filter zero-point for mag -> F_λ conversion
        f0 = np.array([self._vega_zero_flux(b, lam) for b, lam in zip(bands, lam_um)])
        for i, b in enumerate(bands):
            if any(b.startswith(p) for p in self._AB_FAMILIES):
                lam_A = lam_um[i] * 1e4
                f0[i] = self._AB_FNU * 2.99792458e8 * 1e10 / lam_A**2
        F_lam   = f0 * 10.0 ** (-0.4 * mags)        # erg/s/cm²/Å
        F_err   = F_lam * np.log(10) * 0.4 * errs   # propagated 1-σ
        lF_l    = F_lam * (lam_um * 1e4)            # λF_λ in erg/s/cm² (Å as bandpass scale)
        lF_l_err = F_err * (lam_um * 1e4)

        # Approximate bandpass width, 15% of λ_eff if not otherwise known.
        bp = lam_um * 0.15

        # Colour each point by its effective wavelength so the SED shape
        # reads directly off the plot (blue -> short λ, red -> long λ).
        cmap = plt.cm.RdYlBu_r
        lam_log = np.log10(lam_um)
        norm = plt.Normalize(vmin=lam_log.min(), vmax=lam_log.max())
        colours = cmap(norm(lam_log))

        flagged = set(getattr(self, "_qc_flagged_bands", []) +
                      getattr(self, "_qc_bb_flagged_bands", []))

        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        # Blackbody curve from the QC fit, if available. Scale fit to the
        # *observed* λF_λ on the trusted bands (the ones not flagged by
        # either QC check) so the curve actually overlays the data rather
        # than sitting at some abstract zero-point reference.
        bb_T = getattr(self, "_qc_bb_teff", None)
        if bb_T is not None and bb_T > 0:
            h_, c_, k_ = 6.62607015e-34, 2.99792458e8, 1.380649e-23

            def _planck(lam_um_arr):
                lam_m = lam_um_arr * 1e-6
                x = h_ * c_ / (lam_m * k_ * bb_T)
                return 1.0 / (lam_m**5 * (np.exp(np.clip(x, 0, 500)) - 1.0))

            # Trusted bands = those not flagged. Use them to set the scale.
            trusted_mask = np.array([b not in flagged for b in bands])
            if trusted_mask.sum() < 2:
                trusted_mask = np.ones(len(bands), dtype=bool)
            B_at_bands = _planck(lam_um)
            # log-space ratio so the scale is robust to a single bright band.
            log_scale = float(np.median(
                np.log10(F_lam[trusted_mask]) - np.log10(B_at_bands[trusted_mask])
            ))
            scale = 10.0 ** log_scale

            lam_curve = np.logspace(np.log10(max(lam_um.min() * 0.5, 0.1)),
                                    np.log10(lam_um.max() * 2.0), 400)
            F_lam_curve = _planck(lam_curve) * scale
            lF_l_curve = F_lam_curve * (lam_curve * 1e4)
            ax.plot(lam_curve, lF_l_curve,
                    color="0.4", lw=1.0, ls="--",
                    label=f"BB fit, $T={bb_T:.0f}$ K", zorder=1)

        for i, b in enumerate(bands):
            edge = "red" if b in flagged else "black"
            lw_e = 1.2 if b in flagged else 0.5
            ax.errorbar(lam_um[i], lF_l[i],
                        xerr=bp[i], yerr=lF_l_err[i],
                        fmt="none", ecolor=colours[i], elinewidth=0.9,
                        capsize=0, alpha=0.85, zorder=2)
            ax.scatter(lam_um[i], lF_l[i],
                       s=60, c=[colours[i]], edgecolors=edge,
                       linewidths=lw_e, label=b, zorder=3)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength  ($\mu$m)")
        ax.set_ylabel(r"$\lambda F_\lambda$  (erg s$^{-1}$ cm$^{-2}$)")
        ax.tick_params(which="both", direction="in")
        ax.xaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        # x-range from data, modest pad
        ax.set_xlim(lam_um.min() * 0.7, lam_um.max() * 1.4)

        # Per-band legend (one entry per band, as before).
        ncol = 2 if len(bands) <= 16 else 3
        ax.legend(fontsize=7, ncol=ncol, loc="best", framealpha=0.85)

        # Annotate flagged bands in-plot at their data positions so the
        # warning can't collide with the legend.
        for i, b in enumerate(bands):
            if b in flagged:
                ax.annotate(b, (lam_um[i], lF_l[i]),
                            xytext=(4, 6), textcoords="offset points",
                            fontsize=7, color="red", zorder=10)

        fig.tight_layout()
        if outfile:
            fig.savefig(outfile, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @property
    def evolutionary_state(self) -> str:
        """Pre-fit evolutionary classification: 'ms', 'subgiant', 'giant', or
        'unknown'. Used to triage grid coverage before fitting (grids without
        giant coverage, e.g. YaPSI, are dropped up front); never enters the
        likelihood.

        Cascades three tiers, most direct first:

        1. Spectroscopic log g (survey chain): >= 4.0 ms, 3.2-4.0 subgiant,
           < 3.2 giant. Dwarf/giant is a > 1 dex contrast, so survey-quality
           log g (~0.2-0.3 dex) settles it.
        2. Gaia FLAME radius: R > 4 Rsun is safely post-subgiant for FGK.
           FLAME is model-informed, so this tier only flags clear giants.
        3. Dereddened Gaia CMD position: the giant branch sits several
           magnitudes above the MS at fixed colour, so M_G < 3.5 at
           (BP-RP)_0 > 0.95 is unambiguous (an equal-luminosity MS binary
           lifts a star only 0.75 mag, well short of the cut). Extinction
           coefficients A_G = 0.789 Av, E(BP-RP) = 0.413 Av (Wang & Chen
           2019).

        Tiers 2-3 cannot separate ms from subgiant without a model, so when
        they do not flag a giant the state is 'unknown' rather than a guess.
        """
        if self.logg is not None:
            if self.logg < 3.2:
                return "giant"
            if self.logg < 4.0:
                return "subgiant"
            return "ms"
        rf = getattr(self, "radius_flame", None)
        if rf is not None and rf > 4.0:
            return "giant"
        def _mag(*keys):
            for k in keys:
                v = self.magnitudes.get(k)
                if v is not None and v[0] is not None:
                    return float(v[0])
            return None
        g = _mag("Gaia_G", "Gaia_G_EDR3")
        bp = _mag("Gaia_BP", "Gaia_BP_EDR3")
        rp = _mag("Gaia_RP", "Gaia_RP_EDR3")
        plx = self.parallax
        if None not in (g, bp, rp) and plx is not None and plx > 0:
            av = self.Av if self.Av is not None else 0.0
            abs_g = g + 5.0 * np.log10(plx) - 10.0 - 0.789 * av
            bprp0 = (bp - rp) - 0.413 * av
            if abs_g < 3.5 and bprp0 > 0.95:
                return "giant"
        return "unknown"

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
        # Photometric bands, translate pyphot names to BC table names
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

