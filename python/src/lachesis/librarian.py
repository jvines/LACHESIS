"""Librarian — automated photometry and astrometry retrieval.

Queries Gaia DR3 best_neighbour tables for catalog crossmatching,
with VizieR XMatch fallback when Gaia TAP is unavailable.
Uses pyphot filter names internally (ARIADNE-compatible).

Ported from astroARIADNE with cleaner declarative structure.
"""

__all__ = ["Librarian"]

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.ma as ma
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from regions import CircleSkyRegion

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, append=True)
Vizier.ROW_LIMIT = -1
Vizier.columns = ["all"]


# ── Data structures ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MagSpec:
    """One photometric band within a catalog."""

    col: str
    err_col: str
    pyphot: str


@dataclass(frozen=True, slots=True)
class CatalogDef:
    """Declarative catalog definition."""

    vizier_id: str | None
    bands: tuple[MagSpec, ...]
    xmatch_table: str | None = None
    xmatch_id_cols: tuple[str, ...] | None = None
    qc: Callable | None = None


# ── Helpers ──────────────────────────────────────────────────────


def _col(row, col):
    """Extract value from an astropy Row. Returns None if missing or masked."""
    try:
        v = row[col]
    except (KeyError, IndexError):
        return None
    if ma.is_masked(v):
        return None
    return v


# ── QC functions ─────────────────────────────────────────────────


def _qc_mag(mag, err, max_err=1.0):
    """Universal magnitude quality check."""
    if mag is None or ma.is_masked(mag):
        return False
    try:
        if np.isnan(float(mag)):
            return False
    except (TypeError, ValueError):
        return False
    if err is not None and not ma.is_masked(err):
        try:
            if float(err) > max_err:
                return False
        except (TypeError, ValueError):
            pass
    return True


def _qc_2mass_band(row, band_idx):
    """QC for a single 2MASS band (J=0, H=1, Ks=2).

    Checks Qflg (photometric quality) and Cflg (contamination/confusion).
    """
    qflg = str(_col(row, "Qflg") or "UUU")
    cflg = str(_col(row, "Cflg") or "999")
    if band_idx >= len(qflg) or band_idx >= len(cflg):
        return False
    return qflg[band_idx] in "ABCD" and cflg[band_idx] == "0"


def _qc_wise_band(row, band_idx):
    """QC for a single WISE band (W1=0, W2=1)."""
    qph = str(_col(row, "qph") or "UU")
    if band_idx >= len(qph):
        return False
    return qph[band_idx] in "ABC"


def _qc_wise_extended(row):
    """WISE extended source flag. Returns False if extended."""
    ex = _col(row, "ex")
    if ex is None:
        return True
    return int(ex) == 0


def _qc_sdss(row):
    """SDSS: star (class=6), good photometry (Q=2 or 3)."""
    cls = _col(row, "class")
    q = _col(row, "Q")
    if cls is None or q is None:
        return False
    return int(cls) == 6 and int(q) in (2, 3)


def _qc_ps1(row):
    """Pan-STARRS quality bitflag check."""
    qual = _col(row, "Qual")
    if qual is None:
        return False
    q = int(qual)
    is_star = not (q & 1 and q & 2)
    is_good = (q & 4 or q & 16) and not (q & 128)
    return is_star and bool(is_good)


def _qc_galex_band(row, pyphot_name):
    """GALEX per-band artifact/extraction flag check."""
    if pyphot_name == "GALEX_FUV":
        for col_name in ("Fexf", "Fafl"):
            v = _col(row, col_name)
            if v is not None and int(v) > 0:
                return False
    elif pyphot_name == "GALEX_NUV":
        for col_name in ("Nexf", "Nafl"):
            v = _col(row, col_name)
            if v is not None and int(v) > 0:
                return False
    return True


def _qc_skymapper(row):
    """SkyMapper: flags must be 0."""
    flags = _col(row, "flags")
    if flags is None:
        return False
    return int(flags) == 0


# ── Band index maps ─────────────────────────────────────────────

_2MASS_BAND_IDX = {"2MASS_J": 0, "2MASS_H": 1, "2MASS_Ks": 2}
_WISE_BAND_IDX = {"WISE_RSR_W1": 0, "WISE_RSR_W2": 1}


# ── Catalog registry ────────────────────────────────────────────

_CATALOGS = {
    "Gaia": CatalogDef(
        vizier_id="I/355/gaiadr3",
        bands=(
            MagSpec("Gmag", "e_Gmag", "Gaia_G"),
            MagSpec("BPmag", "e_BPmag", "Gaia_BP"),
            MagSpec("RPmag", "e_RPmag", "Gaia_RP"),
        ),
    ),
    "2MASS": CatalogDef(
        vizier_id="II/246/out",
        bands=(
            MagSpec("Jmag", "e_Jmag", "2MASS_J"),
            MagSpec("Hmag", "e_Hmag", "2MASS_H"),
            MagSpec("Kmag", "e_Kmag", "2MASS_Ks"),
        ),
        xmatch_table="tmass_psc_xsc",
        xmatch_id_cols=("2MASS", "_2MASS", "_2M"),
    ),
    "Wise": CatalogDef(
        vizier_id="II/328/allwise",
        bands=(
            MagSpec("W1mag", "e_W1mag", "WISE_RSR_W1"),
            MagSpec("W2mag", "e_W2mag", "WISE_RSR_W2"),
        ),
        xmatch_table="allwise",
        xmatch_id_cols=("AllWISE",),
    ),
    "Pan-STARRS": CatalogDef(
        vizier_id="II/349/ps1",
        bands=(
            MagSpec("gmag", "e_gmag", "PS1_g"),
            MagSpec("rmag", "e_rmag", "PS1_r"),
            MagSpec("imag", "e_imag", "PS1_i"),
            MagSpec("zmag", "e_zmag", "PS1_z"),
            MagSpec("ymag", "e_ymag", "PS1_y"),
        ),
        xmatch_table="panstarrs1",
        xmatch_id_cols=("objID",),
        qc=_qc_ps1,
    ),
    "SDSS": CatalogDef(
        vizier_id="V/147/sdss12",
        bands=(
            MagSpec("umag", "e_umag", "SDSS_u"),
            MagSpec("gmag", "e_gmag", "SDSS_g"),
            MagSpec("rmag", "e_rmag", "SDSS_r"),
            MagSpec("imag", "e_imag", "SDSS_i"),
            MagSpec("zmag", "e_zmag", "SDSS_z"),
        ),
        xmatch_table="sdssdr13",
        xmatch_id_cols=("objID",),
        qc=_qc_sdss,
    ),
    "TYCHO2": CatalogDef(
        vizier_id="I/259/tyc2",
        bands=(
            MagSpec("BTmag", "e_BTmag", "TYCHO_B_MvB"),
            MagSpec("VTmag", "e_VTmag", "TYCHO_V_MvB"),
        ),
        xmatch_table="tycho2tdsc_merge",
    ),
    "APASS": CatalogDef(
        vizier_id=None,
        bands=(
            MagSpec("vmag", "e_vmag", "GROUND_JOHNSON_V"),
            MagSpec("bmag", "e_bmag", "GROUND_JOHNSON_B"),
            MagSpec("g_mag", "e_g_mag", "SDSS_g"),
            MagSpec("r_mag", "e_r_mag", "SDSS_r"),
            MagSpec("i_mag", "e_i_mag", "SDSS_i"),
        ),
        xmatch_table="apassdr9",
    ),
    "SkyMapper": CatalogDef(
        vizier_id=None,
        bands=(
            MagSpec("u_psf", "e_u_psf", "SkyMapper_u"),
            MagSpec("v_psf", "e_v_psf", "SkyMapper_v"),
            MagSpec("g_psf", "e_g_psf", "SkyMapper_g"),
            MagSpec("r_psf", "e_r_psf", "SkyMapper_r"),
            MagSpec("i_psf", "e_i_psf", "SkyMapper_i"),
            MagSpec("z_psf", "e_z_psf", "SkyMapper_z"),
        ),
        xmatch_table="skymapperdr2",
        xmatch_id_cols=("ObjectId", "object_id"),
        qc=_qc_skymapper,
    ),
    "TESS": CatalogDef(
        vizier_id=None,
        bands=(MagSpec("Tmag", "e_Tmag", "TESS"),),
    ),
    "GALEX": CatalogDef(
        vizier_id="II/312/ais",
        bands=(
            MagSpec("FUV", "e_FUV", "GALEX_FUV"),
            MagSpec("NUV", "e_NUV", "GALEX_NUV"),
        ),
    ),
    "GLIMPSE": CatalogDef(
        vizier_id="II/293/glimpse",
        bands=(
            MagSpec("_3.6mag", "e_3.6mag", "SPITZER_IRAC_36"),
            MagSpec("_4.5mag", "e_4.5mag", "SPITZER_IRAC_45"),
        ),
        xmatch_id_cols=("2MASS", "_2MASS", "_2M", "Designation"),
    ),
}

# Standard catalogs: VizieR cone search + crossmatched ID match
_STANDARD_CATALOGS = ("2MASS", "Wise", "Pan-STARRS", "SDSS", "TYCHO2", "GLIMPSE")

# VizieR XMatch fallback config: catalog_name -> (vizier_cat, id_cols)
_XMATCH_CONFIG = {
    "2MASS": ("vizier:II/246/out", ("2MASS", "_2MASS", "_2M")),
    "Wise": ("vizier:II/328/allwise", ("AllWISE",)),
    "Pan-STARRS": ("vizier:II/349/ps1", ("objID",)),
    "SDSS": ("vizier:V/147/sdss12", ("objID",)),
    "TYCHO2": ("vizier:I/259/tyc2", None),
    "APASS": ("vizier:II/336/apass9", None),
}

# Catalogs only available via VizieR cone search (not XMatch)
_CONE_ONLY_FALLBACK = {
    "RAVE": ("III/283/madera", ("ObsID",)),
    "SkyMapper": ("II/379", ("ObjectId", "object_id")),
}

# ── Constants ────────────────────────────────────────────────────

_PLX_ZEROPOINT = 0.037  # Lindegren+21 parallax zero-point (mas)
_PLX_SYS_ERR = 0.02  # systematic floor (mas), added in quadrature


# ── Librarian ────────────────────────────────────────────────────


class Librarian:
    """Automated photometry and astrometry retrieval for isochrone fitting.

    Uses Gaia DR3 best_neighbour tables for catalog crossmatching,
    with VizieR XMatch fallback when Gaia TAP is unavailable.

    Parameters
    ----------
    ra, dec : float
        Right ascension and declination in degrees.
    gaia_id : int, optional
        Gaia DR3 source_id (skips cone search if provided).
    radius : float
        Search radius in arcseconds (default 3).
    ignore : sequence of str
        Catalog names to skip (e.g. ["SDSS", "APASS"]).
    verbose : bool
        Print retrieval summary.
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        *,
        gaia_id: int | None = None,
        radius: float = 3.0,
        ignore: Sequence[str] = (),
        verbose: bool = True,
    ):
        self.ra = ra
        self.dec = dec
        self._search_radius = radius * u.arcsec
        self._coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        self._ignore = set(ignore)

        self._magnitudes: dict[str, tuple[float, float]] = {}
        self._gaia_id: int | None = gaia_id
        self._ids: dict[str, str | int | None] = {}
        self._parallax: float | None = None
        self._parallax_e: float | None = None
        self._teff: float | None = None
        self._teff_e: float | None = None
        self._radius_flame: float | None = None
        self._radius_flame_e: float | None = None
        self._luminosity: float | None = None
        self._luminosity_e: float | None = None
        self._mass: float | None = None
        self._mass_e: float | None = None
        self._age: float | None = None
        self._age_e: float | None = None
        self._distance: float | None = None
        self._distance_e: float | None = None
        self._spectroscopic_params: dict | None = None
        self._tic_id: int | None = None
        self._kic_id: int | None = None

        self._resolve_gaia_id()
        self._query_gaia_params()
        self._query_crossmatches()
        self._query_bailer_jones()
        self._fetch_photometry()
        self._query_spectroscopic_priors()

        if verbose:
            self._print_summary()

    # ── Gaia ID resolution ───────────────────────────────────────

    def _resolve_gaia_id(self):
        """Find Gaia DR3 source_id via VizieR cone search if not provided."""
        if self._gaia_id is not None:
            return
        try:
            cats = Vizier(columns=["Source", "+_r"]).query_region(
                self._coord, radius=self._search_radius, catalog="I/355/gaiadr3"
            )
            if not cats or len(cats[0]) == 0:
                logger.warning("No Gaia DR3 source within search radius")
                return
            cats[0].sort("_r")
            self._gaia_id = int(cats[0]["Source"][0])
            logger.info("Resolved Gaia DR3 ID: %d", self._gaia_id)
        except Exception as e:
            logger.warning("Gaia ID resolution failed: %s", e)

    # ── Gaia stellar parameters ──────────────────────────────────

    def _query_gaia_params(self):
        """Query Gaia DR3 for parallax, Teff, photometry, and FLAME params."""
        if self._gaia_id is None:
            return

        try:
            main_cats = Vizier.query_constraints(
                catalog="I/355/gaiadr3", Source=str(self._gaia_id)
            )
        except Exception as e:
            logger.warning("Gaia DR3 query failed: %s", e)
            return

        if not main_cats or len(main_cats[0]) == 0:
            logger.warning("Gaia DR3 source %d not found", self._gaia_id)
            return
        main = main_cats[0][0]

        # FLAME astrophysical parameters table
        ap = None
        try:
            ap_cats = Vizier.query_constraints(
                catalog="I/355/paramp", Source=str(self._gaia_id)
            )
            if ap_cats and len(ap_cats[0]) > 0:
                ap = ap_cats[0][0]
        except Exception:
            pass

        # Parallax (Lindegren+21 correction)
        plx = _col(main, "Plx")
        plx_e = _col(main, "e_Plx")
        if plx is not None:
            self._parallax = float(plx) + _PLX_ZEROPOINT
            if plx_e is not None:
                self._parallax_e = np.sqrt(float(plx_e) ** 2 + _PLX_SYS_ERR**2)

        # GSP-Phot Teff
        teff = _col(main, "Teff")
        if teff is not None:
            self._teff = float(teff)
            b_teff = _col(main, "b_Teff")
            B_teff = _col(main, "B_Teff")
            if b_teff is not None and B_teff is not None:
                self._teff_e = max(
                    abs(self._teff - float(b_teff)),
                    abs(float(B_teff) - self._teff),
                )
            else:
                self._teff_e = 100.0

        # FLAME radius — 5x error inflation (ARIADNE convention to avoid
        # over-constraining from crude FLAME estimates)
        rad = _col(ap, "Rad-Flame") if ap is not None else None
        if rad is not None:
            self._radius_flame = float(rad)
            lo = _col(ap, "b_Rad-Flame")
            hi = _col(ap, "B_Rad-Flame")
            if lo is not None and hi is not None:
                self._radius_flame_e = 5 * max(
                    abs(float(rad) - float(lo)), abs(float(hi) - float(rad))
                )

        # FLAME luminosity
        lum = _col(ap, "Lum-Flame") if ap is not None else None
        if lum is not None:
            self._luminosity = float(lum)
            lo = _col(ap, "b_Lum-Flame")
            hi = _col(ap, "B_Lum-Flame")
            if lo is not None and hi is not None:
                self._luminosity_e = max(
                    abs(float(lum) - float(lo)), abs(float(hi) - float(lum))
                )

        # FLAME mass
        mass_v = _col(ap, "Mass-Flame") if ap is not None else None
        if mass_v is not None:
            self._mass = float(mass_v)
            lo = _col(ap, "b_Mass-Flame")
            hi = _col(ap, "B_Mass-Flame")
            if lo is not None and hi is not None:
                self._mass_e = max(
                    abs(float(mass_v) - float(lo)), abs(float(hi) - float(mass_v))
                )

        # FLAME age
        age_v = _col(ap, "Age-Flame") if ap is not None else None
        if age_v is not None:
            self._age = float(age_v)
            lo = _col(ap, "b_Age-Flame")
            hi = _col(ap, "B_Age-Flame")
            if lo is not None and hi is not None:
                self._age_e = max(
                    abs(float(age_v) - float(lo)), abs(float(hi) - float(age_v))
                )

        # Gaia photometry
        for band in _CATALOGS["Gaia"].bands:
            mag_v = _col(main, band.col)
            err_v = _col(main, band.err_col)
            if _qc_mag(mag_v, err_v):
                self._add_mag(
                    band.pyphot,
                    float(mag_v),
                    float(err_v) if err_v is not None else 0.01,
                )

    # ── Gaia TAP crossmatch ──────────────────────────────────────

    def _query_crossmatches(self):
        """Query Gaia DR3 best_neighbour tables for external catalog IDs."""
        if self._gaia_id is None:
            return

        self._ids["Gaia"] = self._gaia_id
        tap_down = False
        xmatch_needed = []

        for cat_name, cat_def in _CATALOGS.items():
            if cat_name == "Gaia":
                continue
            if cat_name in self._ignore:
                self._ids[cat_name] = None
                continue
            if cat_def.xmatch_table is None:
                continue

            if tap_down:
                xmatch_needed.append(cat_name)
                continue

            query = (
                f"SELECT xmatch.original_ext_source_id "
                f"FROM gaiadr3.{cat_def.xmatch_table}_best_neighbour AS xmatch "
                f"WHERE xmatch.source_id = {self._gaia_id}"
            )

            try:
                job = Gaia.launch_job_async(query)
                result = job.get_results()
                if len(result) > 0:
                    self._ids[cat_name] = result[0][0]
                else:
                    self._ids[cat_name] = None
                    logger.info("No %s crossmatch for Gaia %d", cat_name, self._gaia_id)
            except Exception as e:
                err_str = str(e).lower()
                is_service_error = any(
                    s in err_str
                    for s in (
                        "service unavailable",
                        "bad gateway",
                        "503",
                        "502",
                        "500",
                        "connection",
                        "timeout",
                        "unavailable",
                    )
                )
                if is_service_error:
                    tap_down = True
                    xmatch_needed.append(cat_name)
                self._ids[cat_name] = None
                logger.warning("Crossmatch failed for %s: %s", cat_name, e)

        if xmatch_needed:
            logger.warning(
                "Gaia TAP down, falling back to VizieR XMatch for: %s", xmatch_needed
            )
            self._xmatch_fallback(xmatch_needed)

    def _xmatch_fallback(self, catalogs_needed):
        """Recover catalog IDs via VizieR XMatch when Gaia TAP is down."""
        region = CircleSkyRegion(self._coord, radius=self._search_radius)

        for cat_name in catalogs_needed:
            if cat_name in self._ignore:
                continue

            # Cone-only fallback (RAVE, SkyMapper — not on XMatch server)
            if cat_name in _CONE_ONLY_FALLBACK:
                viz_cat, id_cols = _CONE_ONLY_FALLBACK[cat_name]
                try:
                    result = Vizier.query_region(
                        self._coord, radius=self._search_radius, catalog=viz_cat
                    )
                    if not result or len(result[0]) == 0:
                        self._ids[cat_name] = None
                        continue
                    result[0].sort("_r")
                    row = result[0][0]
                    found = next(
                        (_col(row, c) for c in id_cols if c in result[0].colnames),
                        None,
                    )
                    self._ids[cat_name] = found
                except Exception as e:
                    logger.warning("Cone fallback failed for %s: %s", cat_name, e)
                    self._ids[cat_name] = None
                continue

            if cat_name not in _XMATCH_CONFIG:
                continue

            vizier_cat, id_cols = _XMATCH_CONFIG[cat_name]

            try:
                xm = XMatch.query(
                    cat1="vizier:I/355/gaiadr3",
                    cat2=vizier_cat,
                    max_distance=self._search_radius,
                    area=region,
                )
                if xm is None or len(xm) == 0:
                    self._ids[cat_name] = None
                    continue

                xm.sort("angDist")
                row = xm[0]
                if "Source" in xm.colnames:
                    src_mask = xm["Source"] == self._gaia_id
                    if src_mask.sum() > 0:
                        row = xm[src_mask][0]

                # Pattern B: APASS — extract mags directly from XMatch row
                if cat_name == "APASS":
                    self._ids["APASS"] = "xmatch_done"
                    for band in _CATALOGS["APASS"].bands:
                        if band.pyphot in self._magnitudes:
                            continue
                        mag_v = _col(row, band.col)
                        err_v = _col(row, band.err_col)
                        if _qc_mag(mag_v, err_v):
                            err_f = float(err_v) if err_v is not None else 0.02
                            self._add_mag(band.pyphot, float(mag_v), err_f)
                    continue

                # Tycho-2: composite TYC1-TYC2-TYC3 ID
                if cat_name == "TYCHO2":
                    try:
                        tyc_id = (
                            f"{int(row['TYC1'])}-{int(row['TYC2'])}-{int(row['TYC3'])}"
                        )
                        self._ids["TYCHO2"] = tyc_id
                    except (KeyError, ValueError):
                        self._ids["TYCHO2"] = None
                    continue

                # Pattern A: extract ID from first matching column
                if id_cols is not None:
                    found = next(
                        (_col(row, c) for c in id_cols if c in xm.colnames),
                        None,
                    )
                    self._ids[cat_name] = found
                else:
                    self._ids[cat_name] = None

            except Exception as e:
                logger.warning("XMatch fallback failed for %s: %s", cat_name, e)
                self._ids[cat_name] = None

    # ── Bailer-Jones distance ────────────────────────────────────

    def _query_bailer_jones(self):
        """Query Bailer-Jones EDR3 geometric distance."""
        if self._gaia_id is None:
            return
        try:
            res = Vizier.query_constraints(
                catalog="I/352/gedr3dis", Source=str(self._gaia_id)
            )
            if not res or len(res[0]) == 0:
                return
            row = res[0][0]
            dist = float(row["rgeo"])
            lo = dist - float(row["b_rgeo"])
            hi = float(row["B_rgeo"]) - dist
            self._distance = dist
            self._distance_e = max(lo, hi)
        except Exception as e:
            logger.warning("Bailer-Jones query failed: %s", e)

    # ── Photometry retrieval ─────────────────────────────────────

    def _fetch_photometry(self):
        """Dispatch photometry retrieval for all catalogs."""
        if self._gaia_id is None:
            return

        # Bulk VizieR cone search for standard catalogs
        viz_ids = [
            _CATALOGS[name].vizier_id
            for name in _STANDARD_CATALOGS
            if name not in self._ignore
            and _CATALOGS[name].vizier_id is not None
            and self._ids.get(name) is not None
        ]
        cats = {}
        if viz_ids:
            try:
                result = Vizier.query_region(
                    self._coord, radius=self._search_radius, catalog=viz_ids
                )
                if result:
                    cats = result
            except Exception as e:
                logger.warning("Bulk VizieR query failed: %s", e)

        for cat_name in _STANDARD_CATALOGS:
            if cat_name in self._ignore:
                continue
            ext_id = self._ids.get(cat_name)
            if ext_id is None:
                continue
            self._fetch_standard(cat_name, _CATALOGS[cat_name], cats, ext_id)

        if "TESS" not in self._ignore:
            self._fetch_tess()
        if "APASS" not in self._ignore:
            self._fetch_apass()
        if "SkyMapper" not in self._ignore:
            self._fetch_skymapper()
        if "GALEX" not in self._ignore:
            self._fetch_galex()
        if "MERMILLIOD" not in self._ignore:
            self._fetch_mermilliod()
        if "STROMGREN" not in self._ignore:
            self._fetch_stromgren()

    def _fetch_standard(self, cat_name, cat_def, cats, ext_id):
        """Fetch photometry from VizieR results, filtered by crossmatched ID."""
        if cat_def.vizier_id is None:
            return

        try:
            cat = cats[cat_def.vizier_id]
            cat.sort("_r")
        except (TypeError, KeyError):
            logger.info("No VizieR data for %s", cat_name)
            return

        # Filter by crossmatched ID
        if cat_name == "TYCHO2":
            try:
                tyc1, tyc2, tyc3 = str(ext_id).split("-")
                mask = (
                    (cat["TYC1"] == int(tyc1))
                    & (cat["TYC2"] == int(tyc2))
                    & (cat["TYC3"] == int(tyc3))
                )
            except (ValueError, KeyError):
                return
        elif cat_name == "GLIMPSE":
            tmass_id = self._ids.get("2MASS")
            if tmass_id is None:
                return
            id_cols = cat_def.xmatch_id_cols or ()
            col_name = next((c for c in id_cols if c in cat.colnames), None)
            if col_name is None:
                return
            mask = cat[col_name] == tmass_id
        else:
            id_cols = cat_def.xmatch_id_cols or ()
            col_name = next((c for c in id_cols if c in cat.colnames), None)
            if col_name is None:
                return
            mask = cat[col_name] == ext_id

        matched = cat[mask]
        if len(matched) == 0:
            return
        row = matched[0]

        # Catalog-level QC
        if cat_def.qc is not None and not cat_def.qc(row):
            logger.info("%s failed catalog QC", cat_name)
            return

        # WISE extended source check
        if cat_name == "Wise" and not _qc_wise_extended(row):
            logger.info("WISE source is extended, skipping")
            return

        for band in cat_def.bands:
            if band.pyphot in self._magnitudes:
                continue

            # Per-band QC
            if band.pyphot in _2MASS_BAND_IDX:
                if not _qc_2mass_band(row, _2MASS_BAND_IDX[band.pyphot]):
                    continue
            if band.pyphot in _WISE_BAND_IDX:
                if not _qc_wise_band(row, _WISE_BAND_IDX[band.pyphot]):
                    continue

            mag_v = _col(row, band.col)
            err_v = _col(row, band.err_col)
            if not _qc_mag(mag_v, err_v):
                continue

            err_f = float(err_v) if err_v is not None else 0.02
            self._add_mag(band.pyphot, float(mag_v), err_f)

    def _fetch_tess(self):
        """Query TESS TIC via MAST, crossmatched on Gaia ID."""
        try:
            result = Catalogs.query_region(
                self._coord, radius=self._search_radius, catalog="TIC"
            )
        except Exception as e:
            logger.warning("TESS TIC query failed: %s", e)
            return

        if result is None or len(result) == 0:
            return

        result.sort("dstArcSec")
        gaia_mask = result["GAIA"] == str(self._gaia_id)
        matched = result[gaia_mask]
        if len(matched) == 0:
            return

        row = matched[0]
        if _col(row, "objType") != "STAR":
            return

        self._tic_id = int(row["ID"])
        kic = _col(row, "KIC")
        if kic is not None:
            self._kic_id = int(kic)

        band = _CATALOGS["TESS"].bands[0]
        if band.pyphot in self._magnitudes:
            return

        mag_v = _col(row, band.col)
        err_v = _col(row, band.err_col)
        if _qc_mag(mag_v, err_v):
            err_f = float(err_v) if err_v is not None else 0.01
            self._add_mag(band.pyphot, float(mag_v), err_f)

    def _fetch_apass(self):
        """Query APASS via Gaia TAP external.apassdr9 table."""
        apass_id = self._ids.get("APASS")
        if apass_id is None or apass_id == "xmatch_done":
            return

        query = (
            "SELECT apass.vmag, apass.e_vmag, "
            "apass.bmag, apass.e_bmag, "
            "apass.g_mag, apass.e_g_mag, "
            "apass.r_mag, apass.e_r_mag, "
            "apass.i_mag, apass.e_i_mag "
            "FROM external.apassdr9 AS apass "
            "INNER JOIN gaiadr3.apassdr9_best_neighbour AS xmatch "
            "ON apass.recno = xmatch.original_ext_source_id "
            f"WHERE xmatch.source_id = {self._gaia_id}"
        )

        try:
            job = Gaia.launch_job_async(query)
            result = job.get_results()
            if len(result) == 0:
                return

            row = result[0]
            for band in _CATALOGS["APASS"].bands:
                if band.pyphot in self._magnitudes:
                    continue
                mag_v = _col(row, band.col)
                err_v = _col(row, band.err_col)
                if _qc_mag(mag_v, err_v):
                    err_f = float(err_v) if err_v is not None else 0.02
                    self._add_mag(band.pyphot, float(mag_v), err_f)
        except Exception as e:
            logger.warning("APASS query via Gaia TAP failed: %s", e)

    def _fetch_skymapper(self):
        """Query SkyMapper DR2 via SkyMapper TAP service."""
        sm_id = self._ids.get("SkyMapper")
        if sm_id is None:
            return

        from astroquery.utils.tap.core import TapPlus

        try:
            tap = TapPlus(url="https://api.skymapper.nci.org.au/public/tap/")
            query = (
                "SELECT object_id, u_psf, e_u_psf, v_psf, e_v_psf, "
                "g_psf, e_g_psf, r_psf, e_r_psf, i_psf, e_i_psf, "
                "z_psf, e_z_psf, flags "
                f"FROM dr2.master WHERE object_id = {sm_id}"
            )
            job = tap.launch_job_async(query)
            result = job.get_results()
            if len(result) == 0:
                return

            row = result[0]
            if not _qc_skymapper(row):
                return

            for band in _CATALOGS["SkyMapper"].bands:
                if band.pyphot in self._magnitudes:
                    continue
                mag_v = _col(row, band.col)
                err_v = _col(row, band.err_col)
                if _qc_mag(mag_v, err_v):
                    err_f = float(err_v) if err_v is not None else 0.02
                    self._add_mag(band.pyphot, float(mag_v), err_f)
        except Exception as e:
            logger.warning("SkyMapper TAP query failed: %s", e)

    def _fetch_galex(self):
        """Query GALEX via VizieR XMatch with Gaia DR3."""
        region = CircleSkyRegion(self._coord, radius=self._search_radius)
        try:
            xm = XMatch.query(
                cat1="vizier:I/355/gaiadr3",
                cat2="vizier:II/312/ais",
                max_distance=self._search_radius,
                area=region,
            )
            if xm is None or len(xm) == 0:
                return

            xm.sort("angDist")
            src_mask = xm["Source"] == self._gaia_id
            if src_mask.sum() == 0:
                return
            row = xm[src_mask][0]

            for band in _CATALOGS["GALEX"].bands:
                if band.pyphot in self._magnitudes:
                    continue
                if not _qc_galex_band(row, band.pyphot):
                    continue
                mag_v = _col(row, band.col)
                err_v = _col(row, band.err_col)
                if _qc_mag(mag_v, err_v):
                    err_f = float(err_v) if err_v is not None else 0.02
                    self._add_mag(band.pyphot, float(mag_v), err_f)
        except Exception as e:
            logger.warning("GALEX XMatch query failed: %s", e)

    def _fetch_mermilliod(self):
        """Query Mermilliod Johnson UBV via VizieR XMatch.

        Color algebra: V direct, B = (B-V) + V, U = (U-B) + B.
        """
        region = CircleSkyRegion(self._coord, radius=5 * u.arcmin)
        try:
            xm = XMatch.query(
                cat1="vizier:I/355/gaiadr3",
                cat2="vizier:II/168/ubvmeans",
                max_distance=3 * u.arcmin,
                area=region,
            )
            if xm is None or len(xm) == 0:
                return

            xm.sort("angDist")
            src_mask = xm["Source"] == self._gaia_id
            if src_mask.sum() == 0:
                return
            row = xm[src_mask][0]

            v = _col(row, "Vmag")
            v_e = _col(row, "e_Vmag")
            if not _qc_mag(v, v_e):
                return

            v, v_e = float(v), float(v_e) if v_e is not None else 0.0

            if v_e > 0 and "GROUND_JOHNSON_V" not in self._magnitudes:
                self._add_mag("GROUND_JOHNSON_V", v, v_e)

            bv = _col(row, "B-V")
            bv_e = _col(row, "e_B-V")
            if _qc_mag(bv, bv_e):
                bv = float(bv)
                bv_e = float(bv_e) if bv_e is not None else 0.0
                b = bv + v
                b_e = np.sqrt(v_e**2 + bv_e**2)
                if b_e > 0 and "GROUND_JOHNSON_B" not in self._magnitudes:
                    self._add_mag("GROUND_JOHNSON_B", b, b_e)

                ub = _col(row, "U-B")
                ub_e = _col(row, "e_U-B")
                if _qc_mag(ub, ub_e):
                    ub = float(ub)
                    ub_e = float(ub_e) if ub_e is not None else 0.0
                    u_mag = ub + b
                    u_e = np.sqrt(b_e**2 + ub_e**2)
                    if u_e > 0 and "GROUND_JOHNSON_U" not in self._magnitudes:
                        self._add_mag("GROUND_JOHNSON_U", u_mag, u_e)

        except Exception as e:
            logger.warning("Mermilliod XMatch query failed: %s", e)

    def _fetch_stromgren(self):
        """Query Stromgren from Paunzen+15 and Hauck+98 via VizieR XMatch.

        Index algebra: b=(b-y)+y, v=m1+2(b-y)+y, u=c1+2m1+3(b-y)+y.
        """
        xm_configs = [
            ("vizier:J/A+A/580/A23/catalog", self._search_radius),
            ("vizier:II/215/catalog", self._search_radius),
        ]
        region = CircleSkyRegion(self._coord, radius=self._search_radius)

        for cat2, max_dist in xm_configs:
            try:
                xm = XMatch.query(
                    cat1="vizier:I/355/gaiadr3",
                    cat2=cat2,
                    max_distance=max_dist,
                    area=region,
                )
                if xm is None or len(xm) == 0:
                    continue

                xm.sort("angDist")
                src_mask = xm["Source"] == self._gaia_id
                if src_mask.sum() == 0:
                    continue
                row = xm[src_mask][0]

                y = _col(row, "Vmag")
                y_e = _col(row, "e_Vmag")
                if not _qc_mag(y, y_e):
                    continue

                y = float(y)
                y_e = float(y_e) if y_e is not None else 0.0

                by = _col(row, "b-y")
                by_e = _col(row, "e_b-y")
                m1 = _col(row, "m1")
                m1_e = _col(row, "e_m1")
                c1 = _col(row, "c1")
                c1_e = _col(row, "e_c1")

                if not all(
                    _qc_mag(x, xe) for x, xe in [(by, by_e), (m1, m1_e), (c1, c1_e)]
                ):
                    continue

                by, by_e = float(by), float(by_e) if by_e is not None else 0.0
                m1, m1_e = float(m1), float(m1_e) if m1_e is not None else 0.0
                c1, c1_e = float(c1), float(c1_e) if c1_e is not None else 0.0

                b_mag = by + y
                v_mag = m1 + 2 * by + y
                u_mag = c1 + 2 * m1 + 3 * by + y

                b_e = np.sqrt(by_e**2 + y_e**2)
                v_e = np.sqrt(m1_e**2 + 4 * by_e**2 + y_e**2)
                u_e = np.sqrt(c1_e**2 + 4 * m1_e**2 + 9 * by_e**2 + y_e**2)

                for name, m, e in [
                    ("STROMGREN_u", u_mag, u_e),
                    ("STROMGREN_v", v_mag, v_e),
                    ("STROMGREN_b", b_mag, b_e),
                    ("STROMGREN_y", y, y_e),
                ]:
                    if name not in self._magnitudes:
                        self._add_mag(name, m, e)

                break  # found in one catalog, skip the other

            except Exception as e:
                logger.warning("Stromgren XMatch failed for %s: %s", cat2, e)

    # ── Spectroscopic parameters ─────────────────────────────────

    def _query_spectroscopic_priors(self):
        """Try spectroscopic catalogs in priority order; stop at first match.

        Priority: APOGEE DR17 > GALAH DR3 > RAVE DR6 > LAMOST DR8 > PASTEL.
        """
        for query_fn, source_name in [
            (self._query_apogee, "APOGEE_DR17"),
            (self._query_galah, "GALAH_DR3"),
            (self._query_rave, "RAVE_DR6"),
            (self._query_lamost, "LAMOST_DR8"),
            (self._query_pastel, "PASTEL"),
        ]:
            result = query_fn()
            if result is not None:
                result["source"] = source_name
                self._spectroscopic_params = result
                logger.info("Spectroscopic params from %s", source_name)
                return

    def _query_apogee(self) -> dict | None:
        """Query APOGEE DR17 via VizieR (III/286).

        Crossmatches via 2MASS ID. Rejects entries with ASPCAPFLAG != 0.
        Uses [M/H] as proxy for [Fe/H] (close enough for priors).
        """
        tmass_id = self._ids.get("2MASS")
        if tmass_id is None:
            return None

        try:
            cat = Vizier.query_constraints(
                catalog="III/286/allstars", **{"2MASS": str(tmass_id)}
            )
            if not cat or len(cat[0]) == 0:
                return None

            row = cat[0][0]

            # Reject flagged entries
            flag = _col(row, "ASPCAPFLAG")
            if flag is not None and int(flag) != 0:
                return None

            teff = _col(row, "Teff")
            teff_e = _col(row, "e_Teff")
            logg = _col(row, "logg")
            logg_e = _col(row, "e_logg")
            mh = _col(row, "__M_H_")
            mh_e = _col(row, "e__M_H_")

            if any(v is None for v in (teff, logg, mh)):
                return None

            return {
                "teff": float(teff),
                "teff_err": float(teff_e) if teff_e is not None else 100.0,
                "logg": float(logg),
                "logg_err": float(logg_e) if logg_e is not None else 0.2,
                "feh": float(mh),
                "feh_err": float(mh_e) if mh_e is not None else 0.1,
            }
        except Exception as e:
            logger.warning("APOGEE DR17 query failed: %s", e)
            return None

    def _query_galah(self) -> dict | None:
        """Query GALAH DR3 via VizieR (J/MNRAS/506/150).

        Pre-matched to Gaia DR3 source_id. Rejects entries with quality flags.
        """
        if self._gaia_id is None:
            return None

        try:
            cat = Vizier.query_constraints(
                catalog="J/MNRAS/506/150/catalog",
                Source=str(self._gaia_id),
            )
            if not cat or len(cat[0]) == 0:
                return None

            row = cat[0][0]

            # Quality flags: reject if flagged
            flag_sp = _col(row, "flag_sp")
            flag_fe_h = _col(row, "flag_fe_h")
            if flag_sp is not None and int(flag_sp) != 0:
                return None
            if flag_fe_h is not None and int(flag_fe_h) != 0:
                return None

            teff = _col(row, "Teff")
            teff_e = _col(row, "e_Teff")
            logg = _col(row, "logg")
            logg_e = _col(row, "e_logg")
            feh = _col(row, "fe_h")
            feh_e = _col(row, "e_fe_h")

            if any(v is None for v in (teff, logg, feh)):
                return None

            return {
                "teff": float(teff),
                "teff_err": float(teff_e) if teff_e is not None else 100.0,
                "logg": float(logg),
                "logg_err": float(logg_e) if logg_e is not None else 0.2,
                "feh": float(feh),
                "feh_err": float(feh_e) if feh_e is not None else 0.1,
            }
        except Exception as e:
            logger.warning("GALAH DR3 query failed: %s", e)
            return None

    def _query_rave(self) -> dict | None:
        """Query RAVE DR6 for spectroscopic Teff, logg, [Fe/H]."""
        rave_id = self._ids.get("RAVE")
        if rave_id is None:
            return None

        try:
            cat = Vizier.query_constraints(
                catalog="III/283/madera", ObsID=str(rave_id)
            )
            if not cat or len(cat[0]) == 0:
                return None

            row = cat[0][0]
            qual = _col(row, "Qual")
            if qual is not None and int(qual) == 1:
                return None

            return {
                "teff": float(row["TeffmC"]),
                "teff_err": float(row["e_Teffm"]),
                "logg": float(row["loggmC"]),
                "logg_err": float(row["e_loggm"]),
                "feh": float(row["[m/H]mC"]),
                "feh_err": float(row["e_[m/H]m"]),
            }
        except Exception as e:
            logger.warning("RAVE DR6 query failed: %s", e)
            return None

    def _query_lamost(self) -> dict | None:
        """Query LAMOST DR8 stellar parameters via VizieR (V/164).

        Crossmatches via positional cone search since LAMOST has no
        pre-matched Gaia source_id. Requires SNR_g > 30.
        """
        try:
            cats = Vizier.query_region(
                self._coord, radius=self._search_radius, catalog="V/164/dr8"
            )
            if not cats or len(cats[0]) == 0:
                return None

            cats[0].sort("_r")
            row = cats[0][0]

            # SNR quality cut
            snrg = _col(row, "snrg")
            if snrg is not None and float(snrg) < 30:
                return None

            teff = _col(row, "Teff")
            teff_e = _col(row, "e_Teff")
            logg = _col(row, "logg")
            logg_e = _col(row, "e_logg")
            feh = _col(row, "__Fe_H_")
            feh_e = _col(row, "e__Fe_H_")

            if any(v is None for v in (teff, logg, feh)):
                return None

            return {
                "teff": float(teff),
                "teff_err": float(teff_e) if teff_e is not None else 100.0,
                "logg": float(logg),
                "logg_err": float(logg_e) if logg_e is not None else 0.2,
                "feh": float(feh),
                "feh_err": float(feh_e) if feh_e is not None else 0.1,
            }
        except Exception as e:
            logger.warning("LAMOST DR8 query failed: %s", e)
            return None

    def _query_pastel(self) -> dict | None:
        """Query PASTEL catalog via VizieR (B/pastel).

        Literature compilation of spectroscopic parameters. Positional match.
        Uses conservative default errors since many entries lack formal errors.
        """
        try:
            cats = Vizier.query_region(
                self._coord, radius=self._search_radius, catalog="B/pastel/pastel"
            )
            if not cats or len(cats[0]) == 0:
                return None

            cats[0].sort("_r")
            row = cats[0][0]

            teff = _col(row, "Teff")
            logg = _col(row, "logg")
            feh = _col(row, "__Fe_H_")

            if any(v is None for v in (teff, logg, feh)):
                return None

            teff_e = _col(row, "e_Teff")
            logg_e = _col(row, "e_logg")
            feh_e = _col(row, "e__Fe_H_")

            return {
                "teff": float(teff),
                "teff_err": float(teff_e) if teff_e is not None else 150.0,
                "logg": float(logg),
                "logg_err": float(logg_e) if logg_e is not None else 0.3,
                "feh": float(feh),
                "feh_err": float(feh_e) if feh_e is not None else 0.15,
            }
        except Exception as e:
            logger.warning("PASTEL query failed: %s", e)
            return None

    # ── Helpers ──────────────────────────────────────────────────

    def _add_mag(self, pyphot_name, mag, err):
        """Add magnitude. First-write-wins (catalog priority by query order)."""
        if pyphot_name not in self._magnitudes:
            self._magnitudes[pyphot_name] = (mag, err)

    def _print_summary(self):
        tab = "\t\t\t\t"
        print(f"{tab}--- Retrieved Photometry ---")
        print(f"{tab}{'Filter':20s}  {'Magnitude':>10s}  {'Error':>8s}")
        print(f"{tab}{'-' * 42}")
        for band, (mag, err) in sorted(self._magnitudes.items()):
            print(f"{tab}{band:20s}  {mag:10.4f}  {err:8.4f}")
        if self._parallax is not None:
            print(
                f"{tab}Parallax : {self._parallax:.3f} "
                f"+/- {self._parallax_e:.3f} mas"
            )
        if self._teff is not None:
            t_str = f"{self._teff:.0f}"
            if self._teff_e:
                t_str += f" +/- {self._teff_e:.0f}"
            print(f"{tab}Gaia Teff : {t_str} K")
        if self._distance is not None:
            print(
                f"{tab}BJ Distance : {self._distance:.1f} "
                f"+/- {self._distance_e:.1f} pc"
            )
        if self._spectroscopic_params:
            sp = self._spectroscopic_params
            src = sp.get("source", "unknown")
            print(f"{tab}Spectroscopic source : {src}")
            print(f"{tab}{src} Teff : {sp['teff']:.0f} +/- {sp['teff_err']:.0f} K")
            print(f"{tab}{src} logg : {sp['logg']:.2f} +/- {sp['logg_err']:.2f}")
            print(f"{tab}{src} [Fe/H] : {sp['feh']:.2f} +/- {sp['feh_err']:.2f}")
        if self._gaia_id:
            print(f"{tab}Gaia DR3 ID : {self._gaia_id}")
        print()

    # ── Properties ───────────────────────────────────────────────

    @property
    def parallax(self) -> tuple[float, float] | None:
        """(parallax_mas, error_mas) with Lindegren+21 correction."""
        if self._parallax is None:
            return None
        return (self._parallax, self._parallax_e)

    @property
    def teff(self) -> tuple[float, float] | None:
        """(Teff_K, error_K) from Gaia GSP-Phot."""
        if self._teff is None:
            return None
        return (self._teff, self._teff_e or 100.0)

    @property
    def radius(self) -> tuple[float, float] | None:
        """(R_Rsun, error_Rsun) from FLAME (5x inflated error)."""
        if self._radius_flame is None:
            return None
        return (self._radius_flame, self._radius_flame_e)

    @property
    def luminosity(self) -> tuple[float, float] | None:
        """(L_Lsun, error_Lsun) from FLAME."""
        if self._luminosity is None:
            return None
        return (self._luminosity, self._luminosity_e)

    @property
    def mass(self) -> tuple[float, float] | None:
        """(M_Msun, error_Msun) from FLAME."""
        if self._mass is None:
            return None
        return (self._mass, self._mass_e)

    @property
    def age(self) -> tuple[float, float] | None:
        """(age_Gyr, error_Gyr) from FLAME."""
        if self._age is None:
            return None
        return (self._age, self._age_e)

    @property
    def distance(self) -> tuple[float, float] | None:
        """(distance_pc, error_pc) from Bailer-Jones EDR3."""
        if self._distance is None:
            return None
        return (self._distance, self._distance_e)

    @property
    def magnitudes(self) -> dict[str, tuple[float, float]]:
        """{'pyphot_filter_name': (mag, error)}."""
        return dict(self._magnitudes)

    @property
    def gaia_id(self) -> int | None:
        return self._gaia_id

    @property
    def tic_id(self) -> int | None:
        return self._tic_id

    @property
    def spectroscopic_params(self) -> dict | None:
        """Spectroscopic params dict with source tag, or None.

        Keys: teff, teff_err, logg, logg_err, feh, feh_err, source.
        """
        return dict(self._spectroscopic_params) if self._spectroscopic_params else None

    @property
    def rave_params(self) -> dict | None:
        """RAVE DR6 spectroscopic params, or None. Backward-compatible."""
        if (
            self._spectroscopic_params is not None
            and self._spectroscopic_params.get("source") == "RAVE_DR6"
        ):
            return dict(self._spectroscopic_params)
        return None

    def to_star(self, starname: str, **kwargs):
        """Convert to a Star object for isochrone fitting.

        Passes through all retrieved parameters. User kwargs override.
        """
        from lachesis.star import Star

        star_kwargs: dict = {}

        if self._parallax is not None:
            star_kwargs["parallax"] = self._parallax
            star_kwargs["parallax_e"] = self._parallax_e
        if self._distance is not None:
            star_kwargs["distance"] = self._distance
            star_kwargs["distance_e"] = self._distance_e
        # GSP-Phot Teff, FLAME radius/luminosity are NOT passed to the Star.
        # They're Gaia pipeline estimates with unrealistically small formal
        # errors that would over-constrain the fit. The photometry + parallax
        # constrain Teff/L/R properly through the isochrone model.
        # Available via Librarian's .teff / .radius / .luminosity properties.
        if self._spectroscopic_params:
            star_kwargs.setdefault("logg", self._spectroscopic_params["logg"])
            star_kwargs.setdefault("logg_e", self._spectroscopic_params["logg_err"])
            star_kwargs.setdefault("feh", self._spectroscopic_params["feh"])
            star_kwargs.setdefault("feh_e", self._spectroscopic_params["feh_err"])

        star_kwargs["magnitudes"] = self.magnitudes
        star_kwargs.update(kwargs)

        return Star(starname, **star_kwargs)
