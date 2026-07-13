"""Tests for the Librarian, all network calls mocked."""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import numpy.ma as ma
import pytest
from astropy.table import Table

from lachesis.librarian import (
    MagSpec,
    CatalogDef,
    _CATALOGS,
    _col,
    _qc_mag,
    _qc_2mass_band,
    _qc_wise_band,
    _qc_wise_extended,
    _qc_sdss,
    _qc_ps1,
    _qc_galex_band,
    _qc_skymapper,
    Librarian,
)


# ── Test helpers ─────────────────────────────────────────────────


def _row(data):
    """Make a single-row astropy Table and return row[0]."""
    return Table(data)[0]


def _masked_row(data, masks=None):
    """Make a row with masked columns."""
    t = Table(data, masked=True)
    if masks:
        for col, m in masks.items():
            t[col].mask = m
    return t[0]


def _make_gaia_main_row(
    source=12345,
    plx=10.0,
    plx_e=0.1,
    teff=5800.0,
    b_teff=5700.0,
    B_teff=5900.0,
    gmag=8.5,
    e_gmag=0.003,
    bpmag=8.9,
    e_bpmag=0.003,
    rpmag=8.0,
    e_rpmag=0.003,
):
    return Table(
        {
            "Source": [source],
            "Plx": [plx],
            "e_Plx": [plx_e],
            "Teff": [teff],
            "b_Teff": [b_teff],
            "B_Teff": [B_teff],
            "Gmag": [gmag],
            "e_Gmag": [e_gmag],
            "BPmag": [bpmag],
            "e_BPmag": [e_bpmag],
            "RPmag": [rpmag],
            "e_RPmag": [e_rpmag],
        }
    )


def _make_flame_row(
    source=12345,
    rad=1.0,
    b_rad=0.9,
    B_rad=1.1,
    lum=1.0,
    b_lum=0.8,
    B_lum=1.2,
    mass=1.0,
    b_mass=0.9,
    B_mass=1.1,
    age=4.6,
    b_age=3.0,
    B_age=6.0,
):
    return Table(
        {
            "Source": [source],
            "Rad-Flame": [rad],
            "b_Rad-Flame": [b_rad],
            "B_Rad-Flame": [B_rad],
            "Lum-Flame": [lum],
            "b_Lum-Flame": [b_lum],
            "B_Lum-Flame": [B_lum],
            "Mass-Flame": [mass],
            "b_Mass-Flame": [b_mass],
            "B_Mass-Flame": [B_mass],
            "Age-Flame": [age],
            "b_Age-Flame": [b_age],
            "B_Age-Flame": [B_age],
        }
    )


def _make_bj_row(source=12345, rgeo=100.0, b_rgeo=95.0, B_rgeo=106.0):
    return Table(
        {"Source": [source], "rgeo": [rgeo], "b_rgeo": [b_rgeo], "B_rgeo": [B_rgeo]}
    )


def _make_tap_result(ext_id):
    """Fake Gaia TAP crossmatch result (1 row, 1 col)."""
    return Table({"original_ext_source_id": [ext_id]})


def _make_tap_job(result_table):
    """Fake async TAP job."""
    job = MagicMock()
    job.get_results.return_value = result_table
    return job


class _FakeTableList(dict):
    """Minimal TableList standin keyed by catalog ID."""

    pass


# ── Catalog registry tests ──────────────────────────────────────


class TestCatalogRegistry:
    def test_all_catalogs_have_bands(self):
        for name, cat in _CATALOGS.items():
            assert len(cat.bands) > 0, f"{name} has no bands"

    def test_all_pyphot_names_unique_across_catalogs(self):
        seen = {}
        for name, cat in _CATALOGS.items():
            for band in cat.bands:
                if band.pyphot in seen:
                    # Some overlap is OK (APASS SDSS_g vs SDSS SDSS_g)
                    pass
                seen[band.pyphot] = name

    def test_magspec_frozen(self):
        ms = MagSpec("Jmag", "e_Jmag", "2MASS_J")
        with pytest.raises(AttributeError):
            ms.col = "Hmag"

    def test_catalogdef_frozen(self):
        cd = CatalogDef(vizier_id="test", bands=())
        with pytest.raises(AttributeError):
            cd.vizier_id = "other"


# ── QC function tests ───────────────────────────────────────────


class TestQCMag:
    def test_valid(self):
        assert _qc_mag(10.5, 0.03) is True

    def test_none_mag(self):
        assert _qc_mag(None, 0.03) is False

    def test_nan_mag(self):
        assert _qc_mag(float("nan"), 0.03) is False

    def test_masked_mag(self):
        t = Table({"m": ma.array([1.0], mask=[True])})
        assert _qc_mag(t["m"][0], 0.03) is False

    def test_large_error(self):
        assert _qc_mag(10.0, 1.5) is False

    def test_none_error_ok(self):
        assert _qc_mag(10.0, None) is True

    def test_masked_error_ok(self):
        t = Table({"e": ma.array([1.0], mask=[True])}, masked=True)
        assert _qc_mag(10.0, t["e"][0]) is True

    def test_custom_max_err(self):
        assert _qc_mag(10.0, 0.5, max_err=0.3) is False
        assert _qc_mag(10.0, 0.5, max_err=1.0) is True


class TestQC2MASS:
    def test_good_J(self):
        row = _row({"Qflg": ["AAA"], "Cflg": ["000"]})
        assert _qc_2mass_band(row, 0) is True

    def test_bad_qflg_J(self):
        row = _row({"Qflg": ["UAA"], "Cflg": ["000"]})
        assert _qc_2mass_band(row, 0) is False

    def test_bad_cflg_H(self):
        row = _row({"Qflg": ["AAA"], "Cflg": ["010"]})
        assert _qc_2mass_band(row, 1) is False

    def test_D_quality_ok(self):
        row = _row({"Qflg": ["DAA"], "Cflg": ["000"]})
        assert _qc_2mass_band(row, 0) is True

    def test_missing_flags(self):
        row = _row({"other": [1]})
        assert _qc_2mass_band(row, 0) is False


class TestQCWISE:
    def test_good_W1(self):
        row = _row({"qph": ["AA"]})
        assert _qc_wise_band(row, 0) is True

    def test_bad_W2(self):
        row = _row({"qph": ["AU"]})
        assert _qc_wise_band(row, 1) is False

    def test_extended_source(self):
        row = _row({"ex": [1]})
        assert _qc_wise_extended(row) is False

    def test_point_source(self):
        row = _row({"ex": [0]})
        assert _qc_wise_extended(row) is True


class TestQCSDSS:
    def test_star_good_quality(self):
        row = _row({"class": [6], "Q": [3]})
        assert _qc_sdss(row) is True

    def test_not_star(self):
        row = _row({"class": [3], "Q": [3]})
        assert _qc_sdss(row) is False

    def test_bad_quality(self):
        row = _row({"class": [6], "Q": [1]})
        assert _qc_sdss(row) is False


class TestQCPS1:
    def test_good_star(self):
        # bit 2 set (detection), no galaxy bits, no artifact
        row = _row({"Qual": [4]})
        assert _qc_ps1(row) is True

    def test_galaxy(self):
        # bits 0 AND 1 set
        row = _row({"Qual": [3]})
        assert _qc_ps1(row) is False

    def test_artifact(self):
        # bit 7 set
        row = _row({"Qual": [128 + 4]})
        assert _qc_ps1(row) is False


class TestQCGALEX:
    def test_clean_fuv(self):
        row = _row({"Fexf": [0], "Fafl": [0]})
        assert _qc_galex_band(row, "GALEX_FUV") is True

    def test_artifact_fuv(self):
        row = _row({"Fexf": [1], "Fafl": [0]})
        assert _qc_galex_band(row, "GALEX_FUV") is False

    def test_clean_nuv(self):
        row = _row({"Nexf": [0], "Nafl": [0]})
        assert _qc_galex_band(row, "GALEX_NUV") is True

    def test_artifact_nuv(self):
        row = _row({"Nexf": [0], "Nafl": [1]})
        assert _qc_galex_band(row, "GALEX_NUV") is False


class TestQCSkyMapper:
    def test_good(self):
        row = _row({"flags": [0]})
        assert _qc_skymapper(row) is True

    def test_bad(self):
        row = _row({"flags": [1]})
        assert _qc_skymapper(row) is False

    def test_missing(self):
        row = _row({"other": [0]})
        assert _qc_skymapper(row) is False


# ── _col helper tests ───────────────────────────────────────────


class TestColHelper:
    def test_existing_col(self):
        row = _row({"x": [42.0]})
        assert _col(row, "x") == 42.0

    def test_missing_col(self):
        row = _row({"x": [42.0]})
        assert _col(row, "y") is None

    def test_masked_col(self):
        row = _masked_row({"x": [42.0]}, masks={"x": [True]})
        assert _col(row, "x") is None


# ── Gaia parameters tests ───────────────────────────────────────


def _patch_all_queries():
    """Return a dict of patch objects for all external queries."""
    return {
        "vizier_qr": patch("lachesis.librarian.Vizier.query_region", return_value=None),
        "vizier_qc": patch(
            "lachesis.librarian.Vizier.query_constraints", return_value=None
        ),
        "gaia_tap": patch(
            "lachesis.librarian.Gaia.launch_job_async", side_effect=Exception("no TAP")
        ),
        "mast": patch(
            "lachesis.librarian.Catalogs.query_region", return_value=None
        ),
        "xmatch": patch(
            "lachesis.librarian.XMatch.query", return_value=None
        ),
    }


class TestGaiaParams:
    def _make_lib(self, main_table, flame_table=None, bj_table=None):
        """Create a Librarian with mocked queries, only Gaia params active."""

        def fake_qc(catalog=None, **constraints):
            if catalog == "I/355/gaiadr3":
                return [main_table] if main_table is not None else None
            if catalog == "I/355/paramp":
                return [flame_table] if flame_table is not None else None
            if catalog == "I/352/gedr3dis":
                return [bj_table] if bj_table is not None else None
            if catalog == "III/283/madera":
                return None
            return None

        patches = _patch_all_queries()
        for p in patches.values():
            p.start()

        # Override the constraint query with our fake
        patches["vizier_qc"].stop()
        p_qc = patch(
            "lachesis.librarian.Vizier.query_constraints",
            side_effect=fake_qc,
        )
        p_qc.start()

        try:
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)
        finally:
            for p in patches.values():
                try:
                    p.stop()
                except RuntimeError:
                    pass
            p_qc.stop()

        return lib

    def test_parallax_lindegren_correction(self):
        main = _make_gaia_main_row(plx=10.0, plx_e=0.1)
        lib = self._make_lib(main)
        assert lib.parallax is not None
        plx, plx_e = lib.parallax
        assert abs(plx - 10.037) < 1e-6
        assert abs(plx_e - np.sqrt(0.1**2 + 0.02**2)) < 1e-6

    def test_gaia_gspphot_teff_not_extracted(self):
        """Gaia GSP-Phot Teff must NOT enter the Librarian's _teff slot.
        The photometric SED constrains Teff via BC tables; injecting
        GSP-Phot as a separate observable double-counts the information
        and pins to a ~1-5 K formal σ that hides ±100-200 K systematic
        scatter vs spectroscopy (Andrae+23).
        """
        main = _make_gaia_main_row(teff=5800.0, b_teff=5700.0, B_teff=5950.0)
        lib = self._make_lib(main)
        assert lib.teff is None, (
            "GSP-Phot Teff leaked into Librarian._teff, would inject as "
            "a tight log_Teff likelihood prior"
        )

    def test_gaia_photometry_extracted(self):
        main = _make_gaia_main_row()
        lib = self._make_lib(main)
        mags = lib.magnitudes
        assert "Gaia_G" in mags
        assert "Gaia_BP" in mags
        assert "Gaia_RP" in mags
        assert mags["Gaia_G"][0] == 8.5

    def test_flame_radius_5x_inflation(self):
        main = _make_gaia_main_row()
        flame = _make_flame_row(rad=1.0, b_rad=0.9, B_rad=1.1)
        lib = self._make_lib(main, flame_table=flame)
        assert lib.radius is not None
        r, r_e = lib.radius
        assert r == 1.0
        # max(|1.0-0.9|, |1.1-1.0|) = 0.1, * 5 = 0.5
        assert abs(r_e - 0.5) < 1e-10

    def test_flame_luminosity(self):
        main = _make_gaia_main_row()
        flame = _make_flame_row(lum=1.5, b_lum=1.2, B_lum=1.8)
        lib = self._make_lib(main, flame_table=flame)
        assert lib.luminosity is not None
        assert lib.luminosity[0] == 1.5
        assert abs(lib.luminosity[1] - 0.3) < 1e-10  # max(|1.5-1.2|, |1.8-1.5|)

    def test_flame_missing_returns_none(self):
        main = _make_gaia_main_row()
        lib = self._make_lib(main, flame_table=None)
        assert lib.radius is None
        assert lib.luminosity is None
        assert lib.mass is None
        assert lib.age is None

    def test_bailer_jones_distance(self):
        main = _make_gaia_main_row()
        bj = _make_bj_row(rgeo=100.0, b_rgeo=95.0, B_rgeo=106.0)
        lib = self._make_lib(main, bj_table=bj)
        assert lib.distance is not None
        d, d_e = lib.distance
        assert d == 100.0
        # max(100-95, 106-100) = max(5, 6) = 6
        assert d_e == 6.0


# ── Crossmatch tests ────────────────────────────────────────────


class TestCrossmatch:
    def _make_lib_with_crossmatch(self, tap_results=None, tap_error=None):
        """Create Librarian with controlled TAP crossmatch behavior."""

        def fake_qc(catalog=None, **constraints):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            return None

        call_count = [0]

        def fake_tap(query):
            call_count[0] += 1
            if tap_error is not None:
                raise tap_error
            if tap_results is not None:
                # Return the next result in sequence, or empty table
                if call_count[0] - 1 < len(tap_results):
                    return _make_tap_job(tap_results[call_count[0] - 1])
            return _make_tap_job(Table())

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch(
                "lachesis.librarian.Vizier.query_constraints",
                side_effect=fake_qc,
            ),
            patch("lachesis.librarian.Gaia.launch_job_async", side_effect=fake_tap),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)

        return lib

    def test_tap_success_stores_id(self):
        # 2MASS is the first catalog with xmatch_table
        results = [_make_tap_result("12345678+1234567")]
        lib = self._make_lib_with_crossmatch(tap_results=results)
        assert lib._ids.get("2MASS") == "12345678+1234567"

    def test_tap_not_found_stores_none(self):
        results = [Table()]  # empty = not found
        lib = self._make_lib_with_crossmatch(tap_results=results)
        assert lib._ids.get("2MASS") is None

    def test_tap_down_triggers_xmatch_fallback(self):
        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch(
                "lachesis.librarian.Vizier.query_constraints",
                side_effect=lambda catalog=None, **kw: (
                    [_make_gaia_main_row()]
                    if catalog == "I/355/gaiadr3"
                    else None
                ),
            ),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("503 Service Unavailable"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None) as mock_xm,
        ):
            Librarian(0.0, 0.0, gaia_id=12345, verbose=False)
            # XMatch should have been called for the fallback
            assert mock_xm.called

    def test_ignored_catalog_gets_none(self):
        lib = self._make_lib_with_crossmatch(
            tap_error=Exception("503 Service Unavailable")
        )
        # All IDs should be None since TAP is down and XMatch returns None
        for cat_name in lib._ids:
            if cat_name == "Gaia":
                continue
            assert lib._ids[cat_name] is None


# ── TESS fetch tests ─────────────────────────────────────────────


class TestFetchTESS:
    def test_tess_matched_by_gaia_id(self):
        tic_result = Table(
            {
                "ID": [999],
                "GAIA": ["12345"],
                "objType": ["STAR"],
                "Tmag": [10.5],
                "e_Tmag": [0.01],
                "dstArcSec": [0.5],
                "KIC": [None],
            }
        )

        def fake_qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch(
                "lachesis.librarian.Vizier.query_constraints",
                side_effect=fake_qc,
            ),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch(
                "lachesis.librarian.Catalogs.query_region", return_value=tic_result
            ),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)

        assert lib.tic_id == 999
        assert "TESS" in lib.magnitudes
        assert lib.magnitudes["TESS"][0] == 10.5

    def test_tess_not_star_skipped(self):
        tic_result = Table(
            {
                "ID": [999],
                "GAIA": ["12345"],
                "objType": ["EXTENDED"],
                "Tmag": [10.5],
                "e_Tmag": [0.01],
                "dstArcSec": [0.5],
                "KIC": [None],
            }
        )

        def fake_qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch(
                "lachesis.librarian.Vizier.query_constraints",
                side_effect=fake_qc,
            ),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch(
                "lachesis.librarian.Catalogs.query_region", return_value=tic_result
            ),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)

        assert "TESS" not in lib.magnitudes


# ── Integration tests ────────────────────────────────────────────


class TestLibrarianIntegration:
    def _make_full_lib(self):
        """Librarian with Gaia params + Gaia photometry only (no external cats)."""

        def fake_qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "I/355/paramp":
                return [_make_flame_row()]
            if catalog == "I/352/gedr3dis":
                return [_make_bj_row()]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch(
                "lachesis.librarian.Vizier.query_constraints",
                side_effect=fake_qc,
            ),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            return Librarian(0.0, 0.0, gaia_id=12345, verbose=False)

    def test_all_properties_populated(self):
        lib = self._make_full_lib()
        assert lib.gaia_id == 12345
        assert lib.parallax is not None
        # Gaia GSP-Phot Teff is intentionally NOT extracted (see
        # test_gaia_gspphot_teff_not_extracted), the photometric SED
        # constrains Teff via BC tables.
        assert lib.teff is None
        assert lib.radius is not None
        assert lib.luminosity is not None
        assert lib.mass is not None
        assert lib.age is not None
        assert lib.distance is not None
        assert len(lib.magnitudes) == 3  # G, BP, RP

    def test_magnitudes_are_copies(self):
        lib = self._make_full_lib()
        m1 = lib.magnitudes
        m2 = lib.magnitudes
        assert m1 is not m2
        assert m1 == m2

    def test_librarian_exposes_params(self):
        lib = self._make_full_lib()
        assert lib.parallax[0] is not None
        assert lib.distance[0] is not None
        assert lib.magnitudes is not None
        assert len(lib.magnitudes) > 0

    def test_ignore_skips_catalog(self):

        def fake_qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch(
                "lachesis.librarian.Vizier.query_constraints",
                side_effect=fake_qc,
            ),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None) as mock_mast,
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(
                0.0, 0.0, gaia_id=12345, verbose=False, ignore=["TESS"]
            )

        # TESS should not have been queried via MAST
        assert "TESS" not in lib.magnitudes

    def test_no_gaia_id_graceful(self):
        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch(
                "lachesis.librarian.Vizier.query_constraints", return_value=None
            ),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, verbose=False)

        assert lib.gaia_id is None
        assert lib.parallax is None
        assert len(lib.magnitudes) == 0


# ── Spectroscopic query tests ────────────────────────────────────


def _make_apogee_row(
    tmass="12345678+1234567",
    teff=4800.0,
    e_teff=50.0,
    logg=2.5,
    e_logg=0.05,
    mh=-0.1,
    e_mh=0.03,
    flag=0,
):
    return Table(
        {
            "2MASS": [tmass],
            "Teff": [teff],
            "e_Teff": [e_teff],
            "logg": [logg],
            "e_logg": [e_logg],
            "__M_H_": [mh],
            "e__M_H_": [e_mh],
            "ASPCAPFLAG": [flag],
        }
    )


def _make_galah_row(
    source=12345,
    teff=5800.0,
    e_teff=60.0,
    logg=4.3,
    e_logg=0.1,
    feh=-0.05,
    e_feh=0.05,
    flag_sp=0,
    flag_fe_h=0,
):
    return Table(
        {
            "Source": [source],
            "Teff": [teff],
            "e_Teff": [e_teff],
            "logg": [logg],
            "e_logg": [e_logg],
            "fe_h": [feh],
            "e_fe_h": [e_feh],
            "flag_sp": [flag_sp],
            "flag_fe_h": [flag_fe_h],
        }
    )


def _make_rave_row(
    teff=5500.0,
    e_teff=80.0,
    logg=4.1,
    e_logg=0.15,
    feh=-0.2,
    e_feh=0.08,
    qual=0,
):
    return Table(
        {
            "TeffmC": [teff],
            "e_Teffm": [e_teff],
            "loggmC": [logg],
            "e_loggm": [e_logg],
            "[m/H]mC": [feh],
            "e_[m/H]m": [e_feh],
            "Qual": [qual],
        }
    )


def _make_lamost_row(
    teff=5700.0,
    e_teff=70.0,
    logg=4.2,
    e_logg=0.12,
    feh=-0.15,
    e_feh=0.09,
    snrg=50.0,
    r=0.5,
):
    return Table(
        {
            "Teff": [teff],
            "e_Teff": [e_teff],
            "logg": [logg],
            "e_logg": [e_logg],
            "__Fe_H_": [feh],
            "e__Fe_H_": [e_feh],
            "snrg": [snrg],
            "_r": [r],
        }
    )


def _make_pastel_row(
    teff=5900.0,
    logg=4.4,
    feh=0.0,
    e_teff=None,
    e_logg=None,
    e_feh=None,
    r=0.3,
):
    data = {
        "Teff": [teff],
        "logg": [logg],
        "__Fe_H_": [feh],
        "_r": [r],
    }
    if e_teff is not None:
        data["e_Teff"] = [e_teff]
    if e_logg is not None:
        data["e_logg"] = [e_logg]
    if e_feh is not None:
        data["e__Fe_H_"] = [e_feh]
    return Table(data)


def _make_spectroscopic_lib(
    vizier_qc_side_effect=None, vizier_qr_return=None, twomass_id=None
):
    """Create Librarian with mocked queries, controlling spectroscopic behavior."""

    default_qc = vizier_qc_side_effect

    if default_qc is None:
        def default_qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            return None

    ids = {}
    if twomass_id is not None:
        ids["2MASS"] = twomass_id

    with (
        patch(
            "lachesis.librarian.Vizier.query_region",
            return_value=vizier_qr_return,
        ),
        patch(
            "lachesis.librarian.Vizier.query_constraints",
            side_effect=default_qc,
        ),
        patch(
            "lachesis.librarian.Gaia.launch_job_async",
            side_effect=Exception("no TAP"),
        ),
        patch("lachesis.librarian.Catalogs.query_region", return_value=None),
        patch("lachesis.librarian.XMatch.query", return_value=None),
    ):
        lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)

    # Inject IDs post-init for testing individual query methods
    for k, v in ids.items():
        lib._ids[k] = v

    return lib


class TestQueryAPOGEE:
    def test_apogee_success(self):
        apogee = _make_apogee_row()

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "III/286/allstars":
                return [apogee]
            return None

        lib = _make_spectroscopic_lib(vizier_qc_side_effect=qc, twomass_id="12345678+1234567")
        # Reset and re-query to test APOGEE
        lib._spectroscopic_params = None
        with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
            result = lib._query_apogee()

        assert result is not None
        assert result["teff"] == 4800.0
        assert result["logg"] == 2.5
        assert result["feh"] == -0.1

    def test_apogee_flagged_rejected(self):
        apogee = _make_apogee_row(flag=1)

        def qc(catalog=None, **kw):
            if catalog == "III/286/allstars":
                return [apogee]
            return None

        lib = _make_spectroscopic_lib(vizier_qc_side_effect=qc, twomass_id="12345678+1234567")
        with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
            result = lib._query_apogee()

        assert result is None

    def test_apogee_no_2mass_id(self):
        lib = _make_spectroscopic_lib()
        result = lib._query_apogee()
        assert result is None

    def test_apogee_query_failure(self):
        lib = _make_spectroscopic_lib(twomass_id="12345678+1234567")
        with patch(
            "lachesis.librarian.Vizier.query_constraints",
            side_effect=Exception("VizieR down"),
        ):
            result = lib._query_apogee()
        assert result is None


class TestQueryGALAH:
    def test_galah_success(self):
        galah = _make_galah_row()

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "J/MNRAS/506/150/catalog":
                return [galah]
            return None

        lib = _make_spectroscopic_lib(vizier_qc_side_effect=qc)
        lib._spectroscopic_params = None
        with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
            result = lib._query_galah()

        assert result is not None
        assert result["teff"] == 5800.0
        assert result["feh"] == -0.05

    def test_galah_sp_flag_rejected(self):
        galah = _make_galah_row(flag_sp=1)

        def qc(catalog=None, **kw):
            if catalog == "J/MNRAS/506/150/catalog":
                return [galah]
            return None

        lib = _make_spectroscopic_lib(vizier_qc_side_effect=qc)
        with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
            result = lib._query_galah()

        assert result is None

    def test_galah_feh_flag_rejected(self):
        galah = _make_galah_row(flag_fe_h=1)

        def qc(catalog=None, **kw):
            if catalog == "J/MNRAS/506/150/catalog":
                return [galah]
            return None

        lib = _make_spectroscopic_lib(vizier_qc_side_effect=qc)
        with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
            result = lib._query_galah()

        assert result is None

    def test_galah_no_gaia_id(self):
        lib = _make_spectroscopic_lib()
        lib._gaia_id = None
        result = lib._query_galah()
        assert result is None


class TestQueryRAVE:
    def test_rave_success(self):
        rave = _make_rave_row()

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "III/283/madera":
                return [rave]
            return None

        lib = _make_spectroscopic_lib(vizier_qc_side_effect=qc)
        lib._ids["RAVE"] = "RAVE_OBS_001"
        lib._spectroscopic_params = None
        with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
            result = lib._query_rave()

        assert result is not None
        assert result["teff"] == 5500.0
        assert result["feh"] == -0.2

    def test_rave_bad_quality_rejected(self):
        rave = _make_rave_row(qual=1)

        def qc(catalog=None, **kw):
            if catalog == "III/283/madera":
                return [rave]
            return None

        lib = _make_spectroscopic_lib(vizier_qc_side_effect=qc)
        lib._ids["RAVE"] = "RAVE_OBS_001"
        with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
            result = lib._query_rave()

        assert result is None

    def test_rave_no_id(self):
        lib = _make_spectroscopic_lib()
        result = lib._query_rave()
        assert result is None


class TestQueryLAMOST:
    def test_lamost_success(self):
        lamost = _make_lamost_row()

        lib = _make_spectroscopic_lib()
        with patch(
            "lachesis.librarian.Vizier.query_region",
            return_value=[lamost],
        ):
            result = lib._query_lamost()

        assert result is not None
        assert result["teff"] == 5700.0
        assert result["feh"] == -0.15

    def test_lamost_low_snr_rejected(self):
        lamost = _make_lamost_row(snrg=10.0)

        lib = _make_spectroscopic_lib()
        with patch(
            "lachesis.librarian.Vizier.query_region",
            return_value=[lamost],
        ):
            result = lib._query_lamost()

        assert result is None

    def test_lamost_no_result(self):
        lib = _make_spectroscopic_lib()
        with patch(
            "lachesis.librarian.Vizier.query_region",
            return_value=None,
        ):
            result = lib._query_lamost()

        assert result is None


class TestQueryPASTEL:
    def test_pastel_success(self):
        pastel = _make_pastel_row(e_teff=40.0, e_logg=0.1, e_feh=0.05)

        lib = _make_spectroscopic_lib()
        with patch(
            "lachesis.librarian.Vizier.query_region",
            return_value=[pastel],
        ):
            result = lib._query_pastel()

        assert result is not None
        assert result["teff"] == 5900.0
        assert result["teff_err"] == 40.0
        assert result["feh_err"] == 0.05

    def test_pastel_default_errors(self):
        pastel = _make_pastel_row()  # no error columns

        lib = _make_spectroscopic_lib()
        with patch(
            "lachesis.librarian.Vizier.query_region",
            return_value=[pastel],
        ):
            result = lib._query_pastel()

        assert result is not None
        assert result["teff_err"] == 150.0
        assert result["logg_err"] == 0.3
        assert result["feh_err"] == 0.15

    def test_pastel_missing_param(self):
        """If a core param is None, return None."""
        data = {"Teff": [5900.0], "__Fe_H_": [0.0], "_r": [0.3]}
        # No logg column at all
        pastel = Table(data)

        lib = _make_spectroscopic_lib()
        with patch(
            "lachesis.librarian.Vizier.query_region",
            return_value=[pastel],
        ):
            result = lib._query_pastel()

        assert result is None


class TestSpectroscopicPriority:
    def test_apogee_wins_over_galah(self):
        """APOGEE should be selected when both APOGEE and GALAH are available."""
        apogee = _make_apogee_row()
        galah = _make_galah_row()

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "III/286/allstars":
                return [apogee]
            if catalog == "J/MNRAS/506/150/catalog":
                return [galah]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)
            # Need to inject 2MASS ID and re-run
            lib._ids["2MASS"] = "12345678+1234567"
            lib._spectroscopic_params = None
            with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
                lib._query_spectroscopic_priors()

        assert lib.spectroscopic_params is not None
        assert lib.spectroscopic_params["source"] == "APOGEE_DR17"

    def test_galah_fallback_when_no_apogee(self):
        """GALAH should be used when APOGEE is not available."""
        galah = _make_galah_row()

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "III/286/allstars":
                return None  # no APOGEE
            if catalog == "J/MNRAS/506/150/catalog":
                return [galah]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)
            lib._spectroscopic_params = None
            with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
                lib._query_spectroscopic_priors()

        assert lib.spectroscopic_params is not None
        assert lib.spectroscopic_params["source"] == "GALAH_DR3"

    def test_no_catalogs_available(self):
        """Spectroscopic params should be None when no catalog matches."""

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)

        assert lib.spectroscopic_params is None

    def test_rave_params_backward_compat(self):
        """rave_params property returns dict only when source is RAVE_DR6."""
        rave = _make_rave_row()

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "III/283/madera":
                return [rave]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)
            lib._ids["RAVE"] = "RAVE_OBS_001"
            lib._spectroscopic_params = None
            with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
                lib._query_spectroscopic_priors()

        assert lib.rave_params is not None
        assert lib.rave_params["source"] == "RAVE_DR6"

    def test_rave_params_none_when_different_source(self):
        """rave_params property returns None when source is not RAVE."""
        apogee = _make_apogee_row()

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "III/286/allstars":
                return [apogee]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)
            lib._ids["2MASS"] = "12345678+1234567"
            lib._spectroscopic_params = None
            with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
                lib._query_spectroscopic_priors()

        assert lib.spectroscopic_params is not None
        assert lib.spectroscopic_params["source"] == "APOGEE_DR17"
        assert lib.rave_params is None

    @pytest.mark.skip(
        reason="Librarian.to_star() is unimplemented; pre-existing failing TDD test."
    )
    def test_to_star_passes_spectroscopic_params(self):
        """to_star should pass logg/feh from any spectroscopic source."""
        galah = _make_galah_row(logg=4.3, e_logg=0.1, feh=-0.05, e_feh=0.05)

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "I/355/paramp":
                return [_make_flame_row()]
            if catalog == "I/352/gedr3dis":
                return [_make_bj_row()]
            if catalog == "J/MNRAS/506/150/catalog":
                return [galah]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)
            lib._spectroscopic_params = None
            with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
                lib._query_spectroscopic_priors()

        with patch("lachesis.star.Star") as MockStar:
            lib.to_star("test_star")
            kw = MockStar.call_args.kwargs
            assert kw["logg"] == 4.3
            assert kw["logg_e"] == 0.1
            assert kw["feh"] == -0.05
            assert kw["feh_e"] == 0.05

    def test_catalog_exception_does_not_crash(self):
        """If one catalog raises, the priority chain continues."""

        def qc(catalog=None, **kw):
            if catalog == "I/355/gaiadr3":
                return [_make_gaia_main_row()]
            if catalog == "III/286/allstars":
                raise Exception("APOGEE server down")
            if catalog == "J/MNRAS/506/150/catalog":
                raise Exception("GALAH server down")
            if catalog == "III/283/madera":
                return [_make_rave_row()]
            return None

        with (
            patch("lachesis.librarian.Vizier.query_region", return_value=None),
            patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc),
            patch(
                "lachesis.librarian.Gaia.launch_job_async",
                side_effect=Exception("no TAP"),
            ),
            patch("lachesis.librarian.Catalogs.query_region", return_value=None),
            patch("lachesis.librarian.XMatch.query", return_value=None),
        ):
            lib = Librarian(0.0, 0.0, gaia_id=12345, verbose=False)
            lib._ids["2MASS"] = "12345678+1234567"
            lib._ids["RAVE"] = "RAVE_OBS_001"
            lib._spectroscopic_params = None
            with patch("lachesis.librarian.Vizier.query_constraints", side_effect=qc):
                lib._query_spectroscopic_priors()

        # Should have fallen through to RAVE
        assert lib.spectroscopic_params is not None
        assert lib.spectroscopic_params["source"] == "RAVE_DR6"
