"""Mirror junk rows and failed archive queries (2026-11-13 CfA outage).

The CfA VizieR mirror put an all-empty row first in some answers and sent a
malformed VOTable for AllWISE. Code that read row 0 got a NaN distance or lost
2MASS; one bad table failed the bulk photometry query, so every standard
catalogue read as absent. All network calls are fakes.
"""
import io
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table

from lachesis.librarian import Librarian
from lachesis.librarian import _api
from lachesis.librarian import _cds
from lachesis.star import Star


def _bj_table(junk_first=True):
    rows = [(0, np.nan, np.nan, np.nan, True),
            (6112358006954796416, 110.17, 109.03, 111.21, False)]
    if not junk_first:
        rows = rows[1:]
    t = Table()
    t["Source"] = [r[0] for r in rows]
    for i, name in enumerate(("rgeo", "b_rgeo", "B_rgeo"), start=1):
        t[name] = MaskedColumn([r[i] for r in rows], mask=[r[4] for r in rows])
    return t


# ── _junk_rows ───────────────────────────────────────────────────────


def test_junk_rows_flags_the_mirror_bailer_jones_row():
    assert list(_cds._junk_rows(_bj_table())) == [True, False]


def test_junk_rows_flags_the_mirror_2mass_row():
    # The 2MASS junk row has RAJ2000 = 0 (not masked) and a filled link column.
    t = Table()
    t["RAJ2000"] = [0.0, 205.729261]
    t["DEJ2000"] = MaskedColumn([0.0, -41.401066], mask=[True, False])
    t["Kmag"] = MaskedColumn([0.0, 3.1], mask=[True, False])
    t["Opt"] = ["Opt", "Opt"]
    assert list(_cds._junk_rows(t)) == [True, False]


def test_junk_rows_keeps_real_rows_and_text_only_tables():
    assert not _cds._junk_rows(_bj_table(junk_first=False)).any()
    assert not _cds._junk_rows(Table({"name": ["a", "b"]})).any()
    assert not _cds._junk_rows(Table({"x": np.array([], dtype=float)})).any()


def test_parse_result_drops_junk_rows():
    buf = io.BytesIO()
    _bj_table().write(buf, format="votable")

    class _Resp:
        content = buf.getvalue()

    result = _cds.MirroredVizier()._parse_result(_Resp())
    assert len(result) == 1
    tbl = result[0]
    assert list(tbl["Source"]) == [6112358006954796416]
    assert float(tbl["rgeo"][0]) == pytest.approx(110.17)


# ── Bailer-Jones distance ────────────────────────────────────────────


def test_bj_distance_rejects_masked_and_non_finite():
    assert _api._bj_distance(_bj_table()[0]) is None
    assert _api._bj_distance({"rgeo": np.nan, "b_rgeo": 1.0, "B_rgeo": 2.0}) is None
    assert _api._bj_distance({"rgeo": None, "b_rgeo": 1.0, "B_rgeo": 2.0}) is None
    assert _api._bj_distance({"rgeo": 0.0, "b_rgeo": 0.0, "B_rgeo": 0.0}) is None
    d, e = _api._bj_distance(_bj_table()[1])
    assert d == pytest.approx(110.17) and e == pytest.approx(1.14)


def _bare_lib(**attrs):
    lib = Librarian.__new__(Librarian)
    lib._gaia_id = 6112358006954796416
    lib._coord = SkyCoord(205.7289, -41.4013, unit="deg")
    lib._search_radius = 3.0 * u.arcsec
    lib._ignore = set()
    lib._ids = {}
    lib._magnitudes = {}
    lib._distance = lib._distance_e = None
    lib.failed_queries = []
    for k, v in attrs.items():
        setattr(lib, k, v)
    return lib


def test_masked_vizier_distance_is_read_from_esa():
    # Without the parse-level filter (e.g. a cached answer) row 0 is the junk
    # row; it must not become a NaN distance.
    lib = _bare_lib()
    esa = {"r_med_geo": 110.17, "r_lo_geo": 109.03, "r_hi_geo": 111.21}
    with patch.object(_api.Vizier, "query_constraints",
                      return_value=[_bj_table()]), \
            patch.object(_api, "_esa_first_row", return_value=esa) as q:
        lib._query_bailer_jones()
    assert q.called
    assert lib._distance == pytest.approx(110.17)
    assert np.isfinite(lib._distance_e)
    assert lib.failed_queries == []


def test_distance_failure_is_recorded():
    lib = _bare_lib()
    with patch.object(_api.Vizier, "query_constraints",
                      side_effect=Exception("mirror down")), \
            patch.object(_api, "_esa_first_row", return_value=None):
        lib._query_bailer_jones()
    assert lib._distance is None
    assert lib.failed_queries == ["Bailer-Jones distance"]


def test_absent_distance_is_not_a_failure():
    lib = _bare_lib()
    with patch.object(_api.Vizier, "query_constraints", return_value=[]), \
            patch.object(_api, "_esa_first_row", return_value={}):
        lib._query_bailer_jones()
    assert lib._distance is None
    assert lib.failed_queries == []


# ── Bulk photometry query ────────────────────────────────────────────


class _TL(dict):
    """TableList stand-in keyed by catalogue ID."""


def _fetch_with(query_region):
    lib = _bare_lib(_ids={"2MASS": "13425501-4124040", "Wise": "J134255"})
    seen = {}

    def fake_standard(cat_name, cat_def, cats, ext_id):
        seen[cat_name] = cat_def.vizier_id in cats

    with patch.object(_api.Vizier, "query_region", side_effect=query_region), \
            patch.object(Librarian, "_fetch_standard", side_effect=fake_standard,
                         autospec=False), \
            patch.multiple(Librarian, _fetch_tess=lambda s: None,
                           _fetch_apass=lambda s: None,
                           _fetch_skymapper=lambda s: None,
                           _fetch_galex=lambda s: None,
                           _fetch_mermilliod=lambda s: None,
                           _fetch_stromgren=lambda s: None):
        lib._fetch_photometry()
    return lib, seen


def test_one_bad_table_does_not_cost_the_other_catalogues():
    def query_region(coord, radius=None, catalog=None):
        if isinstance(catalog, list) or catalog == "II/328/allwise":
            raise ValueError("Failed to parse VIZIER result")
        return _TL({catalog: Table({"Jmag": [3.5]})})

    lib, seen = _fetch_with(query_region)
    assert seen == {"2MASS": True, "Wise": False}
    assert lib.failed_queries == ["II/328/allwise"]


def test_working_bulk_query_records_nothing():
    def query_region(coord, radius=None, catalog=None):
        return _TL({c: Table({"x": [1.0]}) for c in catalog})

    lib, seen = _fetch_with(query_region)
    assert seen == {"2MASS": True, "Wise": True}
    assert lib.failed_queries == []


# ── Single-catalogue fetchers: failure vs absence ────────────────────


def test_tess_failure_is_recorded_but_absence_is_not():
    lib = _bare_lib()
    with patch.object(_api.Catalogs, "query_region",
                      side_effect=Exception("MAST down")):
        lib._fetch_tess()
    assert lib.failed_queries == ["TIC"]

    lib = _bare_lib()
    with patch.object(_api.Catalogs, "query_region",
                      return_value=Table({"GAIA": np.array([], dtype=str)})):
        lib._fetch_tess()
    assert lib.failed_queries == []


def test_mermilliod_failure_is_recorded():
    lib = _bare_lib()
    with patch.object(_api, "_xmatch", side_effect=Exception("xMatch down")):
        lib._fetch_mermilliod()
    assert lib.failed_queries == ["Mermilliod UBV"]


# ── Star ─────────────────────────────────────────────────────────────


def test_star_ignores_a_nan_distance_and_uses_the_parallax():
    s = Star("x", 10.0, -20.0, magnitudes={"Gaia_G": (6.0, 0.003)},
             plx=10.0, plx_e=0.1, dist=float("nan"), dist_e=float("nan"),
             Av=0.0, offline=True, verbose=False)
    assert s.distance == pytest.approx(100.0)
    assert s.distance_e == pytest.approx(1.0)


def test_star_carries_the_librarian_failures():
    class _FakeLib:
        def __init__(self, *a, **k):
            self.magnitudes = {"Gaia_G": (6.0, 0.003)}
            self._parallax, self._parallax_e = 10.0, 0.1
            self._distance, self._distance_e = 100.0, 1.0
            self.Av = 0.0
            self._radius_flame = self._radius_flame_e = None
            self._spectroscopic_params = None
            self.failed_queries = ["II/328/allwise"]

    with patch("lachesis.librarian.Librarian", _FakeLib):
        s = Star("x", 10.0, -20.0, Av=0.0, verbose=False)
    assert s.failed_queries == ["II/328/allwise"]


# ── Catalogue-ID fallback (Gaia TAP down) ────────────────────────────


def test_failed_id_crossmatch_is_recorded_but_no_counterpart_is_not():
    # With ESA's best_neighbour tables down the IDs come from xMatch; if that
    # fails too, the catalogue is never queried, so it must not read as absent.
    def xmatch(cat1=None, cat2=None, **kw):
        if cat2 == "vizier:II/246/out":
            raise ValueError("Failed to parse VIZIER result")
        return Table({"angDist": np.array([], dtype=float)})

    lib = _bare_lib()
    with patch.object(_api, "_xmatch", side_effect=xmatch):
        lib._xmatch_fallback(["2MASS", "Wise"])
    assert lib._ids == {"2MASS": None, "Wise": None}
    assert lib.failed_queries == ["2MASS ID"]


def test_failed_cone_only_id_lookup_is_recorded():
    lib = _bare_lib()
    with patch.object(_api.Vizier, "query_region",
                      side_effect=Exception("VizieR down")):
        lib._xmatch_fallback(["SkyMapper"])
    assert lib.failed_queries == ["SkyMapper ID"]


# ── Crossmatched source missing from the VizieR answer ───────────────


def test_crossmatched_source_missing_from_mirror_answer_is_a_failure():
    from lachesis.librarian._api import _CATALOGS, _PRIMARY_VIZIER
    lib = _bare_lib()
    _cds.breaker.trip(_PRIMARY_VIZIER, "test")
    lib._fetch_standard("2MASS", _CATALOGS["2MASS"], {}, "13425502-4124038")
    assert lib.failed_queries == [
        "II/246/out (mirror answer lacks the crossmatched 2MASS source)"]


def test_crossmatched_source_missing_from_cds_answer_is_not_a_failure():
    from lachesis.librarian._api import _CATALOGS
    lib = _bare_lib()
    tbl = Table({"_r": [0.5], "_2MASS": ["00000000-0000000"], "Jmag": [5.0]})
    lib._fetch_standard("2MASS", _CATALOGS["2MASS"], {"II/246/out": tbl},
                        "13425502-4124038")
    assert lib.failed_queries == []
