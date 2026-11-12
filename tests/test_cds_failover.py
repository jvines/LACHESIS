"""Offline tests for the CDS failover module (fakes only, no network)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.query import BaseQuery
from regions import CircleSkyRegion

from lachesis.librarian import _cds

PRIMARY = "vizier.cds.unistra.fr"
MIRROR = _cds.VIZIER_MIRRORS[0]
URL = f"https://{PRIMARY}/viz-bin/votable"


@pytest.fixture
def real_local_xmatch(monkeypatch):
    """Undo the autouse fixture that disables local_xmatch."""
    monkeypatch.setattr(_cds, "local_xmatch", _cds.local_xmatch.real)


def _resp(status=200):
    return SimpleNamespace(status_code=status, content=b"")


class FakeNet:
    """Replacement for BaseQuery._request, scripted per host."""

    def __init__(self, script):
        self.script = script  # host -> status int or Exception
        self.calls = []

    def __call__(self, qself, method, url, *a, **kw):
        host = url.split("/")[2]
        self.calls.append((host, kw.get("timeout")))
        act = self.script.get(host, 200)
        if isinstance(act, Exception):
            raise act
        return _resp(act)


@pytest.fixture
def net(monkeypatch):
    def make(script):
        fake = FakeNet(script)
        monkeypatch.setattr(
            BaseQuery, "_request",
            lambda qself, method, url, *a, **kw: fake(qself, method, url, *a, **kw))
        return fake
    return make


def _viz():
    return _cds.MirroredVizier()


def test_primary_ok_no_failover(net):
    fake = net({})
    assert _viz()._request("GET", URL).status_code == 200
    assert [h for h, _ in fake.calls] == [PRIMARY]


def test_timeout_is_connect_read(net):
    fake = net({})
    _viz()._request("GET", URL, timeout=60)
    assert fake.calls[0][1] == (10.0, 60.0)


def test_failover_on_404(net):
    fake = net({PRIMARY: 404})
    assert _viz()._request("GET", URL).status_code == 200
    assert [h for h, _ in fake.calls] == [PRIMARY, MIRROR]
    assert _cds.breaker.is_down(PRIMARY)


def test_failover_on_connection_error(net):
    fake = net({PRIMARY: requests.ConnectionError("x")})
    assert _viz()._request("GET", URL).status_code == 200
    assert [h for h, _ in fake.calls] == [PRIMARY, MIRROR]


def test_all_hosts_down_returns_last_response_or_raises(net):
    net({PRIMARY: 404, MIRROR: 503})
    assert _viz()._request("GET", URL).status_code == 503
    _cds.breaker.clear()
    net({PRIMARY: requests.ConnectionError("a"), MIRROR: requests.Timeout("b")})
    with pytest.raises(requests.Timeout):
        _viz()._request("GET", URL)


def test_empty_result_not_retried(net):
    """An empty (but 200) answer is valid; the mirror must not be asked."""
    fake = net({})
    _viz()._request("GET", URL)
    assert len(fake.calls) == 1


def test_breaker_skips_tripped_host_and_resets_on_success(net):
    fake = net({PRIMARY: 404})
    _viz()._request("GET", URL)
    fake.calls.clear()
    _viz()._request("GET", URL)
    assert [h for h, _ in fake.calls] == [MIRROR]  # primary skipped


def test_breaker_reset_on_success(net):
    _cds.breaker.trip(PRIMARY, "test")
    _cds.breaker.trip(MIRROR, "test")
    net({})
    _viz()._request("GET", URL)
    assert not _cds.breaker.is_down(PRIMARY)


def test_cached_error_purged(net, tmp_path, monkeypatch):
    removed = []
    monkeypatch.setattr(_cds.os, "remove", lambda p: removed.append(p))
    monkeypatch.setattr(_cds.os.path, "exists", lambda p: True)
    net({PRIMARY: 404})
    v = _viz()
    v._request("GET", URL, cache=True, params={"a": 1})
    assert len(removed) == 1
    removed.clear()
    _cds.breaker.clear()
    net({})
    v._request("GET", URL, cache=True, params={"a": 1})
    assert removed == []


# ── local xMatch ─────────────────────────────────────────────────────


def _patch_cones(monkeypatch, t1, t2):
    tables = {"I/355/gaiadr3": t1, "II/x": t2}

    def fake_cone(catalog, center, radius, timeout):
        return tables[catalog]
    monkeypatch.setattr(_cds, "_cone", fake_cone)


AREA = CircleSkyRegion(SkyCoord(10 * u.deg, 20 * u.deg), radius=5 * u.arcmin)


def test_local_xmatch_pairs_and_layout(real_local_xmatch, monkeypatch):
    t1 = Table({"Source": [1, 2], "RA_ICRS": [10.0, 10.01], "DE_ICRS": [20.0, 20.0],
                "Gmag": [10.0, 11.0]})
    d = 1.0 / 3600
    t2 = Table({"RA_ICRS": [10.0, 10.0], "DE_ICRS": [20.0 + d, 21.0],
                "Gmag": [9.0, 8.0], "id": ["a", "b"]})
    _patch_cones(monkeypatch, t1, t2)
    out = _cds.local_xmatch("vizier:I/355/gaiadr3", "vizier:II/x",
                            5 * u.arcsec, area=AREA)
    assert len(out) == 1
    # The counterpart keeps clashing names; Gaia's become <name>_1, except Source.
    assert out.colnames == ["angDist", "Source", "RA_ICRS_1", "DE_ICRS_1", "Gmag_1",
                            "RA_ICRS", "DE_ICRS", "Gmag", "id"]
    assert out["Source"][0] == 1 and out["id"][0] == "a"
    assert out["angDist"][0] == pytest.approx(1.0, abs=1e-3)
    assert out["Gmag"][0] == 9.0 and out["Gmag_1"][0] == 10.0


def test_local_xmatch_empty(real_local_xmatch, monkeypatch):
    t1 = Table({"RA_ICRS": [10.0], "DE_ICRS": [20.0]})
    t2 = Table({"RA_ICRS": [11.0], "DE_ICRS": [21.0]})
    _patch_cones(monkeypatch, t1, t2)
    out = _cds.local_xmatch("vizier:I/355/gaiadr3", "vizier:II/x",
                            5 * u.arcsec, area=AREA)
    assert len(out) == 0 and "angDist" in out.colnames
    _patch_cones(monkeypatch, None, t2)
    assert len(_cds.local_xmatch("vizier:I/355/gaiadr3", "vizier:II/x",
                                 5 * u.arcsec, area=AREA)) == 0


def test_local_xmatch_failure_raises(real_local_xmatch, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(_cds, "_cone", boom)
    with pytest.raises(requests.ConnectionError):
        _cds.local_xmatch("vizier:I/355/gaiadr3", "vizier:II/x",
                          5 * u.arcsec, area=AREA)


def test_local_xmatch_needs_area_and_vizier_ids(real_local_xmatch):
    with pytest.raises(ValueError):
        _cds.local_xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=None)
    with pytest.raises(ValueError):
        _cds.local_xmatch("simbad", "vizier:b", 1 * u.arcsec, area=AREA)


# ── xmatch wrapper ───────────────────────────────────────────────────


def test_xmatch_uses_cds_when_up():
    client = MagicMock()
    client.query.return_value = "ok"
    assert _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA,
                       client=client) == "ok"


def test_xmatch_falls_back_on_request_exception(monkeypatch):
    client = MagicMock()
    client.query.side_effect = requests.ConnectionError("tcp")
    monkeypatch.setattr(_cds, "local_xmatch", lambda *a, **k: "local")
    assert _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA,
                       client=client) == "local"
    assert _cds.breaker.is_down(_cds.XMATCH_HOST)
    # breaker open: CDS is not asked again
    client.query.reset_mock()
    _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA, client=client)
    client.query.assert_not_called()


def test_api_xmatch_skips_cds_when_tcp_connect_fails(monkeypatch):
    """astroquery retries a dead xMatch host for >60 s; detect it up front."""
    from lachesis.librarian import _api

    client = MagicMock()
    monkeypatch.setattr(_api, "XMatch", client)
    monkeypatch.setattr(_cds, "_reachable", lambda *a, **k: False)
    monkeypatch.setattr(_cds, "local_xmatch", lambda *a, **k: "local")
    assert _api._xmatch(cat1="vizier:a", cat2="vizier:b",
                        max_distance=1 * u.arcsec, area=AREA) == "local"
    client.query.assert_not_called()
    assert _cds.breaker.is_down(_cds.XMATCH_HOST)


def test_cds_reachable_false_on_refused(monkeypatch):
    from lachesis.librarian import _api

    def refuse(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(_cds.socket, "create_connection", refuse)
    assert _cds._reachable.real("http://example.invalid/x", timeout=1) is False


def test_xmatch_does_not_fall_back_on_other_errors(monkeypatch):
    client = MagicMock()
    client.query.side_effect = ValueError("bad query")
    monkeypatch.setattr(_cds, "local_xmatch", lambda *a, **k: "local")
    with pytest.raises(ValueError):
        _cds.xmatch("vizier:a", "vizier:b", 1 * u.arcsec, area=AREA, client=client)


def test_connect_read():
    assert _cds.connect_read(60) == (10.0, 60.0)
    assert _cds.connect_read(5) == (5.0, 5.0)
    assert _cds.connect_read((1, 2)) == (1, 2)
    assert _cds.connect_read(None) is None


# ── SIMBAD ───────────────────────────────────────────────────────────


class FakeSimbad:
    def __init__(self, server, behaviour):
        self.server = server
        self.ROW_LIMIT = 0
        self.columns_in_output = []
        self.behaviour = behaviour

    def query_objectids(self, name):
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


def test_simbad_ok_on_primary():
    c = FakeSimbad("simbad.cds.unistra.fr", "ids")
    assert _cds.simbad("query_objectids", "x", client=c) == "ids"


def test_simbad_mirror_failover(monkeypatch):
    c = FakeSimbad("simbad.cds.unistra.fr", requests.ConnectionError("x"))
    seen = []

    class Mirror(FakeSimbad):
        def __init__(self):
            super().__init__(None, "mirror-ids")

        def query_objectids(self, name):
            seen.append(self.server)
            return "mirror-ids"

    monkeypatch.setattr(_cds, "SimbadClass", Mirror)
    assert _cds.simbad("query_objectids", "x", client=c) == "mirror-ids"
    assert seen == [_cds.SIMBAD_MIRRORS[0]]
    assert _cds.breaker.is_down("simbad.cds.unistra.fr")


def test_simbad_no_retry_on_valueerror(monkeypatch):
    c = FakeSimbad("simbad.cds.unistra.fr", ValueError("bad"))
    monkeypatch.setattr(_cds, "SimbadClass",
                        lambda: pytest.fail("mirror must not be tried"))
    with pytest.raises(ValueError):
        _cds.simbad("query_objectids", "x", client=c)


def test_simbad_all_down_raises(monkeypatch):
    c = FakeSimbad("simbad.cds.unistra.fr", requests.ConnectionError("a"))

    class Mirror(FakeSimbad):
        def __init__(self):
            super().__init__(None, requests.Timeout("b"))
    monkeypatch.setattr(_cds, "SimbadClass", Mirror)
    with pytest.raises(requests.Timeout):
        _cds.simbad("query_objectids", "x", client=c)


# ── wiring ───────────────────────────────────────────────────────────


def test_api_wiring():
    from lachesis.librarian import _api

    assert isinstance(_api.Vizier, _cds.MirroredVizier)
    assert _api.XMatch.TIMEOUT == (10.0, 60.0)


def test_api_xmatch_helper_uses_patched_client(monkeypatch):
    from lachesis.librarian import _api

    fake = MagicMock()
    fake.query.return_value = "cds"
    monkeypatch.setattr(_api, "XMatch", fake)
    assert _api._xmatch(cat1="vizier:a", cat2="vizier:b",
                        max_distance=1 * u.arcsec, area=AREA) == "cds"
    kw = fake.query.call_args.kwargs
    assert kw["cat1"] == "vizier:a" and kw["area"] is AREA


def test_star_name_lookup_goes_through_simbad_failover(monkeypatch):
    """lachesis.fit resolves the Gaia ID via _cds.simbad, not Simbad directly."""
    import lachesis
    from astropy.table import Table as T

    calls = {}

    def fake_simbad(method, *args, client=None, **kw):
        calls["args"] = (method, args)
        return T({"ID": ["Gaia DR3 123456789", "HD 1"]})

    monkeypatch.setattr(_cds, "simbad", fake_simbad)
    monkeypatch.setattr("astropy.coordinates.SkyCoord.from_name",
                        classmethod(lambda cls, n: SkyCoord(1 * u.deg, 2 * u.deg)))
    captured = {}

    class Stop(Exception):
        pass

    class FakeStar:
        def __init__(self, name, ra, dec, g_id=None, verbose=False):
            captured["g_id"] = g_id
            raise Stop

    monkeypatch.setattr(lachesis, "Star", FakeStar)
    with pytest.raises(Stop):
        lachesis.fit("HD 1")
    assert calls["args"] == ("query_objectids", ("HD 1",))
    assert captured["g_id"] == 123456789


def test_tycho_id_reads_the_tycho_column_not_gaias():
    """Gaia DR3 has its own TYC2; the counterpart keeps the bare name."""
    from lachesis.librarian import _api

    xm = Table({"angDist": [0.5], "Source": [7], "TYC2_1": ["1707-00767-1"],
                "TYC1": [1707], "TYC2": [767], "TYC3": [1]})
    lib = _api.Librarian.__new__(_api.Librarian)
    lib._coord = SkyCoord(1 * u.deg, 2 * u.deg)
    lib._search_radius = 3 * u.arcsec
    lib._gaia_id = 7
    lib._ignore = ()
    lib._ids = {}
    lib._magnitudes = {}
    fake = MagicMock()
    fake.query.return_value = xm
    import lachesis.librarian._api as api_mod
    orig = api_mod.XMatch
    api_mod.XMatch = fake
    try:
        lib._xmatch_fallback(["TYCHO2"])
    finally:
        api_mod.XMatch = orig
    assert lib._ids["TYCHO2"] == "1707-767-1"
