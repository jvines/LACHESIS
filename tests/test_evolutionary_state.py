"""Pre-fit evolutionary classification (Star.evolutionary_state) and the
giant-coverage grid probe (_grid_min_logg)."""

import numpy as np
import pytest

from lachesis.fitter import _grid_min_logg
from lachesis.star import Star


def _star(logg=None, magnitudes=None, plx=None, Av=0.0, radius_flame=None):
    s = Star(
        starname="test_star", ra=0.0, dec=0.0,
        magnitudes=magnitudes if magnitudes is not None else {},
        plx=plx, plx_e=0.01 if plx else None,
        Av=Av, logg=logg, logg_e=0.1 if logg is not None else None,
        verbose=False, offline=True,
    )
    if radius_flame is not None:
        s.radius_flame = radius_flame
    return s


# ── tier 1: spectroscopic log g ──────────────────────────────────────────

def test_tier1_logg_giant():
    assert _star(logg=2.5).evolutionary_state == "giant"


def test_tier1_logg_subgiant():
    assert _star(logg=3.6).evolutionary_state == "subgiant"


def test_tier1_logg_ms():
    assert _star(logg=4.4).evolutionary_state == "ms"


def test_tier1_wins_over_tier23():
    # spectroscopic log g overrides FLAME and CMD
    mags = {"Gaia_G": (8.0, 0.01), "Gaia_BP": (9.0, 0.01), "Gaia_RP": (7.5, 0.01)}
    s = _star(logg=4.5, magnitudes=mags, plx=10.0, radius_flame=10.0)
    assert s.evolutionary_state == "ms"


# ── tier 2: FLAME radius ─────────────────────────────────────────────────

def test_tier2_flame_giant():
    assert _star(radius_flame=8.0).evolutionary_state == "giant"


def test_tier2_flame_dwarf_falls_through():
    # a small FLAME radius cannot separate ms from subgiant -> unknown
    assert _star(radius_flame=1.0).evolutionary_state == "unknown"


# ── tier 3: dereddened CMD position ──────────────────────────────────────

def test_tier3_cmd_giant():
    # plx=10 mas -> d=100 pc -> M_G = G - 5; G=8 -> M_G=3.0 < 3.5,
    # BP-RP = 1.5 > 0.95 -> giant
    mags = {"Gaia_G": (8.0, 0.01), "Gaia_BP": (9.0, 0.01), "Gaia_RP": (7.5, 0.01)}
    assert _star(magnitudes=mags, plx=10.0).evolutionary_state == "giant"


def test_tier3_cmd_dwarf_unknown():
    # M_G = 5.5 (faint): below the giant cut -> unknown, not 'ms'
    mags = {"Gaia_G": (10.5, 0.01), "Gaia_BP": (11.2, 0.01), "Gaia_RP": (10.0, 0.01)}
    assert _star(magnitudes=mags, plx=10.0).evolutionary_state == "unknown"


def test_tier3_equal_binary_not_flagged():
    # an equal-luminosity MS pair at BP-RP ~ 1.0 sits ~0.75 mag above the
    # single-star MS (M_G ~ 5.8 -> 5.05), still far from the 3.5 cut
    mags = {"Gaia_G": (10.05, 0.01), "Gaia_BP": (10.85, 0.01), "Gaia_RP": (9.8, 0.01)}
    assert _star(magnitudes=mags, plx=10.0).evolutionary_state == "unknown"


def test_tier3_extinction_dereddening():
    # heavily reddened dwarf: raw colours mimic a giant, dereddening must
    # pull it back below the cut. Av=3: A_G=2.37, E(BP-RP)=1.24.
    mags = {"Gaia_G": (5.8, 0.01), "Gaia_BP": (6.9, 0.01), "Gaia_RP": (5.3, 0.01)}
    s = _star(magnitudes=mags, plx=10.0, Av=3.0)
    # M_G = 5.8 - 5 - 2.37 = -1.57 < 3.5 BUT (BP-RP)0 = 1.6 - 1.24 = 0.36 < 0.95
    assert s.evolutionary_state == "unknown"


def test_no_information_unknown():
    assert _star().evolutionary_state == "unknown"


def test_edr3_key_variant():
    mags = {"Gaia_G_EDR3": (8.0, 0.01), "Gaia_BP_EDR3": (9.0, 0.01),
            "Gaia_RP_EDR3": (7.5, 0.01)}
    assert _star(magnitudes=mags, plx=10.0).evolutionary_state == "giant"


# ── grid probe ───────────────────────────────────────────────────────────

class _FakeGrid:
    def __init__(self, logg_values, columns=("log_Teff", "log_g")):
        self.columns = list(columns)
        n = len(logg_values)
        self._data = np.full((2, 3, n, len(columns)), np.nan)
        if "log_g" in self.columns:
            self._data[0, 0, :, self.columns.index("log_g")] = logg_values


def test_grid_min_logg_dwarf_only():
    g = _FakeGrid([3.7, 4.2, 4.7])
    assert _grid_min_logg(g) == pytest.approx(3.7)


def test_grid_min_logg_reaches_giants():
    g = _FakeGrid([0.5, 2.0, 4.5])
    assert _grid_min_logg(g) == pytest.approx(0.5)


def test_grid_min_logg_no_column():
    g = _FakeGrid([4.0], columns=("log_Teff", "log_L"))
    assert _grid_min_logg(g) is None
