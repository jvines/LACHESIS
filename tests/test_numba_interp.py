"""Tests for numba-JIT trilinear interpolator, TDD."""

import numpy as np
import pytest

from lachesis.grid.mist import MISTModelGrid
from lachesis.interp import GridInterpolator

try:
    from lachesis.interp_numba import NumbaGridInterpolator
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


@pytest.fixture(scope="module")
def grids(full_iso_path):
    mg = MISTModelGrid(full_iso_path.parent)
    scipy_interp = GridInterpolator(mg)
    numba_interp = NumbaGridInterpolator(mg)
    return mg, scipy_interp, numba_interp


class TestNumbaInterpolator:

    def test_construct(self, grids):
        _, _, numba_interp = grids
        assert numba_interp is not None

    def test_matches_scipy_at_grid_point(self, grids):
        """Numba and scipy should give identical results at grid points."""
        mg, scipy_interp, numba_interp = grids
        feh = mg.feh_values[0]
        age = mg.age_values[50]
        ci = mg.columns.index("log_Teff")
        valid = np.where(~np.isnan(mg._data[0, 50, :, ci]))[0]
        eep = float(mg.eep_values[valid[len(valid) // 2]])

        r_scipy = scipy_interp(eep=eep, log_age=age, feh=feh)
        r_numba = numba_interp(eep=eep, log_age=age, feh=feh)

        for col in mg.columns:
            if np.isfinite(r_scipy[col]):
                assert r_numba[col] == pytest.approx(r_scipy[col], rel=1e-10), \
                    f"Mismatch on {col}"

    def test_radius_recomputed_from_log_R_post_interp(self):
        """Radius must be recomputed from log_R after interpolation, not
        interpolated linearly. Otherwise cubes that span orders of magnitude
        in radius (e.g. PARSEC's regridded EEP layout, where the same EEP
        integer maps to MS for one isochrone and RGB tip for another) inject
        a bimodal/inflated radius posterior. Smoking-gun case: KIC_7871531
        v0.0.6 returned R=8 R☉ for a 0.76 M☉ logg=4.3 main-sequence dwarf.
        """
        from types import SimpleNamespace
        from lachesis.interp_numba import NumbaGridInterpolator

        # Pathological synthetic grid: at fixed EEP, radius spans 0.1-130 R☉
        # across (feh, age) corners while log_R stays well-behaved.
        feh_values = np.array([-0.25, 0.0])
        age_values = np.array([9.50, 9.55])
        eep_values = np.array([322.0, 323.0])
        columns = ["initial_mass", "log_Teff", "log_g", "log_L", "log_R",
                   "Teff", "Mbol", "radius", "density"]
        # Cube: 8 corners (feh, age, eep). At each corner pick log_R values
        # that differ by ~3 dex between adjacent (feh, age) slices, # mimicking PARSEC's regrid pathology.
        log_R = np.array([
            [[2.10, 2.13],   # feh=-.25, age=9.50 -> R=125, 135
             [-0.93, -0.93]], # feh=-.25, age=9.55 -> R=0.12
            [[-0.020, -0.018], # feh=0.0,  age=9.50 -> R=0.96
             [-0.049, -0.047]] # feh=0.0,  age=9.55 -> R=0.89
        ])
        mass = np.full(log_R.shape, 0.93)
        log_g = np.full(log_R.shape, 4.30)
        log_T = np.full(log_R.shape, 3.72)
        log_L = np.full(log_R.shape, -0.13)
        Teff = 10**log_T
        Mbol = -2.5 * log_L + 4.74
        # Bake the *bug-prone* radius / density columns: radius == 10**log_R,
        # consistent with the constructor, but interpolating these linearly
        # would inflate the result for queries near (feh=0, age=9.5).
        radius = 10**log_R
        density = mass / radius**3 * 1.41  # rough scale
        data = np.stack([mass, log_T, log_g, log_L, log_R,
                         Teff, Mbol, radius, density], axis=-1)

        grid = SimpleNamespace(
            columns=columns, feh_values=feh_values, age_values=age_values,
            eep_values=eep_values, _data=data, vini_values=None,
        )
        interp = NumbaGridInterpolator(grid)

        # Query far from the inflated corners: feh=-0.033, mostly weighted
        # toward feh=0 where log_R ≈ 0 -> radius ≈ 1 R☉.
        out = interp(eep=322.9, log_age=9.528, feh=-0.033)
        log_R_interp = out["log_R"]
        radius_interp = out["radius"]

        # The fix: radius == 10**log_R after interpolation, not the linear
        # cube interp (which would give ~8 R☉ here).
        assert radius_interp == pytest.approx(10**log_R_interp, rel=1e-6), (
            f"Radius decoupled from log_R after interp: "
            f"radius={radius_interp:.3f}, 10**log_R={10**log_R_interp:.3f}. "
            f"This is the v0.0.6 PARSEC-cube bug, radius column is being "
            f"interpolated linearly through cells with R spanning orders of "
            f"magnitude. Recompute from log_R post-interp."
        )
        # Sanity: should be O(1) R☉, not O(8) R☉.
        assert radius_interp < 2.0, (
            f"Radius={radius_interp:.2f} suggests linear-radius "
            f"interpolation through the inflated corners is still active."
        )

    def test_matches_scipy_between_points(self, grids):
        """Interpolated values between grid points should match."""
        mg, scipy_interp, numba_interp = grids
        eep, age, feh = 325.5, 9.33, 0.0

        r_scipy = scipy_interp(eep=eep, log_age=age, feh=feh)
        r_numba = numba_interp(eep=eep, log_age=age, feh=feh)

        for col in mg.columns:
            if np.isfinite(r_scipy[col]):
                assert r_numba[col] == pytest.approx(r_scipy[col], rel=1e-6), \
                    f"Mismatch on {col}"

    def test_oob_returns_nan(self, grids):
        _, _, numba_interp = grids
        result = numba_interp(eep=9999.0, log_age=9.0, feh=0.0)
        assert np.isnan(result["log_Teff"])

    def test_vectorized(self, grids):
        mg, scipy_interp, numba_interp = grids
        ci = mg.columns.index("log_Teff")
        valid = np.where(~np.isnan(mg._data[0, 50, :, ci]))[0]
        interior = valid[5:-5]
        n = min(20, len(interior))
        eeps = mg.eep_values[interior[:n]].astype(float)
        ages = np.full(n, mg.age_values[50])
        fehs = np.full(n, mg.feh_values[0])

        r_scipy = scipy_interp(eep=eeps, log_age=ages, feh=fehs)
        r_numba = numba_interp(eep=eeps, log_age=ages, feh=fehs)

        for col in mg.columns:
            s = r_scipy[col]
            nb = r_numba[col]
            valid_mask = np.isfinite(s)
            if valid_mask.any():
                np.testing.assert_allclose(
                    nb[valid_mask], s[valid_mask], rtol=1e-6,
                    err_msg=f"Mismatch on {col}",
                )

    def test_faster_than_scipy(self, grids):
        """Numba should be faster than scipy for repeated scalar calls."""
        import time
        mg, scipy_interp, numba_interp = grids

        ci = mg.columns.index("log_Teff")
        valid = np.where(~np.isnan(mg._data[0, 50, :, ci]))[0]
        eep = float(mg.eep_values[valid[len(valid) // 2]])
        age = mg.age_values[50]
        feh = mg.feh_values[0]

        # Warmup numba
        numba_interp(eep=eep, log_age=age, feh=feh)

        n_calls = 1000
        t0 = time.perf_counter()
        for _ in range(n_calls):
            scipy_interp(eep=eep, log_age=age, feh=feh)
        t_scipy = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_calls):
            numba_interp(eep=eep, log_age=age, feh=feh)
        t_numba = time.perf_counter() - t0

        print(f"\n  scipy: {t_scipy:.3f}s, numba: {t_numba:.3f}s, "
              f"speedup: {t_scipy/t_numba:.1f}x")
        # Numba should be at least 2x faster
        assert t_numba < t_scipy
