"""Tests for numba-JIT trilinear interpolator — TDD."""

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
