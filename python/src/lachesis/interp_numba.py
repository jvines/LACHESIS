"""Numba-JIT trilinear interpolator for isochrone grids.

Drop-in replacement for GridInterpolator with same API but ~10-50x faster
for repeated scalar evaluations (nested sampling inner loop).
"""


import numpy as np
import numba as nb


@nb.njit(cache=True)
def _searchsorted(arr, val):
    """Binary search returning index of interval containing val.
    Returns i such that arr[i] <= val < arr[i+1].
    Returns -1 if val < arr[0], len(arr)-1 if val >= arr[-1].
    """
    n = len(arr)
    if val < arr[0] or val > arr[n - 1]:
        return -1
    lo, hi = 0, n - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid
    return lo


@nb.njit(cache=True)
def _trilinear(grid, ax0, ax1, ax2, x0, x1, x2, n_cols):
    """Trilinear interpolation on a regular 3D grid.

    grid: (n0, n1, n2, n_cols) array
    ax0, ax1, ax2: 1D arrays of grid coordinates
    x0, x1, x2: query point
    n_cols: number of output columns

    Returns array of length n_cols, NaN if out of bounds.
    """
    result = np.empty(n_cols)

    i0 = _searchsorted(ax0, x0)
    i1 = _searchsorted(ax1, x1)
    i2 = _searchsorted(ax2, x2)

    if i0 < 0 or i0 >= len(ax0) - 1:
        result[:] = np.nan
        return result
    if i1 < 0 or i1 >= len(ax1) - 1:
        result[:] = np.nan
        return result
    if i2 < 0 or i2 >= len(ax2) - 1:
        result[:] = np.nan
        return result

    # Normalized distances
    d0 = (x0 - ax0[i0]) / (ax0[i0 + 1] - ax0[i0])
    d1 = (x1 - ax1[i1]) / (ax1[i1 + 1] - ax1[i1])
    d2 = (x2 - ax2[i2]) / (ax2[i2 + 1] - ax2[i2])

    # 8-point trilinear interpolation
    for c in range(n_cols):
        c000 = grid[i0, i1, i2, c]
        c001 = grid[i0, i1, i2 + 1, c]
        c010 = grid[i0, i1 + 1, i2, c]
        c011 = grid[i0, i1 + 1, i2 + 1, c]
        c100 = grid[i0 + 1, i1, i2, c]
        c101 = grid[i0 + 1, i1, i2 + 1, c]
        c110 = grid[i0 + 1, i1 + 1, i2, c]
        c111 = grid[i0 + 1, i1 + 1, i2 + 1, c]

        # Weight-aware NaN: only NaN if a corner with non-zero weight is NaN
        w000 = (1 - d0) * (1 - d1) * (1 - d2)
        w001 = (1 - d0) * (1 - d1) * d2
        w010 = (1 - d0) * d1 * (1 - d2)
        w011 = (1 - d0) * d1 * d2
        w100 = d0 * (1 - d1) * (1 - d2)
        w101 = d0 * (1 - d1) * d2
        w110 = d0 * d1 * (1 - d2)
        w111 = d0 * d1 * d2

        has_nan = False
        if w000 > 0 and np.isnan(c000): has_nan = True
        if w001 > 0 and np.isnan(c001): has_nan = True
        if w010 > 0 and np.isnan(c010): has_nan = True
        if w011 > 0 and np.isnan(c011): has_nan = True
        if w100 > 0 and np.isnan(c100): has_nan = True
        if w101 > 0 and np.isnan(c101): has_nan = True
        if w110 > 0 and np.isnan(c110): has_nan = True
        if w111 > 0 and np.isnan(c111): has_nan = True

        if has_nan:
            result[c] = np.nan
        else:
            # Replace NaN corners (zero weight) with 0 for arithmetic
            if np.isnan(c000): c000 = 0.0
            if np.isnan(c001): c001 = 0.0
            if np.isnan(c010): c010 = 0.0
            if np.isnan(c011): c011 = 0.0
            if np.isnan(c100): c100 = 0.0
            if np.isnan(c101): c101 = 0.0
            if np.isnan(c110): c110 = 0.0
            if np.isnan(c111): c111 = 0.0
            result[c] = (w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011 +
                         w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111)

    return result


@nb.njit(cache=True)
def _trilinear_batch(grid, ax0, ax1, ax2, x0s, x1s, x2s, n_cols):
    """Vectorized trilinear interpolation."""
    n = len(x0s)
    result = np.empty((n, n_cols))
    for i in range(n):
        result[i] = _trilinear(grid, ax0, ax1, ax2, x0s[i], x1s[i], x2s[i], n_cols)
    return result


class NumbaGridInterpolator:
    """Drop-in replacement for GridInterpolator using numba JIT.

    Same API: __call__(eep, log_age, feh) → dict
    """

    def __init__(self, grid):
        self._columns = grid.columns
        self._n_cols = len(self._columns)

        feh = grid.feh_values.astype(np.float64)
        ages = grid.age_values.astype(np.float64)
        eeps = grid.eep_values.astype(np.float64)
        data = grid._data.astype(np.float64)

        # Pad length-1 axes (trilinear needs at least 2 points per axis)
        if len(feh) == 1:
            feh = np.array([feh[0] - 0.01, feh[0] + 0.01])
            data = np.concatenate([data, data], axis=0)
        if len(ages) == 1:
            ages = np.array([ages[0] - 0.01, ages[0] + 0.01])
            data = np.concatenate([data, data], axis=1)
        if len(eeps) == 1:
            eeps = np.array([eeps[0] - 0.5, eeps[0] + 0.5])
            data = np.concatenate([data, data], axis=2)

        self._feh = np.ascontiguousarray(feh)
        self._ages = np.ascontiguousarray(ages)
        self._eeps = np.ascontiguousarray(eeps)
        self._data = np.ascontiguousarray(data)

        # Warmup JIT
        _trilinear(
            self._data, self._feh, self._ages, self._eeps,
            self._feh[0], self._ages[0], self._eeps[0], self._n_cols,
        )

    def __call__(self, eep, log_age, feh):
        eep = np.asarray(eep, dtype=np.float64)
        log_age = np.asarray(log_age, dtype=np.float64)
        feh = np.asarray(feh, dtype=np.float64)

        scalar = eep.ndim == 0
        if scalar:
            vals = _trilinear(
                self._data, self._feh, self._ages, self._eeps,
                feh.item(), log_age.item(), eep.item(), self._n_cols,
            )
            return {col: float(vals[i]) for i, col in enumerate(self._columns)}
        else:
            vals = _trilinear_batch(
                self._data, self._feh, self._ages, self._eeps,
                feh, log_age, eep, self._n_cols,
            )
            return {col: vals[:, i] for i, col in enumerate(self._columns)}
