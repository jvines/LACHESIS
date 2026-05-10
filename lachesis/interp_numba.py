"""Numba-JIT trilinear interpolator for isochrone grids.

Drop-in replacement for GridInterpolator with same API but ~10-50x faster
for repeated scalar evaluations (nested sampling inner loop).
"""


import numpy as np
import numba as nb
from numba import prange


@nb.njit(cache=True)
def _searchsorted(arr, val):
    """Binary search returning index of interval containing val.
    Returns i such that arr[i] <= val < arr[i+1].
    Returns -1 if val is non-finite or out of [arr[0], arr[-1]].
    """
    if not np.isfinite(val):
        return -1
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


@nb.njit(cache=True)  # parallel=True removed — numba TBB pool races with dynesty teardown across grids
def _trilinear_batch(grid, ax0, ax1, ax2, x0s, x1s, x2s, n_cols):
    """Vectorised trilinear interpolation; parallel over batch rows."""
    n = len(x0s)
    result = np.empty((n, n_cols))
    for i in range(n):
        result[i] = _trilinear(grid, ax0, ax1, ax2, x0s[i], x1s[i], x2s[i], n_cols)
    return result


@nb.njit(cache=True)
def _quadlinear(grid, ax0, ax1, ax2, ax3, x0, x1, x2, x3, n_cols):
    """Quadrilinear interpolation on a regular 4D grid.

    grid: (n0, n1, n2, n3, n_cols) array
    ax0..ax3: 1D arrays of grid coordinates
    x0..x3: query point
    """
    result = np.empty(n_cols)

    i0 = _searchsorted(ax0, x0)
    i1 = _searchsorted(ax1, x1)
    i2 = _searchsorted(ax2, x2)
    i3 = _searchsorted(ax3, x3)

    if (i0 < 0 or i0 >= len(ax0) - 1 or
        i1 < 0 or i1 >= len(ax1) - 1 or
        i2 < 0 or i2 >= len(ax2) - 1 or
        i3 < 0 or i3 >= len(ax3) - 1):
        result[:] = np.nan
        return result

    d0 = (x0 - ax0[i0]) / (ax0[i0 + 1] - ax0[i0])
    d1 = (x1 - ax1[i1]) / (ax1[i1 + 1] - ax1[i1])
    d2 = (x2 - ax2[i2]) / (ax2[i2 + 1] - ax2[i2])
    d3 = (x3 - ax3[i3]) / (ax3[i3 + 1] - ax3[i3])

    for c in range(n_cols):
        val = 0.0
        has_nan = False
        for di0 in range(2):
            for di1 in range(2):
                for di2 in range(2):
                    for di3 in range(2):
                        w = (
                            (d0 if di0 else 1 - d0)
                            * (d1 if di1 else 1 - d1)
                            * (d2 if di2 else 1 - d2)
                            * (d3 if di3 else 1 - d3)
                        )
                        corner = grid[i0 + di0, i1 + di1, i2 + di2, i3 + di3, c]
                        if w > 0 and np.isnan(corner):
                            has_nan = True
                            break
                        if not np.isnan(corner):
                            val += w * corner
                    if has_nan:
                        break
                if has_nan:
                    break
            if has_nan:
                break
        result[c] = np.nan if has_nan else val

    return result


@nb.njit(cache=True)  # parallel=True removed — numba TBB pool races with dynesty teardown across grids
def _quadlinear_batch(grid, ax0, ax1, ax2, ax3, x0s, x1s, x2s, x3s, n_cols):
    """Vectorised quadrilinear interpolation; parallel over batch rows."""
    n = len(x0s)
    result = np.empty((n, n_cols))
    for i in range(n):
        result[i] = _quadlinear(
            grid, ax0, ax1, ax2, ax3,
            x0s[i], x1s[i], x2s[i], x3s[i], n_cols,
        )
    return result


def _pad_axis(arr, data, axis, half_width: float = 5.0):
    """Pad a length-1 axis to length 2 so the interpolator has an interval.

    The duplicated slice represents a constant-along-axis grid; widening the
    pad keeps queries inside the interpolation interval for any plausible
    coordinate value (single-feh grids would otherwise NaN outside ±0.01).
    Memory is doubled along the affected axis only when padding is needed,
    not unconditionally.
    """
    if len(arr) == 1:
        arr = np.array([arr[0] - half_width, arr[0] + half_width])
        data = np.concatenate([data, data], axis=axis)
    return arr, data


class NumbaGridInterpolator:
    """Drop-in replacement for GridInterpolator using numba JIT.

    Handles both 3D grids (feh, age, eep) and 4D grids with a rotation
    axis (feh, vini, age, eep). Detects the dimensionality from the grid.
    """

    def __init__(self, grid):
        self._columns = grid.columns
        self._n_cols = len(self._columns)
        self._has_vini = hasattr(grid, "vini_values") and grid.vini_values is not None

        feh = grid.feh_values.astype(np.float64)
        ages = grid.age_values.astype(np.float64)
        eeps = grid.eep_values.astype(np.float64)
        data = grid._data.astype(np.float64)

        if self._has_vini:
            # 5D data: (n_feh, n_vini, n_age, n_eep, n_cols)
            vini = grid.vini_values.astype(np.float64)
            feh, data = _pad_axis(feh, data, 0)
            vini, data = _pad_axis(vini, data, 1)
            ages, data = _pad_axis(ages, data, 2)
            eeps, data = _pad_axis(eeps, data, 3)

            self._vini = np.ascontiguousarray(vini)
            self._feh = np.ascontiguousarray(feh)
            self._ages = np.ascontiguousarray(ages)
            self._eeps = np.ascontiguousarray(eeps)
            self._data = np.ascontiguousarray(data)

            # Warmup 4D JIT
            _quadlinear(
                self._data, self._feh, self._vini, self._ages, self._eeps,
                self._feh[0], self._vini[0], self._ages[0], self._eeps[0],
                self._n_cols,
            )
        else:
            # 4D data: (n_feh, n_age, n_eep, n_cols)
            feh, data = _pad_axis(feh, data, 0)
            ages, data = _pad_axis(ages, data, 1)
            eeps, data = _pad_axis(eeps, data, 2)

            self._feh = np.ascontiguousarray(feh)
            self._ages = np.ascontiguousarray(ages)
            self._eeps = np.ascontiguousarray(eeps)
            self._data = np.ascontiguousarray(data)

            # Warmup 3D JIT
            _trilinear(
                self._data, self._feh, self._ages, self._eeps,
                self._feh[0], self._ages[0], self._eeps[0], self._n_cols,
            )

    def __call__(self, eep, log_age, feh, vini=None):
        eep = np.asarray(eep, dtype=np.float64)
        log_age = np.asarray(log_age, dtype=np.float64)
        feh = np.asarray(feh, dtype=np.float64)

        scalar = eep.ndim == 0

        from lachesis.interp import _recompute_linear_from_logs

        if self._has_vini:
            if vini is None:
                raise ValueError("Grid has rotation axis — vini is required")
            vini = np.asarray(vini, dtype=np.float64)
            if scalar:
                vals = _quadlinear(
                    self._data, self._feh, self._vini, self._ages, self._eeps,
                    feh.item(), vini.item(), log_age.item(), eep.item(),
                    self._n_cols,
                )
                result = {col: float(vals[i]) for i, col in enumerate(self._columns)}
            else:
                vals = _quadlinear_batch(
                    self._data, self._feh, self._vini, self._ages, self._eeps,
                    feh, vini, log_age, eep, self._n_cols,
                )
                result = {col: vals[:, i] for i, col in enumerate(self._columns)}
        else:
            if scalar:
                vals = _trilinear(
                    self._data, self._feh, self._ages, self._eeps,
                    feh.item(), log_age.item(), eep.item(), self._n_cols,
                )
                result = {col: float(vals[i]) for i, col in enumerate(self._columns)}
            else:
                vals = _trilinear_batch(
                    self._data, self._feh, self._ages, self._eeps,
                    feh, log_age, eep, self._n_cols,
                )
                result = {col: vals[:, i] for i, col in enumerate(self._columns)}

        return _recompute_linear_from_logs(result)
