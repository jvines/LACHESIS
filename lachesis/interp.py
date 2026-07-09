"""Grid interpolation over isochrone grids (3D or 4D with rotation)."""


import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from lachesis.grid.derived import compute_density, M_SUN, R_SUN

_DENSITY_K = M_SUN / (4.0 / 3.0 * math.pi * R_SUN ** 3)


def _recompute_linear_from_logs(result):
    """Replace linearly-interpolated radius/density with values recomputed
    from the log-quantities in the same result.

    Background: regridded grids (notably PARSEC) can place wildly different
    evolutionary states at the same EEP across (feh, age) — e.g. EEP=322
    is end-of-MS for one isochrone but RGB tip for another, so the
    interpolation cube's `radius` corners span 0.1–130 R☉. Linear
    interpolation across that cube produces nonsense (e.g. R=8 R☉ for a
    main-sequence query). The log-quantities (`log_R`) interpolate cleanly
    because they're well-behaved across the same cube.

    The fix: trust `log_R` as the radius source-of-truth and recompute
    linear `radius` (and `density`, which depends on radius) post-interp.
    Mass, log_g, log_L, log_Teff are unaffected — they don't span orders
    of magnitude across the cube. Mbol = -2.5*log_L + const is a linear
    function of log_L, so it's also unaffected. Teff = 10**log_Teff
    *could* show the same pathology in principle, but log_Teff varies by
    less than ±0.3 dex across cube corners in practice, so the linear vs
    log discrepancy is below the BC-table grid resolution and we leave
    `Teff` interpolated as-is for likelihood compatibility.
    """
    if "log_R" not in result or "radius" not in result:
        return result
    logR = result["log_R"]
    is_scalar = np.isscalar(logR) or (hasattr(logR, "ndim") and logR.ndim == 0)
    if is_scalar:
        # Pure-Python scalar path (the nested-sampling inner loop): avoids
        # np.asarray / compute_density array machinery on every call.
        radius = 10.0 ** float(logR)
        result["radius"] = radius
        if "initial_mass" in result and "density" in result:
            mass = float(result["initial_mass"])
            result["density"] = _DENSITY_K * mass / (radius ** 3) if radius > 0 else float("nan")
        return result
    radius = 10.0 ** np.asarray(logR)
    result["radius"] = radius
    if "initial_mass" in result and "density" in result:
        result["density"] = compute_density(
            np.asarray(result["initial_mass"]), np.asarray(radius)
        )
    return result


def make_interpolator(grid):
    """Factory: returns NumbaGridInterpolator if numba is available, else scipy."""
    try:
        from lachesis.interp_numba import NumbaGridInterpolator
        return NumbaGridInterpolator(grid)
    except ImportError:
        return GridInterpolator(grid)


class GridInterpolator:
    """Interpolates an IsochroneGrid in (EEP, log_age, [Fe/H]) space,
    or (EEP, log_age, [Fe/H], Vini) for grids with a rotation axis.

    Builds one scipy RegularGridInterpolator per output column.
    """

    def __init__(self, grid):
        self._grid = grid
        self._columns = grid.columns
        self._has_vini = hasattr(grid, "vini_values") and grid.vini_values is not None

        feh = grid.feh_values
        ages = grid.age_values
        eeps = grid.eep_values.astype(float)

        self._interpolators = {}

        if self._has_vini:
            vini = grid.vini_values
            for ci, col in enumerate(self._columns):
                data_4d = grid._data[:, :, :, :, ci]
                self._interpolators[col] = RegularGridInterpolator(
                    (feh, vini, ages, eeps),
                    data_4d,
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
        else:
            for ci, col in enumerate(self._columns):
                data_3d = grid._data[:, :, :, ci]
                self._interpolators[col] = RegularGridInterpolator(
                    (feh, ages, eeps),
                    data_3d,
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )

    def __call__(self, eep, log_age, feh, vini=None):
        """Interpolate at (EEP, log_age, [Fe/H], [Vini])."""
        eep = np.asarray(eep, dtype=float)
        log_age = np.asarray(log_age, dtype=float)
        feh = np.asarray(feh, dtype=float)

        scalar = eep.ndim == 0
        if scalar:
            eep = eep.reshape(1)
            log_age = log_age.reshape(1)
            feh = feh.reshape(1)

        if self._has_vini:
            if vini is None:
                raise ValueError("Grid has rotation axis — vini is required")
            vini = np.asarray(vini, dtype=float)
            if scalar:
                vini = vini.reshape(1)
            points = np.column_stack([feh, vini, log_age, eep])
        else:
            points = np.column_stack([feh, log_age, eep])

        result = {}
        for col, interp in self._interpolators.items():
            vals = interp(points)
            result[col] = float(vals[0]) if scalar else vals

        return _recompute_linear_from_logs(result)
