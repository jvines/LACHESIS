"""Grid interpolation over isochrone grids (3D or 4D with rotation)."""


import numpy as np
from scipy.interpolate import RegularGridInterpolator


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

        return result
