"""3D grid interpolation over isochrone grids."""


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
    """Interpolates an IsochroneGrid in (EEP, log_age, [Fe/H]) space.

    Builds one scipy RegularGridInterpolator per output column.
    Out-of-bounds and NaN regions return NaN.
    Supports both scalar and vectorized calls.
    """

    def __init__(self, grid):
        self._grid = grid
        self._columns = grid.columns

        # Grid axes (must be 1D, strictly ascending)
        feh = grid.feh_values
        ages = grid.age_values
        eeps = grid.eep_values.astype(float)

        # Build one interpolator per column
        # Data shape: (n_feh, n_age, n_eep, n_cols)
        self._interpolators = {}
        for ci, col in enumerate(self._columns):
            # Extract 3D slice: (n_feh, n_age, n_eep)
            data_3d = grid._data[:, :, :, ci]
            self._interpolators[col] = RegularGridInterpolator(
                (feh, ages, eeps),
                data_3d,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

    def __call__(
        self,
        eep: float | np.ndarray,
        log_age: float | np.ndarray,
        feh: float | np.ndarray,
    ) -> dict[str, float | np.ndarray]:
        """Interpolate at (EEP, log_age, [Fe/H]).

        Accepts scalars or arrays. Returns dict of column → value(s).
        """
        eep = np.asarray(eep, dtype=float)
        log_age = np.asarray(log_age, dtype=float)
        feh = np.asarray(feh, dtype=float)

        scalar = eep.ndim == 0
        if scalar:
            eep = eep.reshape(1)
            log_age = log_age.reshape(1)
            feh = feh.reshape(1)

        # RegularGridInterpolator expects (n_points, n_dims)
        # Axes order in the interpolator: (feh, age, eep)
        points = np.column_stack([feh, log_age, eep])

        result = {}
        for col, interp in self._interpolators.items():
            vals = interp(points)
            result[col] = float(vals[0]) if scalar else vals

        return result
