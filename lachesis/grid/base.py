"""Abstract grid interface — every isochrone grid implements this."""

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class IsochroneGrid(Protocol):
    """Protocol for isochrone grids (MIST, PARSEC, Dartmouth, ...)."""

    @property
    def name(self) -> str: ...

    @property
    def eep_range(self) -> tuple[int, int]: ...

    @property
    def fitting_eep_range(self) -> tuple[int, int]:
        """EEP range for fitting (may exclude PMS/unphysical regions).

        Defaults to eep_range. Override per grid if the full range
        includes regions that shouldn't be sampled (e.g., PMS in MIST).
        """
        ...

    @property
    def feh_values(self) -> NDArray[np.float64]: ...

    @property
    def age_values(self) -> NDArray[np.float64]: ...

    @property
    def columns(self) -> list[str]: ...

    def __call__(
        self, eep: float, log_age: float, feh: float
    ) -> dict[str, float]:
        """Interpolate grid at (EEP, log_age, [Fe/H]) → observables."""
        ...


def _reconstruct_sparse(f) -> np.ndarray:
    """Reconstruct a dense NaN-padded array from sparse_v1 format."""
    shape = tuple(f.attrs["shape"])
    values = f["values"][:]
    offsets = f["offsets"][:]
    lengths = f["lengths"][:]

    n_eep = shape[-2]
    n_col = shape[-1]
    n_slices = len(offsets)
    flat = np.full((n_slices, n_eep, n_col), np.nan, dtype=values.dtype)

    pos = 0
    for i in range(n_slices):
        length = int(lengths[i])
        if length == 0:
            continue
        offset = int(offsets[i])
        flat[i, offset : offset + length, :] = values[pos : pos + length]
        pos += length

    return flat.reshape(shape)


def _mask_sentinel_plateaus(data, m_idx, eep_axis, min_run=5):
    """NaN-mask cells that are obviously sentinel-filled by the regrid.

    PARSEC's shipped HDF5 has many (Fe/H, age) slices where the EEP axis is
    padded out with copies of the last valid mass value — long ``min_run+``
    runs of identical mass that aren't physical evolutionary tracks. They
    would otherwise contaminate the prior volume (zero |dM/dEEP| from the
    plateau, but inside the grid box, so the per-grid evidence integral
    treats them as truncated rather than as out-of-grid). Masking them as
    NaN makes the rejection consistent with how MIST/Dartmouth/etc. handle
    their unphysical extents.

    A plateau is flagged as sentinel only if it extends to the trailing
    NaN/edge or is itself ≥ ``min_run`` cells long. Short plateaus (≤
    ``min_run-1``) are kept and handled by ``compute_dm_deep``'s linear
    interpolation across the plateau.
    """
    arr = np.moveaxis(data[..., m_idx], eep_axis, -1)
    *batch_shape, n_eep = arr.shape
    flat = arr.reshape(-1, n_eep)

    for b in range(flat.shape[0]):
        m = flat[b]
        # Walk EEP axis; identify runs of equal-finite values
        i = 0
        while i < n_eep:
            if not np.isfinite(m[i]):
                i += 1
                continue
            j = i + 1
            while j < n_eep and np.isfinite(m[j]) and m[j] == m[i]:
                j += 1
            run = j - i
            # j is now first index whose mass differs OR is NaN OR off-end.
            tail_is_unphysical = (j == n_eep) or not np.isfinite(m[j])
            if run >= min_run and tail_is_unphysical:
                # Sentinel: mass plateau that extends to the end of the
                # finite block. Mask the whole run as NaN.
                m[i:j] = np.nan
            i = j

    # `flat` is a view into `arr` which is a view via moveaxis; assign back.
    data[..., m_idx] = np.moveaxis(arr, -1, eep_axis)


def _refresh_dm_deep_inplace(data, columns, eep_axis):
    """Recompute the dm_deep column on load (after sentinel masking).

    Shipped HDF5 grids have dm_deep baked in from the original
    ``np.gradient`` implementation, which returns 0 across constant-mass
    plateaus that arise from regrid artifacts (notably ~37% of PARSEC's MS
    cells). We:

    1. NaN-mask sentinel-filled trailing plateaus in initial_mass — these
       are unphysical fills (e.g., 261.66 Msun stretched over hundreds of
       EEPs in PARSEC), and treating them as out-of-grid is more honest
       than treating them as in-grid-but-zero-gradient.
    2. Recompute dm_deep with the patched ``compute_dm_deep`` that
       linearly interpolates across short (real) regrid plateaus.

    This way the shipped HDF5 files don't need to be rebuilt.
    """
    if "dm_deep" not in columns or "initial_mass" not in columns:
        return
    m_idx = columns.index("initial_mass")
    _mask_sentinel_plateaus(data, m_idx, eep_axis)

    from lachesis.grid.derived import compute_dm_deep
    dm_idx = columns.index("dm_deep")
    data[..., dm_idx] = compute_dm_deep(data[..., m_idx], eep_axis=eep_axis)


def load_grid_hdf5(path):
    """Load a grid HDF5 file, handling both dense and sparse_v1 formats.

    Returns (data, feh_values, age_values, eep_values, columns).

    dm_deep is recomputed from initial_mass after load so any fix to
    ``compute_dm_deep`` takes effect without rebuilding the HDF5 file.
    """
    import h5py
    from pathlib import Path

    with h5py.File(Path(path), "r") as f:
        feh_values = f["feh_values"][:]
        age_values = f["age_values"][:]
        eep_values = f["eep_values"][:]
        columns = list(f.attrs.get("columns", []))

        storage = f.attrs.get("storage", "dense")
        if storage == "sparse_v1":
            data = _reconstruct_sparse(f)
        else:
            data = f["data"][:]

    # Layout (n_feh, n_age, n_eep, n_cols) -> EEP axis = 2
    _refresh_dm_deep_inplace(data, columns, eep_axis=2)
    return data, feh_values, age_values, eep_values, columns


def load_grid_hdf5_starevol(path):
    """Like load_grid_hdf5 but also returns vini_values for STAREVOL."""
    import h5py
    from pathlib import Path

    with h5py.File(Path(path), "r") as f:
        feh_values = f["feh_values"][:]
        vini_values = f["vini_values"][:]
        age_values = f["age_values"][:]
        eep_values = f["eep_values"][:]
        columns = list(f.attrs.get("columns", []))

        storage = f.attrs.get("storage", "dense")
        if storage == "sparse_v1":
            data = _reconstruct_sparse(f)
        else:
            data = f["data"][:]

    # Layout (n_feh, n_vini, n_age, n_eep, n_cols) -> EEP axis = 3
    _refresh_dm_deep_inplace(data, columns, eep_axis=3)
    return data, feh_values, vini_values, age_values, eep_values, columns
