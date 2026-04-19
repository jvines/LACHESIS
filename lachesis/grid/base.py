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


def load_grid_hdf5(path):
    """Load a grid HDF5 file, handling both dense and sparse_v1 formats.

    Returns (data, feh_values, age_values, eep_values, columns).
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

    return data, feh_values, vini_values, age_values, eep_values, columns
