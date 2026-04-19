"""Verify sparse_v1 format roundtrips to identical dense arrays."""
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from lachesis.grid.base import load_grid_hdf5


def _make_dense_grid(n_feh=3, n_age=5, n_eep=100, n_col=4, nan_frac=0.4):
    """Create a synthetic dense grid with contiguous NaN padding."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_feh, n_age, n_eep, n_col)).astype(np.float32)
    for i in range(n_feh):
        for j in range(n_age):
            cut = rng.integers(10, n_eep // 2)
            data[i, j, :cut, :] = np.nan
            data[i, j, n_eep - rng.integers(5, 30) :, :] = np.nan
    return data


def _write_dense(path, data):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("feh_values", data=np.linspace(-1, 0.5, data.shape[0]))
        f.create_dataset("age_values", data=np.linspace(8, 10, data.shape[1]))
        f.create_dataset("eep_values", data=np.arange(data.shape[2]))
        f.attrs["columns"] = [f"col{i}" for i in range(data.shape[3])]


def _write_sparse(path, data):
    shape = data.shape
    n_eep, n_col = shape[-2], shape[-1]
    flat = data.reshape(-1, n_eep, n_col)
    n_slices = flat.shape[0]
    offsets = np.zeros(n_slices, dtype=np.int32)
    lengths = np.zeros(n_slices, dtype=np.int32)
    chunks = []
    for i in range(n_slices):
        valid = ~np.isnan(flat[i, :, 0])
        if not valid.any():
            continue
        first = int(np.argmax(valid))
        last = n_eep - 1 - int(np.argmax(valid[::-1]))
        offsets[i] = first
        lengths[i] = last - first + 1
        chunks.append(flat[i, first : last + 1, :])
    values = np.concatenate(chunks) if chunks else np.empty((0, n_col), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("values", data=values, compression="gzip", compression_opts=9, shuffle=True)
        f.create_dataset("offsets", data=offsets)
        f.create_dataset("lengths", data=lengths)
        f.create_dataset("feh_values", data=np.linspace(-1, 0.5, data.shape[0]))
        f.create_dataset("age_values", data=np.linspace(8, 10, data.shape[1]))
        f.create_dataset("eep_values", data=np.arange(data.shape[2]))
        f.attrs["shape"] = list(shape)
        f.attrs["storage"] = "sparse_v1"
        f.attrs["columns"] = [f"col{i}" for i in range(n_col)]


def test_sparse_roundtrip():
    data = _make_dense_grid()
    with tempfile.TemporaryDirectory() as tmp:
        sparse_path = Path(tmp) / "grid_sparse.h5"
        dense_path = Path(tmp) / "grid_dense.h5"

        _write_sparse(sparse_path, data)
        _write_dense(dense_path, data)

        sparse_data, *_ = load_grid_hdf5(sparse_path)
        dense_data, *_ = load_grid_hdf5(dense_path)

        np.testing.assert_array_equal(sparse_data, dense_data)
        np.testing.assert_array_equal(sparse_data, data)


def test_dense_format_still_works():
    data = _make_dense_grid()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grid.h5"
        _write_dense(path, data)
        loaded, *_ = load_grid_hdf5(path)
        np.testing.assert_array_equal(loaded, data)
