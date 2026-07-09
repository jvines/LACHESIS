"""Tests for the grid protocol."""

from lachesis.grid.base import IsochroneGrid


def test_protocol_exists():
    assert IsochroneGrid is not None


def test_load_grid_hdf5_sorts_descending_feh(tmp_path):
    """lachesis-grids 0.0.3 ships the PARSEC eeprebuild cube with a
    DESCENDING [Fe/H] axis; loading must sort the axis and permute the data
    along dim 0 instead of raising."""
    import h5py
    import numpy as np
    from lachesis.grid.base import load_grid_hdf5

    feh = np.array([0.5, 0.0, -0.5])           # descending-ish (unsorted)
    age = np.array([1.0, 2.0])
    eep = np.array([100.0, 200.0])
    data = np.zeros((3, 2, 2, 1))
    for i in range(3):
        data[i, ..., 0] = feh[i]               # value tags its feh slice

    p = tmp_path / "toy.h5"
    with h5py.File(p, "w") as f:
        f["feh_values"] = feh
        f["age_values"] = age
        f["eep_values"] = eep
        f["data"] = data
        f.attrs["columns"] = ["dummy"]

    out, feh_s, age_s, eep_s, cols = load_grid_hdf5(p)
    assert list(feh_s) == [-0.5, 0.0, 0.5]
    # data slices must follow their axis values
    assert out[0, 0, 0, 0] == -0.5 and out[2, 0, 0, 0] == 0.5


def test_load_grid_hdf5_still_raises_on_duplicate_axis(tmp_path):
    import h5py
    import numpy as np
    import pytest
    from lachesis.grid.base import load_grid_hdf5

    p = tmp_path / "dup.h5"
    with h5py.File(p, "w") as f:
        f["feh_values"] = np.array([0.0, 0.0, 0.5])
        f["age_values"] = np.array([1.0, 2.0])
        f["eep_values"] = np.array([100.0, 200.0])
        f["data"] = np.zeros((3, 2, 2, 1))
        f.attrs["columns"] = ["dummy"]
    with pytest.raises(ValueError, match="not strictly increasing"):
        load_grid_hdf5(p)
