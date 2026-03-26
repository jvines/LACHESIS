"""MIST isochrone grid: download, parse, cache."""


import re
from pathlib import Path

import numpy as np


class MISTIsoFile:
    """Parser for a single MIST .iso file (one [Fe/H])."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._columns: list[str] = []
        self._isochrones: list[np.ndarray] = []
        self._log_ages: list[float] = []
        self._version = ""
        self._feh = 0.0
        self._afe = 0.0
        self._vvcrit = 0.0
        self._num_isochrones = 0
        self._parse()

    def _parse(self):
        with open(self._path) as f:
            lines = f.readlines()

        i = 0
        expect_params = False
        # Parse file header
        while i < len(lines) and lines[i].startswith("#"):
            line = lines[i].strip()

            m = re.search(r"MIST version number\s*=\s*([\d.]+)", line)
            if m:
                self._version = m.group(1)

            # The line with "Yinit" labels precedes the values line
            if "Yinit" in line:
                expect_params = True
                i += 1
                continue
            if expect_params:
                vals = line.lstrip("#").split()
                # Yinit, Zinit, [Fe/H], [a/Fe], v/vcrit
                self._feh = float(vals[2])
                self._afe = float(vals[3])
                self._vvcrit = float(vals[4])
                expect_params = False

            m = re.search(r"number of isochrones\s*=\s*(\d+)", line)
            if m:
                self._num_isochrones = int(m.group(1))

            # First block header appears in the file header
            if line.startswith("# number of EEPs"):
                break

            i += 1

        # Parse all isochrone blocks (including the first one found above)
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("# number of EEPs"):
                m = re.search(r"number of EEPs, cols\s*=\s*(\d+)\s+(\d+)", line)
                n_eeps = int(m.group(1))
                n_cols = int(m.group(2))
                # Skip 2 header lines (column numbers + column names)
                i += 1
                # Column numbers line
                i += 1
                # Column names line
                col_line = lines[i].strip()
                if col_line.startswith("# EEP"):
                    self._columns = col_line[2:].split()
                i += 1
                # Read n_eeps data rows
                block = np.empty((n_eeps, n_cols))
                for j in range(n_eeps):
                    block[j] = [float(x) for x in lines[i].split()]
                    i += 1
                self._isochrones.append(block)
                age_col = self._columns.index("log10_isochrone_age_yr")
                self._log_ages.append(block[0, age_col])
            else:
                i += 1

    @property
    def version(self) -> str:
        return self._version

    @property
    def feh(self) -> float:
        return self._feh

    @property
    def afe(self) -> float:
        return self._afe

    @property
    def vvcrit(self) -> float:
        return self._vvcrit

    @property
    def num_isochrones(self) -> int:
        return self._num_isochrones

    @property
    def log_ages(self) -> np.ndarray:
        return np.array(self._log_ages)

    @property
    def columns(self) -> list[str]:
        return self._columns

    @property
    def all_isochrones(self) -> list[np.ndarray]:
        return self._isochrones

    def get_isochrone(self, log_age: float) -> np.ndarray:
        """Get isochrone data at a specific log(age)."""
        idx = np.argmin(np.abs(self.log_ages - log_age))
        if abs(self.log_ages[idx] - log_age) > 0.001:
            raise ValueError(
                f"No isochrone at log_age={log_age}. "
                f"Nearest is {self.log_ages[idx]}"
            )
        return self._isochrones[idx]


class MISTGrid:
    """Full MIST isochrone grid (all [Fe/H] files → regular 4D array).

    Loads all .iso files from a directory, aligns them onto a common
    EEP axis, and stores as a (n_feh, n_age, n_eep, n_cols) array
    padded with NaN where EEPs don't exist.
    """

    # Columns we keep for interpolation
    _KEEP_COLS = [
        "initial_mass", "star_mass", "log_Teff", "log_g",
        "log_L", "log_R", "phase",
    ]

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.suffix == ".iso":
            # Single file
            self._build_from_files([path])
        elif path.is_dir():
            iso_files = sorted(path.glob("*.iso"))
            if not iso_files:
                raise FileNotFoundError(f"No .iso files in {path}")
            self._build_from_files(iso_files)
        else:
            raise ValueError(f"Expected directory or .iso file, got {path}")

    def _build_from_files(self, iso_files: list[Path]):
        parsed = [MISTIsoFile(f) for f in iso_files]
        self._feh_values = np.array(sorted(p.feh for p in parsed))
        # All files should share the same ages
        self._age_values = parsed[0].log_ages
        self._all_columns = parsed[0].columns

        # Figure out which columns to keep (indices into the full column list)
        self._col_indices = [
            self._all_columns.index(c) for c in self._KEEP_COLS
        ]
        self._columns = list(self._KEEP_COLS)

        # Find global EEP range across all files and ages
        all_eeps = set()
        eep_idx = self._all_columns.index("EEP")
        for p in parsed:
            for iso_data in p.all_isochrones:
                eeps = iso_data[:, eep_idx].astype(int)
                all_eeps.update(eeps)
        self._eep_values = np.array(sorted(all_eeps))
        eep_to_idx = {int(e): i for i, e in enumerate(self._eep_values)}

        n_feh = len(self._feh_values)
        n_age = len(self._age_values)
        n_eep = len(self._eep_values)
        n_cols = len(self._columns)

        # Build 4D array: (feh, age, eep, col) — NaN where data doesn't exist
        self._data = np.full((n_feh, n_age, n_eep, n_cols), np.nan)

        feh_order = np.argsort([p.feh for p in parsed])
        for fi, pi in enumerate(feh_order):
            p = parsed[pi]
            for ai, iso_data in enumerate(p.all_isochrones):
                eeps = iso_data[:, eep_idx].astype(int)
                for row_i, eep_val in enumerate(eeps):
                    ei = eep_to_idx[int(eep_val)]
                    for ci, col_idx in enumerate(self._col_indices):
                        self._data[fi, ai, ei, ci] = iso_data[row_i, col_idx]

    @property
    def name(self) -> str:
        return "MIST"

    @property
    def feh_values(self) -> np.ndarray:
        return self._feh_values

    @property
    def age_values(self) -> np.ndarray:
        return self._age_values

    @property
    def eep_range(self) -> tuple[int, int]:
        return int(self._eep_values[0]), int(self._eep_values[-1])

    @property
    def eep_values(self) -> np.ndarray:
        return self._eep_values

    @property
    def columns(self) -> list[str]:
        return self._columns

    def to_hdf5(self, path: str | Path):
        import h5py

        path = Path(path)
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=self._data)
            f.create_dataset("feh_values", data=self._feh_values)
            f.create_dataset("age_values", data=self._age_values)
            f.create_dataset("eep_values", data=self._eep_values)
            f.attrs["columns"] = self._columns

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "MISTGrid":
        import h5py

        obj = object.__new__(cls)
        with h5py.File(path, "r") as f:
            obj._data = f["data"][:]
            obj._feh_values = f["feh_values"][:]
            obj._age_values = f["age_values"][:]
            obj._eep_values = f["eep_values"][:]
            obj._columns = list(f.attrs["columns"])
        return obj


class MISTModelGrid:
    """Proper model grid built from full MIST isochrones with derived columns.

    16 columns: 11 from raw MIST + 5 derived.
    Dense 4D array: (n_feh, n_age, n_eep, 16), NaN-padded.
    """

    # Raw columns to extract from full isochrones
    _RAW_COLS = [
        "initial_mass", "star_mass", "log_Teff", "log_g",
        "log_L", "log_R", "phase", "delta_nu", "nu_max",
    ]
    # Derived columns computed after loading
    _DERIVED_COLS = ["Teff", "Mbol", "radius", "density", "dm_deep"]
    # eep and age are axes, not stored as columns in the grid —
    # but we keep them for the column list to match isochrones convention
    _AXIS_COLS = ["eep", "age"]

    def __init__(self, path: str | Path):
        from lachesis.grid.derived import (
            compute_density,
            compute_dm_deep,
            compute_mbol,
            compute_radius,
            compute_teff,
        )

        path = Path(path)
        if path.is_dir():
            # Prefer full iso files, fall back to basic
            iso_files = sorted(path.glob("*full*.iso"))
            if not iso_files:
                iso_files = sorted(path.glob("*.iso"))
            if not iso_files:
                raise FileNotFoundError(f"No .iso files in {path}")
        else:
            raise ValueError(f"Expected directory, got {path}")

        parsed = [MISTIsoFile(f) for f in iso_files]
        self._feh_values = np.array(sorted(p.feh for p in parsed))
        self._age_values = parsed[0].log_ages
        all_raw_columns = parsed[0].columns

        # Column indices in the raw file
        raw_col_indices = [all_raw_columns.index(c) for c in self._RAW_COLS]
        eep_idx = all_raw_columns.index("EEP")

        # Find global EEP set
        all_eeps = set()
        for p in parsed:
            for iso_data in p.all_isochrones:
                all_eeps.update(iso_data[:, eep_idx].astype(int))
        self._eep_values = np.array(sorted(all_eeps))
        eep_to_idx = {int(e): i for i, e in enumerate(self._eep_values)}

        n_feh = len(self._feh_values)
        n_age = len(self._age_values)
        n_eep = len(self._eep_values)

        # Full column list: axis cols + raw + derived
        self._columns = self._AXIS_COLS + self._RAW_COLS + self._DERIVED_COLS
        n_cols = len(self._columns)

        self._data = np.full((n_feh, n_age, n_eep, n_cols), np.nan)

        # Column index helpers
        ci_eep = self._columns.index("eep")
        ci_age = self._columns.index("age")
        ci_raw_start = len(self._AXIS_COLS)

        # Fill raw data
        feh_order = np.argsort([p.feh for p in parsed])
        for fi, pi in enumerate(feh_order):
            p = parsed[pi]
            for ai, iso_data in enumerate(p.all_isochrones):
                eeps = iso_data[:, eep_idx].astype(int)
                age_val = self._age_values[ai]
                for row_i, eep_val in enumerate(eeps):
                    ei = eep_to_idx[int(eep_val)]
                    self._data[fi, ai, ei, ci_eep] = eep_val
                    self._data[fi, ai, ei, ci_age] = age_val
                    for rci, src_idx in enumerate(raw_col_indices):
                        self._data[fi, ai, ei, ci_raw_start + rci] = (
                            iso_data[row_i, src_idx]
                        )

        # Compute derived columns
        ci = {c: self._columns.index(c) for c in self._columns}

        self._data[:, :, :, ci["Teff"]] = compute_teff(
            self._data[:, :, :, ci["log_Teff"]]
        )
        self._data[:, :, :, ci["Mbol"]] = compute_mbol(
            self._data[:, :, :, ci["log_L"]]
        )
        self._data[:, :, :, ci["radius"]] = compute_radius(
            self._data[:, :, :, ci["log_R"]]
        )
        self._data[:, :, :, ci["density"]] = compute_density(
            self._data[:, :, :, ci["star_mass"]],
            self._data[:, :, :, ci["radius"]],
        )
        # dm_deep: gradient of initial_mass along EEP axis (axis=2)
        self._data[:, :, :, ci["dm_deep"]] = compute_dm_deep(
            self._data[:, :, :, ci["initial_mass"]], eep_axis=2
        )

    @property
    def name(self) -> str:
        return "MIST"

    @property
    def feh_values(self) -> np.ndarray:
        return self._feh_values

    @property
    def age_values(self) -> np.ndarray:
        return self._age_values

    @property
    def eep_range(self) -> tuple[int, int]:
        return int(self._eep_values[0]), int(self._eep_values[-1])

    @property
    def eep_values(self) -> np.ndarray:
        return self._eep_values

    @property
    def columns(self) -> list[str]:
        return self._columns

    def to_hdf5(self, path: str | Path):
        import h5py

        path = Path(path)
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=self._data, compression="gzip")
            f.create_dataset("feh_values", data=self._feh_values)
            f.create_dataset("age_values", data=self._age_values)
            f.create_dataset("eep_values", data=self._eep_values)
            f.attrs["columns"] = self._columns
            f.attrs["version"] = "1.2"
            f.attrs["grid_name"] = "MIST"

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "MISTModelGrid":
        import h5py

        obj = object.__new__(cls)
        with h5py.File(path, "r") as f:
            obj._data = f["data"][:]
            obj._feh_values = f["feh_values"][:]
            obj._age_values = f["age_values"][:]
            obj._eep_values = f["eep_values"][:]
            obj._columns = list(f.attrs["columns"])
        return obj


def download(dest=None):
    """Download all MIST v1.2 grids from the official website."""
    raise NotImplementedError
