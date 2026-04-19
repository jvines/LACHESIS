"""BHAC15 (Baraffe+ 2015) isochrone grid.

Parses BHAC15_iso.* files from the Baraffe et al. (2015) models.
BHAC15 is solar metallicity only, covering 0.01-1.4 Msun and 0.5 Myr - 10 Gyr.
Mass-sorted row index serves as the EEP coordinate (same as BaSTI/YAPSI/Geneva).

Raw file columns per age block:
    M/Ms  Teff  L/Ls(log)  g(log)  R/Rs  Li/Li0  [photometry...]

Grid shape: (n_feh, n_age, n_eep, n_cols).
"""

import re
from pathlib import Path

import numpy as np

from lachesis.grid.derived import (
    compute_density,
    compute_dm_deep,
    compute_mbol,
    compute_teff,
)

_LOG_TSUN = np.log10(5772.0)

_COLUMNS = [
    "eep",
    "age",
    "initial_mass",
    "star_mass",
    "log_Teff",
    "log_g",
    "log_L",
    "log_R",
    "phase",
    "Teff",
    "Mbol",
    "radius",
    "density",
    "dm_deep",
]


def _parse_bhac15_file(path: Path) -> list[tuple[float, np.ndarray]]:
    """Parse a BHAC15 isochrone file.

    Returns
    -------
    list of (age_gyr, ndarray) pairs
        Each ndarray has columns [M/Ms, Teff, log_L, log_g, R/Rs].
    """
    text = path.read_text()
    lines = text.splitlines()

    age_pattern = re.compile(r"!\s+t \(Gyr\)\s*=\s*([\d.]+)")
    isochrones: list[tuple[float, np.ndarray]] = []

    i = 0
    while i < len(lines):
        m = age_pattern.match(lines[i])
        if m:
            age_gyr = float(m.group(1))
            i += 1

            # Skip separator and header lines
            while i < len(lines) and lines[i].startswith("!"):
                i += 1

            # Read data rows
            rows = []
            while i < len(lines) and not lines[i].startswith("!") and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= 5:
                    mass = float(parts[0])
                    teff = float(parts[1].rstrip("."))
                    log_l = float(parts[2])
                    log_g = float(parts[3])
                    r_rsun = float(parts[4])
                    rows.append([mass, teff, log_l, log_g, r_rsun])
                i += 1

            if rows:
                isochrones.append((age_gyr, np.array(rows)))
        else:
            i += 1

    return isochrones


class BHAC15ModelGrid:
    """BHAC15 (Baraffe+ 2015) model grid.

    Solar metallicity only. Grid shape: (1, n_age, n_eep, n_cols).
    Same interface as DartmouthModelGrid / BaSTIModelGrid.
    """

    _COLUMNS = _COLUMNS

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        # Accept any BHAC15_iso.* file
        iso_files = sorted(directory.glob("BHAC15_iso.*"))
        if not iso_files:
            raise FileNotFoundError(f"No BHAC15_iso.* files in {directory}")

        # Parse the first available file (fundamental params are identical
        # across photometric-system variants)
        isochrones = _parse_bhac15_file(iso_files[0])

        if not isochrones:
            raise FileNotFoundError(
                f"No isochrone blocks found in {iso_files[0]}"
            )

        # Solar metallicity only
        self._feh_values = np.array([0.0])

        # Collect age axis
        age_set = set()
        max_rows = 0
        for age_gyr, data in isochrones:
            log_age = np.log10(age_gyr * 1e9)
            age_set.add(round(log_age, 4))
            max_rows = max(max_rows, len(data))

        self._age_values = np.array(sorted(age_set))
        # EEPs are simply 0, 1, 2, ..., max_rows-1
        self._eep_values = np.arange(max_rows, dtype=float)

        age_to_idx = {round(a, 4): i for i, a in enumerate(self._age_values)}

        n_feh = 1
        n_age = len(self._age_values)
        n_eep = len(self._eep_values)
        n_cols = len(self._COLUMNS)

        self._data = np.full((n_feh, n_age, n_eep, n_cols), np.nan)
        ci = {c: i for i, c in enumerate(self._COLUMNS)}

        for age_gyr, data in isochrones:
            log_age = round(np.log10(age_gyr * 1e9), 4)
            ai = age_to_idx.get(log_age)
            if ai is None:
                continue

            n_rows = len(data)
            eep_idx = np.arange(n_rows)

            self._data[0, ai, :n_rows, ci["eep"]] = eep_idx
            self._data[0, ai, :n_rows, ci["age"]] = log_age
            # BHAC15: mass is both initial and current (isochrone)
            self._data[0, ai, :n_rows, ci["initial_mass"]] = data[:, 0]
            self._data[0, ai, :n_rows, ci["star_mass"]] = data[:, 0]
            # Teff is linear in the file -> store as log
            self._data[0, ai, :n_rows, ci["log_Teff"]] = np.log10(data[:, 1])
            # log_g is already log
            self._data[0, ai, :n_rows, ci["log_g"]] = data[:, 3]
            # L/Ls is already log
            self._data[0, ai, :n_rows, ci["log_L"]] = data[:, 2]
            # R/Rs is linear -> store as log
            self._data[0, ai, :n_rows, ci["log_R"]] = np.log10(data[:, 4])
            self._data[0, ai, :n_rows, ci["phase"]] = np.nan

        # Derived columns
        self._data[:, :, :, ci["Teff"]] = compute_teff(
            self._data[:, :, :, ci["log_Teff"]]
        )
        self._data[:, :, :, ci["Mbol"]] = compute_mbol(
            self._data[:, :, :, ci["log_L"]]
        )
        self._data[:, :, :, ci["radius"]] = (
            10.0 ** self._data[:, :, :, ci["log_R"]]
        )
        self._data[:, :, :, ci["density"]] = compute_density(
            self._data[:, :, :, ci["star_mass"]],
            self._data[:, :, :, ci["radius"]],
        )
        self._data[:, :, :, ci["dm_deep"]] = compute_dm_deep(
            self._data[:, :, :, ci["initial_mass"]], eep_axis=2
        )

    @property
    def name(self) -> str:
        return "BHAC15"

    @property
    def feh_values(self) -> np.ndarray:
        return self._feh_values

    @property
    def age_values(self) -> np.ndarray:
        return self._age_values

    @property
    def eep_values(self) -> np.ndarray:
        return self._eep_values

    @property
    def eep_range(self) -> tuple[int, int]:
        return int(self._eep_values[0]), int(self._eep_values[-1])

    @property
    def fitting_eep_range(self) -> tuple[int, int]:
        """Full range -- mass-index EEPs are already compact."""
        return self.eep_range

    @property
    def columns(self) -> list[str]:
        return list(self._COLUMNS)

    def to_hdf5(self, path: str | Path):
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=self._data, compression="gzip")
            f.create_dataset("feh_values", data=self._feh_values)
            f.create_dataset("age_values", data=self._age_values)
            f.create_dataset("eep_values", data=self._eep_values)
            f.attrs["columns"] = self._COLUMNS
            f.attrs["grid_name"] = "BHAC15"

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "BHAC15ModelGrid":
        from lachesis.grid.base import load_grid_hdf5

        obj = object.__new__(cls)
        obj._data, obj._feh_values, obj._age_values, obj._eep_values, obj._COLUMNS = load_grid_hdf5(path)
        return obj
