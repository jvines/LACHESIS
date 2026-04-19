"""Dartmouth (DSEP) isochrone grid.

Parses .iso files from the Dartmouth Stellar Evolution Database.
Dartmouth outputs EEPs natively, so no phase-to-EEP translation is needed
(unlike PARSEC). Grid shape: (n_feh, n_age, n_eep, n_cols).
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


def _parse_iso_file(path: Path) -> dict:
    """Parse a single Dartmouth .iso file (one [Fe/H]).

    Returns
    -------
    dict with keys:
        feh : float
        afe : float
        isochrones : list of (age_gyr, ndarray) pairs
            Each ndarray has columns [EEP, mass, log_Teff, log_g, log_L].
    """
    text = path.read_text()
    lines = text.splitlines()

    # Extract metadata from header
    feh = 0.0
    afe = 0.0
    for line in lines:
        if line.startswith("#MIX-LEN") or line.startswith("#EEP") or line.startswith("#AGE"):
            continue
        if line.startswith("#") and "[Fe/H]" not in line and "[a/Fe]" not in line:
            # Check if this is the metadata line (starts with # followed by numbers)
            stripped = line.lstrip("# ")
            parts = stripped.split()
            if len(parts) >= 6:
                try:
                    float(parts[0])  # MIX-LEN
                    feh = float(parts[4])
                    afe = float(parts[5])
                    break
                except (ValueError, IndexError):
                    pass

    # Split into age blocks
    isochrones = []
    age_pattern = re.compile(r"#AGE=\s*([\d.]+)\s+EEPS=\s*(\d+)")

    i = 0
    while i < len(lines):
        m = age_pattern.match(lines[i])
        if m:
            age_gyr = float(m.group(1))
            n_eeps = int(m.group(2))
            i += 1  # skip #AGE line

            # Skip column header line(s)
            while i < len(lines) and lines[i].startswith("#"):
                i += 1

            # Read data rows
            rows = []
            while i < len(lines) and not lines[i].startswith("#") and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= 5:
                    eep = int(parts[0])
                    mass = float(parts[1])
                    log_teff = float(parts[2])
                    log_g = float(parts[3])
                    log_l = float(parts[4])
                    rows.append([eep, mass, log_teff, log_g, log_l])
                i += 1

            if rows:
                isochrones.append((age_gyr, np.array(rows)))
        else:
            i += 1

    return {"feh": feh, "afe": afe, "isochrones": isochrones}


class DartmouthModelGrid:
    """Dartmouth (DSEP) model grid.

    Grid shape: (n_feh, n_age, n_eep, n_cols) — same interface as MIST/PARSEC.
    """

    _COLUMNS = _COLUMNS

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        iso_files = sorted(directory.glob("dartmouth_feh*.iso"))
        if not iso_files:
            raise FileNotFoundError(f"No Dartmouth .iso files in {directory}")

        # Parse all files
        parsed = [_parse_iso_file(f) for f in iso_files]

        # Collect axes
        feh_set = sorted({p["feh"] for p in parsed})
        age_set = set()
        eep_set = set()
        for p in parsed:
            for age_gyr, data in p["isochrones"]:
                log_age = np.log10(age_gyr * 1e9)
                age_set.add(round(log_age, 4))
                eep_set.update(data[:, 0].astype(int))

        self._feh_values = np.array(feh_set)
        self._age_values = np.array(sorted(age_set))
        self._eep_values = np.arange(
            int(min(eep_set)), int(max(eep_set)) + 1, dtype=float
        )

        feh_to_idx = {round(f, 4): i for i, f in enumerate(self._feh_values)}
        age_to_idx = {round(a, 4): i for i, a in enumerate(self._age_values)}
        eep_to_idx = {int(e): i for i, e in enumerate(self._eep_values)}

        n_feh = len(self._feh_values)
        n_age = len(self._age_values)
        n_eep = len(self._eep_values)
        n_cols = len(self._COLUMNS)

        self._data = np.full((n_feh, n_age, n_eep, n_cols), np.nan)
        ci = {c: i for i, c in enumerate(self._COLUMNS)}

        for p in parsed:
            fi = feh_to_idx.get(round(p["feh"], 4))
            if fi is None:
                continue

            for age_gyr, data in p["isochrones"]:
                log_age = round(np.log10(age_gyr * 1e9), 4)
                ai = age_to_idx.get(log_age)
                if ai is None:
                    continue

                for row in data:
                    eep_int = int(row[0])
                    ei = eep_to_idx.get(eep_int)
                    if ei is None:
                        continue

                    self._data[fi, ai, ei, ci["eep"]] = eep_int
                    self._data[fi, ai, ei, ci["age"]] = log_age
                    # Dartmouth M/Mo is the current mass on the isochrone,
                    # which is also the initial mass (isochrone = single age).
                    self._data[fi, ai, ei, ci["initial_mass"]] = row[1]
                    self._data[fi, ai, ei, ci["star_mass"]] = row[1]
                    self._data[fi, ai, ei, ci["log_Teff"]] = row[2]
                    self._data[fi, ai, ei, ci["log_g"]] = row[3]
                    self._data[fi, ai, ei, ci["log_L"]] = row[4]
                    self._data[fi, ai, ei, ci["phase"]] = np.nan

        # Compute log_R from Stefan-Boltzmann: L = 4π R² σ T⁴
        log_l = self._data[:, :, :, ci["log_L"]]
        log_te = self._data[:, :, :, ci["log_Teff"]]
        self._data[:, :, :, ci["log_R"]] = (
            0.5 * log_l + 2.0 * (_LOG_TSUN - log_te)
        )

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
        return "Dartmouth"

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
        """Dartmouth's full range — its EEPs are already compact (2-279)."""
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
            f.attrs["grid_name"] = "Dartmouth"

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "DartmouthModelGrid":
        from lachesis.grid.base import load_grid_hdf5

        obj = object.__new__(cls)
        obj._data, obj._feh_values, obj._age_values, obj._eep_values, obj._COLUMNS = load_grid_hdf5(path)
        return obj
