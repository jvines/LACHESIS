"""BaSTI isochrone grid.

Parses combined .dat files produced by ``basti_download.py``.
BaSTI outputs mass-sorted isochrones without EEPs, so the row index
along each isochrone serves as the EEP coordinate.

Columns in the raw file (per age block):
    log(L/Lo)  logTe  M/Mo(ini)  M/Mo(fin)

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
_LOG_G_SUN = np.log10(2.7427e4)  # log10(g_sun) in cgs

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


def _parse_basti_file(path: Path) -> dict:
    """Parse a single BaSTI combined .dat file (one [Fe/H]).

    Returns
    -------
    dict with keys:
        feh : float  (extracted from filename)
        isochrones : list of (age_gyr, ndarray) pairs
            Each ndarray has columns [log_L, log_Teff, M_ini, M_fin].
    """
    text = path.read_text()
    lines = text.splitlines()

    # Extract [Fe/H] from filename: basti_feh+0.00.dat
    feh_match = re.search(r"feh([+-]?\d+\.\d+)", path.name)
    feh = float(feh_match.group(1)) if feh_match else 0.0

    # Parse age blocks
    age_pattern = re.compile(r"#AGE=(\d+)\s+NPTS=(\d+)")
    isochrones = []

    i = 0
    while i < len(lines):
        m = age_pattern.match(lines[i])
        if m:
            age_myr = int(m.group(1))
            age_gyr = age_myr / 1000.0
            n_pts = int(m.group(2))
            i += 1

            rows = []
            while i < len(lines) and not lines[i].startswith("#") and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= 4:
                    log_l = float(parts[0])
                    log_teff = float(parts[1])
                    m_ini = float(parts[2])
                    m_fin = float(parts[3])
                    rows.append([log_l, log_teff, m_ini, m_fin])
                i += 1

            if rows:
                isochrones.append((age_gyr, np.array(rows)))
        else:
            i += 1

    return {"feh": feh, "isochrones": isochrones}


class BaSTIModelGrid:
    """BaSTI model grid.

    Grid shape: (n_feh, n_age, n_eep, n_cols) -- same interface as
    DartmouthModelGrid / MISTModelGrid.
    """

    _COLUMNS = _COLUMNS

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        dat_files = sorted(directory.glob("basti_feh*.dat"))
        if not dat_files:
            raise FileNotFoundError(f"No BaSTI .dat files in {directory}")

        # Parse all files
        parsed = [_parse_basti_file(f) for f in dat_files]

        # Collect axes
        feh_set = sorted({p["feh"] for p in parsed})
        age_set = set()
        max_rows = 0
        for p in parsed:
            for age_gyr, data in p["isochrones"]:
                log_age = np.log10(age_gyr * 1e9)
                age_set.add(round(log_age, 4))
                max_rows = max(max_rows, len(data))

        self._feh_values = np.array(feh_set)
        self._age_values = np.array(sorted(age_set))
        # EEPs are simply 0, 1, 2, ..., max_rows-1
        self._eep_values = np.arange(max_rows, dtype=float)

        feh_to_idx = {round(f, 4): i for i, f in enumerate(self._feh_values)}
        age_to_idx = {round(a, 4): i for i, a in enumerate(self._age_values)}

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

                n_rows = len(data)
                eep_idx = np.arange(n_rows)

                self._data[fi, ai, :n_rows, ci["eep"]] = eep_idx
                self._data[fi, ai, :n_rows, ci["age"]] = log_age
                self._data[fi, ai, :n_rows, ci["log_L"]] = data[:, 0]
                self._data[fi, ai, :n_rows, ci["log_Teff"]] = data[:, 1]
                self._data[fi, ai, :n_rows, ci["initial_mass"]] = data[:, 2]
                self._data[fi, ai, :n_rows, ci["star_mass"]] = data[:, 3]
                self._data[fi, ai, :n_rows, ci["phase"]] = np.nan

        # Compute log_R from Stefan-Boltzmann: L = 4pi R^2 sigma T^4
        log_l = self._data[:, :, :, ci["log_L"]]
        log_te = self._data[:, :, :, ci["log_Teff"]]
        self._data[:, :, :, ci["log_R"]] = (
            0.5 * log_l + 2.0 * (_LOG_TSUN - log_te)
        )

        # Compute log_g: log_g = log_g_sun + log(M/Msun) - 2*log(R/Rsun)
        log_m = np.log10(
            np.where(
                self._data[:, :, :, ci["star_mass"]] > 0,
                self._data[:, :, :, ci["star_mass"]],
                np.nan,
            )
        )
        log_r = self._data[:, :, :, ci["log_R"]]
        self._data[:, :, :, ci["log_g"]] = (
            _LOG_G_SUN + log_m - 2.0 * log_r
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
        return "BaSTI"

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
            f.attrs["grid_name"] = "BaSTI"

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "BaSTIModelGrid":
        import h5py

        obj = object.__new__(cls)
        with h5py.File(path, "r") as f:
            obj._data = f["data"][:]
            obj._feh_values = f["feh_values"][:]
            obj._age_values = f["age_values"][:]
            obj._eep_values = f["eep_values"][:]
            obj._COLUMNS = list(f.attrs["columns"])
        return obj
