"""Geneva (Ekstroem+ 2012) isochrone grid.

Parses individual isochrone files from the Geneva stellar evolution group.
Only Z=0.014 (solar) precomputed isochrones are available from the server;
other metallicities only have evolutionary tracks.

Each file is a single (Z, age) isochrone. No EEPs are provided, mass-sorted row index is used as a proxy EEP.

Grid shape: (n_feh, n_age, n_eep, n_cols).
"""

import re
from pathlib import Path

import numpy as np

from lachesis.grid.derived import (
    R_SUN,
    compute_density,
    compute_dm_deep,
    compute_mbol,
    compute_teff,
)

_LOG_TSUN = np.log10(5772.0)
_Z_SUN = 0.014

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


def _parse_geneva_file(path: Path) -> dict:
    """Parse a single Geneva isochrone file.

    Returns
    -------
    dict with keys:
        z : float, metal mass fraction
        log_age : float, log10(age/yr) extracted from filename
        data : ndarray, shape (n_rows, 6) with columns:
            [M_ini, M, logL, logTe_c, g_pol, r_pol_cm]
    """
    # Extract Z and log_age from filename
    fname = path.name
    m = re.match(
        r"Isochr_Z([\d.]+)_Vini[\d.]+_t([\d.]+)\.dat", fname
    )
    if not m:
        raise ValueError(f"Cannot parse Geneva filename: {fname}")
    z = float(m.group(1))
    log_age = float(m.group(2))

    text = path.read_text()
    lines = text.splitlines()

    # First line is header, second is blank, then data rows
    header = lines[0].split()
    # Locate column indices
    col_idx = {name: i for i, name in enumerate(header)}

    i_mini = col_idx["M_ini"]
    i_m = col_idx["M"]
    i_logl = col_idx["logL"]
    i_logte = col_idx["logTe_c"]
    i_gpol = col_idx["g_pol"]
    i_rpol = col_idx["r_pol"]
    i_mbol = col_idx["MBol"]

    rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < len(header):
            continue
        try:
            m_ini = float(parts[i_mini])
        except ValueError:
            continue

        # Clip to M_ini <= 5 Msun
        if m_ini > 5.0:
            continue

        rows.append([
            m_ini,
            float(parts[i_m]),
            float(parts[i_logl]),
            float(parts[i_logte]),
            float(parts[i_gpol]),
            float(parts[i_rpol]),
            float(parts[i_mbol]),
        ])

    return {
        "z": z,
        "log_age": log_age,
        "data": np.array(rows) if rows else np.empty((0, 7)),
    }


class GenevaModelGrid:
    """Geneva (Ekstroem+ 2012) model grid.

    Grid shape: (n_feh, n_age, n_eep, n_cols), same interface as
    Dartmouth/MIST/PARSEC.

    Since Geneva provides no EEPs, mass-sorted row index is used.
    """

    _COLUMNS = _COLUMNS

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        dat_files = sorted(directory.glob("Isochr_Z*_Vini*_t*.dat"))
        if not dat_files:
            raise FileNotFoundError(
                f"No Geneva isochrone .dat files in {directory}"
            )

        # Parse all files
        parsed = [_parse_geneva_file(f) for f in dat_files]

        # Collect axes
        # Convert Z to [Fe/H]
        z_set = sorted({p["z"] for p in parsed})
        feh_set = [np.log10(z / _Z_SUN) for z in z_set]

        age_set = sorted({round(p["log_age"], 4) for p in parsed})

        # Find max number of rows (after mass clipping) across all files
        max_rows = max(len(p["data"]) for p in parsed)

        self._feh_values = np.array(feh_set)
        self._age_values = np.array(age_set)
        # EEP = 0-based row index (mass-sorted)
        self._eep_values = np.arange(max_rows, dtype=float)

        z_to_feh_idx = {
            round(z, 6): i for i, z in enumerate(z_set)
        }
        age_to_idx = {
            round(a, 4): i for i, a in enumerate(self._age_values)
        }

        n_feh = len(self._feh_values)
        n_age = len(self._age_values)
        n_eep = len(self._eep_values)
        n_cols = len(self._COLUMNS)

        self._data = np.full((n_feh, n_age, n_eep, n_cols), np.nan)
        ci = {c: i for i, c in enumerate(self._COLUMNS)}

        for p in parsed:
            fi = z_to_feh_idx.get(round(p["z"], 6))
            ai = age_to_idx.get(round(p["log_age"], 4))
            if fi is None or ai is None:
                continue

            data = p["data"]
            if len(data) == 0:
                continue

            # Data is already mass-sorted from the file; assign EEP = row idx
            for ei, row in enumerate(data):
                m_ini, m_cur, log_l, log_te, g_pol, r_pol_cm, mbol = row
                self._data[fi, ai, ei, ci["eep"]] = ei
                self._data[fi, ai, ei, ci["age"]] = p["log_age"]
                self._data[fi, ai, ei, ci["initial_mass"]] = m_ini
                self._data[fi, ai, ei, ci["star_mass"]] = m_cur
                self._data[fi, ai, ei, ci["log_Teff"]] = log_te
                self._data[fi, ai, ei, ci["log_g"]] = g_pol
                self._data[fi, ai, ei, ci["log_L"]] = log_l
                self._data[fi, ai, ei, ci["Mbol"]] = mbol
                self._data[fi, ai, ei, ci["phase"]] = np.nan

                # log_R from r_pol (in cm) -> Rsun
                r_rsun = r_pol_cm / R_SUN
                if r_rsun > 0:
                    self._data[fi, ai, ei, ci["log_R"]] = np.log10(
                        r_rsun
                    )
                else:
                    # Fallback: Stefan-Boltzmann
                    self._data[fi, ai, ei, ci["log_R"]] = (
                        0.5 * log_l + 2.0 * (_LOG_TSUN - log_te)
                    )

        # Derived columns
        self._data[:, :, :, ci["Teff"]] = compute_teff(
            self._data[:, :, :, ci["log_Teff"]]
        )
        # Mbol already set from file; overwrite NaN entries from derived
        nan_mbol = np.isnan(self._data[:, :, :, ci["Mbol"]])
        self._data[:, :, :, ci["Mbol"]] = np.where(
            nan_mbol,
            compute_mbol(self._data[:, :, :, ci["log_L"]]),
            self._data[:, :, :, ci["Mbol"]],
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
        return "Geneva"

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
        """Geneva's full range, row-index EEPs cover the whole isochrone."""
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
            f.attrs["grid_name"] = "Geneva"

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "GenevaModelGrid":
        from lachesis.grid.base import load_grid_hdf5

        obj = object.__new__(cls)
        obj._data, obj._feh_values, obj._age_values, obj._eep_values, obj._COLUMNS = load_grid_hdf5(path)
        return obj
