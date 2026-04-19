"""STAREVOL (Amard+ 2019) isochrone grid with rotation.

Parses precomputed isochrone files from CDS (J/A+A/631/A77).
Each file is a single (Z, Vini, age) isochrone. Rotation rate (Vini)
is a fitted parameter. Mass-sorted row index serves as EEP.

Grid shape: (n_feh, n_vini, n_age, n_eep, n_cols) -- 5D with rotation axis.
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

_Z_SUN = 0.0134  # Amard+ 2019 solar composition (Asplund+ 2009)
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


def _parse_starevol_file(path: Path) -> dict:
    """Parse a single STAREVOL isochrone file.

    Returns dict with z, vini, log_age, data (n_rows x 7).
    """
    fname = path.name
    m = re.match(
        r"Isochr_Z([\d.]+)_Vini([\d.]+)_t([\d.]+)\.dat", fname
    )
    if not m:
        raise ValueError(f"Cannot parse STAREVOL filename: {fname}")
    z = float(m.group(1))
    vini = float(m.group(2))
    log_age = float(m.group(3))

    text = path.read_text()
    lines = text.splitlines()

    header = None
    for line in lines:
        if line.startswith("#"):
            header = line.lstrip("#").split()
            break

    if header is None:
        return {"z": z, "vini": vini, "log_age": log_age, "data": np.empty((0, 7))}

    col_idx = {}
    for i, name in enumerate(header):
        clean = name.replace("\\", "").replace("_", "")
        col_idx[clean] = i
        col_idx[name] = i

    def _find(names):
        for n in names:
            if n in col_idx:
                return col_idx[n]
        return None

    i_mini = _find(["Mini", "M_ini", "M\\_ini"])
    i_m = _find(["M"])
    i_logl = _find(["logL"])
    i_logte = _find(["logTeff"])
    i_logg = _find(["logg"])
    i_r = _find(["R"])
    i_mbol = _find(["Mbol"])

    if i_mini is None or i_logl is None or i_logte is None:
        raise ValueError(f"Missing required columns in {fname}")

    rows = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < max(
            filter(None, [i_mini, i_m, i_logl, i_logte, i_logg, i_r, i_mbol])
        ) + 1:
            continue
        try:
            m_ini = float(parts[i_mini])
        except (ValueError, IndexError):
            continue
        if m_ini > 1.6:
            continue

        m_cur = float(parts[i_m]) if i_m is not None else m_ini
        log_l = float(parts[i_logl])
        log_te = float(parts[i_logte])
        logg = float(parts[i_logg]) if i_logg is not None else np.nan
        radius = float(parts[i_r]) if i_r is not None else np.nan
        mbol = float(parts[i_mbol]) if i_mbol is not None else np.nan

        rows.append([m_ini, m_cur, log_l, log_te, logg, radius, mbol])

    return {
        "z": z,
        "vini": vini,
        "log_age": log_age,
        "data": np.array(rows) if rows else np.empty((0, 7)),
    }


class STAREVOLModelGrid:
    """STAREVOL (Amard+ 2019) model grid with rotation.

    Grid shape: (n_feh, n_vini, n_age, n_eep, n_cols).
    Rotation rate (Vini) is an interpolation axis.
    """

    _COLUMNS = _COLUMNS

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        dat_files = sorted(directory.glob("Isochr_Z*_Vini*_t*.dat"))
        if not dat_files:
            raise FileNotFoundError(
                f"No STAREVOL isochrone .dat files in {directory}"
            )

        parsed = [_parse_starevol_file(f) for f in dat_files]

        # Collect axes
        z_set = sorted({p["z"] for p in parsed})
        vini_set = sorted({round(p["vini"], 2) for p in parsed})
        feh_set = [round(np.log10(z / _Z_SUN), 4) for z in z_set]
        age_set = sorted({round(p["log_age"], 4) for p in parsed})

        max_rows = 0
        for p in parsed:
            if len(p["data"]) > 0:
                max_rows = max(max_rows, len(p["data"]))

        self._feh_values = np.array(feh_set)
        self._vini_values = np.array(vini_set)
        self._age_values = np.array(age_set)
        self._eep_values = np.arange(max_rows, dtype=float)

        z_to_feh_idx = {round(z, 6): i for i, z in enumerate(z_set)}
        vini_to_idx = {round(v, 2): i for i, v in enumerate(vini_set)}
        age_to_idx = {round(a, 4): i for i, a in enumerate(self._age_values)}

        n_feh = len(self._feh_values)
        n_vini = len(self._vini_values)
        n_age = len(self._age_values)
        n_eep = len(self._eep_values)
        n_cols = len(self._COLUMNS)

        self._data = np.full((n_feh, n_vini, n_age, n_eep, n_cols), np.nan)
        ci = {c: i for i, c in enumerate(self._COLUMNS)}

        for p in parsed:
            fi = z_to_feh_idx.get(round(p["z"], 6))
            vi = vini_to_idx.get(round(p["vini"], 2))
            ai = age_to_idx.get(round(p["log_age"], 4))
            if fi is None or vi is None or ai is None:
                continue

            data = p["data"]
            if len(data) == 0:
                continue

            n_rows = len(data)
            eep_idx = np.arange(n_rows)

            self._data[fi, vi, ai, :n_rows, ci["eep"]] = eep_idx
            self._data[fi, vi, ai, :n_rows, ci["age"]] = p["log_age"]
            self._data[fi, vi, ai, :n_rows, ci["initial_mass"]] = data[:, 0]
            self._data[fi, vi, ai, :n_rows, ci["star_mass"]] = data[:, 1]
            self._data[fi, vi, ai, :n_rows, ci["log_L"]] = data[:, 2]
            self._data[fi, vi, ai, :n_rows, ci["log_Teff"]] = data[:, 3]
            self._data[fi, vi, ai, :n_rows, ci["log_g"]] = data[:, 4]
            self._data[fi, vi, ai, :n_rows, ci["phase"]] = np.nan

            # Radius
            radii = data[:, 5]
            valid_r = np.isfinite(radii) & (radii > 0)
            log_r = np.full(n_rows, np.nan)
            log_r[valid_r] = np.log10(radii[valid_r])
            bad = ~valid_r
            if np.any(bad):
                log_r[bad] = 0.5 * data[bad, 2] + 2.0 * (_LOG_TSUN - data[bad, 3])
            self._data[fi, vi, ai, :n_rows, ci["log_R"]] = log_r

            # Mbol
            mbol = data[:, 6]
            valid_mbol = np.isfinite(mbol)
            self._data[fi, vi, ai, :n_rows, ci["Mbol"]] = np.where(
                valid_mbol, mbol, compute_mbol(data[:, 2])
            )

        # Derived columns
        self._data[..., ci["Teff"]] = compute_teff(self._data[..., ci["log_Teff"]])
        nan_mbol = np.isnan(self._data[..., ci["Mbol"]])
        self._data[..., ci["Mbol"]] = np.where(
            nan_mbol,
            compute_mbol(self._data[..., ci["log_L"]]),
            self._data[..., ci["Mbol"]],
        )
        self._data[..., ci["radius"]] = 10.0 ** self._data[..., ci["log_R"]]
        self._data[..., ci["density"]] = compute_density(
            self._data[..., ci["star_mass"]],
            self._data[..., ci["radius"]],
        )
        self._data[..., ci["dm_deep"]] = compute_dm_deep(
            self._data[..., ci["initial_mass"]], eep_axis=3  # EEP is axis 3 in 5D
        )

    @property
    def name(self) -> str:
        return "STAREVOL"

    @property
    def feh_values(self) -> np.ndarray:
        return self._feh_values

    @property
    def vini_values(self) -> np.ndarray:
        return self._vini_values

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
        return self.eep_range

    @property
    def columns(self) -> list[str]:
        return list(self._COLUMNS)

    def to_hdf5(self, path: str | Path):
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=self._data, compression="gzip")
            f.create_dataset("feh_values", data=self._feh_values)
            f.create_dataset("vini_values", data=self._vini_values)
            f.create_dataset("age_values", data=self._age_values)
            f.create_dataset("eep_values", data=self._eep_values)
            f.attrs["columns"] = self._COLUMNS
            f.attrs["grid_name"] = "STAREVOL"

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "STAREVOLModelGrid":
        from lachesis.grid.base import load_grid_hdf5_starevol

        obj = object.__new__(cls)
        obj._data, obj._feh_values, obj._vini_values, obj._age_values, obj._eep_values, obj._COLUMNS = load_grid_hdf5_starevol(path)
        return obj
