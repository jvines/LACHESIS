"""PARSEC isochrone grid with EEP translation.

Translates PARSEC phase labels to MIST-compatible EEP numbers so both
grids share the same (feh, age, eep) -> observables parameterization.

PARSEC label -> EEP mapping:
  0 (PMS) -> EEP   1-201
  1 (MS) -> EEP 202-454
  2 (SGB) -> EEP 454-500
  3 (RGB) -> EEP 500-605
  4 (CHeB) -> EEP 631-707
  5 (E-AGB) -> EEP 707-808
  6 (TP-AGB) -> EEP 808-1200

PRODUCTION CUBE. The shipped cube (parsec_v1.2S_eeprebuild.h5, loaded via
``from_hdf5``) is built by ``scripts/rebuild_parsec_eep.py`` with a HOMOLOGOUS
track-anchored EEP axis: PMS/MS EEPs come from reconstructed fixed-mass tracks
(Dotter metric distance), post-MS EEPs from per-isochrone metric distance along
the near-vertical giant branch. That removes the non-homologous mass-rank EEP
that fabricated young-massive solutions (DEBcat spec-only bias), so no PMS mask
is needed on load.

LEGACY BUILDER. ``__init__`` + ``_assign_eep`` below build a cube from a
directory of per-isochrone CSVs using per-phase linspace over mass-rank. That
axis is NON-homologous and is retained only for the from-CSV API and the unit
tests; it NaN-masks the PMS as a safety band-aid. Do not use it to regenerate
the production cube, use the rebuild script.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lachesis.grid.derived import (
    compute_density,
    compute_mbol,
    compute_teff,
)

# Phase label -> (EEP_start, EEP_end)
_LABEL_TO_EEP = {
    0: (1, 201),     # PMS
    1: (202, 454),   # MS
    2: (454, 500),   # SGB
    3: (500, 605),   # RGB
    4: (631, 707),   # CHeB
    5: (707, 808),   # E-AGB
    6: (808, 1200),  # TP-AGB
}


def _assign_eep(iso_df: pd.DataFrame) -> np.ndarray:
    """Assign EEP values to each row of a single PARSEC isochrone.

    Within each phase, EEP is linearly spaced from start to end.
    Phases not in the mapping (7=post-AGB, 8=WD, 9=naked He) get NaN.
    """
    eeps = np.full(len(iso_df), np.nan)
    labels = iso_df["label"].values

    for label, (eep_lo, eep_hi) in _LABEL_TO_EEP.items():
        mask = labels == label
        n = mask.sum()
        if n == 0:
            continue
        if n == 1:
            eeps[mask] = (eep_lo + eep_hi) / 2
        else:
            eeps[mask] = np.linspace(eep_lo, eep_hi, n)

    return eeps


class PARSECModelGrid:
    """PARSEC model grid with EEP translation.

    Grid shape: (n_feh, n_age, n_eep, n_cols), same as MIST.
    Axes: [M/H], log_age, EEP (translated from PARSEC phase labels).
    """

    _COLUMNS = [
        "eep", "age", "initial_mass", "star_mass", "log_Teff", "log_g",
        "log_L", "log_R", "phase", "Teff", "Mbol", "radius", "density",
        "dm_deep",
    ]

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        csv_files = sorted(directory.glob("parsec_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No PARSEC CSV files in {directory}")

        # Load all files
        frames = []
        for f in csv_files:
            frames.append(pd.read_csv(f))
        all_data = pd.concat(frames, ignore_index=True)

        # PARSEC returns ages like 9.00001, 9.00002, different per [M/H].
        # Round to 2 decimals to build a common grid.
        all_data["logAge"] = np.round(all_data["logAge"], 2)
        all_data["MH"] = np.round(all_data["MH"], 2)

        self._feh_values = np.array(sorted(all_data["MH"].unique()))
        self._age_values = np.array(sorted(all_data["logAge"].unique()))

        # Assign EEPs to every row
        all_eeps = []
        for (mh, age), group in all_data.groupby(["MH", "logAge"]):
            eeps = _assign_eep(group)
            all_eeps.append(eeps)
        all_data["EEP"] = np.concatenate(all_eeps)

        # Drop rows with NaN EEP (post-AGB, WD, etc.)
        all_data = all_data[np.isfinite(all_data["EEP"])].copy()

        # Build global EEP axis, integer EEPs spanning all isochrones
        eep_min = int(np.floor(all_data["EEP"].min()))
        eep_max = int(np.ceil(all_data["EEP"].max()))
        self._eep_values = np.arange(eep_min, eep_max + 1, dtype=float)
        eep_to_idx = {int(e): i for i, e in enumerate(self._eep_values)}

        n_feh = len(self._feh_values)
        n_age = len(self._age_values)
        n_eep = len(self._eep_values)
        n_cols = len(self._COLUMNS)

        self._data = np.full((n_feh, n_age, n_eep, n_cols), np.nan)
        ci = {c: i for i, c in enumerate(self._COLUMNS)}

        # Column mapping from PARSEC CSV
        col_map = {
            "initial_mass": "Mini",
            "star_mass": "Mass",
            "log_Teff": "logTe",
            "log_g": "logg",
            "log_L": "logL",
            "phase": "label",
        }

        # Fill grid: for each isochrone, interpolate onto integer EEP grid
        for (mh, age), group in all_data.groupby(["MH", "logAge"]):
            fi = int(np.argmin(np.abs(self._feh_values - mh)))
            ai = int(np.argmin(np.abs(self._age_values - age)))
            if abs(self._feh_values[fi] - mh) > 0.01:
                continue
            if abs(self._age_values[ai] - age) > 0.01:
                continue

            iso_eeps = group["EEP"].values
            if len(iso_eeps) < 2:
                continue

            # Integer EEPs within this isochrone's range
            eep_lo = int(np.ceil(iso_eeps.min()))
            eep_hi = int(np.floor(iso_eeps.max()))

            for raw_col, csv_col in col_map.items():
                vals = group[csv_col].values
                for eep_int in range(eep_lo, eep_hi + 1):
                    if eep_int not in eep_to_idx:
                        continue
                    ei = eep_to_idx[eep_int]
                    # Interpolate this column at this integer EEP
                    self._data[fi, ai, ei, ci[raw_col]] = np.interp(
                        eep_int, iso_eeps, vals
                    )
                    self._data[fi, ai, ei, ci["eep"]] = eep_int
                    self._data[fi, ai, ei, ci["age"]] = age

        # Fill inter-phase NaN gaps by interpolating along EEP axis.
        # E.g., the He flash gap (EEP 605-631) gets bridged.
        for fi in range(n_feh):
            for ai in range(n_age):
                for col_idx in range(n_cols):
                    col_slice = self._data[fi, ai, :, col_idx]
                    valid = np.isfinite(col_slice)
                    if valid.sum() < 2:
                        continue
                    first = np.argmax(valid)
                    last = len(valid) - 1 - np.argmax(valid[::-1])
                    interior = slice(first, last + 1)
                    x = np.where(valid[interior])[0]
                    y = col_slice[interior][valid[interior]]
                    xnew = np.arange(last - first + 1)
                    col_slice[interior] = np.interp(xnew, x, y)

        # Compute log_R from L and Teff (PARSEC doesn't provide it directly)
        log_l = self._data[:, :, :, ci["log_L"]]
        log_te = self._data[:, :, :, ci["log_Teff"]]
        log_tsun = np.log10(5772.0)
        self._data[:, :, :, ci["log_R"]] = (
            0.5 * log_l + 2.0 * (log_tsun - log_te)
        )

        # Derived columns
        self._data[:, :, :, ci["Teff"]] = compute_teff(
            self._data[:, :, :, ci["log_Teff"]]
        )
        self._data[:, :, :, ci["Mbol"]] = compute_mbol(
            self._data[:, :, :, ci["log_L"]]
        )
        self._data[:, :, :, ci["radius"]] = 10.0 ** self._data[:, :, :, ci["log_R"]]
        self._data[:, :, :, ci["density"]] = compute_density(
            self._data[:, :, :, ci["star_mass"]],
            self._data[:, :, :, ci["radius"]],
        )

        # dm_deep: gradient of initial_mass along EEP axis (axis=2)
        from lachesis.grid.derived import compute_dm_deep
        self._data[:, :, :, ci["dm_deep"]] = compute_dm_deep(
            self._data[:, :, :, ci["initial_mass"]], eep_axis=2
        )

        # Mask pre-main-sequence (phase 0) nodes. PARSEC's regridded EEP axis
        # is non-homologous: a given EEP maps to PMS at young ages but to the
        # MS at old ages, so the trilinear interpolator blends high-mass PMS
        # corners into the MS locus and fabricates a spurious young-massive
        # solution that passes through a star's pinned (Teff, log g, [Fe/H]).
        # This drove the DEBcat spec-only mass bias (PARSEC +6.6% on the MS,
        # +54% at sub-solar [Fe/H]) and contaminates the photometric fits too.
        # NaN-ing the PMS rows (done last, after gap-filling, so they are not
        # bridged) makes any interpolation cell that touches the PMS region
        # return NaN, which the likelihood rejects, removing the fabricated
        # branch without affecting clean MS / evolved cells, which never touch
        # a PMS node. LACHESIS targets field stars, not pre-MS objects, and the
        # other grids retain PMS coverage if ever needed.
        phase = self._data[:, :, :, ci["phase"]]
        self._data[np.isfinite(phase) & (phase < 0.5)] = np.nan

    @property
    def name(self) -> str:
        return "PARSEC"

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
        """ZAMS (202) to TPAGB (808), PARSEC uses MIST's EEP scheme."""
        return (max(202, int(self._eep_values[0])), min(808, int(self._eep_values[-1])))

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
            f.attrs["grid_name"] = "PARSEC"

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "PARSECModelGrid":
        from lachesis.grid.base import load_grid_hdf5

        obj = object.__new__(cls)
        obj._data, obj._feh_values, obj._age_values, obj._eep_values, obj._COLUMNS = load_grid_hdf5(path)
        return obj
