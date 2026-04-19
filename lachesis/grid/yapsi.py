"""YAPSI (Yale-Potsdam Stellar Isochrones) grid.

Parses the FITS isochrone bundle from
http://www.astro.yale.edu/yapsi/data/bundles/

The FITS file contains evolutionary tracks stored as 4D arrays:
  (n_comp, n_mass, n_step, 8)
where column 0 of each HDU gives the direct value at each time step,
and columns 1-7 are Chebyshev interpolation coefficients.

HDU layout:
  CAGE  — age (Gyr)
  CLOGT — log(Teff)
  CLOGG — log(g)
  CLOGL — log(L/Lsun)
  CFEHS — surface [Fe/H]
  CAMAX — (n_comp, n_mass, 4) max age info

Isochrones are constructed by interpolating each track at target ages.
YAPSI has no EEPs — we use mass-sorted row index as EEP (0, 1, 2, ...).
Grid shape: (n_feh, n_age, n_eep, n_cols).
"""

from pathlib import Path

import numpy as np
from astropy.io import fits as pyfits

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

# Default age grid: 0.5 to 15 Gyr in 0.5 Gyr steps
_DEFAULT_AGES_GYR = np.arange(0.5, 15.5, 0.5)


def _parse_yapsi_fits(path: Path) -> list[dict]:
    """Parse a YAPSI FITS track bundle into per-composition track data.

    The FITS file stores evolutionary tracks as 4D ImageHDU arrays
    with shape (n_comp, n_mass, n_step, 8). Column 0 of each HDU
    holds the direct value (age, log_Teff, etc.) at each time step.

    Returns
    -------
    list of dict, each with keys:
        feh : float
            Initial [Fe/H] for this composition.
        isochrones : list of (age_gyr, ndarray) pairs
            Pre-built isochrones. Each ndarray has columns
            [mass_proxy, log_Teff, log_g, log_L] sorted by mass.
    """
    results = []

    with pyfits.open(path) as hdul:
        cage = hdul["CAGE"].data     # (n_comp, n_mass, n_step, 8)
        clogt = hdul["CLOGT"].data
        clogg = hdul["CLOGG"].data
        clogl = hdul["CLOGL"].data
        crad = hdul["CRAD"].data     # R/Rsun (linear, NOT log)
        cfehs = hdul["CFEHS"].data
        camax = hdul["CAMAX"].data   # (n_comp, n_mass, 4)

        n_comp, n_mass = cage.shape[0], cage.shape[1]

        # Derive initial masses from logg + R at an early MS time step.
        # M/Msun = 10^(logg - logg_sun) * (R/Rsun)^2
        _LOGG_SUN = 4.4377
        n_step = cage.shape[2]
        _ref_step = min(100, n_step - 1)  # early MS, past PMS contraction

        for ci in range(n_comp):
            # Initial [Fe/H] from surface metallicity at step 0
            feh = float(np.round(np.median(cfehs[ci, :, 0, 0]), 2))

            # Derive initial mass per bin from logg + R
            masses = np.array([
                10 ** (clogg[ci, mi, _ref_step, 0] - _LOGG_SUN)
                * crad[ci, mi, _ref_step, 0] ** 2
                for mi in range(n_mass)
            ])

            # Determine which ages are reachable by at least some masses
            max_ages = camax[ci, :, 0]  # max age per mass
            global_max = float(max_ages.max())

            # Build isochrones at default age grid
            isochrones = []
            for target_age in _DEFAULT_AGES_GYR:
                if target_age > global_max:
                    break

                rows = []
                for mi in range(n_mass):
                    if target_age > max_ages[mi]:
                        continue  # star already dead at this age

                    ages = cage[ci, mi, :, 0]
                    lt = clogt[ci, mi, :, 0]
                    lg = clogg[ci, mi, :, 0]
                    ll = clogl[ci, mi, :, 0]

                    # Interpolate at target age
                    interp_lt = float(np.interp(target_age, ages, lt))
                    interp_lg = float(np.interp(target_age, ages, lg))
                    interp_ll = float(np.interp(target_age, ages, ll))

                    rows.append([masses[mi], interp_lt, interp_lg, interp_ll])

                if rows:
                    isochrones.append(
                        (float(target_age), np.array(rows))
                    )

            results.append({"feh": feh, "isochrones": isochrones})

    return results


class YAPSIModelGrid:
    """YAPSI (Yale-Potsdam Stellar Isochrones) model grid.

    Grid shape: (n_feh, n_age, n_eep, n_cols) -- same interface as
    Dartmouth/MIST/PARSEC. EEPs are mass-sorted row indices (0, 1, 2, ...).
    """

    _COLUMNS = _COLUMNS

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        fits_files = sorted(directory.glob("*.fits"))
        if not fits_files:
            raise FileNotFoundError(f"No YAPSI FITS files in {directory}")

        # Parse all FITS files (usually just one bundle)
        all_parsed = []
        for f in fits_files:
            all_parsed.extend(_parse_yapsi_fits(f))

        # Collect axes
        feh_set = sorted({round(p["feh"], 4) for p in all_parsed})
        age_set = set()
        max_eep = 0
        for p in all_parsed:
            for age_gyr, data in p["isochrones"]:
                log_age = round(np.log10(age_gyr * 1e9), 4)
                age_set.add(log_age)
                max_eep = max(max_eep, data.shape[0])

        self._feh_values = np.array(feh_set)
        self._age_values = np.array(sorted(age_set))
        self._eep_values = np.arange(max_eep, dtype=float)

        feh_to_idx = {round(f, 4): i for i, f in enumerate(self._feh_values)}
        age_to_idx = {round(a, 4): i for i, a in enumerate(self._age_values)}

        n_feh = len(self._feh_values)
        n_age = len(self._age_values)
        n_eep = len(self._eep_values)
        n_cols = len(self._COLUMNS)

        self._data = np.full((n_feh, n_age, n_eep, n_cols), np.nan)
        ci = {c: i for i, c in enumerate(self._COLUMNS)}

        for p in all_parsed:
            fi = feh_to_idx.get(round(p["feh"], 4))
            if fi is None:
                continue

            for age_gyr, data in p["isochrones"]:
                log_age = round(np.log10(age_gyr * 1e9), 4)
                ai = age_to_idx.get(log_age)
                if ai is None:
                    continue

                n_rows = data.shape[0]
                for ei in range(n_rows):
                    self._data[fi, ai, ei, ci["eep"]] = ei
                    self._data[fi, ai, ei, ci["age"]] = log_age
                    self._data[fi, ai, ei, ci["initial_mass"]] = data[ei, 0]
                    self._data[fi, ai, ei, ci["star_mass"]] = data[ei, 0]
                    self._data[fi, ai, ei, ci["log_Teff"]] = data[ei, 1]
                    self._data[fi, ai, ei, ci["log_g"]] = data[ei, 2]
                    self._data[fi, ai, ei, ci["log_L"]] = data[ei, 3]
                    self._data[fi, ai, ei, ci["phase"]] = np.nan

        # Compute log_R from Stefan-Boltzmann: L = 4pi R^2 sigma T^4
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
        return "YAPSI"

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
        """YAPSI's full range -- mass-sorted indices, all usable."""
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
            f.attrs["grid_name"] = "YAPSI"

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "YAPSIModelGrid":
        from lachesis.grid.base import load_grid_hdf5

        obj = object.__new__(cls)
        obj._data, obj._feh_values, obj._age_values, obj._eep_values, obj._COLUMNS = load_grid_hdf5(path)
        return obj
