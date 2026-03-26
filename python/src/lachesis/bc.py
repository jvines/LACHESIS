"""Bolometric correction tables for photometric magnitude computation.

BC convention (MIST): M_band = Mbol - BC_band
Apparent magnitude: m_band = M_band + 5 * log10(distance_pc / 10)
"""


from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class BCTable:
    """Bolometric correction interpolator for a photometric system.

    Parses MIST BC table files (one per [Fe/H]) and builds 4D interpolators
    over (Teff, logg, [Fe/H], Av) → BC for each photometric band.
    """

    def __init__(self, directory: str | Path, system: str = "UBVRIplus"):
        directory = Path(directory)
        self._system = system

        # Find BC files matching the system
        files = sorted(directory.glob(f"feh*.{system}"))
        if not files:
            raise FileNotFoundError(
                f"No BC files for system '{system}' in {directory}"
            )

        # Parse header from first file to get band names
        self._bands = self._parse_header(files[0])

        # Parse all files, collecting [Fe/H] values and data
        all_data = []
        fehs = []
        for f in files:
            feh, data = self._parse_file(f)
            fehs.append(feh)
            all_data.append(data)

        # Sort by [Fe/H]
        order = np.argsort(fehs)
        fehs = [fehs[i] for i in order]
        all_data = [all_data[i] for i in order]

        self._feh_values = np.array(fehs)

        # Extract unique grid axes from first file
        d0 = all_data[0]
        self._teff_values = np.unique(d0[:, 0])
        self._logg_values = np.unique(d0[:, 1])
        self._av_values = np.unique(d0[:, 3])

        n_teff = len(self._teff_values)
        n_logg = len(self._logg_values)
        n_feh = len(self._feh_values)
        n_av = len(self._av_values)
        n_bands = len(self._bands)

        # Build 5D array: (teff, logg, feh, av, band)
        grid = np.full((n_teff, n_logg, n_feh, n_av, n_bands), np.nan)

        for fi, data in enumerate(all_data):
            for row in data:
                ti = np.searchsorted(self._teff_values, row[0])
                li = np.searchsorted(self._logg_values, row[1])
                ai = np.searchsorted(self._av_values, row[3])
                grid[ti, li, fi, ai, :] = row[5:]

        # Build one interpolator per band
        self._interpolators = {}
        for bi, band in enumerate(self._bands):
            self._interpolators[band] = RegularGridInterpolator(
                (self._teff_values, self._logg_values,
                 self._feh_values, self._av_values),
                grid[:, :, :, :, bi],
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

    def _parse_header(self, path: Path) -> list[str]:
        """Extract band names from the header."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Teff" in line and "logg" in line:
                    parts = line.lstrip("#").split()
                    # Skip Teff, logg, [Fe/H], Av, Rv → rest are band names
                    return parts[5:]
        raise ValueError(f"Could not find band names in {path}")

    def _parse_file(self, path: Path) -> tuple[float, np.ndarray]:
        """Parse a single BC file, return ([Fe/H], data_array)."""
        data = np.loadtxt(path, comments="#")
        feh = data[0, 2]  # [Fe/H] column
        return feh, data

    @property
    def bands(self) -> list[str]:
        return self._bands

    @property
    def teff_values(self) -> np.ndarray:
        return self._teff_values

    @property
    def logg_values(self) -> np.ndarray:
        return self._logg_values

    @property
    def feh_values(self) -> np.ndarray:
        return self._feh_values

    @property
    def av_values(self) -> np.ndarray:
        return self._av_values

    def get_bc(
        self,
        teff: float,
        logg: float,
        feh: float,
        av: float,
    ) -> dict[str, float]:
        """Interpolate BC at (Teff, logg, [Fe/H], Av) for all bands.

        Returns dict of band_name → BC value.
        M_band = Mbol - BC_band
        """
        point = np.array([[teff, logg, feh, av]])
        result = {}
        for band, interp in self._interpolators.items():
            result[band] = float(interp(point)[0])
        return result

    def get_absolute_mag(
        self,
        mbol: float,
        teff: float,
        logg: float,
        feh: float,
        av: float,
        bands: list[str] | None = None,
    ) -> dict[str, float]:
        """Absolute magnitude: M_band = Mbol - BC."""
        bcs = self.get_bc(teff, logg, feh, av)
        if bands is not None:
            bcs = {k: v for k, v in bcs.items() if k in bands}
        return {band: mbol - bc for band, bc in bcs.items()}

    def get_apparent_mag(
        self,
        mbol: float,
        teff: float,
        logg: float,
        feh: float,
        av: float,
        distance_pc: float,
        bands: list[str] | None = None,
    ) -> dict[str, float]:
        """Apparent magnitude: m = M + 5*log10(d/10)."""
        abs_mags = self.get_absolute_mag(mbol, teff, logg, feh, av, bands)
        dm = 5.0 * np.log10(distance_pc / 10.0)
        return {band: m + dm for band, m in abs_mags.items()}
