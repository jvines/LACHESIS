"""Bolometric correction tables for photometric magnitude computation.

BC convention (MIST): M_band = Mbol - BC_band
Apparent magnitude: m_band = M_band + 5 * log10(distance_pc / 10)
"""


from pathlib import Path

import numpy as np

try:
    import numba as nb

    @nb.njit(cache=True)
    def _searchsorted_bc(arr, val):
        n = len(arr)
        if val < arr[0] or val > arr[n - 1]:
            return -1
        lo, hi = 0, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if arr[mid] <= val:
                lo = mid
            else:
                hi = mid
        return lo

    @nb.njit(cache=True)
    def _quadlinear(grid, ax0, ax1, ax2, ax3, x0, x1, x2, x3, n_bands):
        """4D linear interpolation returning all bands at once."""
        result = np.empty(n_bands)

        i0 = _searchsorted_bc(ax0, x0)
        i1 = _searchsorted_bc(ax1, x1)
        i2 = _searchsorted_bc(ax2, x2)
        i3 = _searchsorted_bc(ax3, x3)

        if (i0 < 0 or i0 >= len(ax0) - 1 or
            i1 < 0 or i1 >= len(ax1) - 1 or
            i2 < 0 or i2 >= len(ax2) - 1 or
            i3 < 0 or i3 >= len(ax3) - 1):
            result[:] = np.nan
            return result

        d0 = (x0 - ax0[i0]) / (ax0[i0 + 1] - ax0[i0])
        d1 = (x1 - ax1[i1]) / (ax1[i1 + 1] - ax1[i1])
        d2 = (x2 - ax2[i2]) / (ax2[i2 + 1] - ax2[i2])
        d3 = (x3 - ax3[i3]) / (ax3[i3 + 1] - ax3[i3])

        for b in range(n_bands):
            val = 0.0
            for di0 in range(2):
                for di1 in range(2):
                    for di2 in range(2):
                        for di3 in range(2):
                            w = (
                                (d0 if di0 else 1 - d0) *
                                (d1 if di1 else 1 - d1) *
                                (d2 if di2 else 1 - d2) *
                                (d3 if di3 else 1 - d3)
                            )
                            c = grid[i0 + di0, i1 + di1, i2 + di2, i3 + di3, b]
                            if w > 0 and np.isnan(c):
                                val = np.nan
                                break
                            if not np.isnan(c):
                                val += w * c
                        if np.isnan(val):
                            break
                    if np.isnan(val):
                        break
                if np.isnan(val):
                    break
            result[b] = val

        return result

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


_DEFAULT_SYSTEMS = ["UBVRIplus", "WISE", "SDSSugriz", "PanSTARRS", "SkyMapper", "GALEX"]


class BCTable:
    """Bolometric correction interpolator.

    Loads from a single HDF5 file (bc_tables.h5) containing all photometric
    systems, compressed with gzip. Uses numba JIT for fast 4D interpolation
    returning all bands in a single call.
    """

    def __init__(self, directory: str | Path, system: str = "UBVRIplus"):
        directory = Path(directory)
        h5_path = directory / "bc_tables.h5"

        if not h5_path.exists():
            raise FileNotFoundError(f"BC table not found: {h5_path}")

        import h5py
        with h5py.File(h5_path, "r") as f:
            if system not in f:
                raise ValueError(
                    f"System '{system}' not in bc_tables.h5. "
                    f"Available: {list(f.keys())}"
                )
            grp = f[system]
            raw = grp["data"][:].astype(np.float64)  # (n_feh, n_rows, n_cols)
            self._feh_values = grp["feh_values"][:]
            self._bands = [s.decode() for s in grp["band_names"][:]]

        self._system = system

        # raw shape: (n_feh, n_rows, n_cols) where cols = Teff,logg,feh,Av,Rv,bands...
        d0 = raw[0]
        self._teff_values = np.unique(d0[:, 0])
        self._logg_values = np.unique(d0[:, 1])
        self._av_values = np.unique(d0[:, 3])

        n_teff = len(self._teff_values)
        n_logg = len(self._logg_values)
        n_feh = len(self._feh_values)
        n_av = len(self._av_values)
        n_bands = len(self._bands)

        # 5D array: (teff, logg, feh, av, band)
        self._grid = np.full(
            (n_teff, n_logg, n_feh, n_av, n_bands), np.nan, dtype=np.float64
        )

        for fi in range(n_feh):
            data = raw[fi]
            for row in data:
                ti = np.searchsorted(self._teff_values, row[0])
                li = np.searchsorted(self._logg_values, row[1])
                ai = np.searchsorted(self._av_values, row[3])
                self._grid[ti, li, fi, ai, :] = row[5:]

        self._finalize()

    def _finalize(self):
        """Build contiguous arrays, band index, scipy fallback, JIT warmup."""
        self._ax_teff = np.ascontiguousarray(self._teff_values, dtype=np.float64)
        self._ax_logg = np.ascontiguousarray(self._logg_values, dtype=np.float64)
        self._ax_feh = np.ascontiguousarray(self._feh_values, dtype=np.float64)
        self._ax_av = np.ascontiguousarray(self._av_values, dtype=np.float64)
        self._grid = np.ascontiguousarray(self._grid)
        self._n_bands = len(self._bands)
        self._band_indices = {b: i for i, b in enumerate(self._bands)}

        self._active_bands: list[str] | None = None
        self._active_idx: np.ndarray | None = None
        self._active_grid: np.ndarray | None = None

        if _HAS_NUMBA:
            grid = self._active_grid if self._active_grid is not None else self._grid
            _quadlinear(
                grid,
                self._ax_teff, self._ax_logg, self._ax_feh, self._ax_av,
                self._ax_teff[0], self._ax_logg[0], self._ax_feh[0], self._ax_av[0],
                grid.shape[4],
            )

        self._scipy_interps = None
        if not _HAS_NUMBA:
            from scipy.interpolate import RegularGridInterpolator
            self._scipy_interps = {}
            for bi, band in enumerate(self._bands):
                self._scipy_interps[band] = RegularGridInterpolator(
                    (self._teff_values, self._logg_values,
                     self._feh_values, self._av_values),
                    self._grid[:, :, :, :, bi],
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )

    @classmethod
    def multi_system(cls, directory: str | Path, systems: list[str] | None = None):
        """Load multiple BC systems from bc_tables.h5 and merge.

        All systems share identical (Teff, logg, [Fe/H], Av) axes,
        so their band columns are concatenated along axis 4 of the 5D grid.
        """
        if systems is None:
            systems = list(_DEFAULT_SYSTEMS)

        directory = Path(directory)
        h5_path = directory / "bc_tables.h5"

        if not h5_path.exists():
            raise FileNotFoundError(f"BC table not found: {h5_path}")

        import h5py
        with h5py.File(h5_path, "r") as f:
            available = [s for s in systems if s in f]

        if not available:
            raise FileNotFoundError(
                f"No BC systems found for {systems} in {h5_path}"
            )

        # Load first system to get axes
        first = cls(directory, system=available[0])

        if len(available) == 1:
            return first

        # Merge additional systems
        merged_bands = list(first._bands)
        seen = set(merged_bands)
        grids = [first._grid]

        for sysname in available[1:]:
            extra = cls(directory, system=sysname)

            new_idx = []
            new_names = []
            for i, b in enumerate(extra._bands):
                if b not in seen:
                    new_idx.append(i)
                    new_names.append(b)
                    seen.add(b)

            if not new_names:
                continue

            grids.append(extra._grid[:, :, :, :, new_idx])
            merged_bands.extend(new_names)

        first._system = "+".join(available)
        first._bands = merged_bands
        first._grid = np.concatenate(grids, axis=4)
        first._finalize()
        return first

    @property
    def bands(self) -> list[str]:
        return self._active_bands if self._active_bands is not None else self._bands

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

    def set_active_bands(self, bands: list[str]):
        """Restrict interpolation to only these bands (major speedup)."""
        idx = []
        valid = []
        for b in bands:
            if b in self._band_indices:
                idx.append(self._band_indices[b])
                valid.append(b)
        if not idx:
            self._active_bands = None
            self._active_idx = None
            self._active_grid = None
            return
        self._active_bands = valid
        self._active_idx = np.array(idx)
        self._active_grid = np.ascontiguousarray(
            self._grid[:, :, :, :, self._active_idx]
        )

    def get_bc(
        self,
        teff: float,
        logg: float,
        feh: float,
        av: float,
    ) -> dict[str, float]:
        """Interpolate BC at (Teff, logg, [Fe/H], Av)."""
        band_names = self._active_bands if self._active_bands is not None else self._bands
        grid = self._active_grid if self._active_grid is not None else self._grid
        n = len(band_names)

        if _HAS_NUMBA:
            vals = _quadlinear(
                grid,
                self._ax_teff, self._ax_logg, self._ax_feh, self._ax_av,
                teff, logg, feh, av, n,
            )
            return {band_names[i]: float(vals[i]) for i in range(n)}
        else:
            point = np.array([[teff, logg, feh, av]])
            result = {}
            for band in band_names:
                result[band] = float(self._scipy_interps[band](point)[0])
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
