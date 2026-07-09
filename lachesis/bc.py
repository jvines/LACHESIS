"""Bolometric correction tables for photometric magnitude computation.

BC convention (MIST): M_band = Mbol - BC_band
Apparent magnitude: m_band = M_band + 5 * log10(distance_pc / 10)
"""


import math
from pathlib import Path

import numpy as np

try:
    import numba as nb

    @nb.njit(cache=True)
    def _searchsorted_bc(arr, val):
        if not np.isfinite(val):
            return -1
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

        # Mirror lachesis.interp_numba._quadlinear: track has_nan via a flag
        # so the final value is well-defined even when NaN corners coincide
        # with zero weights.
        for b in range(n_bands):
            val = 0.0
            has_nan = False
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
                                has_nan = True
                                break
                            if not np.isnan(c):
                                val += w * c
                        if has_nan:
                            break
                    if has_nan:
                        break
                if has_nan:
                    break
            result[b] = np.nan if has_nan else val

        return result

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


_DEFAULT_SYSTEMS = ["UBVRIplus", "WISE", "SDSSugriz", "PanSTARRS", "SkyMapper", "GALEX"]

# Photometric zero-point convention per MIST BC system.
# MIST distributes UBVRIplus and 2MASS in Vega magnitudes; the survey
# systems (SDSS, PS1, SkyMapper, GALEX) are in AB. The catalogue must use
# the same zero point as the BC table or magnitudes are off by the
# Vega-AB offset (~0.1 mag in Gaia G, ~0.5 mag in u/g, etc).
_SYSTEM_ZERO_POINT: dict[str, str] = {
    "UBVRIplus": "Vega",
    "WISE": "Vega",
    "SDSSugriz": "AB",
    "PanSTARRS": "AB",
    "SkyMapper": "AB",
    "GALEX": "AB",
}

# Per-band override map for systems that mix conventions (e.g. UBVRIplus
# bundles Gaia EDR3 + 2MASS + Tycho/Bessell — all Vega — alongside Hipparcos
# Hp which is also Vega; nothing to override for now). Kept here so
# additions are obvious.
_BAND_ZERO_POINT_OVERRIDES: dict[str, str] = {}


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
        sys_zp = _SYSTEM_ZERO_POINT.get(system, "Vega")
        self._band_systems = {
            b: _BAND_ZERO_POINT_OVERRIDES.get(b, sys_zp) for b in self._bands
        }

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

        # Vectorised rectilinear assignment with explicit equality checks.
        # The previous loop used np.searchsorted insertion indices without
        # confirming the row coordinate exactly matches an axis value, so
        # any floating-point drift in the raw data silently scrambled the
        # grid by writing the row into a neighbouring cell.
        teff_idx = np.searchsorted(self._teff_values, raw[..., 0])
        logg_idx = np.searchsorted(self._logg_values, raw[..., 1])
        av_idx = np.searchsorted(self._av_values, raw[..., 3])
        feh_idx = np.broadcast_to(
            np.arange(n_feh)[:, None], teff_idx.shape
        )

        if (
            (self._teff_values[teff_idx] != raw[..., 0]).any()
            or (self._logg_values[logg_idx] != raw[..., 1]).any()
            or (self._av_values[av_idx] != raw[..., 3]).any()
        ):
            raise ValueError(
                "BC table coordinates do not match the inferred axis "
                "values exactly; refusing to silently misplace rows."
            )

        self._grid[
            teff_idx, logg_idx, feh_idx, av_idx, :
        ] = raw[..., 5:]

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

        Opens the HDF5 file once and constructs a single instance instead of
        repeatedly invoking ``cls(...)`` per system (each of which would
        re-open the file and rebuild the grid from scratch).
        """
        if systems is None:
            systems = list(_DEFAULT_SYSTEMS)

        directory = Path(directory)
        h5_path = directory / "bc_tables.h5"

        if not h5_path.exists():
            raise FileNotFoundError(f"BC table not found: {h5_path}")

        import h5py
        merged_bands: list[str] = []
        seen: set[str] = set()
        grids: list[np.ndarray] = []
        band_systems: dict[str, str] = {}
        teff_values = logg_values = feh_values = av_values = None

        with h5py.File(h5_path, "r") as f:
            available = [s for s in systems if s in f]
            if not available:
                raise FileNotFoundError(
                    f"No BC systems found for {systems} in {h5_path}"
                )

            for sysname in available:
                grp = f[sysname]
                raw = grp["data"][:].astype(np.float64)
                feh_vals = grp["feh_values"][:]
                bands = [s.decode() for s in grp["band_names"][:]]

                d0 = raw[0]
                t_vals = np.unique(d0[:, 0])
                l_vals = np.unique(d0[:, 1])
                a_vals = np.unique(d0[:, 3])

                if teff_values is None:
                    teff_values, logg_values, feh_values, av_values = (
                        t_vals, l_vals, feh_vals, a_vals,
                    )
                else:
                    if (
                        not np.array_equal(t_vals, teff_values)
                        or not np.array_equal(l_vals, logg_values)
                        or not np.array_equal(feh_vals, feh_values)
                        or not np.array_equal(a_vals, av_values)
                    ):
                        raise ValueError(
                            f"System '{sysname}' axes differ from the others; "
                            "BC systems must share identical axes."
                        )

                new_idx = [i for i, b in enumerate(bands) if b not in seen]
                if not new_idx:
                    continue

                n_t, n_l, n_f, n_a = (
                    len(t_vals), len(l_vals), len(feh_vals), len(a_vals)
                )
                n_b = len(new_idx)
                grid = np.full((n_t, n_l, n_f, n_a, n_b), np.nan, dtype=np.float64)

                t_idx = np.searchsorted(t_vals, raw[..., 0])
                l_idx = np.searchsorted(l_vals, raw[..., 1])
                a_idx = np.searchsorted(a_vals, raw[..., 3])
                f_idx = np.broadcast_to(np.arange(n_f)[:, None], t_idx.shape)
                if (
                    (t_vals[t_idx] != raw[..., 0]).any()
                    or (l_vals[l_idx] != raw[..., 1]).any()
                    or (a_vals[a_idx] != raw[..., 3]).any()
                ):
                    raise ValueError(
                        f"System '{sysname}': BC table coordinates do not "
                        "match the inferred axis values exactly."
                    )
                grid[t_idx, l_idx, f_idx, a_idx, :] = raw[..., 5 + np.array(new_idx)]
                grids.append(grid)
                sys_zp = _SYSTEM_ZERO_POINT.get(sysname, "Vega")
                for i in new_idx:
                    band = bands[i]
                    merged_bands.append(band)
                    band_systems[band] = _BAND_ZERO_POINT_OVERRIDES.get(band, sys_zp)
                    seen.add(band)

        # Build instance directly without re-running __init__
        inst = cls.__new__(cls)
        inst._teff_values = teff_values
        inst._logg_values = logg_values
        inst._feh_values = feh_values
        inst._av_values = av_values
        inst._bands = merged_bands
        inst._band_systems = band_systems
        inst._system = "+".join(available)
        inst._grid = np.concatenate(grids, axis=4) if len(grids) > 1 else grids[0]
        inst._finalize()
        return inst

    @property
    def bands(self) -> list[str]:
        return self._active_bands if self._active_bands is not None else self._bands

    def zero_point(self, band: str) -> str:
        """Return the zero-point convention ('Vega' or 'AB') for a band."""
        if not hasattr(self, "_band_systems"):
            return "Vega"
        return self._band_systems.get(band, "Vega")

    @property
    def band_systems(self) -> dict[str, str]:
        """Per-band zero-point ('Vega'/'AB') for every loaded band."""
        return dict(getattr(self, "_band_systems", {}))

    def assert_zero_point(self, band: str, expected: str) -> None:
        """Raise ValueError when the BC table convention for *band*
        differs from *expected* ('Vega' or 'AB')."""
        actual = self.zero_point(band)
        if actual != expected:
            raise ValueError(
                f"BC table convention mismatch for band '{band}': "
                f"BC table is {actual}, caller expects {expected}. "
                "Mixing Vega and AB magnitudes leads to band-specific "
                "biases of ~0.1-0.5 mag."
            )

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
        """Apparent magnitude: m = Mbol - BC + 5*log10(d/10).

        Single-pass hot path: interpolates the BC once and folds the bolometric
        magnitude and distance modulus straight in, avoiding the two
        intermediate dicts of get_bc → get_absolute_mag (this is called once per
        likelihood evaluation, O(10^5–10^6) times per fit).
        """
        if _HAS_NUMBA and bands is None:
            band_names = self._active_bands if self._active_bands is not None else self._bands
            grid = self._active_grid if self._active_grid is not None else self._grid
            n = len(band_names)
            vals = _quadlinear(
                grid, self._ax_teff, self._ax_logg, self._ax_feh, self._ax_av,
                teff, logg, feh, av, n,
            )
            offset = mbol + 5.0 * math.log10(distance_pc / 10.0)
            return {band_names[i]: offset - float(vals[i]) for i in range(n)}
        # General path (scipy fallback or band subset)
        abs_mags = self.get_absolute_mag(mbol, teff, logg, feh, av, bands)
        dm = 5.0 * math.log10(distance_pc / 10.0)
        return {band: m + dm for band, m in abs_mags.items()}
