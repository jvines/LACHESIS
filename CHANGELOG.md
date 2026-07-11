# Changelog

## [1.0.5] - 2026-07-11

### Changed
- `prior_transform` hot-path optimizations (~10-15% faster fits, results
  bit-identical): the per-band jitter draw is vectorized (was a Python loop over
  bands), and the truncated-normal distance-prior CDF bounds are precomputed
  once instead of two `ndtr` evaluations per proposal.

## [1.0.4] - 2026-07-11

### Changed
- Removed the inner per-likelihood `ThreadPool` (previously built when
  `setup['threads'] > 1`). dynesty calls a GIL-holding Python likelihood closure
  around a ~20 us njit kernel, so the pool added per-proposal dispatch overhead
  with no real parallelism, making fits ~1.4x slower and worse the more
  photometric bands (likelihood calls) there are. Per-grid fits now run
  single-threaded; BMA parallelism comes from the grid-level process pool
  (`n_grid_jobs`). A 16-dim per-band fit drops from ~6.5 s to ~4.8 s.

## [1.0.3] - 2026-07-11

### Changed
- Photometric excess noise is now fit as ONE white-noise term PER photometric
  band (ARIADNE-style), replacing the single global jitter. Each band's variance
  is inflated independently: `sigma_eff[k]^2 = sigma_cat[k]^2 + noise[k]^2`,
  with an independent log-uniform prior per band. The parameter vector is now
  `5 + n_bands` and the terms are named `<band>_noise` (e.g. `2MASS_J_noise`) to
  match ARIADNE's output convention.

## [1.0.2] - 2026-07-10

### Fixed
- The `.nc` posterior `age` variable stored log10(age) instead of age in Gyr:
  the derived-quantities loop in `to_inference_data` overwrote the correct Gyr
  conversion with the grid's log-valued `derived['age']`. Affected single-grid
  and BMA netCDF outputs (the `.dat` `Age(Gyr)` column was already correct).
  Downstream consumers reading `posterior['age']` from the `.nc` now get Gyr.

## [1.0.1] - 2026-07-09

### Fixed
- PARSEC grid registry pointed at `parsec_v1.2S.h5`, but lachesis-grids 0.0.3
  ships the EEP-rebuilt cube as `parsec_v1.2S_eeprebuild.h5`, so any fit
  including PARSEC crashed with `GridError` at `initialize()`. Broken in 1.0.0.
- Sort unsorted grid axes at load, permuting the cube (fixes the PARSEC
  `feh_values` ordering shipped in lachesis-grids 0.0.3).
- `__version__` now reports the correct version (was `0.0.10` at the 1.0.0 tag).

## [1.0.0] - 2026-07-09

First public release, accompanying the LACHESIS paper (Vines et al., submitted
to Astronomy & Astrophysics).

### Added
- Hypatia Catalog as the default [Fe/H] prior source, with the survey chain
  (PASTEL, APOGEE, GALAH, RAVE, LAMOST) as a fallback.
- Gaia DR3 RUWE retrieval and an unresolved-binary warning (RUWE > 1.4).
- Per-grid nested-sampling log-evidence uncertainty (`log_evidence_err`),
  persisted to `model_weights.dat` and the `.nc` output.
- Numba-compiled log-likelihood and parallel per-grid fitting.
- Blackbody and bolometric-correction photometry quality control with
  errorbar-independent outlier flagging, plus SED plotting.
- Evolutionary-state classification.

### Changed
- Missing catalogue magnitude errors now fall through to zero and receive the
  conservative imputed uncertainty (previously a 0.01/0.02 mag placeholder).
- Extinction applied through a Fitzpatrick (1999) law with R_V = 3.1.

### Earlier (0.0.x)
- 0.0.10: drop silent Av=0.1 fallback; raise ExtinctionError on dustmap failure.
- 0.0.9: auto-drop grids whose [Fe/H] axis is incompatible with the star's prior.
- 0.0.8: drop Gaia GSP-Phot Teff prior; retry on dynesty/scipy kmeans crash.
- 0.0.7: fix radius/density linear interpolation through cube corners.
- 0.0.6: fix `_BMA.dat` sampled-row loss and plotter `gaussian_kde` NameError.
