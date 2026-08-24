# Changelog

## [1.0.7] - 2026-08-24

### Added
- `Fitter.include_pms` (default `False`) samples the pre-main-sequence on grids
  that carry it. MIST and PARSEC both hold PMS models down to log t = 5.0, but
  `fitting_eep_range` was hard-coded to (202, 808) -- ZAMS onward -- so a young
  star could not be fitted at all: at log t = 7.0 the allowed band holds no cool
  dwarfs, only 576-1125 Rsun supergiants, and below log t ~ 6.7 it is empty.
  Combined with the hard-walled KDE priors that `Star.from_ariadne` installs on
  Teff, log g, radius, distance, Av and luminosity, an upstream posterior for an
  inflated young star put every proposal at -inf, so every grid was dropped and
  the fit died reporting the star as outside all grid coverage. Without those
  priors the same fit instead rails to the old end. Only grids declaring
  `pms_eep_max` opt in, since EEP is not a common coordinate --
  basti/bhac15/geneva/yapsi/dartmouth/starevol index by mass or row.

  Turn it on only for targets independently known to be young. The PMS is
  degenerate with the post-main-sequence in Teff/log g, and for intermediate
  masses the posterior goes bimodal; on an old control the spurious sub-100 Myr
  tail ran to 7-14% across realisations.

### Changed
- The age interval is now the intersection across the selected grids instead of
  each grid falling back to its own native minimum. The ceiling was already
  equalised by the `age_range` default sitting below every grid's maximum, but
  the floor was not, so mist/parsec sampled from log t = 8.0 while
  dartmouth/basti started at 9.0 -- unequal support inside an evidence
  comparison. The default five-grid set is now pinned to 9.0 by dartmouth/basti,
  which also keeps the PMS unreachable there; a deliberate mist+parsec selection
  keeps the young range. **This moves results for field stars between 0.1 and
  1 Gyr.**

- PARSEC does not opt into `include_pms`. Its EEP < 202 band is anchored on
  PARSEC's own phase label 0->1 transition, which stays 0 well past the ZAMS, so
  the band is contaminated with main-sequence models: measured against MIST's
  PMS mass ceiling it is 1.3x too massive at log t = 7, 2.0x at log t = 8 and
  4.7x at log t = 9, where a 0.65 Msun star has been on the main sequence for
  ~800 Myr. Opening it drove a B9V main-sequence star (3.06 Msun, 280 Myr) to a
  2 Myr solution with PARSEC taking 0.66 of the BMA weight, while on a genuinely
  young target PARSEC scored 14 log-units below MIST. PMS fits are therefore
  MIST-only for now.

### Removed
- `Fitter.eep_range`. It never had any effect -- every shipped grid defines
  `fitting_eep_range`, so the branch reading it was unreachable -- and it cannot
  work as a global knob because EEP means different things per grid.

## [1.0.6] - 2026-08-20

### Changed
- Default age prior upper bound lowered from log t = 10.3 (19.95 Gyr) to
  10.1399 (13.8 Gyr, Planck 2018 age of the universe). Besides removing
  unphysical super-Hubble ages, this equalizes the age support across grids
  with different native ceilings (MIST 10.30, PARSEC 10.25,
  DSEP/BaSTI/YaPSI 10.176), closing a BMA bias where the common-measure
  evidence correction (`ln(age box)`) rewarded the widest-axis grid exactly
  on stars whose likelihood rails old, e.g. HD 191939: MIST weight 0.88 ->
  0.10 and BMA median 18.7 -> 12.1 Gyr after the change. Override via
  `Fitter.age_range` for grid-systematics experiments.

- Geneva is excluded from BMA, as the README has always documented but nothing
  enforced, and is removed from the default grid list so `bma = True` on the
  defaults does not trip the guard. Its coverage (0.5-3.5 Msun, log t 7.5-10.2)
  is too narrow to put on a common evidence scale with the other grids. Until
  now the a-posteriori [Fe/H] rail drop removed it by accident, and only while
  lachesis-grids shipped a solar-only Geneva.

### Fixed
- The shipped Geneva grid was cited as Ekstroem+ 2012 (0.8-120 Msun, solar
  metallicity, rotating). It is built by `scripts/build_geneva_mowlavi.py`
  from the non-rotating Mowlavi+ 2012 grid (VizieR J/A+A/541/A41, Z = 0.006
  to 0.040, M = 0.5-3.5 Msun), so anyone citing Geneva from a LACHESIS run was
  citing the wrong paper. `citations.md`, the README grid table and the
  `geneva.py` docstring are corrected.
- A metallicity FIXED in the upstream ARIADNE fit arrives as a constant
  posterior column, and every layer that consumed it failed differently.
  `gaussian_kde` raised `LinAlgError` only when the covariance evaluated to
  exactly zero, which depends on the bit pattern of the fixed value: a
  constant -0.5 raised, a constant -0.13 built a KDE of bandwidth ~1e-18
  whose mass fell between two nodes of the tabulated inverse CDF, so the
  normalisation was 0/0 and every draw came back as `feh_lo`. That fit ran
  to completion with [Fe/H] pinned at the grid's metallicity floor and a zero
  error bar. [Fe/H] priors are now screened on `np.ptp` and a degenerate one
  becomes an explicit fixed prior. `prior_setup={'feh': ('fixed', v)}` is
  recognised (it was silently ignored, falling through to the ARIADNE KDE).
- The inverse CDF for a KDE [Fe/H] prior is tabulated on the sample support
  rather than the full prior range. A posterior narrower than one grid cell
  (2.4e-3 dex on the default range) resolved to one or two nodes, which
  `np.interp` turns into a pin or a uniform ramp: a posterior at -0.13 with
  sigma 2e-4 drew a mean of -1.07 with a minimum of -2.0. Realistic
  posteriors are unaffected (verified to sub-millidex against the old
  tabulation).
- Observable uncertainties are validated before use. `np.std` of a constant
  column is 0 for some values and a ~5.6e-17 residue for others; the first
  raised a bare `math domain error` out of `log(sigma)` well after
  `initialize()` had succeeded, and the second displaced each grid's
  log-evidence by ~1e32 with no error at all, collapsing the BMA onto one
  grid. `Star.from_ariadne` now reports no uncertainty for a parameter that
  was fixed upstream, and `Star.observed` / `Star.uncertainties` agree on
  membership.
- An upstream Av prior is no longer applied to a star inside the Local
  Bubble, where `av_range` is None and Av is not sampled. The sampler read it
  back as None and returned `-inf` for every proposal, so the fit died in
  dynesty's live-point initialisation and was reported as the star lying
  outside the coverage of every grid. This affected any ARIADNE-sourced star
  within 70 pc, whether or not [Fe/H] was fixed.
- External priors built from a degenerate posterior column are rejected by
  name instead of skipped with a bare `except LinAlgError: continue`. The
  silent skip deleted a whole constraint from the likelihood, and tabulating
  it instead collapsed the 2048-node table onto a single point, after which
  the sampler rejected every proposal.
- The [Fe/H] grid-coverage drop was gated on a Gaussian prior, so it never
  ran on the ARIADNE path, which produces a KDE prior. It now keys on the
  prior's central value, warns unconditionally rather than only when verbose,
  and records to `dropped_grids`.
- Zero-width prior boxes no longer produce infinities. `-log(0)` made
  `log_prior` `+inf` for a fixed Av, and `log(0)` made the BMA common-scale
  correction `-inf` for a single-metallicity grid, which reaches
  `bayesian_model_average` as NaN weights, zero samples drawn from every
  model, and an empty combined posterior returned as a success.
  `bayesian_model_average` now refuses a non-finite log-evidence.
- `_eep_to_state` returned "TP-AGB / post-AGB" for a NaN EEP, because every
  `eep < threshold` comparison is False for NaN and the loop fell through to
  the last label. A fit with no posterior reported the most evolved phase
  there is.
- A distance prior with a non-positive sigma raises instead of dividing by
  it, which gave `ZeroDivisionError` for a Python float and silent NaN
  parameter vectors for a `np.float64`.
- `show_priors` reported a fixed [Fe/H] as a wide uniform, and a
  single-metallicity grid as `U(0.00, 0.00)`.
- `librarian/_api.py`: the RUWE > 1.4 unresolved-binary warning crashed with
  `NameError` (missing local `termcolor` import), the only unconditional
  `colored()` call site without one. Any target with RUWE > 1.4 failed to
  build a `Star` at all.

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
