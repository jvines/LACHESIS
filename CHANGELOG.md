# Changelog

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
