# LACHESIS (modeL Averaging for isoChrone cHaractErization of Stellar propertIeS)
## Stellar masses, ages, and evolutionary states from photometry

**LACHESIS** is a Python code that estimates stellar mass, age, and evolutionary
state by fitting broadband photometry to isochrone grids using nested sampling,
with Bayesian Model Averaging across multiple stellar evolution models.

It is designed as the natural companion to
[**ARIADNE**](https://github.com/jvines/astroARIADNE)
([Vines & Jenkins 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.2719V/abstract)):
where ARIADNE characterizes stellar atmospheres (Teff, logg, [Fe/H], radius)
via SED fitting, **LACHESIS** takes over for the evolutionary parameters that
require isochrone models.

# Installation

```bash
pip install lachesis
```

For development:

```bash
git clone https://github.com/jvines/LACHESIS.git
cd LACHESIS
pip install -e .
```

All isochrone grids and bolometric correction tables ship with the package
(~120 MB) — no extra downloads or environment variables needed.

# Quick start

The fastest way to use **LACHESIS** is the one-liner interface. Just give it a
star name:

```python
import lachesis

result = lachesis.fit("HD 209458")
```

This will:
1. Resolve the name via Simbad to get Gaia DR3 coordinates
2. Query Gaia, 2MASS, WISE, SDSS, PanSTARRS, and other catalogs for photometry
3. Look for spectroscopic priors in APOGEE, GALAH, RAVE, LAMOST, and PASTEL
4. Fit the photometry against 5 isochrone grids (MIST, PARSEC, Dartmouth, BaSTI, YAPSI)
5. Combine the results via Bayesian Model Averaging

The output is a `BMAResult` object containing the combined posterior samples and
derived physical parameters:

```python
import numpy as np

# Posterior samples for mass and age
mass = result.derived["initial_mass"]
age = 10 ** result.samples[:, 1] / 1e9  # log_age → Gyr

print(f"Mass: {np.median(mass):.3f} Msun")
print(f"Age:  {np.median(age):.2f} Gyr")
```

If you already know the coordinates or Gaia ID (useful for high proper-motion
stars where a cone search might grab the wrong source):

```python
result = lachesis.fit("HD 103095", gaia_id=4034171629042489088)
result = lachesis.fit("WASP-19", ra=148.417, dec=-45.659)
```

# Stellar information setup

To use **LACHESIS** start by setting up the stellar information, this is done by
importing the Star module.

```python
from lachesis.star import Star
```

Stars are defined in **LACHESIS** by their RA and DEC in degrees, a name, and
optionally the Gaia DR3 source id, for example:

```python
starname = 'HD 209458'
ra = 330.795
dec = 18.884
gaia_id = 1779546757669063552

s = Star(starname, ra, dec, g_id=gaia_id)
```

The starname is used for identification, and the `g_id` is provided to make sure
the automatic photometry retrieval collects the correct magnitudes, otherwise
**LACHESIS** will try and get the `g_id` by itself using a cone search centered
around the RA and DEC.

Executing the previous block will start the photometry and stellar parameter
retrieval routine. **LACHESIS** will query Gaia DR3 for parallax, then cross-match
against 2MASS, WISE, SDSS, PanSTARRS, SkyMapper, GALEX, APASS, and Tycho-2 for
broadband photometry. It will also search for spectroscopic priors in APOGEE,
GALAH, RAVE, LAMOST, and PASTEL (in that priority order).

If you want to check the retrieved magnitudes you can call the `print_mags`
method from Star:

```python
s.print_mags()
```

If you already have the magnitudes and wish to override the on-line search, you
can provide a dictionary where the keys are the filters and values are the
(mag, mag_err) tuples:

```python
s = Star(starname, ra, dec, g_id=gaia_id, magnitudes={
    '2MASS_J': (7.49, 0.02),
    '2MASS_H': (7.18, 0.03),
    '2MASS_Ks': (7.09, 0.02),
})
```

If **LACHESIS** found a bad photometry point, you can remove it:

```python
s.remove_mag('WISE_RSR_W2')
```

Or add one manually:

```python
s.add_mag(7.49, 0.02, '2MASS_J')
```

### Interstellar extinction

**LACHESIS** has an incorporated prior for the interstellar extinction in the
Visual band, $A_{\rm V}$ which consists of a uniform prior between 0 and the
maximum line-of-sight value provided by the
[SFD dust maps](https://ui.adsabs.harvard.edu/abs/2011ApJ...737..103S/abstract).
This, however, can be changed by providing a different dustmap:

```python
s = Star(starname, ra, dec, g_id=gaia_id, dustmap='Bayestar')
```

We provide the following dustmaps (same as ARIADNE):

- [SFD (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJ...737..103S/abstract)
- [Planck Collaboration (2013)](http://adsabs.harvard.edu/abs/2014A%26A...571A..11P)
- [Planck Collaboration (2016; GNILC)](https://ui.adsabs.harvard.edu/abs/2016A%26A...596A.109P/abstract)
- [Lenz, Hensley & Doré (2017)](https://arxiv.org/abs/1706.00011)
- [Bayestar (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G)

**These maps are all implemented through the
[dustmaps](https://dustmaps.readthedocs.io/en/latest/index.html) package and
need to be downloaded. Instructions to download the dustmaps can be found in
its documentation.**

## From an ARIADNE result

If you've already run **ARIADNE** on a star, you can pass its `.nc` output
directly to **LACHESIS**. This loads the full ARIADNE posterior and builds
KDE-based external priors for [Fe/H] and logg (the parameters where ARIADNE's
posterior collapses to the spectroscopic prior, so no double-counting occurs).
Provide `ra`/`dec` so that **LACHESIS** can retrieve photometry for the
isochrone fit:

```python
s = Star.from_ariadne("ariadne_result.nc", starname="HD 209458",
                       ra=330.795, dec=18.884)
```

In this mode, **LACHESIS** fits the photometry with informed [Fe/H] and logg
priors from ARIADNE. The final reported parameters combine the best of both:
Teff, radius, Av, and distance from ARIADNE; mass, age, and evolutionary state
from LACHESIS; [Fe/H] and logg from either (both collapse to the spectroscopic
prior).

# Fitter setup

In this section we'll detail how to set up the fitter for the Bayesian Model
Averaging (BMA) mode of **LACHESIS**. For single grids the procedure is very
similar.

First, import the fitter from **LACHESIS**:

```python
from lachesis.fitter import Fitter
```

There are several configuration parameters we have to set up, the first one is
the output folder where we want **LACHESIS** to output the fitting files and
results, next we have to select the fitting engine (only dynesty is supported),
number of live points to use, evidence tolerance threshold, bounding method,
sampling method, threads, and dynamic nested sampler. After selecting all of
those, we need to select the grids we want to use and finally, we feed them all
to the fitter:

```python
out_folder = 'your folder here'

engine = 'dynesty'
nlive = 500
dlogz = 0.5
bound = 'multi'
sample = 'rwalk'
threads = 4
dynamic = False

setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

# Feel free to comment out any unneeded/unwanted grids
grids = [
    'mist',
    'parsec',
    'dartmouth',
    'basti',
    'yapsi',
]

f = Fitter()
f.star = s
f.setup = setup
f.av_law = 'fitzpatrick'
f.out_folder = out_folder
f.bma = True
f.grids = grids
```

We allow the use of four different extinction laws:

- fitzpatrick
- cardelli
- odonnell
- calzetti

The next step is setting up the priors to use:

```python
f.prior_setup = {
    'eep': ('default'),
    'log_age': ('default'),
    'feh': ('default'),
    'dist': ('default'),
    'Av': ('default'),
}
```

A quick explanation on the priors:

The default prior for the distance is a truncated normal drawn from the
[Bailer-Jones](https://ui.adsabs.harvard.edu/abs/2021AJ....161..147B/abstract)
distance estimate from Gaia DR3 (matching ARIADNE's behavior). The default
prior for [Fe/H] is uniform unless spectroscopic priors are found in APOGEE,
GALAH, RAVE, LAMOST, or PASTEL, in which case a Gaussian prior is used
automatically. The default prior for Av is a flat prior that ranges from 0 to
the maximum line-of-sight value from the selected dustmap.

We offer customization on the priors as well:

| Prior | Hyperparameters |
| :------: | :----------: |
| Fixed | value |
| Normal | mean, std |
| Uniform | ini, end |
| Default | --- |

So if you knew from a spectroscopic analysis that [Fe/H] = 0.09 +/- 0.05 and
the star is nearby (< 70 pc) so you wanted to fix Av to 0, your prior
dictionary should look like this:

```python
f.prior_setup = {
    'eep': ('default'),
    'log_age': ('default'),
    'feh': ('normal', 0.09, 0.05),
    'dist': ('default'),
    'Av': ('fixed', 0),
}
```

Though leaving everything at default usually works well enough.

After having set up everything we can finally initialize the fitter and start
fitting:

```python
f.initialize()
f.fit_bma()
```

Now we wait for our results!

To see the full prior configuration:

```python
f.show_priors()
```

## Single-grid fits

Sometimes you don't want BMA — either because you're targeting a specific
stellar population a single grid is best suited for, or because the grid
itself can't participate in BMA (Geneva/BHAC15/STAREVOL). Set `f.bma = False`
and a single grid name:

```python
f = Fitter()
f.star = s
f.grids = ["bhac15"]
f.bma = False
f.setup = ["dynesty", 1000, 0.01, "multi", "rwalk", 4, False]
f.initialize()
f.fit()
```

The routine display will show `Selected engine : Single model (BHAC15)`
(or whichever grid you chose) to make the mode explicit.

Three grids ship with **LACHESIS** that are only available as single-grid
fits — they have intentionally narrow coverage that would bias BMA evidence
comparisons, but they're the right tool for their target populations:

- **BHAC15** (Baraffe+ 2015) — M dwarfs, 0.01–1.4 Msun, solar metallicity.
  See `test_bhac15.py` for a Proxima Centauri example.
- **Geneva** (Ekstroem+ 2012) — massive/intermediate-mass tracks, solar
  metallicity only. See `test_geneva.py`.
- **STAREVOL** (Amard+ 2019) — includes stellar rotation (Vini) as a 6th
  sampled parameter. See `test_starevol.py`.

You can also run any of the BMA grids (MIST, PARSEC, etc.) as a single-grid
fit if you just want the posterior from one specific model.

# Available grids

| Grid | [Fe/H] range | Age range | EEPs | BMA | Notes |
|------|:---:|:---:|:---:|:---:|-------|
| **MIST** | -4.0 to +0.5 | 5.0–10.3 | 202–808 | Yes | Default reference grid |
| **PARSEC** | -2.2 to +0.5 | 6.6–10.1 | 1–1200 | Yes | Gap-filled inter-phase regions |
| **Dartmouth** | -2.5 to +0.5 | 9.0–10.1 | 2–279 | Yes | Native DSEP EEPs |
| **BaSTI** | -3.2 to +0.4 | 7.9–10.2 | 1–2100 | Yes | Mass-index EEPs |
| **YAPSI** | -0.75 to +0.55 | 8.0–10.2 | 1–71 | Yes | Yale-Potsdam |
| **Geneva** | 0.0 only | 6.0–10.1 | 1–363 | No | Solar metallicity only |
| **BHAC15** | 0.0 only | 6.0–10.0 | 1–30 | No | M dwarfs, 0.01–1.4 Msun |
| **STAREVOL** | -2.14 to +0.41 | 7.0–10.1 | 1–422 | No | Includes rotation (Vini) |

Geneva, BHAC15, and STAREVOL are excluded from BMA because they have limited
parameter coverage (single metallicity, rotation parameter) that would bias the
evidence comparison. They work well as standalone fits for their target
populations.

# Visualization

**LACHESIS** includes a publication-quality plotter that follows ARIADNE's visual
style (serif fonts, consistent sizing). Like ARIADNE's `SEDPlotter`, you point
it at the results file and an output folder:

```python
from lachesis.plotter import ISOPlotter

in_file = out_folder + '/lachesis_HD_209458_BMA.nc'
plots_out = out_folder + '/plots'

artist = ISOPlotter(in_file, plots_out)
artist.plot_corner()
artist.plot_histograms()
artist.plot_hr()
artist.plot_mass_age()
artist.plot_model_weights()
artist.summary()
```

Since the plotter loads from the .nc file, you can re-plot at any time without
re-running the fit. If you're iterating through many stars, call
`artist.clean()` to close opened figures.

## LaTeX output

For papers, you can get formatted parameter strings for the parameters
**LACHESIS** actually constrains (mass, age, [Fe/H], logg):

```python
latex = artist.to_latex()
for param, val in latex.items():
    print(f"{param}: {val}")
# initial_mass: $1.148^{+0.037}_{-0.034}$
# age_gyr: $3.42^{+1.21}_{-0.89}$
# [Fe/H]: $-0.01^{+0.05}_{-0.04}$
# log_g: $4.35^{+0.02}_{-0.03}$
```

For Teff, radius, distance, and Av, use **ARIADNE**'s output — those are better
constrained by the SED fit.

# Output files

When `f.out_folder` is set, **LACHESIS** writes:

- `lachesis_{starname}_BMA.nc` — Full posterior as an arviz InferenceData
  (netCDF4). This is the canonical format for passing results to downstream
  tools.
- `lachesis_{starname}_BMA.dat` — Summary statistics (median, 16th/84th
  percentiles) in human-readable format.
- `model_weights.dat` — BMA posterior model probabilities and log-evidences per
  grid.

## Reading results

```python
import arviz as az

idata = az.from_netcdf("output/lachesis_HD_209458_BMA.nc")
print(idata.posterior)
```

# Using ARIADNE and LACHESIS together

**LACHESIS** and
[**ARIADNE**](https://github.com/jvines/astroARIADNE)
([Vines & Jenkins 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.2719V/abstract))
are designed to work together for end-to-end stellar characterization. Both
exchange full posteriors via arviz InferenceData in netCDF4 format. Each tool
is standalone — the pipeline is the happy path, not the only path.

When used together, the recommended pipeline is:

1. **ARIADNE** fits the SED → Teff, radius, distance, Av, luminosity
2. **LACHESIS** fits isochrones with ARIADNE's [Fe/H] and logg posteriors as KDE priors → mass, age, evolutionary state
3. Final parameters combine: atmosphere from ARIADNE, evolution from LACHESIS

| Parameter | Best source |
|-----------|------------|
| Teff | ARIADNE |
| Radius | ARIADNE |
| Distance | ARIADNE |
| Av | ARIADNE |
| Luminosity | ARIADNE |
| [Fe/H] | Either (spectroscopic prior) |
| logg | Either (spectroscopic prior) |
| **Mass** | **LACHESIS** |
| **Age** | **LACHESIS** |
| **Evol. state** | **LACHESIS** |

# Known limitations

- **Very bright stars (V < 2)**: Survey photometry (2MASS, WISE, etc.) is
  saturated for the brightest stars. **LACHESIS** will warn you when this
  happens. Use manually curated photometry instead.
- **Young ages (< ~1 Gyr)**: Isochrone fitting has limited age sensitivity for
  young main-sequence stars where the HR diagram is nearly degenerate. Cluster
  membership, lithium, or gyrochronology are better age indicators in this
  regime.

# Citing LACHESIS

If you use **LACHESIS** in your research, please cite:

> Vines et al. (in prep.)

A BibTeX entry will be provided when the paper is published.
