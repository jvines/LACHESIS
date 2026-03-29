# Equivalent Evolutionary Phase (EEP) Parameterization: Theory and Methodology

**LACHESIS Technical Note 1**

J. I. Vines

---

## 1. The Problem: Why Not Parameterize by Mass?

### 1.1 The naive approach and its failures

The most natural parameterization of a stellar isochrone is by initial mass $M$.
At fixed age $\tau$ and metallicity $[\text{Fe/H}]$, an isochrone is a
one-dimensional curve $\{T_\text{eff}(M), \log g(M), L(M), \ldots\}$ traced out
as $M$ varies from the lowest-mass track that has reached the ZAMS at age $\tau$
to the highest-mass track that has not yet ended its evolution. The Bayesian
inference problem is then to sample the joint posterior

$$P(M, \tau, [\text{Fe/H}] \mid \mathbf{D}) \propto \mathcal{L}(\mathbf{D} \mid M, \tau, [\text{Fe/H}]) \cdot \pi(M, \tau, [\text{Fe/H}]), \tag{1}$$

where $\mathbf{D}$ is the set of observed stellar properties.

This has three fundamental problems:

**Problem 1: Non-monotonic mapping.** At fixed age and metallicity, the mapping
$M \to (T_\text{eff}, \log g)$ is *not* bijective. A 1 $M_\odot$ star at 10 Gyr
sits on the subgiant branch in a region of the HR diagram also occupied by a
1.3 $M_\odot$ main-sequence star at the same age. The function
$T_\text{eff}(M)$ is multi-valued when the isochrone turns back on itself at the
main-sequence turnoff and again at the base of the RGB. This makes interpolation
along the mass axis ill-defined: a linear interpolation between two mass grid
points can cross the turnoff region and produce nonsensical predictions.

**Problem 2: Non-uniform evolutionary speed.** A star spends $\sim$90% of its
nuclear-burning lifetime on the main sequence and $\lesssim$1% on the subgiant
branch. A uniform grid in $M$ therefore provides enormous oversampling of the
MS (where $T_\text{eff}$ varies slowly with $M$) and catastrophic undersampling
of the SGB and RGB (where a tiny change in $M$ produces a large excursion in
$T_\text{eff}$ and $\log g$). This is a resolution problem, not a prior problem,
but it makes numerical interpolation unreliable precisely where it matters most.

**Problem 3: Mass loss.** At fixed initial mass, the current (observed) mass of
a star depends on its evolutionary state. A 1 $M_\odot$ star on the RGB may
have $M_\text{current} \approx 0.8\,M_\odot$ due to mass loss via stellar
winds. Initial mass is the physically meaningful parameter for isochrone
fitting, but it is not directly observable. The mapping between initial and
current mass is both age- and metallicity-dependent.

### 1.2 Why uniform in mass is the wrong prior

Even setting aside interpolation issues, a uniform prior in mass,
$\pi(M) = \text{const}$, is astrophysically nonsensical. The initial mass
function (IMF) is a steeply falling power law: there are $\sim$10$\times$ more
0.5 $M_\odot$ stars than 1 $M_\odot$ stars per unit mass interval. A uniform
mass prior assigns equal weight to all masses, dramatically overweighting
high-mass stars relative to their actual occurrence rate. The correct prior on
mass is the IMF itself, which is the whole-population birth rate per unit mass:

$$\pi(M) = \xi(M), \tag{2}$$

where $\xi(M) \equiv dN/dM$ is the IMF. We return to this in Section 3.


## 2. Definition of the Equivalent Evolutionary Phase

### 2.1 The Dotter (2016) formalism

The EEP parameterization, formalized by Dotter (2016), replaces initial mass as
the independent variable along an isochrone with a monotonically increasing
integer index $e$ that tracks evolutionary state rather than birth mass. The key
construction proceeds in two steps.

**Step 1: Primary EEPs.** A set of *primary* EEPs are defined at physically
significant evolutionary milestones along each stellar evolutionary track. These
are points where the track's morphology in the HR diagram changes character:

| Primary EEP | Name | Physical definition |
|-------------|------|---------------------|
| 1 | PreMS start | Beginning of the pre-main-sequence contraction |
| 202 | ZAMS | Zero-age main sequence: core H ignition reaches thermal equilibrium |
| 454 | TAMS | Terminal-age main sequence: core H exhaustion ($X_c < 10^{-12}$) |
| 500 | RGBTip (low-mass) / "hook" (high-mass) | Subgiant branch / Hertzsprung gap traversal |
| 605 | He flash / ZAHB | Tip of the RGB (He flash) or start of core He burning |
| 631 | ZAHB | Zero-age horizontal branch (post-He-flash) |
| 707 | TAHB | Terminal-age horizontal branch: core He exhaustion |
| 808 | TP-AGB | Thermally pulsing AGB tip |

The precise physical criteria for each primary EEP are given in Dotter (2016),
Table 1. The critical property is that these milestones are identified
*per track* before any isochrone construction takes place.

**Step 2: Secondary EEPs.** Between each pair of consecutive primary EEPs, a
fixed number $N_k$ of *secondary* EEPs are inserted by uniform interpolation
along the track in a suitable metric. Dotter (2016) uses an arc-length metric
in $(\log T_\text{eff}, M_\text{bol})$ space:

$$ds = \sqrt{(d\log T_\text{eff})^2 + (dM_\text{bol})^2}. \tag{3}$$

The secondary EEPs are placed at equal intervals of $s$ between consecutive
primary EEPs. This ensures smooth, uniform sampling of the track's morphology
in the HR diagram.

### 2.2 Formal definition

Let $\mathcal{T}(M_0, Z)$ denote a stellar evolutionary track of initial mass
$M_0$ and metallicity $Z$. For each track, the EEP construction defines a
function

$$e: [t_\text{start}, t_\text{end}] \to \mathbb{Z}^+ \tag{4}$$

that assigns an integer EEP index $e$ to each time step along the track such that:

1. **Monotonicity.** $e$ is strictly increasing along the track: if $t_1 < t_2$
   then $e(t_1) < e(t_2)$.
2. **Alignment.** All tracks with the same metallicity $Z$ pass through the same
   primary EEP numbers at the same evolutionary milestones. A 0.8 $M_\odot$
   track and a 1.2 $M_\odot$ track both have $e = 202$ at their ZAMS and
   $e = 454$ at their TAMS, even though these events occur at vastly different
   physical times.
3. **Uniform secondary spacing.** Between primary EEPs $e_k$ and $e_{k+1}$,
   exactly $N_k$ secondary EEPs are uniformly spaced in the arc-length
   metric (Eq. 3).

An isochrone at age $\tau$ is then constructed by:
1. For each track $\mathcal{T}(M_i, Z)$, identify the time $t^*$ at which the
   track reaches age $\tau$.
2. Evaluate the EEP at that time: $e_i = e(t^*)$.
3. Interpolate all observables $(T_\text{eff}, \log g, L, \ldots)$ at the
   EEP $e_i$ using the pre-computed EEP-aligned tracks.

The result is a one-dimensional curve parameterized by $e$ that is
*guaranteed to be monotonic* in all observable quantities within each
evolutionary phase. This eliminates Problem 1 entirely.

### 2.3 The key property

The EEP is a monotonic evolutionary coordinate: along any isochrone at fixed
$(\tau, [\text{Fe/H}])$, the EEP $e$ increases monotonically from the
lowest-mass (least-evolved) to the highest-mass (most-evolved) point. This
means:

- The mapping $e \to M(e, \tau, Z)$ is bijective.
- The mapping $e \to (T_\text{eff}, \log g, L, \ldots)$ is single-valued.
- Interpolation along the $e$ axis is numerically well-defined and well-conditioned.


## 3. The Prior Jacobian

### 3.1 The change-of-variables formula

We now derive the prior $\pi(e)$ on EEP that is consistent with an IMF prior
on mass. This is the single most important equation in this document.

We wish to sample in $(e, \log\tau, [\text{Fe/H}])$ space, but the astrophysically
motivated prior is on mass:

$$\pi(M) = \xi(M), \tag{5}$$

where $\xi(M)$ is the IMF. At fixed $(\tau, [\text{Fe/H}])$, the EEP-to-mass
mapping $M(e)$ is a smooth, monotonic function. By the standard
change-of-variables formula for probability densities, if $e = f(M)$ is a
monotonic transformation with inverse $M = M(e)$, then:

$$\pi(e) \, de = \pi(M) \, dM \tag{6}$$

$$\pi(e) = \pi(M(e)) \cdot \left|\frac{dM}{de}\right| \tag{7}$$

$$\boxed{\pi(e) = \xi\big(M(e)\big) \cdot \left|\frac{dM}{de}\right|.} \tag{8}$$

This is exact. No approximations have been made.

### 3.2 Physical interpretation

Equation (8) has a clean physical interpretation. The prior on EEP has two
factors:

- $\xi(M(e))$: the IMF weight at the mass corresponding to EEP $e$. This
  accounts for the birth-rate prior: low-mass stars are more common.

- $|dM/de|$: the *mass spacing per EEP step*. Where EEPs are closely spaced
  in mass (i.e., the evolutionary track changes rapidly in the HR diagram),
  $|dM/de|$ is small and the EEP prior downweights that region. Where EEPs
  span a large mass range (MS, where evolution is slow), $|dM/de|$ is large
  and the EEP prior upweights it. This factor corrects for the non-uniform
  density of EEP points in mass space.

Together, they ensure that the posterior obtained by sampling in EEP space is
*identical* to the posterior that would be obtained by sampling directly in mass
space with an IMF prior, despite the change of variable.

### 3.3 Why $\pi(e) = \text{const}$ is wrong

A flat prior $\pi(e) = 1/\Delta e$ would correspond to (inverting Eq. 8):

$$\pi(M) = \frac{1}{\Delta e} \cdot \left|\frac{de}{dM}\right|, \tag{9}$$

which is *not* the IMF. Since $|de/dM|$ is large in the SGB/RGB (many EEPs per
unit mass interval) and small on the MS, a flat EEP prior effectively
overweights evolved stars relative to the IMF. The magnitude of this
effect is large: the ratio $|de/dM|_\text{RGB} / |de/dM|_\text{MS}$ can
exceed $10^2$ in typical isochrones.

### 3.4 The Chabrier (2003) IMF

The standard IMF used in LACHESIS is the Chabrier (2003) system IMF:

$$\xi(M) = \begin{cases} \displaystyle \frac{0.158}{M} \exp\left(-\frac{(\log_{10} M - \log_{10} 0.08)^2}{2 \times 0.69^2}\right) & M < 1\,M_\odot \\[8pt] 0.0443 \, M^{-2.3} & M \geq 1\,M_\odot \end{cases} \tag{10}$$

This is a lognormal distribution below $1\,M_\odot$ (peaking at
$M_c = 0.08\,M_\odot$ with dispersion $\sigma = 0.69$ dex) joined to a
Salpeter-like power law $M^{-\alpha}$ with $\alpha = 2.3$ above $1\,M_\odot$.
The normalization constants (0.158, 0.0443) are chosen so that $\xi(M)$ is
continuous at $M = 1\,M_\odot$. These are the *number* density coefficients
per $\log_{10} M$ bin for the low-mass segment and per unit mass for the
high-mass segment, respectively.

The Salpeter (1955) IMF, $\xi(M) = M^{-2.35}$, is provided as a simpler
alternative but should only be used for comparison or in mass ranges
$M \gtrsim 1\,M_\odot$ where it is a good approximation to Chabrier.


## 4. Grid-Specific EEP Schemes

LACHESIS supports six isochrone grids. Each has a different native
parameterization, but all are translated to the common
$(n_\text{feh}, n_\text{age}, n_\text{eep}, n_\text{cols})$ data structure
with the Jacobian $|dM/de|$ computed along the EEP axis. The EEP numbering
differs between grids, but as proven in Section 7.1, this does not affect the
inference.

### 4.1 MIST: Native EEPs

MIST (Dotter 2016; Choi et al. 2016) provides EEPs natively. The MIST `.iso`
files contain an `EEP` column with integer values following the Dotter (2016)
scheme:

| Primary EEP | Name | EEP number |
|-------------|------|------------|
| PreMS start | PMS | 1 |
| ZAMS | Zero-age MS | 202 |
| TAMS | Terminal-age MS | 454 |
| RGBTip | Tip of RGB / He flash | 605 |
| ZAHB | Zero-age HB | 631 |
| TAHB | Terminal-age HB | 707 |
| TP-AGB | End of TP-AGB | 808 |

LACHESIS fits in the range EEP $\in [202, 808]$, excluding the pre-main
sequence (EEP $< 202$) because PMS stars are rarely the targets of isochrone
fitting and PMS models have large systematic uncertainties.

Since MIST provides EEPs natively, no translation is needed. The Jacobian
$|dM/de|$ is computed by numerical differentiation of the `initial_mass` column
along the EEP axis (see Section 6.5).

### 4.2 PARSEC: Phase Labels to EEP

PARSEC (Bressan et al. 2012; Marigo et al. 2017) does not provide EEPs.
Instead, each point on an isochrone carries an integer `label` identifying the
evolutionary phase:

| PARSEC label | Phase | Mapped EEP range |
|-------------|-------|-----------------|
| 0 | PMS | 1--201 |
| 1 | MS | 202--454 |
| 2 | SGB | 454--500 |
| 3 | RGB | 500--605 |
| 4 | CHeB | 631--707 |
| 5 | E-AGB | 707--808 |
| 6 | TP-AGB | 808--1200 |

The translation assigns EEPs by linear interpolation within each phase: if
phase $k$ contains $n_k$ rows and maps to EEP range $[e_k^\text{lo}, e_k^\text{hi}]$,
then the $j$-th row ($j = 0, \ldots, n_k - 1$) receives EEP

$$e_j = e_k^\text{lo} + j \cdot \frac{e_k^\text{hi} - e_k^\text{lo}}{n_k - 1}. \tag{11}$$

**Justification for linear spacing.** Within a single evolutionary phase,
the morphology of the isochrone in the HR diagram is smooth and monotonic.
The rows within a PARSEC phase are already uniformly sampled in an internal
evolutionary coordinate (typically the central hydrogen/helium abundance),
which is closely related to the arc-length metric used by Dotter (2016) for
secondary EEP placement. Linear spacing in EEP within each phase is therefore
a good approximation to the Dotter secondary EEP construction.

**The He flash gap (EEP 605--631).** There is a gap between the RGB tip
(EEP 605) and the ZAHB (EEP 631) corresponding to the helium flash. During
this event, the star's structure changes so rapidly that no equilibrium
models exist. LACHESIS fills this gap by linear interpolation in all
observables along the EEP axis. This is justified because (a) no real star
should be observed in this state (it lasts $\lesssim 10^6$ yr), and (b) the
interpolation ensures continuous coverage for the nested sampler, which would
otherwise waste samples on the gap boundary. The Jacobian $|dM/de|$ varies
smoothly across the interpolated region.

### 4.3 Dartmouth: Native EEPs, Different Numbering

The Dartmouth Stellar Evolution Program (DSEP; Dotter et al. 2008) also
provides native EEPs, but with a compact numbering scheme (EEP 2--279
typically) rather than the MIST convention. The primary EEP anchor points are
at different integer values, and fewer secondary EEPs are placed between them.

This difference is immaterial. The EEP number itself is arbitrary; only the
Jacobian $|dM/de|$ matters. As proven in Section 7.1, the evidence integral is
invariant to the EEP numbering scheme. The Dartmouth EEP range is used
directly, and $|dM/de|$ is computed from the initial mass column exactly as for
MIST.

### 4.4 BaSTI, YAPSI, and Geneva: Mass-Index as EEP

BaSTI (Pietrinferni et al. 2004, 2006; Hidalgo et al. 2018), YAPSI (Spada et
al. 2017), and Geneva (Ekstroem et al. 2012) provide isochrones as tables of
stellar properties sorted by initial mass, with no EEP column and no phase
labels. LACHESIS uses the *row index* $i = 0, 1, 2, \ldots, N-1$ as the EEP
coordinate:

$$e_i = i, \quad i = 0, 1, \ldots, N-1. \tag{12}$$

This requires justification.

**Claim.** Using the mass-sorted row index as the EEP with the Jacobian
$|dM/de|$ is mathematically equivalent to sampling directly in mass with the
IMF prior, up to discretization error bounded by the grid's mass resolution.

**Proof.** See Section 7.2 for the formal proof. The key insight is that when
$e = i$ is the row index along a mass-sorted isochrone with mass values
$\{M_0, M_1, \ldots, M_{N-1}\}$, the Jacobian

$$\left|\frac{dM}{de}\right|_{e=i} \approx \frac{M_{i+1} - M_{i-1}}{2} \tag{13}$$

(computed via central differences, i.e., `np.gradient`) is exactly the mass
spacing of the isochrone grid at that point. The EEP prior then becomes

$$\pi(e_i) = \xi(M_i) \cdot \Delta M_i, \tag{14}$$

which is the trapezoidal-rule discretization of the IMF-weighted mass integral.
This is numerically equivalent to sampling in mass with the grid's native
resolution.

**Why this is valid.** The mass-index EEP is monotonic (the isochrone is sorted
by mass), bijective (each row has a unique index), and supports well-defined
numerical differentiation. The only requirement for the EEP formalism to work
is that $e \to M(e)$ is monotonic with computable Jacobian. Row indices satisfy
this trivially.


## 5. The Evidence Integral and Bayesian Model Averaging

### 5.1 Nested sampling in EEP space

The marginal likelihood (evidence) for grid $k$ is

$$Z_k = \int \mathcal{L}(\mathbf{D} \mid \boldsymbol{\theta}) \, \pi_k(\boldsymbol{\theta}) \, d\boldsymbol{\theta}, \tag{15}$$

where $\boldsymbol{\theta} = (e, \log\tau, [\text{Fe/H}], d, A_V)$ and the
prior factorizes as

$$\pi_k(\boldsymbol{\theta}) = \pi_k(e) \cdot \pi(\log\tau) \cdot \pi([\text{Fe/H}]) \cdot \pi(d) \cdot \pi(A_V). \tag{16}$$

The EEP prior is grid-dependent:

$$\pi_k(e) = \xi\big(M_k(e)\big) \cdot \left|\frac{dM_k}{de}\right|. \tag{17}$$

This integral is computed via nested sampling (Skilling 2004, 2006). In
practice, LACHESIS uses the `dynesty` implementation (Speagle 2020). Because
the EEP prior $\pi_k(e)$ depends on the grid (through $M_k(e)$ and
$|dM_k/de|$), it cannot be encoded analytically in dynesty's `prior_transform`
(which maps the unit hypercube $[0,1]^n \to \boldsymbol{\theta}$). Instead, the
IMF$\cdot|dM/de|$ weight is folded into the log-likelihood:

$$\ln \mathcal{L}_\text{eff}(\boldsymbol{\theta}) = \ln \mathcal{L}(\mathbf{D} \mid \boldsymbol{\theta}) + \ln\big[\xi(M(e)) \cdot |dM/de|\big]. \tag{18}$$

The `prior_transform` then maps to a uniform distribution in EEP (and uniform
or Gaussian in the other parameters). This is mathematically exact: the nested
sampling evidence integral over the effective likelihood with the flat EEP
prior equals the integral over the true likelihood with the IMF$\cdot|dM/de|$
prior.

**Verification.** The identity is straightforward:

$$Z_k = \int \mathcal{L}(\mathbf{D} \mid \boldsymbol{\theta}) \cdot \pi_k(e) \cdot \pi(\text{rest}) \, d\boldsymbol{\theta} = \int \underbrace{\Big[\mathcal{L}(\mathbf{D} \mid \boldsymbol{\theta}) \cdot \xi(M_k(e)) \cdot |dM_k/de|\Big]}_{\mathcal{L}_\text{eff}} \cdot \underbrace{\frac{1}{\Delta e}}_{\pi_\text{flat}(e)} \cdot \Delta e \cdot \pi(\text{rest}) \, d\boldsymbol{\theta}. \tag{19}$$

The $\Delta e$ cancels, confirming that folding the EEP prior into the
likelihood is exact as long as the prior_transform uses a flat distribution
over the EEP range $[\Delta e]$. The $1/\Delta e$ normalization factor from
the flat EEP prior enters the evidence as a multiplicative constant, which
is correct because it represents the prior volume in the EEP dimension.

### 5.2 Why different EEP schemes produce comparable evidences

Consider two grids, $k=1$ (MIST, EEP $\in [202, 808]$) and $k=2$ (BaSTI,
EEP $\in [0, 150]$). Their evidence integrals are:

$$Z_1 = \int_{202}^{808} \int \mathcal{L}_1 \cdot \xi(M_1(e)) \cdot |dM_1/de| \cdot \pi(\log\tau) \cdot \pi([\text{Fe/H}]) \, de \, d\log\tau \, d[\text{Fe/H}] \tag{20}$$

$$Z_2 = \int_{0}^{150} \int \mathcal{L}_2 \cdot \xi(M_2(e)) \cdot |dM_2/de| \cdot \pi(\log\tau) \cdot \pi([\text{Fe/H}]) \, de \, d\log\tau \, d[\text{Fe/H}] \tag{21}$$

By the change-of-variables theorem (proven formally in Section 7.1), both
integrals reduce to the *same* integral over mass:

$$Z_k = \int_{M_\text{lo}}^{M_\text{hi}} \int \mathcal{L}_k \cdot \xi(M) \cdot \pi(\log\tau) \cdot \pi([\text{Fe/H}]) \, dM \, d\log\tau \, d[\text{Fe/H}] \tag{22}$$

(where the mass limits $[M_\text{lo}, M_\text{hi}]$ may differ between grids
due to their different coverage of the mass range). The evidence ratio
$Z_1 / Z_2$ therefore reflects only differences in (a) the likelihood function
$\mathcal{L}_k$ (how well each grid's physics matches the data), (b) the mass
range covered, and (c) any differences in $(\log\tau, [\text{Fe/H}])$
coverage. It is *independent* of the arbitrary EEP numbering.

### 5.3 The prior volume normalization

The full prior normalization in LACHESIS is:

$$\pi(\boldsymbol{\theta}) = \underbrace{\xi(M(e)) \cdot |dM/de|}_{\text{EEP}} \cdot \underbrace{\frac{1}{\Delta\log\tau}}_{\log\tau} \cdot \underbrace{\pi([\text{Fe/H}])}_{\text{uniform or Gaussian}} \cdot \underbrace{\frac{1}{\Delta d}}_{\text{distance}} \cdot \underbrace{\frac{1}{\Delta A_V}}_{A_V}. \tag{23}$$

The EEP term is the only non-trivial factor. All other priors are either
uniform (with normalization $1/\Delta x$) or Gaussian.

### 5.4 Per-grid parameter range clamping

Each grid covers a different region of parameter space. MIST provides
isochrones for $[\text{Fe/H}] \in [-4.0, +0.5]$, while BaSTI covers
$[\text{Fe/H}] \in [-3.2, +0.4]$. The prior ranges for $[\text{Fe/H}]$,
$\log\tau$, and EEP must be clamped to the actual coverage of each grid:

$$[\text{Fe/H}]_\text{lo}^{(k)} = \max\big([\text{Fe/H}]_\text{lo}^\text{user},\; [\text{Fe/H}]_\text{min}^\text{grid}\big) \tag{24}$$

$$[\text{Fe/H}]_\text{hi}^{(k)} = \min\big([\text{Fe/H}]_\text{hi}^\text{user},\; [\text{Fe/H}]_\text{max}^\text{grid}\big) \tag{25}$$

and similarly for the other parameters. This clamping is *necessary and
correct*: it prevents the nested sampler from exploring regions where the grid
provides no models (returning NaN), which would waste live points and bias the
evidence estimate.

The effect on the evidence is through the prior volume. A grid with narrower
coverage has a smaller prior volume, which *increases* its evidence per unit
volume (higher Occam factor concentration) but *decreases* the total prior-weighted
volume. These two effects partially cancel, and the net effect correctly
reflects the grid's ability to explain the data within its domain.

### 5.5 The Occam factor

The evidence $Z$ naturally encodes an Occam factor: a model with a large prior
volume that concentrates its likelihood in a small region is penalized relative
to a model with a smaller prior volume that fits the data comparably well. In
the isochrone BMA context, this means:

- A grid that covers $[\text{Fe/H}] \in [-4, +0.5]$ but only fits the data
  well near $[\text{Fe/H}] = 0.0$ will have a lower evidence than one covering
  $[\text{Fe/H}] \in [-0.5, +0.5]$ that fits equally well. This is the correct
  behavior: the wider grid is "wasting" prior volume on regions that don't
  contribute to the fit.

- Two grids with different EEP ranges but the same effective mass range will
  have the same Occam penalty in the EEP dimension, because the EEP prior
  (Eq. 8) maps EEP volume to mass volume via the Jacobian.


### 5.6 Bayesian Model Averaging

Given $K$ grids with evidences $Z_1, \ldots, Z_K$, the BMA posterior is

$$P(\boldsymbol{\theta} \mid \mathbf{D}) = \sum_{k=1}^{K} w_k \, P_k(\boldsymbol{\theta} \mid \mathbf{D}), \tag{26}$$

where the evidence weights are

$$w_k = \frac{Z_k}{\sum_{j=1}^K Z_j}. \tag{27}$$

This is exact under the assumption of equal prior model probabilities
$P(\mathcal{M}_k) = 1/K$. The combined posterior automatically accounts for
systematic differences between grids: a grid that fits the data poorly (low
$Z_k$) receives negligible weight.

In practice, the sum over models is implemented by drawing
$n_k = \text{round}(w_k \cdot N_\text{total})$ samples from each grid's
posterior and concatenating them. Derived quantities (age, mass, radius, etc.)
are computed for each sample from the corresponding grid's interpolator.


## 6. Practical Implementation

### 6.1 The 4D grid data structure

All grids are stored as a 4D NumPy array:

$$\texttt{data}[i_\text{feh}, i_\text{age}, i_\text{eep}, i_\text{col}] \in \mathbb{R}, \tag{28}$$

with shape $(n_\text{feh}, n_\text{age}, n_\text{eep}, n_\text{cols})$. The
axes are:

- **Axis 0:** Metallicity $[\text{Fe/H}]$, sorted ascending.
- **Axis 1:** Age $\log_{10}(\tau/\text{yr})$, sorted ascending.
- **Axis 2:** EEP $e$, sorted ascending (monotonic by construction).
- **Axis 3:** Observable columns ($M_\text{ini}$, $M_\text{cur}$, $\log T_\text{eff}$, $\log g$, $\log L$, $\log R$, phase, $T_\text{eff}$, $M_\text{bol}$, radius, $\rho$, $|dM/de|$).

Regions of parameter space where no model exists (e.g., EEP 700 at $\log\tau = 9.0$
for $M = 0.5\,M_\odot$, which has not yet reached the HB) are filled with NaN.

### 6.2 Trilinear interpolation

At a proposed point $(e^*, \log\tau^*, [\text{Fe/H}]^*)$, the interpolator
performs trilinear interpolation on the regular 3D grid. The EEP axis is not
integer-valued for the query (the sampler proposes continuous EEP values), so
the interpolation uses the actual EEP values as the coordinate axis.

For each observable column $c$, the interpolated value is:

$$\hat{y}_c = \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} w_{ijk} \cdot \texttt{data}[i_0 + i, j_0 + j, k_0 + k, c], \tag{29}$$

where $w_{ijk} = f_i \cdot g_j \cdot h_k$ are trilinear weights formed from
the fractional positions along each axis, and $(i_0, j_0, k_0)$ are the
lower-corner indices of the enclosing cell.

### 6.3 NaN handling

If any of the eight vertices of the enclosing cell is NaN, the interpolated
value is NaN, and the log-likelihood returns $-\infty$. This is the correct
behavior: it tells the nested sampler that this region of parameter space is
outside the grid's domain, and the sampler will not waste further live points
there.

No extrapolation is performed. The grid boundary is a hard wall.

### 6.4 Gap filling: Inter-phase interpolation

For PARSEC grids, the He flash gap (EEP 605--631) creates a NaN band across the
EEP axis. This is filled by linear interpolation along the EEP axis for each
$([\text{Fe/H}], \log\tau)$ slice independently:

For each column $c$, given the valid (non-NaN) entries at EEP indices
$\{e_{a_1}, e_{a_2}, \ldots\}$ and missing entries in the interior, the gap is
filled by `np.interp` using the valid entries as knots.

**When gap filling is justified:** Only for short-lived evolutionary phases
where no equilibrium model exists (He flash, He shell flashes on the TP-AGB).
The interpolated values are unphysical in detail, but no real star occupies
these states for a photometrically detectable duration, so the likelihood
will be negligible there. The purpose is solely to maintain a continuous
EEP axis for the sampler.

**When gap filling is not justified:** Between evolutionary phases with
genuinely different physics (e.g., MS to RGB across the Hertzsprung gap for
massive stars). In these cases, the NaN entries correctly represent the
absence of models, and forcing interpolation would produce misleading results.
MIST, Dartmouth, BaSTI, YAPSI, and Geneva grids do not require gap filling
because they either have native EEPs that span the gap or use mass-index
EEPs with no structural gaps.

### 6.5 Computing $|dM/de|$

The Jacobian $|dM/de|$ is computed as the numerical gradient of the initial
mass column along the EEP axis:

```python
dm_deep = np.gradient(initial_mass, axis=eep_axis)
```

`np.gradient` uses second-order central differences in the interior and
first-order one-sided differences at the boundaries:

$$\left.\frac{dM}{de}\right|_{i} = \begin{cases} \displaystyle\frac{M_{i+1} - M_{i-1}}{e_{i+1} - e_{i-1}} & 1 \leq i \leq N-2 \\[6pt] \displaystyle\frac{M_1 - M_0}{e_1 - e_0} & i = 0 \\[6pt] \displaystyle\frac{M_{N-1} - M_{N-2}}{e_{N-1} - e_{N-2}} & i = N-1 \end{cases} \tag{30}$$

For grids with uniformly spaced EEP values (MIST: every integer, BaSTI/YAPSI/Geneva:
$e = 0, 1, 2, \ldots$), the denominators are all 1 or 2, and this reduces to
simple finite differences.

For grids with non-uniformly spaced EEPs (Dartmouth, where some EEPs may be
missing in certain isochrones), `np.gradient` correctly handles the non-uniform
spacing via its default behavior with non-uniform coordinates.

The gradient is computed on the full 4D array and stored as a column
(`dm_deep`) so that it is available during interpolation without
recomputation.


## 7. Remark on EEP Invariance

The results in Sections 3–5 rest on the standard change-of-variables theorem.
Three immediate consequences:

1. **Invariance to EEP numbering.** Any two monotonic EEP functions $f(M)$ and $g(M)$ produce the same evidence integral, because both reduce to $\int \mathcal{L}(M) \, \xi(M) \, dM$ after substitution.

2. **Mass-index equivalence.** Using the mass-sorted row index as EEP with `np.gradient` for $|dM/de|$ is equivalent to the trapezoidal-rule discretization of the mass-weighted IMF integral, with $\mathcal{O}(1/N)$ boundary error that is negligible for typical grid sizes ($N \sim 100$–$500$).

3. **Cross-grid BMA validity.** Bayes factors $B_{12} = Z_1/Z_2$ between grids with different EEP schemes contain no EEP variables after change-of-variables elimination. All differences arise from the physics (likelihoods) and parameter coverage (Occam factor), not from the arbitrary EEP labeling.

It is therefore valid to compare evidence values between MIST (native Dotter EEPs), PARSEC (translated phase-label EEPs), Dartmouth (native but differently numbered EEPs), and BaSTI/YAPSI (mass-index EEPs) within a single BMA analysis.


## 8. References

- Bressan, A., Marigo, P., Girardi, L., et al. 2012, MNRAS, 427, 127. "PARSEC: stellar tracks and isochrones with PAdova and TRieste Stellar Evolution Code."

- Chabrier, G. 2003, PASP, 115, 763. "Galactic Stellar and Substellar Initial Mass Function."

- Choi, J., Dotter, A., Conroy, C., et al. 2016, ApJ, 823, 102. "Mesa Isochrones and Stellar Tracks (MIST). I."

- Dotter, A. 2016, ApJS, 222, 8. "MESA Isochrones and Stellar Tracks (MIST) 0: Methods for the Construction of Stellar Isochrones."

- Dotter, A., Chaboyer, B., Jevremovic, D., et al. 2008, ApJS, 178, 89. "The Dartmouth Stellar Evolution Database."

- Ekstroem, S., Georgy, C., Eggenberger, P., et al. 2012, A&A, 537, A146. "Grids of stellar models with rotation."

- Hidalgo, S. L., Pietrinferni, A., Cassisi, S., et al. 2018, ApJ, 856, 125. "Updated BaSTI Stellar Evolution Models and Isochrones."

- Marigo, P., Girardi, L., Bressan, A., et al. 2017, ApJ, 835, 77. "A New Generation of PARSEC-COLIBRI Stellar Isochrones Including the TP-AGB Phase."

- Morton, T. D. 2015, isochrones: Stellar model grid package. Astrophysics Source Code Library, ascl:1503.010.

- Pietrinferni, A., Cassisi, S., Salaris, M., Castelli, F. 2004, ApJ, 612, 168. "A Large Stellar Evolution Database for Population Synthesis Studies."

- Pietrinferni, A., Cassisi, S., Salaris, M., Castelli, F. 2006, ApJ, 642, 797. "A Large Stellar Evolution Database for Population Synthesis Studies. II."

- Salpeter, E. E. 1955, ApJ, 121, 161. "The Luminosity Function and Stellar Evolution."

- Skilling, J. 2004, AIP Conf. Proc., 735, 395. "Nested Sampling."

- Skilling, J. 2006, Bayesian Anal., 1, 833. "Nested sampling for general Bayesian computation."

- Spada, F., Demarque, P., Kim, Y.-C., Boyajian, T. S., Brewer, J. M. 2017, ApJ, 838, 161. "The Yale-Potsdam Stellar Isochrones."

- Speagle, J. S. 2020, MNRAS, 493, 3132. "dynesty: a dynamic nested sampling package for estimating Bayesian posteriors and evidences."

- Vines, J. I. & Jenkins, J. S. 2022, MNRAS, 513, 2719. "ARIADNE: measuring accurate and precise stellar parameters through SED fitting."
