# Spectroscopic Metallicity Catalogs for [Fe/H] Priors

Reference for integration into LACHESIS Librarian and ARIADNE.

## Priority 1 (Immediate)

### Gaia DR3 GSP-Spec (Recio-Blanco+ 2023)
- **Stars:** 5.6M | **Sky:** All-sky | **Precision:** ~0.03 dex
- **VizieR:** I/355 | **Key:** Gaia `source_id` (native, no crossmatch)
- **Columns:** `mh_gspspec`, `teff_gspspec`, `logg_gspspec` + errors
- **Note:** Already in Gaia DR3 — zero overhead. Quality degrades for faint stars.

### APOGEE DR17 (Abdurro'uf+ 2023)
- **Stars:** 730k | **Sky:** All-sky (IR) | **Precision:** ~0.08 dex
- **VizieR:** III/286 | **Key:** 2MASS ID → Gaia via `tmass_psc_xsc_best_neighbour`
- **Columns:** `M_H`, `TEFF`, `LOGG` + errors. Filter: `ASPCAP_FLAG == 0`
- **Note:** Asteroseismically calibrated logg. [M/H] not strictly [Fe/H].

### LAMOST DR9 (Wang+ 2022)
- **Stars:** 6.9M spectra | **Sky:** Northern | **Precision:** 0.05–0.15 dex
- **Access:** https://www.lamost.org/dr9/v2.0/ (direct download, not VizieR)
- **Columns:** `feh`, `teff`, `logg` + errors
- **Note:** Largest spectroscopic survey. Crossmatch via position+PM to Gaia.

## Priority 2

### GALAH DR3 (Buder+ 2021)
- **Stars:** 588k | **Sky:** Southern | **Precision:** ~0.10 dex
- **VizieR:** VI/180 | **Key:** Gaia EDR3 `source_id` (pre-matched)
- **Columns:** `fe_h`, `teff`, `logg` + errors. Filter: `flag_sp == 0`

### RAVE DR6 (Steinmetz+ 2020) — already partially in LACHESIS
- **Stars:** 451k | **Sky:** All-sky | **Precision:** 0.02–0.15 dex
- **VizieR:** III/283 | **Key:** Gaia source_id (pre-matched)
- **Columns:** `Met`, `Teff`, `logg` (both spectroscopic and Bayesian-enhanced)

## Priority 3 (Validation)

### PASTEL (Soubiran+ 2016)
- **Stars:** 31k | **Precision:** ~0.05 dex (literature compilation, high-res)
- **VizieR:** V/136 | **Key:** HD/HIP → SIMBAD → Gaia

## Summary Table

| Catalog | Stars | Sky | [Fe/H] σ | Teff/logg | VizieR | Crossmatch |
|---------|-------|-----|----------|-----------|--------|------------|
| Gaia GSP-Spec | 5.6M | All | 0.03 | Yes | I/355 | Native |
| APOGEE DR17 | 730k | All | 0.08 | Yes (astero) | III/286 | 2MASS→Gaia |
| LAMOST DR9 | 6.9M | North | 0.05–0.15 | Yes | Direct | Pos+PM→Gaia |
| GALAH DR3 | 588k | South | 0.10 | Yes | VI/180 | Gaia (pre-matched) |
| RAVE DR6 | 451k | All | 0.02–0.15 | Yes | III/283 | Gaia (pre-matched) |
| PASTEL | 31k | Hetero | 0.05 | Subset | V/136 | HD→SIMBAD→Gaia |
