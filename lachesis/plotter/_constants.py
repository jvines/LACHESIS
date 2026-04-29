"""Plotter module-level constants.

Per-model colours, default parameter sets, LaTeX axis labels, and the
shipped settings file path.
"""

from pathlib import Path

# Per-model colors (same as ARIADNE BMA palette)
_MODEL_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown",
]

# Default parameters for corner / to_latex.
# Note: we report the *current* stellar mass (`mass`, which comes from the
# grid's `star_mass` column), not `initial_mass`. For main-sequence stars
# they're nearly identical, but for evolved stars (RGB/AGB) they differ due
# to mass loss, and the current mass is what astronomers want reported.
_DEFAULT_CORNER_PARAMS = [
    "mass", "Teff", "log_g", "[Fe/H]", "age_gyr",
]

# Histograms show every sampled parameter + useful derived ones
_DEFAULT_HIST_PARAMS = [
    "mass", "Teff", "log_g", "[Fe/H]", "age_gyr",
    "distance", "Av", "eep",
]

_DEFAULT_LATEX_PARAMS = [
    "mass", "age_gyr", "[Fe/H]", "log_g",
]

# LaTeX-formatted axis labels
_PARAM_LABELS = {
    "mass": r"$M_\star$ [$M_\odot$]",
    "initial_mass": r"$M_{\mathrm{init}}$ [$M_\odot$]",
    "Teff": r"$T_{\mathrm{eff}}$ [K]",
    "log_g": r"$\log g$ [dex]",
    "[Fe/H]": r"[Fe/H] [dex]",
    "age_gyr": r"Age [Gyr]",
    "star_mass": r"$M_\star$ [$M_\odot$]",
    "log_L": r"$\log L/L_\odot$",
    "log_R": r"$\log R/R_\odot$",
    "log_Teff": r"$\log T_{\mathrm{eff}}$",
    "radius": r"$R_\star$ [$R_\odot$]",
    "density": r"$\rho$ [g cm$^{-3}$]",
    "distance": r"Distance [pc]",
    "Av": r"$A_V$ [mag]",
    "eep": "EEP",
}

# Default settings file shipped with the package (sibling of the plotter pkg)
_SETTINGS_FILE = Path(__file__).resolve().parent.parent / "plot_settings.dat"
