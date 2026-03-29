"""Filter name translation: pyphot ↔ MIST BC table.

LACHESIS uses pyphot filter names internally (compatible with ARIADNE).
The BC table uses MIST's naming convention. This module translates between them.

Bands without an extracted BC table yet are mapped here for forward
compatibility — the Librarian retrieves all photometry, and the fitter
silently skips bands that lack a BC table entry.
"""

# pyphot name → bc_band_name (as it appears in the MIST BC table header)
PYPHOT_TO_BC = {
    # Gaia EDR3/DR3  [UBVRIplus]
    "Gaia_G": "Gaia_G_EDR3",
    "Gaia_BP": "Gaia_BP_EDR3",
    "Gaia_RP": "Gaia_RP_EDR3",
    # Gaia DR2 (legacy)  [UBVRIplus]
    "GaiaDR2v2_G": "Gaia_G_DR2Rev",
    "GaiaDR2v2_BP": "Gaia_BP_DR2Rev",
    "GaiaDR2v2_RP": "Gaia_RP_DR2Rev",
    # 2MASS  [UBVRIplus]
    "2MASS_J": "2MASS_J",
    "2MASS_H": "2MASS_H",
    "2MASS_Ks": "2MASS_Ks",
    # Tycho  [UBVRIplus]
    "TYCHO_B_MvB": "Tycho_B",
    "TYCHO_V_MvB": "Tycho_V",
    # Johnson/Bessell  [UBVRIplus]
    "GROUND_JOHNSON_U": "Bessell_U",
    "GROUND_JOHNSON_B": "Bessell_B",
    "GROUND_JOHNSON_V": "Bessell_V",
    "GROUND_COUSINS_R": "Bessell_R",
    "GROUND_COUSINS_I": "Bessell_I",
    # Kepler  [UBVRIplus]
    "KEPLER_Kp": "Kepler_Kp",
    # TESS  [UBVRIplus]
    "TESS": "TESS",
    # WISE  [WISE BC system]
    "WISE_RSR_W1": "WISE_W1",
    "WISE_RSR_W2": "WISE_W2",
    # SDSS  [SDSSugriz BC system]
    "SDSS_u": "SDSS_u",
    "SDSS_g": "SDSS_g",
    "SDSS_r": "SDSS_r",
    "SDSS_i": "SDSS_i",
    "SDSS_z": "SDSS_z",
    # Pan-STARRS  [PanSTARRS BC system]
    "PS1_g": "PS_g",
    "PS1_r": "PS_r",
    "PS1_i": "PS_i",
    "PS1_z": "PS_z",
    "PS1_y": "PS_y",
    # SkyMapper  [SkyMapper BC system]
    "SkyMapper_u": "SkyMapper_u",
    "SkyMapper_v": "SkyMapper_v",
    "SkyMapper_g": "SkyMapper_g",
    "SkyMapper_r": "SkyMapper_r",
    "SkyMapper_i": "SkyMapper_i",
    "SkyMapper_z": "SkyMapper_z",
    # GALEX  [GALEX BC system]
    "GALEX_FUV": "GALEX_FUV",
    "GALEX_NUV": "GALEX_NUV",
    # Spitzer IRAC  [SPITZER BC system — not loaded by default]
    "SPITZER_IRAC_36": "IRAC_3.6",
    "SPITZER_IRAC_45": "IRAC_4.5",
}

# Reverse mapping: BC table name → pyphot name
BC_TO_PYPHOT = {v: k for k, v in PYPHOT_TO_BC.items()}

# Catalog column → pyphot filter name (for external code; the Librarian
# uses its own CatalogDef registry for this mapping).
CATALOG_TO_PYPHOT = {
    # Gaia DR3 (TAP column names)
    "phot_g_mean_mag": "Gaia_G",
    "phot_bp_mean_mag": "Gaia_BP",
    "phot_rp_mean_mag": "Gaia_RP",
    # Gaia DR3 (VizieR column names)
    "Gmag": "Gaia_G",
    "BPmag": "Gaia_BP",
    "RPmag": "Gaia_RP",
    # 2MASS (TAP)
    "j_m": "2MASS_J",
    "h_m": "2MASS_H",
    "ks_m": "2MASS_Ks",
    # 2MASS (VizieR)
    "Jmag": "2MASS_J",
    "Hmag": "2MASS_H",
    "Kmag": "2MASS_Ks",
    # Tycho-2
    "BTmag": "TYCHO_B_MvB",
    "VTmag": "TYCHO_V_MvB",
    # TESS TIC
    "Tmag": "TESS",
    # APASS / Johnson
    "Vmag": "GROUND_JOHNSON_V",
    "Bmag": "GROUND_JOHNSON_B",
    # WISE
    "W1mag": "WISE_RSR_W1",
    "W2mag": "WISE_RSR_W2",
    # Pan-STARRS (VizieR lowercase)
    "gmag": "PS1_g",
    "rmag": "PS1_r",
    "imag": "PS1_i",
    "zmag": "PS1_z",
    "ymag": "PS1_y",
    # GALEX
    "FUV": "GALEX_FUV",
    "NUV": "GALEX_NUV",
}


def pyphot_to_bc(pyphot_name: str) -> str | None:
    """Translate pyphot filter name to MIST BC table band name."""
    return PYPHOT_TO_BC.get(pyphot_name)


def bc_to_pyphot(bc_name: str) -> str | None:
    """Translate MIST BC table band name to pyphot filter name."""
    return BC_TO_PYPHOT.get(bc_name)
