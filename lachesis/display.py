"""Console display functions — matching ARIADNE's output style."""

import time

import numpy as np

from lachesis.config import EEP_PHASES

_TAB = "\t\t\t\t"


def display_banner(starname: str):
    print(f"{_TAB}#####################################")
    print(f"{_TAB}##            LACHESIS             ##")
    print(f"{_TAB}#####################################")
    print(f"   isochrone bAyesian modeL avEraging")
    print(f"   CHaracterization of stEllar age, maSs")
    print(f"   and evolutIonary State")
    print()
    print(f"{_TAB}Author : Jose Vines")
    print(f"{_TAB}Star : {starname}")
    print()


def display_star_info(star):
    print(f"{_TAB}--- Input Stellar Properties ---")
    props = [
        ("Teff", star.teff, star.teff_e, "K"),
        ("log(g)", star.logg, star.logg_e, "dex"),
        ("[Fe/H]", star.feh, star.feh_e, "dex"),
        ("Luminosity", star.luminosity, star.luminosity_e, "Lsun"),
        ("Radius", star.radius, star.radius_e, "Rsun"),
        ("Distance", star.distance, star.distance_e, "pc"),
        ("Parallax", star.parallax, star.parallax_e, "mas"),
    ]
    for name, val, err, unit in props:
        if val is not None:
            if err is not None:
                print(f"{_TAB}{name:15s}:  {val:.4g} +/- {err:.4g} {unit}")
            else:
                print(f"{_TAB}{name:15s}:  {val:.4g} {unit}")
    if star.magnitudes:
        print(f"{_TAB}{'Magnitudes':15s}:  {len(star.magnitudes)} bands")
    print(f"{_TAB}{'Mode':15s}:  {star.mode}")
    print()


def display_routine(fitter):
    setup = fitter._parsed_setup
    print(f"{_TAB}*** EXECUTING ISOCHRONE FITTING ROUTINE ***")
    print(f"{_TAB}Selected engine : {setup['engine']}")
    print(f"{_TAB}Live points : {setup['nlive']}")
    print(f"{_TAB}log Evidence tolerance : {setup['dlogz']}")
    print(f"{_TAB}Bounding : {setup['bound']}")
    print(f"{_TAB}Sampling : {setup['sample']}")
    print(f"{_TAB}N threads : {setup['threads']}")
    print(f"{_TAB}Mode : {fitter.star.mode}")
    print(f"{_TAB}Grids : {', '.join(g.upper() for g in fitter.grids)}")
    print(f"{_TAB}BMA : {fitter.bma}")
    if fitter.binary:
        print(f"{_TAB}Binary : True")
    print()


def display_fitting_grid(name: str):
    print(f"{_TAB}FITTING GRID : {name.upper()}")


def display_summary(samples, derived, param_names):
    """Print parameter summary table with median, 1σ, 3σ."""
    print()
    print(f"{_TAB}Fitting finished.")
    print(f"{_TAB}Best fit parameters are:")
    print()

    # Derived quantities to display
    display_params = [
        ("Age (Gyr)", 10.0 ** samples[:, param_names.index("log_age")] / 1e9
         if "log_age" in param_names else None),
        ("Initial mass (Msun)", derived.get("initial_mass")),
        ("EEP", samples[:, param_names.index("eep")]
         if "eep" in param_names else None),
        ("Teff (K)", derived.get("Teff")),
        ("log(g) (dex)", derived.get("log_g")),
        ("[Fe/H] (dex)", samples[:, param_names.index("feh")]
         if "feh" in param_names else None),
        ("Radius (Rsun)", derived.get("radius")),
        ("Luminosity (Lsun)", _safe_pow10(derived.get("log_L"))),
    ]

    # Add distance/Av if present
    if "distance" in param_names:
        idx = param_names.index("distance")
        display_params.append(("Distance (pc)", samples[:, idx]))
    if "av" in param_names:
        idx = param_names.index("av")
        display_params.append(("Av (mag)", samples[:, idx]))

    fmt = f"{_TAB}{{:20s}}  {{:>10s}}  {{:>20s}}  {{:>20s}}"
    print(fmt.format("Parameter", "Median", "1σ CI", "3σ CI"))
    print(f"{_TAB}{'-' * 74}")

    for name, arr in display_params:
        if arr is None:
            continue
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        med = np.median(arr)
        lo1 = med - np.percentile(arr, 15.87)
        hi1 = np.percentile(arr, 84.13) - med
        lo3 = np.percentile(arr, 0.13)
        hi3 = np.percentile(arr, 99.87)
        print(fmt.format(
            name,
            f"{med:.4g}",
            f"+{hi1:.4g}  -{lo1:.4g}",
            f"[{lo3:.4g}, {hi3:.4g}]",
        ))

    # Evolutionary state from median EEP
    if "eep" in param_names:
        med_eep = np.median(samples[:, param_names.index("eep")])
        state = _eep_to_state(med_eep)
        print(f"{_TAB}{'Evol. state':20s}  {state}")

    print()


def display_model_weights(bma_result):
    print(f"{_TAB}Model weights:")
    for name, w, lz in zip(
        bma_result.model_names, bma_result.weights, bma_result.log_evidences
    ):
        print(f"{_TAB}{name} probability : {w:.4f}  (logZ = {lz:.3f})")
    print()


def display_elapsed(t_start):
    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    if minutes > 0:
        print(f"{_TAB}Elapsed time : {minutes} minutes and {seconds:.2f} seconds")
    else:
        print(f"{_TAB}Elapsed time : {seconds:.2f} seconds")
    print()


def _eep_to_state(eep: float) -> str:
    if eep < 202:
        return "Pre-Main Sequence"
    elif eep < 353:
        return "Main Sequence (early)"
    elif eep < 454:
        return "Main Sequence (late/turnoff)"
    elif eep < 605:
        return "Subgiant/RGB"
    elif eep < 631:
        return "RGB Tip"
    elif eep < 707:
        return "Core He Burning"
    elif eep < 808:
        return "Early AGB"
    else:
        return "TP-AGB / post-AGB"


def _safe_pow10(arr):
    if arr is None:
        return None
    return 10.0 ** arr
