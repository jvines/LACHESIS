"""Console display functions — matching ARIADNE's output style."""

import os
import random
import sys
import time

import numpy as np
from termcolor import colored as _termcolor_colored

from lachesis.config import EEP_PHASES

_COLORS = ['red', 'green', 'blue', 'yellow', 'grey', 'magenta', 'cyan', 'white']
_T2 = "\t\t"      # section headers (banner box, routine header)
_T3 = "\t\t\t"    # detail lines (author, params)


def _color_enabled() -> bool:
    """Honour NO_COLOR / non-tty / TERM=dumb so logs don't fill with ANSI."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def colored(text, color=None, *args, **kwargs):
    """termcolor wrapper that strips colour when output is not a TTY."""
    if not _color_enabled():
        return text
    return _termcolor_colored(text, color, *args, **kwargs)


def _pick_color():
    return random.choice(_COLORS) if _color_enabled() else None


def display_retrieved_photometry(star, c=None):
    """Print retrieved photometry table + Gaia params (before banner)."""
    if c is None:
        c = _pick_color()

    # Photometry table
    if star.magnitudes:
        print(colored(f'{_T3}--- Retrieved Photometry ---', c))
        print(colored(f'{_T3}{"Filter":23s}{"Magnitude":>14s}{"Error":>10s}', c))
        print(colored(f'{_T3}{"-" * 42}', c))
        for filt in sorted(star.magnitudes.keys()):
            mag, err = star.magnitudes[filt]
            print(colored(f'{_T3}{filt:23s}{mag:14.4f}{err:10.4f}', c))

    # Gaia params
    if star.parallax is not None:
        print(colored(
            f'{_T3}Parallax : {star.parallax:.3f} +/- {star.parallax_e:.3f} mas', c))
    if star.teff is not None:
        print(colored(
            f'{_T3}Gaia Teff : {star.teff:.0f} +/- {star.teff_e:.0f} K', c))
    if star.distance is not None:
        print(colored(
            f'{_T3}BJ Distance : {star.distance:.1f} +/- {star.distance_e:.1f} pc', c))
    if star.g_id is not None:
        print(colored(f'{_T3}Gaia DR3 ID : {star.g_id}', c))
    if star.Av is not None:
        print(colored(f'{_T3}Maximum Av : {star.Av:.3f}', c))
    print()
    return c


def display_banner(starname: str, c=None):
    if c is None:
        c = _pick_color()
    print(colored(f'\n{_T2}#####################################', c))
    print(colored(f'{_T2}##            LACHESIS             ##', c))
    print(colored(f'{_T2}#####################################', c))
    print(colored('   modeL Averaging for isoChrone', c), end=' ')
    print(colored('cHaractErization of Stellar propertIeS', c))
    print()
    print(colored(f'{_T3}Author : Jose Vines', c))
    print(colored(f'{_T3}Contact : jose . vines at ug . uchile . cl', c))
    print(colored(f'{_T3}Star : {starname}', c))
    return c


def display_routine(fitter):
    c = _pick_color()
    setup = fitter._parsed_setup
    if fitter.bma:
        engine = 'Bayesian Model Averaging'
    else:
        grid = fitter.grids[0].upper() if fitter.grids else '?'
        engine = f'Single model ({grid})'
    print(colored(f'\n{_T2}*** EXECUTING ISOCHRONE FITTING ROUTINE ***', c))
    print(colored(f'{_T3}Selected engine : ', c), end='')
    print(colored(engine, c))
    print(colored(f'{_T3}Live points : ', c), end='')
    print(colored(str(setup["nlive"]), c))
    print(colored(f'{_T3}log Evidence tolerance : ', c), end='')
    print(colored(str(setup["dlogz"]), c))
    print(colored(f'{_T3}Bounding : ', c), end='')
    print(colored(setup["bound"], c))
    print(colored(f'{_T3}Sampling : ', c), end='')
    print(colored(setup["sample"], c))
    print(colored(f'{_T3}N threads : ', c), end='')
    print(colored(str(setup["threads"]), c))
    print()


def display_fitting_grid(name: str):
    c = _pick_color()
    print(colored(f'{_T3}FITTING GRID : {name.upper()}', c))


def display_summary(samples, derived, param_names):
    """Print parameter summary table with median, 1 sigma, 3 sigma."""
    c = _pick_color()
    print()
    print(colored(f'{_T3}Fitting finished.', c))
    print(colored(f'{_T3}Best fit parameters are:', c))
    print()

    display_params = [
        ("Age (Gyr)", 10.0 ** samples[:, param_names.index("log_age")] / 1e9
         if "log_age" in param_names else None),
        # Current stellar mass (after mass loss), not ZAMS mass.
        ("Mass (Msun)", derived.get("star_mass", derived.get("mass"))),
        ("EEP", samples[:, param_names.index("eep")]
         if "eep" in param_names else None),
        ("Teff (K)", derived.get("Teff")),
        ("log(g) (dex)", derived.get("log_g")),
        ("[Fe/H] (dex)", samples[:, param_names.index("feh")]
         if "feh" in param_names else None),
        ("Radius (Rsun)", derived.get("radius")),
        ("Luminosity (Lsun)", _safe_pow10(derived.get("log_L"))),
    ]

    if "distance" in param_names:
        idx = param_names.index("distance")
        display_params.append(("Distance (pc)", samples[:, idx]))
    if "Av" in param_names:
        idx = param_names.index("Av")
        display_params.append(("Av (mag)", samples[:, idx]))
    if "jitter" in param_names:
        idx = param_names.index("jitter")
        display_params.append(("Jitter (mag)", samples[:, idx]))

    fmt = f"{_T3}{{:20s}}  {{:>10s}}  {{:>20s}}  {{:>20s}}"
    print(colored(fmt.format("Parameter", "Median", "1\u03c3 CI", "3\u03c3 CI"), c))
    print(colored(f"{_T3}{'-' * 74}", c))

    for name, arr in display_params:
        if arr is None:
            continue
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        med = np.median(arr)
        lo1 = med - np.percentile(arr, 15.87)
        hi1 = np.percentile(arr, 84.13) - med
        lo3 = med - np.percentile(arr, 0.13)
        hi3 = np.percentile(arr, 99.87) - med
        # Both 1σ and 3σ columns now report half-widths around the median
        # (positive numbers); the table header treats them consistently.
        print(colored(fmt.format(
            name,
            f"{med:.4g}",
            f"+{hi1:.4g}  -{lo1:.4g}",
            f"+{hi3:.4g}  -{lo3:.4g}",
        ), c))

    if "eep" in param_names:
        med_eep = np.median(samples[:, param_names.index("eep")])
        state = _eep_to_state(med_eep)
        print(colored(f"{_T3}{'Evol. state':20s}  {state}", c))

    print()


def display_model_weights(bma_result):
    c = _pick_color()
    print(colored(f'{_T3}Model weights:', c))
    print(colored(f'{_T3}{"Grid":25s}{"Probability":>14s}{"log(Z)":>14s}', c))
    print(colored(f'{_T3}{"-" * 53}', c))
    for name, w, lz in zip(
        bma_result.model_names, bma_result.weights, bma_result.log_evidences
    ):
        print(colored(f'{_T3}{name:25s}{w:14.4f}{lz:14.3f}', c))
    print()


def display_elapsed(t_start):
    c = _pick_color()
    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    if minutes > 0:
        print(colored(
            f'{_T3}Elapsed time : {minutes} minutes and {seconds:.2f} seconds', c))
    else:
        print(colored(f'{_T3}Elapsed time : {seconds:.2f} seconds', c))
    print()


_EEP_PHASE_LABELS = {
    "PreMS": "Pre-Main Sequence",
    "ZAMS": "Main Sequence (early)",
    "IAMS": "Main Sequence (late/turnoff)",
    "TAMS": "Subgiant/RGB",
    "RGBTip": "RGB Tip",
    "ZACHeB": "Core He Burning",
    "TAHeB": "Early AGB",
    "TPAGB": "TP-AGB / post-AGB",
}


def _eep_to_state(eep: float) -> str:
    """Map an EEP value to its phase using config.EEP_PHASES boundaries."""
    last = "Pre-Main Sequence"
    for phase, threshold in EEP_PHASES.items():
        if eep < threshold:
            return last
        last = _EEP_PHASE_LABELS.get(phase, last)
    return last


def _safe_pow10(arr):
    if arr is None:
        return None
    return 10.0 ** arr
