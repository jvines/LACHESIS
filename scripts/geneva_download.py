"""Download Geneva (Ekstroem+ 2012) isochrones.

Individual text files from:
https://obswww.unige.ch/Research/evol/tables_grids2011/isochrones/

Only Z=0.014 (solar) isochrones are precomputed by the Geneva group.
Other metallicities (Z=0.002, 0.006, 0.020) only have evolutionary tracks,
not precomputed isochrones, on this server.

Non-rotating models only (Vini=0.00).
Ages: log(age/yr) from 9.0 to 10.1 in steps of 0.1 (>= 1 Gyr).
"""

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://obswww.unige.ch/Research/evol/tables_grids2011/isochrones"
)

# Only Z=0.014 has precomputed isochrones on the Geneva server
_DEFAULT_Z = [0.014]

# log(age/yr) from 9.0 to 10.1 in steps of 0.1
_DEFAULT_LOG_AGES = [round(9.0 + 0.1 * i, 1) for i in range(12)]


def _filename(z: float, log_age: float) -> str:
    """Build the Geneva isochrone filename.

    The server uses zero-padded ages: t09.000, t10.000, etc.
    """
    return f"Isochr_Z{z:.3f}_Vini0.00_t{log_age:06.3f}.dat"


def download_geneva(
    output_dir: str | Path,
    z_values: list[float] | None = None,
    log_ages: list[float] | None = None,
    delay: float = 1.0,
):
    """Download Geneva isochrone files.

    Parameters
    ----------
    output_dir : path
        Directory to save .dat files.
    z_values : list of float
        Metal mass fractions. Default: [0.014] (only solar available).
    log_ages : list of float
        log(age/yr) values. Default: 9.0 to 10.1 step 0.1.
    delay : float
        Seconds between requests.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if z_values is None:
        z_values = _DEFAULT_Z
    if log_ages is None:
        log_ages = _DEFAULT_LOG_AGES

    total = len(z_values) * len(log_ages)
    downloaded = 0
    skipped = 0

    for z in z_values:
        for log_age in log_ages:
            fname = _filename(z, log_age)
            outfile = output_dir / fname

            if outfile.exists():
                skipped += 1
                continue

            url = f"{_BASE_URL}/{fname}"
            logger.info("Downloading %s", fname)

            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.error("Failed %s: %s", fname, e)
                continue

            outfile.write_text(resp.text)
            downloaded += 1

            time.sleep(delay)

    logger.info(
        "Done. %d downloaded, %d skipped, %d total expected. Files in %s",
        downloaded,
        skipped,
        total,
        output_dir,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else "data/geneva/raw"
    download_geneva(out)
