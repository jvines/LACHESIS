"""Download STAREVOL (Amard+ 2019) isochrones.

Fetches precomputed isochrone files from the CDS VizieR archive
(catalog J/A+A/631/A77). Downloads non-rotating (Vini=0.00) by default,
with option to download rotating models (Vini=0.20, 0.40, 0.60).

Grid coverage:
    Metallicities: Z = 0.0020, 0.0060, 0.0080, 0.0100, 0.0130, 0.0190, 0.0260
                   ([Fe/H] ~ -0.83 to +0.29)
    Ages: log(age/yr) from 9.0 to 10.1 in steps of 0.1
    Masses: 0.2 to 1.5 Msun (per isochrone)

Reference: Amard, Palacios, Charbonnel et al. 2019, A&A, 631, A77
"""

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# CDS mirrors -- try main first, fall back to China mirror
_MIRROR_URLS = [
    "https://cdsarc.cds.unistra.fr/ftp/cats/J/A+A/631/A77/iso",
    "http://vizier.china-vo.org/ftp/cats/J/A+A/631/A77/iso",
]

# Seven metallicities from the Amard+ 2019 grid
_DEFAULT_Z = [0.0020, 0.0060, 0.0080, 0.0100, 0.0130, 0.0190, 0.0260]

# log(age/yr) from 9.0 to 10.1 in steps of 0.1 (>= 1 Gyr)
_DEFAULT_LOG_AGES = [round(9.0 + 0.1 * i, 1) for i in range(12)]


_DEFAULT_VINI = [0.00, 0.20, 0.40, 0.60]


def _filename(z: float, vini: float, log_age: float) -> str:
    """Build the CDS isochrone filename."""
    return f"Isochr_Z{z:.4f}_Vini{vini:.2f}_t{log_age:06.3f}.dat"


def _try_download(fname: str, timeout: float = 60) -> str | None:
    """Try downloading from each mirror until one succeeds."""
    for base_url in _MIRROR_URLS:
        url = f"{base_url}/{fname}"
        try:
            resp = requests.get(url, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200 and len(resp.text) > 100:
                # Sanity: real data files don't start with <!DOCTYPE
                if not resp.text.strip().startswith("<!"):
                    return resp.text
        except requests.RequestException:
            continue
    return None


def download_starevol(
    output_dir: str | Path,
    z_values: list[float] | None = None,
    vini_values: list[float] | None = None,
    log_ages: list[float] | None = None,
    delay: float = 0.5,
):
    """Download STAREVOL isochrone files from CDS.

    Parameters
    ----------
    output_dir : path
        Directory to save .dat files.
    z_values : list of float
        Metal mass fractions. Default: all 7 Amard+ 2019 values.
    vini_values : list of float
        Initial rotation rates (V/Vcrit). Default: [0.00, 0.20, 0.40, 0.60].
    log_ages : list of float
        log(age/yr) values. Default: 9.0 to 10.1 step 0.1.
    delay : float
        Seconds between requests (polite scraping).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if z_values is None:
        z_values = _DEFAULT_Z
    if vini_values is None:
        vini_values = _DEFAULT_VINI
    if log_ages is None:
        log_ages = _DEFAULT_LOG_AGES

    total = len(z_values) * len(vini_values) * len(log_ages)
    downloaded = 0
    skipped = 0
    failed = 0

    for z in z_values:
        for vini in vini_values:
            for log_age in log_ages:
                fname = _filename(z, vini, log_age)
                outfile = output_dir / fname

                if outfile.exists() and outfile.stat().st_size > 0:
                    skipped += 1
                    continue

                logger.info("Downloading %s", fname)
                text = _try_download(fname)

                if text is None:
                    logger.error("Failed %s from all mirrors", fname)
                    failed += 1
                    continue

                outfile.write_text(text)
                downloaded += 1

                if delay > 0:
                    time.sleep(delay)

    logger.info(
        "Done. %d downloaded, %d skipped, %d failed, %d total expected. "
        "Files in %s",
        downloaded,
        skipped,
        failed,
        total,
        output_dir,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else "data/starevol/raw"
    download_starevol(out)
