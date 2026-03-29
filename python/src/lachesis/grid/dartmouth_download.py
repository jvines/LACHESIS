"""Download Dartmouth (DSEP) isochrones from the web interface.

Scrapes https://rcweb.dartmouth.edu/stellar/isolf_new.php
one [Fe/H] value at a time, saving raw .iso files.
"""

import logging
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://rcweb.dartmouth.edu/stellar"
_PHP_URL = f"{_BASE_URL}/isolf_new.php"

# Default [Fe/H] grid — widest Dartmouth range
_DEFAULT_FEH = [-2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5]

# Default ages: 1.0–15.0 Gyr in 0.5 Gyr steps
_DEFAULT_AGES = [a / 2 for a in range(2, 31)]  # 1.0, 1.5, 2.0, ..., 15.0


def download_dartmouth(
    output_dir: str | Path,
    feh_values: list[float] | None = None,
    ages_gyr: list[float] | None = None,
    afe: int = 2,
    hel: int = 1,
    clr: int = 1,
    delay: float = 2.0,
):
    """Download Dartmouth isochrones for a grid of [Fe/H] values.

    Parameters
    ----------
    output_dir : path
        Directory to save .iso files.
    feh_values : list of float
        [Fe/H] values to download. Default: -2.5 to +0.5.
    ages_gyr : list of float
        Ages in Gyr (max 50 per request). Default: 1.0–15.0 in 0.5 Gyr steps.
    afe : int
        Alpha enhancement (2=solar-scaled).
    hel : int
        Helium option (1=standard Y=0.245+1.5Z).
    clr : int
        Photometric system (1=UBV+2MASS+Kepler).
    delay : float
        Seconds between requests (polite scraping).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if feh_values is None:
        feh_values = _DEFAULT_FEH
    if ages_gyr is None:
        ages_gyr = _DEFAULT_AGES

    age_str = " ".join(f"{a:.1f}" for a in ages_gyr)

    for feh in feh_values:
        outfile = output_dir / f"dartmouth_feh{feh:+.2f}.iso"
        if outfile.exists():
            logger.info("Already exists: %s", outfile.name)
            continue

        logger.info("Downloading [Fe/H]=%+.2f ...", feh)

        params = {
            "int": "1",
            "out": "1",
            "age": age_str,
            "feh": f"{feh:.2f}",
            "hel": str(hel),
            "afe": str(afe),
            "clr": str(clr),
        }

        # Step 1: GET the PHP page → HTML with temp file link
        try:
            resp = requests.get(_PHP_URL, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("[Fe/H]=%+.2f request failed: %s", feh, e)
            continue

        # Step 2: extract temp file URL from HTML
        match = re.search(r'href="(tmp/[^"]+\.iso)"', resp.text)
        if not match:
            logger.error("[Fe/H]=%+.2f: no .iso link in response", feh)
            continue

        iso_url = f"{_BASE_URL}/{match.group(1)}"

        # Step 3: download the actual .iso file
        try:
            iso_resp = requests.get(iso_url, timeout=60)
            iso_resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("[Fe/H]=%+.2f .iso download failed: %s", feh, e)
            continue

        outfile.write_text(iso_resp.text)
        logger.info("Saved %s (%d bytes)", outfile.name, len(iso_resp.text))

        time.sleep(delay)

    logger.info("Done. Files in %s", output_dir)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else "data/dartmouth/raw"
    download_dartmouth(out)
