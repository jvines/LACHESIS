"""Download BHAC15 (Baraffe+ 2015) isochrone file.

Downloads from http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/
We grab a single photometric-system file (2MASS) since the fundamental
parameters (M/Ms, Teff, L/Ls, g, R/Rs) are identical across all files.
The photometry columns are ignored by the grid parser.

BHAC15 is solar metallicity only. Mass range 0.01-1.4 Msun.
Ages 0.5 Myr - 10 Gyr (30 isochrones).
"""

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir"
_FILENAME = "BHAC15_iso.2mass"


def download_bhac15(output_dir: str | Path):
    """Download the BHAC15 isochrone file.

    Parameters
    ----------
    output_dir : path
        Directory to save the downloaded file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outfile = output_dir / _FILENAME
    if outfile.exists():
        logger.info("Already exists: %s", outfile.name)
        return outfile

    url = f"{_BASE_URL}/{_FILENAME}"
    logger.info("Downloading %s ...", url)

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    outfile.write_text(resp.text)
    logger.info(
        "Saved %s (%d bytes)", outfile.name, outfile.stat().st_size
    )
    return outfile


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else "data/bhac15/raw"
    download_bhac15(out)
