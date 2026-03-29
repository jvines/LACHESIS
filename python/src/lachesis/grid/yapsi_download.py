"""Download YAPSI (Yale-Potsdam Stellar Isochrones) FITS bundle.

Downloads the precomputed isochrone bundle for Y=0.28 from
http://www.astro.yale.edu/yapsi/data/bundles/
"""

import logging
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_BUNDLE_URL = (
    "http://www.astro.yale.edu/yapsi/data/bundles/yapsi_Y0p28.fits.zip"
)


def download_yapsi(output_dir: str | Path):
    """Download the YAPSI Y=0.28 FITS isochrone bundle.

    Parameters
    ----------
    output_dir : path
        Directory to save the extracted FITS file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "yapsi_Y0p28.fits.zip"

    # Check if already extracted
    fits_files = list(output_dir.glob("*.fits"))
    if fits_files:
        logger.info("FITS file already exists: %s", fits_files[0].name)
        return fits_files[0]

    # Download
    logger.info("Downloading YAPSI bundle from %s ...", _BUNDLE_URL)
    resp = requests.get(_BUNDLE_URL, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                logger.info("  %.1f%% (%d / %d MB)", pct, downloaded >> 20, total >> 20)

    logger.info("Downloaded %s (%d MB)", zip_path.name, zip_path.stat().st_size >> 20)

    # Extract
    logger.info("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Clean up zip
    zip_path.unlink()

    fits_files = list(output_dir.glob("*.fits"))
    if not fits_files:
        raise FileNotFoundError("No FITS file found after extraction")

    logger.info("Extracted: %s", fits_files[0].name)
    return fits_files[0]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else "data/yapsi/raw"
    download_yapsi(out)
