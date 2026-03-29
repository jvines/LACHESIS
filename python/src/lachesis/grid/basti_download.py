"""Download BaSTI isochrones from the IAC web interface.

Hits http://basti-iac.oa-abruzzo.inaf.it/cgi-bin/isoc-get.py
one [Fe/H] value at a time (all ages in one request).

The CGI returns a .tar.gz with one file per age (e.g. 1000z...isc_hr,
2000z...isc_hr). Each file has columns: log(L/Lo), logTe, M/Mo(ini),
M/Mo(fin). We combine them into a single output file with
``#AGE=<myr>`` separator lines.
"""

import io
import logging
import re
import tarfile
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_CGI_URL = "http://basti-iac.oa-abruzzo.inaf.it/cgi-bin/isoc-get.py"
_BASE_URL = "http://basti-iac.oa-abruzzo.inaf.it"

# Default [Fe/H] grid (9 values spanning BaSTI's range)
_DEFAULT_FEH = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.30]


def _extract_age_myr(filename: str) -> int | None:
    """Extract age in Myr from BaSTI filename like '1000z0150014y269...'."""
    basename = filename.rsplit("/", maxsplit=1)[-1]
    m = re.match(r"(\d+)z", basename)
    if m:
        return int(m.group(1))
    return None


def download_basti(
    output_dir: str | Path,
    feh_values: list[float] | None = None,
    delay: float = 3.0,
):
    """Download BaSTI isochrones for a grid of [Fe/H] values.

    Parameters
    ----------
    output_dir : path
        Directory to save .dat files.
    feh_values : list of float
        [Fe/H] values to download. Default: -2.0 to +0.45.
    delay : float
        Seconds between requests (polite scraping).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if feh_values is None:
        feh_values = _DEFAULT_FEH

    for feh in feh_values:
        outfile = output_dir / f"basti_feh{feh:+.2f}.dat"
        if outfile.exists():
            logger.info("Already exists: %s", outfile.name)
            continue

        logger.info("Downloading [Fe/H]=%+.2f ...", feh)

        params = {
            "alpha": "P00",
            "grid": "P00O1D1E1Y247",
            "metal": "None",
            "imetal": "",
            "imetalh": f"{feh}",
            "iage": "1000--15000,500",
            "bcsel": "HR",
        }

        # Step 1: GET the CGI page -> HTML with download link
        try:
            resp = requests.get(_CGI_URL, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("[Fe/H]=%+.2f request failed: %s", feh, e)
            continue

        # Step 2: extract .tar.gz link from HTML
        match = re.search(r'href="([^"]+\.tar\.gz)"', resp.text)
        if not match:
            logger.error(
                "[Fe/H]=%+.2f: no .tar.gz link in response. "
                "First 500 chars: %s",
                feh,
                resp.text[:500],
            )
            continue

        tar_path = match.group(1)
        if tar_path.startswith("/"):
            tar_url = f"{_BASE_URL}{tar_path}"
        elif tar_path.startswith("http"):
            tar_url = tar_path
        else:
            tar_url = f"{_BASE_URL}/{tar_path}"

        logger.info("  Downloading tar: %s", tar_url)

        # Step 3: download the .tar.gz
        try:
            tar_resp = requests.get(tar_url, timeout=120)
            tar_resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("[Fe/H]=%+.2f tar download failed: %s", feh, e)
            continue

        # Step 4: extract all age files from tar.gz, sorted by age
        try:
            age_blocks: list[tuple[int, str]] = []
            with tarfile.open(
                fileobj=io.BytesIO(tar_resp.content), mode="r:gz"
            ) as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    age_myr = _extract_age_myr(m.name)
                    if age_myr is None:
                        logger.warning("  Skipping unknown member: %s", m.name)
                        continue
                    raw = tf.extractfile(m)
                    if raw is None:
                        continue
                    text = raw.read().decode("utf-8", errors="replace")
                    # Strip the per-file header lines (lines starting with #)
                    data_lines = [
                        l for l in text.splitlines() if not l.startswith("#")
                    ]
                    age_blocks.append((age_myr, "\n".join(data_lines)))

            age_blocks.sort(key=lambda x: x[0])

            if not age_blocks:
                logger.error("[Fe/H]=%+.2f: no age files in tar", feh)
                continue

            # Write combined file with age headers
            with open(outfile, "w") as out:
                out.write(
                    "#==================================================\n"
                )
                out.write(
                    "#  log(L/Lo)       logTe          M/Mo(ini)      "
                    " M/Mo(fin)\n"
                )
                out.write(
                    "#==================================================\n"
                )
                for age_myr, data in age_blocks:
                    n_rows = len(data.strip().splitlines())
                    out.write(f"#AGE={age_myr} NPTS={n_rows}\n")
                    out.write(data)
                    out.write("\n")

            logger.info(
                "Saved %s (%d ages, %d bytes)",
                outfile.name,
                len(age_blocks),
                outfile.stat().st_size,
            )
        except (tarfile.TarError, OSError) as e:
            logger.error("[Fe/H]=%+.2f: tar extraction failed: %s", feh, e)
            continue

        time.sleep(delay)

    logger.info("Done. Files in %s", output_dir)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = sys.argv[1] if len(sys.argv) > 1 else "data/basti/raw"
    download_basti(out)
