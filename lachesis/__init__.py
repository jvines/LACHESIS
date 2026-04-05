"""LACHESIS — Isochrone fitting with Bayesian Model Averaging."""

__version__ = "0.1.0"

from lachesis.star import Star
from lachesis.fitter import Fitter
from lachesis.plotter import ISOPlotter


def fit(starname, ra=None, dec=None, gaia_id=None,
        grids=None, bma=True, nlive=500, dlogz=0.01, verbose=True):
    """One-liner isochrone BMA fit.

    Parameters
    ----------
    starname : str or Star
        Star name (resolved via Simbad if no ra/dec given) or a Star object.
    ra, dec : float, optional
        Coordinates in degrees. If not given, resolved from starname via Simbad.
    gaia_id : int, optional
        Gaia DR3 source_id. Bypasses cone search for high-PM stars.
    grids : list of str, optional
        Grid names. Default: ["mist", "parsec", "dartmouth", "basti", "yapsi"].
    bma : bool
        Use Bayesian Model Averaging (default True).
    nlive : int
        Number of live points for nested sampling.
    dlogz : float
        Stopping criterion for nested sampling.
    verbose : bool
        Print progress.

    Returns
    -------
    result : BMAResult or dict
        BMA result if bma=True, single-grid result dict otherwise.

    Examples
    --------
    >>> import lachesis
    >>> result = lachesis.fit("HD 209458")
    >>> result = lachesis.fit("HD 209458", ra=330.795, dec=18.884)
    >>> result = lachesis.fit("HD 103095", gaia_id=4034171629042489088)
    """
    if isinstance(starname, Star):
        star = starname
    else:
        if ra is None or dec is None:
            # Resolve coordinates from name via Simbad
            from astropy.coordinates import SkyCoord
            coord = SkyCoord.from_name(starname)
            ra, dec = coord.ra.deg, coord.dec.deg
            # Try to get Gaia DR3 ID from Simbad if not provided
            if gaia_id is None:
                try:
                    from astroquery.simbad import Simbad
                    ids = Simbad.query_objectids(starname)
                    if ids is not None:
                        for row in ids:
                            rid = str(row[0]) if hasattr(row, '__getitem__') else str(row)
                            if 'Gaia DR3' in rid:
                                gaia_id = int(rid.split()[-1])
                                break
                except Exception:
                    pass
        star = Star(starname, ra, dec, g_id=gaia_id, verbose=verbose)

    f = Fitter()
    f.star = star
    if grids is not None:
        f.grids = grids
    f.bma = bma
    f.setup = ["dynesty", nlive, dlogz, "multi", "rwalk", 1, False]
    f.verbose = verbose
    f.initialize()

    if bma:
        return f.fit_bma()
    else:
        return f.fit()


__all__ = ["Star", "Fitter", "ISOPlotter", "fit", "__version__"]
