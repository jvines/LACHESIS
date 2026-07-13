"""Test the ARIADNE -> LACHESIS pipeline with a real ARIADNE .nc file."""
from lachesis.star import Star
from lachesis.fitter import Fitter
from lachesis.plotter import ISOPlotter

if __name__ == '__main__':

    # Load stellar info from ARIADNE posterior + photometry from Librarian
    s = Star.from_ariadne(
        'astroARIADNE/HD209458/ariadne_result.nc',
        starname='HD 209458',
        ra=330.795,
        dec=18.884,
        g_id=1779546757669063552,
    )

    out_folder = 'test_output_ariadne'

    # Same fitter setup as test_bma.py
    setup = ['dynesty', 500, 0.5, 'multi', 'rwalk', 4, False]
    grids = ['mist', 'parsec', 'dartmouth', 'basti', 'yapsi']

    f = Fitter()
    f.star = s
    f.setup = setup
    f.av_law = 'fitzpatrick'
    f.verbose = True
    f.out_folder = out_folder
    f.bma = True
    f.grids = grids

    # Let ARIADNE's KDE priors on [Fe/H] and logg handle those automatically
    f.prior_setup = {
        'eep': ('default'),
        'log_age': ('default'),
        'feh': ('default'),
        'dist': ('default'),
        'Av': ('default'),
    }

    f.initialize()
    f.show_priors()
    f.fit_bma()

    # Plot results
    in_file = out_folder + '/lachesis_HD_209458_BMA.nc'
    plots_out = out_folder + '/plots'

    artist = ISOPlotter(in_file, plots_out)
    artist.plot_corner()
    artist.plot_histograms()
    artist.plot_hr()
    artist.plot_mass_age()
    artist.plot_model_weights()
    artist.summary()
