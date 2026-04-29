from lachesis.star import Star
from lachesis.fitter import Fitter
from lachesis.plotter import ISOPlotter

if __name__ == '__main__':

    ra = 330.795
    dec = 18.884
    starname = 'HD 209458'
    gaia_id = 1779546757669063552

    s = Star(starname, ra, dec, g_id=gaia_id)

    # Output setup
    out_folder = 'test_output'
    in_file = out_folder + '/lachesis_HD_209458_BMA.nc'
    plots_out_folder = out_folder + '/plots'

    # Setup parameters
    engine = 'dynesty'   # Only dynesty is supported
    nlive = 500          # Number of live points to use
    dlogz = 0.5          # Evidence tolerance
    bound = 'multi'      # Unit cube bounds. Options are multi, single
    sample = 'rwalk'     # Sampling method. Options are rwalk, unif
    threads = 4          # Number of threads to use
    dynamic = False      # Use dynamic nested sampling?
    setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

    # Isochrone grids for BMA
    # Feel free to comment out any unneeded/unwanted grids
    grids = [
        'mist',
        'parsec',
        'dartmouth',
        'basti',
        'yapsi',
    ]

    # Now to setup the fitter and run the modelling.
    f = Fitter()
    f.star = s
    f.setup = setup
    f.av_law = 'fitzpatrick'
    f.verbose = True
    f.out_folder = out_folder
    f.bma = True
    f.grids = grids
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

    # Setting up plotter, which is independent to the main fitting routine
    artist = ISOPlotter(in_file, plots_out_folder)
    artist.plot_corner()
    artist.plot_histograms()
    artist.plot_hr()
    artist.plot_mass_age()
    artist.plot_model_weights()
    artist.summary()

