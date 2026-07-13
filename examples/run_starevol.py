"""Single-grid fit with STAREVOL (Amard+ 2019).

STAREVOL includes stellar rotation (Vini) as a free parameter, sampled as
a 6th dimension. We demonstrate it on HD 209458, a G0V star with modest
rotation (v sin i ~ 4 km/s).
"""
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
    out_folder = 'test_output_starevol'
    in_file = out_folder + '/lachesis_HD_209458.nc'
    plots_out_folder = out_folder + '/plots'

    # Setup parameters
    engine = 'dynesty'
    nlive = 1000
    dlogz = 0.01
    bound = 'multi'
    sample = 'rwalk'
    threads = 4
    dynamic = False
    setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

    f = Fitter()
    f.star = s
    f.setup = setup
    f.av_law = 'fitzpatrick'
    f.verbose = True
    f.out_folder = out_folder
    f.bma = False
    f.grids = ['starevol']
    f.prior_setup = {
        'eep': ('default'),
        'log_age': ('default'),
        'feh': ('default'),
        'dist': ('default'),
        'Av': ('default'),
    }

    f.initialize()
    f.show_priors()
    f.fit()

    # Plotting
    artist = ISOPlotter(in_file, plots_out_folder)
    artist.plot_corner()
    artist.plot_hr()
    artist.plot_mass_age()
    artist.summary()
