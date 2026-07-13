"""Single-grid fit with BHAC15 (M dwarf isochrones).

BHAC15 covers 0.01-1.4 Msun at solar metallicity, so we demonstrate it on
Proxima Centauri, a nearby, well-studied M5.5V dwarf with a literature
mass of 0.122 Msun (Kervella+ 2017, orbital dynamics with Proxima b).
"""
from lachesis.star import Star
from lachesis.fitter import Fitter
from lachesis.plotter import ISOPlotter

if __name__ == '__main__':

    ra = 217.4289
    dec = -62.6795
    starname = 'Proxima Centauri'
    gaia_id = 5853498713190525696

    s = Star(starname, ra, dec, g_id=gaia_id)

    # Output setup
    out_folder = 'test_output_bhac15'
    in_file = out_folder + '/lachesis_Proxima_Centauri.nc'
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
    f.grids = ['bhac15']
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
