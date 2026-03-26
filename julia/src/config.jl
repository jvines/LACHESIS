const DATA_DIR = joinpath(dirname(dirname(@__DIR__)), "data")
const MIST_DIR = joinpath(DATA_DIR, "mist")
const MIST_RAW_DIR = joinpath(MIST_DIR, "raw")
const MIST_GRID_DIR = joinpath(MIST_DIR, "grids")

const MIST_BASE_URL = "https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2"
const MIST_BC_URL = "https://waps.cfa.harvard.edu/MIST/BC_tables"

const EEP_PHASES = Dict{String,Int}(
    "PreMS" => 1,
    "ZAMS" => 202,
    "IAMS" => 353,
    "TAMS" => 454,
    "RGBTip" => 605,
    "ZACHeB" => 631,
    "TAHeB" => 707,
    "TPAGB" => 808,
)

# IAU 2015 nominal solar values
const M_SUN = 1.98892e33   # g
const R_SUN = 6.9566e10    # cm
const MBOL_SUN = 4.74      # absolute bolometric magnitude
