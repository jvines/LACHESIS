"""MIST isochrone grid: parse, cache, model grid."""

using HDF5

# ---------------------------------------------------------------------------- #
# MISTIsoFile — parser for a single .iso file (one [Fe/H])
# ---------------------------------------------------------------------------- #

struct MISTIsoFile
    version::String
    feh::Float64
    afe::Float64
    vvcrit::Float64
    num_isochrones::Int
    columns::Vector{String}
    log_ages::Vector{Float64}
    isochrones::Vector{Matrix{Float64}}
end

function MISTIsoFile(path::AbstractString)
    lines = readlines(path)
    version = ""
    feh = 0.0
    afe = 0.0
    vvcrit = 0.0
    num_iso = 0
    col_names = String[]
    ages = Float64[]
    isos = Matrix{Float64}[]

    i = 1
    expect_params = false

    # Parse file header
    while i <= length(lines) && startswith(lines[i], "#")
        line = strip(lines[i])

        m = match(r"MIST version number\s*=\s*([\d.]+)", line)
        if m !== nothing
            version = m.captures[1]
        end

        if occursin("Yinit", line)
            expect_params = true
            i += 1
            continue
        end
        if expect_params
            vals = split(replace(line, r"^#\s*" => ""))
            feh = parse(Float64, vals[3])
            afe = parse(Float64, vals[4])
            vvcrit = parse(Float64, vals[5])
            expect_params = false
        end

        m = match(r"number of isochrones\s*=\s*(\d+)", line)
        if m !== nothing
            num_iso = parse(Int, m.captures[1])
        end

        if startswith(line, "# number of EEPs")
            break
        end

        i += 1
    end

    # Parse all isochrone blocks
    while i <= length(lines)
        line = strip(lines[i])

        if startswith(line, "# number of EEPs")
            m = match(r"number of EEPs, cols\s*=\s*(\d+)\s+(\d+)", line)
            n_eeps = parse(Int, m.captures[1])
            n_cols = parse(Int, m.captures[2])

            # Skip column numbers line
            i += 1
            # Skip / parse column names line
            i += 1
            col_line = strip(lines[i])
            if startswith(col_line, "# EEP")
                col_names = split(col_line[3:end])
            end
            i += 1

            # Read data rows
            block = Matrix{Float64}(undef, n_eeps, n_cols)
            for j in 1:n_eeps
                block[j, :] = parse.(Float64, split(lines[i]))
                i += 1
            end

            push!(isos, block)
            age_col = findfirst(==("log10_isochrone_age_yr"), col_names)
            push!(ages, block[1, age_col])
        else
            i += 1
        end
    end

    return MISTIsoFile(version, feh, afe, vvcrit, num_iso, col_names, ages, isos)
end

function get_isochrone(iso::MISTIsoFile, log_age::Float64)
    _, idx = findmin(abs.(iso.log_ages .- log_age))
    if abs(iso.log_ages[idx] - log_age) > 0.001
        error("No isochrone at log_age=$log_age. Nearest is $(iso.log_ages[idx])")
    end
    return iso.isochrones[idx]
end

# ---------------------------------------------------------------------------- #
# MISTModelGrid — full model grid with derived columns
# ---------------------------------------------------------------------------- #

const RAW_COLS = [
    "initial_mass", "star_mass", "log_Teff", "log_g",
    "log_L", "log_R", "phase", "delta_nu", "nu_max",
]
const DERIVED_COLS = ["Teff", "Mbol", "radius", "density", "dm_deep"]
const AXIS_COLS = ["eep", "age"]
const MODEL_COLS = vcat(AXIS_COLS, RAW_COLS, DERIVED_COLS)

struct MISTModelGrid <: IsochroneGrid
    feh_vals::Vector{Float64}
    age_vals::Vector{Float64}
    eep_vals::Vector{Float64}
    cols::Vector{String}
    data::Array{Float64,4}  # (n_feh, n_age, n_eep, n_cols)
end

grid_name(::MISTModelGrid) = "MIST"
feh_values(g::MISTModelGrid) = g.feh_vals
age_values(g::MISTModelGrid) = g.age_vals
eep_values(g::MISTModelGrid) = g.eep_vals
columns(g::MISTModelGrid) = g.cols
grid_data(g::MISTModelGrid) = g.data

function MISTModelGrid(dir::AbstractString)
    # Prefer full iso files, fall back to any .iso
    files = sort(filter(f -> occursin("full", f) && endswith(f, ".iso"), readdir(dir; join=true)))
    if isempty(files)
        files = sort(filter(f -> endswith(f, ".iso"), readdir(dir; join=true)))
    end
    isempty(files) && error("No .iso files in $dir")

    parsed = [MISTIsoFile(f) for f in files]
    feh_vals = sort([p.feh for p in parsed])
    age_vals = parsed[1].log_ages
    all_columns = parsed[1].columns

    raw_col_indices = [findfirst(==(c), all_columns) for c in RAW_COLS]
    eep_col = findfirst(==("EEP"), all_columns)

    # Global EEP set
    all_eeps = Set{Int}()
    for p in parsed, iso_data in p.isochrones
        for row in eachrow(iso_data)
            push!(all_eeps, Int(row[eep_col]))
        end
    end
    eep_vals = sort(collect(all_eeps))
    eep_to_idx = Dict(Int(e) => i for (i, e) in enumerate(eep_vals))

    n_feh = length(feh_vals)
    n_age = length(age_vals)
    n_eep = length(eep_vals)
    n_cols = length(MODEL_COLS)

    data = fill(NaN, n_feh, n_age, n_eep, n_cols)

    ci_eep = findfirst(==("eep"), MODEL_COLS)
    ci_age = findfirst(==("age"), MODEL_COLS)
    ci_raw_start = length(AXIS_COLS)

    feh_order = sortperm([p.feh for p in parsed])
    for (fi, pi) in enumerate(feh_order)
        p = parsed[pi]
        for (ai, iso_data) in enumerate(p.isochrones)
            age_val = age_vals[ai]
            for row_i in axes(iso_data, 1)
                eep_val = Int(iso_data[row_i, eep_col])
                ei = eep_to_idx[eep_val]
                data[fi, ai, ei, ci_eep] = eep_val
                data[fi, ai, ei, ci_age] = age_val
                for (rci, src_idx) in enumerate(raw_col_indices)
                    data[fi, ai, ei, ci_raw_start + rci] = iso_data[row_i, src_idx]
                end
            end
        end
    end

    # Compute derived columns
    ci = Dict(c => findfirst(==(c), MODEL_COLS) for c in MODEL_COLS)

    data[:, :, :, ci["Teff"]] = compute_teff(data[:, :, :, ci["log_Teff"]])
    data[:, :, :, ci["Mbol"]] = compute_mbol(data[:, :, :, ci["log_L"]])
    data[:, :, :, ci["radius"]] = compute_radius(data[:, :, :, ci["log_R"]])
    data[:, :, :, ci["density"]] = compute_density(
        data[:, :, :, ci["star_mass"]],
        data[:, :, :, ci["radius"]],
    )
    data[:, :, :, ci["dm_deep"]] = compute_dm_deep(
        data[:, :, :, ci["initial_mass"]]; dim=3,
    )

    return MISTModelGrid(Float64.(feh_vals), age_vals, Float64.(eep_vals), copy(MODEL_COLS), data)
end

# ---------------------------------------------------------------------------- #
# HDF5 I/O
# ---------------------------------------------------------------------------- #

function save_hdf5(g::MISTModelGrid, path::AbstractString)
    h5open(path, "w") do f
        f["data"] = g.data
        f["feh_values"] = g.feh_vals
        f["age_values"] = g.age_vals
        f["eep_values"] = g.eep_vals
        # Store columns as dataset (more portable than string attrs)
        f["columns"] = g.cols
        attrs(f)["version"] = "1.2"
        attrs(f)["grid_name"] = "MIST"
    end
end

function load_hdf5_model_grid(path::AbstractString)::MISTModelGrid
    h5open(path, "r") do f
        data = read(f["data"])
        feh_vals = read(f["feh_values"])
        age_vals = read(f["age_values"])
        eep_vals = read(f["eep_values"])
        cols = String.(read(f["columns"]))
        MISTModelGrid(feh_vals, age_vals, eep_vals, cols, data)
    end
end
