"""3D grid interpolation over isochrone grids."""

using Interpolations

struct GridInterpolator{G<:IsochroneGrid}
    grid::G
    interpolators::Dict{String, Any}
end

"""Pad length-1 axes so Interpolations.jl can handle them."""
function _pad_axis(vals::Vector{Float64}, data::Array{Float64,3}, dim::Int)
    if length(vals) > 1
        return vals, data
    end
    # Create a 2-point axis with tiny offset, duplicate data
    v = vals[1]
    new_vals = [v - 0.01, v + 0.01]
    new_data = cat(data, data; dims=dim)
    return new_vals, new_data
end

function GridInterpolator(g::IsochroneGrid)
    feh = copy(feh_values(g))
    ages = copy(age_values(g))
    eeps = Float64.(eep_values(g))
    cols = columns(g)
    data = grid_data(g)

    interpolators = Dict{String, Any}()
    for (ci, col) in enumerate(cols)
        data_3d = data[:, :, :, ci]
        # Handle length-1 axes (Interpolations.jl requires length >= 2)
        f, d = _pad_axis(feh, data_3d, 1)
        a, d = _pad_axis(ages, d, 2)
        e, d = _pad_axis(eeps, d, 3)

        itp = interpolate((f, a, e), d, Gridded(Linear()))
        interpolators[col] = extrapolate(itp, NaN)
    end

    return GridInterpolator(g, interpolators)
end

"""
    (interp::GridInterpolator)(; eep, log_age, feh) → Dict{String, Float64}

Interpolate at scalar (EEP, log_age, [Fe/H]) → dict of observables.
"""
function (interp::GridInterpolator)(; eep::Float64, log_age::Float64, feh::Float64)
    result = Dict{String, Float64}()
    for (col, itp) in interp.interpolators
        result[col] = itp(feh, log_age, eep)
    end
    return result
end

"""
    interpolate_batch(interp, eeps, log_ages, fehs) → Dict{String, Vector{Float64}}

Vectorized interpolation.
"""
function interpolate_batch(
    interp::GridInterpolator,
    eeps::AbstractVector{Float64},
    log_ages::AbstractVector{Float64},
    fehs::AbstractVector{Float64},
)
    n = length(eeps)
    result = Dict{String, Vector{Float64}}()
    for (col, itp) in interp.interpolators
        vals = Vector{Float64}(undef, n)
        for i in 1:n
            vals[i] = itp(fehs[i], log_ages[i], eeps[i])
        end
        result[col] = vals
    end
    return result
end
