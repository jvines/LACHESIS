"""Abstract isochrone grid interface."""
abstract type IsochroneGrid end

function grid_name(::IsochroneGrid)::String
    error("Not implemented")
end

function feh_values(::IsochroneGrid)::Vector{Float64}
    error("Not implemented")
end

function age_values(::IsochroneGrid)::Vector{Float64}
    error("Not implemented")
end

function eep_values(::IsochroneGrid)::Vector{Float64}
    error("Not implemented")
end

function eep_range(g::IsochroneGrid)::Tuple{Int,Int}
    ev = eep_values(g)
    return (Int(first(ev)), Int(last(ev)))
end

function columns(::IsochroneGrid)::Vector{String}
    error("Not implemented")
end

function grid_data(::IsochroneGrid)::Array{Float64,4}
    error("Not implemented")
end
