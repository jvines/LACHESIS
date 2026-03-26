"""Derived stellar quantities from raw isochrone columns."""

compute_teff(log_teff) = 10.0 .^ log_teff

compute_mbol(log_l) = MBOL_SUN .- 2.5 .* log_l

compute_radius(log_r) = 10.0 .^ log_r

function compute_density(star_mass, radius)
    r_cm = radius .* R_SUN
    m_g = star_mass .* M_SUN
    return m_g ./ (4.0 / 3.0 .* π .* r_cm .^ 3)
end

"""Gradient of initial_mass along the EEP axis (dimension `dim`)."""
function compute_dm_deep(initial_mass::AbstractArray; dim::Int=3)
    out = similar(initial_mass)
    fill!(out, NaN)
    n = size(initial_mass, dim)
    n < 2 && return out

    # Use selectdim for generic N-D gradient along `dim`
    for i in 1:n
        if i == 1
            selectdim(out, dim, 1) .= selectdim(initial_mass, dim, 2) .- selectdim(initial_mass, dim, 1)
        elseif i == n
            selectdim(out, dim, n) .= selectdim(initial_mass, dim, n) .- selectdim(initial_mass, dim, n - 1)
        else
            selectdim(out, dim, i) .= (selectdim(initial_mass, dim, i + 1) .- selectdim(initial_mass, dim, i - 1)) ./ 2.0
        end
    end
    return out
end
