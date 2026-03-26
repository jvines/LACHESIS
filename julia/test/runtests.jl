using Test
using Lachesis

# Test data lives in the Python tests/data directory (shared)
const TEST_DATA = joinpath(dirname(dirname(@__DIR__)), "python", "tests", "data")
const BASIC_ISO = joinpath(TEST_DATA, "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso")
const FULL_ISO = joinpath(TEST_DATA, "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_full.iso")

@testset "Lachesis.jl" begin

    @testset "Derived quantities" begin
        @test Lachesis.compute_teff(log10(5772.0)) ≈ 5772.0 rtol=1e-6
        @test Lachesis.compute_teff.([3.5, 4.0]) ≈ [10^3.5, 10^4.0]
        @test isnan(Lachesis.compute_teff(NaN))

        @test Lachesis.compute_mbol(0.0) ≈ Lachesis.MBOL_SUN
        @test Lachesis.compute_mbol(1.0) ≈ Lachesis.MBOL_SUN - 2.5

        @test Lachesis.compute_radius(0.0) ≈ 1.0
        @test Lachesis.compute_radius(2.0) ≈ 100.0

        ρ = Lachesis.compute_density(1.0, 1.0)
        @test ρ ≈ 1.41 rtol=0.02

        masses = collect(range(0.1, 2.0; length=100))
        dm = Lachesis.compute_dm_deep(masses; dim=1)
        @test length(dm) == 100
        @test all(x -> isapprox(x, dm[2]; rtol=0.05), dm[2:end-1])
    end

    @testset "MISTIsoFile — basic (25 col)" begin
        @test isfile(BASIC_ISO)
        iso = MISTIsoFile(BASIC_ISO)

        @test iso.version == "1.2"
        @test iso.feh ≈ 0.00
        @test iso.afe ≈ 0.00
        @test iso.vvcrit ≈ 0.40
        @test iso.num_isochrones == 107
        @test length(iso.columns) == 25
        @test "EEP" ∈ iso.columns
        @test "log_Teff" ∈ iso.columns
        @test "log_g" ∈ iso.columns

        @test length(iso.log_ages) == 107
        @test iso.log_ages[1] ≈ 5.0
        @test iso.log_ages[end] ≈ 10.3

        data = get_isochrone(iso, 5.0)
        @test size(data, 2) == 25
        @test size(data, 1) > 0

        # EEPs should be monotonically increasing
        eep_col = findfirst(==("EEP"), iso.columns)
        eeps = data[:, eep_col]
        @test all(diff(eeps) .> 0)

        # Invalid age should throw
        @test_throws ErrorException get_isochrone(iso, 99.0)
    end

    @testset "MISTIsoFile — full (79 col)" begin
        @test isfile(FULL_ISO)
        iso = MISTIsoFile(FULL_ISO)

        @test length(iso.columns) == 79
        @test "delta_nu" ∈ iso.columns
        @test "nu_max" ∈ iso.columns
        @test length(iso.log_ages) == 107
    end

    @testset "MISTModelGrid" begin
        mg = MISTModelGrid(TEST_DATA)

        @test grid_name(mg) == "MIST"
        @test length(columns(mg)) == 16
        @test length(feh_values(mg)) >= 1
        @test any(abs.(feh_values(mg)) .< 0.01)
        @test length(age_values(mg)) == 107
        @test age_values(mg)[1] ≈ 5.0
        @test age_values(mg)[end] ≈ 10.3

        lo, hi = eep_range(mg)
        @test lo > 0
        @test hi > lo

        # Check expected columns
        for col in ["initial_mass", "star_mass", "log_Teff", "log_g",
                     "log_L", "log_R", "phase", "delta_nu", "nu_max",
                     "Teff", "Mbol", "radius", "density", "dm_deep"]
            @test col ∈ columns(mg)
        end

        # Shape
        data = grid_data(mg)
        @test ndims(data) == 4
        @test size(data, 1) >= 1    # n_feh
        @test size(data, 2) == 107  # n_age
        @test size(data, 4) == 16   # n_cols

        # Derived Teff matches log_Teff
        ci_lt = findfirst(==("log_Teff"), columns(mg))
        ci_t = findfirst(==("Teff"), columns(mg))
        slice = data[1, 51, :, :]
        valid = .!isnan.(slice[:, ci_lt])
        @test all(isapprox.(slice[valid, ci_t], 10.0 .^ slice[valid, ci_lt]; rtol=1e-10))

        # Density positive
        ci_rho = findfirst(==("density"), columns(mg))
        rho = data[1, 51, :, ci_rho]
        valid_rho = .!isnan.(rho)
        @test all(rho[valid_rho] .> 0)

        # Fill fraction ~50%
        total = length(data)
        filled = count(.!isnan.(data))
        frac = filled / total
        @test 0.3 < frac < 0.8
    end

    @testset "MISTModelGrid HDF5 roundtrip" begin
        mg = MISTModelGrid(TEST_DATA)
        h5_path = tempname() * ".h5"
        try
            save_hdf5(mg, h5_path)
            @test isfile(h5_path)

            mg2 = load_hdf5_model_grid(h5_path)
            @test columns(mg) == columns(mg2)
            @test feh_values(mg) ≈ feh_values(mg2)
            @test age_values(mg) ≈ age_values(mg2)

            d1 = grid_data(mg)
            d2 = grid_data(mg2)
            # Compare non-NaN values
            valid = .!isnan.(d1)
            @test all(isapprox.(d1[valid], d2[valid]; rtol=1e-10))
        finally
            rm(h5_path; force=true)
        end
    end

    @testset "GridInterpolator" begin
        mg = MISTModelGrid(TEST_DATA)
        interp = GridInterpolator(mg)

        # Returns all 16 columns
        result = interp(eep=300.0, log_age=9.0, feh=0.0)
        @test length(result) == 16
        @test haskey(result, "Teff")
        @test haskey(result, "density")
        @test haskey(result, "dm_deep")

        # Exact grid point test
        fv = feh_values(mg)
        av = age_values(mg)
        ev = eep_values(mg)
        data = grid_data(mg)
        ci_lt = findfirst(==("log_Teff"), columns(mg))

        # Find a valid interior point
        slice = data[1, 51, :, ci_lt]
        valid_idx = findall(.!isnan.(slice))
        mid = valid_idx[div(length(valid_idx), 2)]
        eep = ev[mid]
        expected = data[1, 51, mid, ci_lt]
        result = interp(eep=eep, log_age=av[51], feh=fv[1])
        @test result["log_Teff"] ≈ expected rtol=1e-6

        # Out of bounds → NaN
        result_oob = interp(eep=9999.0, log_age=9.0, feh=0.0)
        @test isnan(result_oob["log_Teff"])

        # Vectorized
        n = 5
        interior = valid_idx[5:min(5+n-1, end)]
        eeps = Float64.(ev[interior])
        ages = fill(av[51], n)
        fehs = fill(fv[1], n)
        batch = interpolate_batch(interp, eeps, ages, fehs)
        @test length(batch["Teff"]) == n
    end
end
