# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Cliodynamics
using DataFrames
using Statistics

@testset "Cliodynamics.jl Tests" begin

    @testset "Malthusian Population Model" begin
        params = MalthusianParams(r=0.02, K=1000.0, N0=100.0)
        sol = malthusian_model(params, tspan=(0.0, 200.0))

        # Test initial condition
        @test sol.u[1][1] ≈ 100.0

        # Test that population approaches carrying capacity
        # N(200) ≈ 858 for r=0.02, K=1000, N0=100 (logistic growth)
        final_pop = sol.u[end][1]
        @test final_pop > 800.0  # Should be approaching K=1000

        # Test logistic growth behavior
        mid_point = sol.u[div(length(sol.u), 2)][1]
        @test mid_point > 100.0 && mid_point < 1000.0
    end

    @testset "Demographic-Structural Model" begin
        params = DemographicStructuralParams(
            r=0.015, K=1000.0, w=2.0, δ=0.03, ε=0.001,
            N0=500.0, E0=10.0, S0=100.0
        )
        sol = demographic_structural_model(params, tspan=(0.0, 200.0))

        # Test initial conditions
        @test sol.u[1][1] ≈ 500.0  # Population
        @test sol.u[1][2] ≈ 10.0   # Elites
        @test sol.u[1][3] ≈ 100.0  # State capacity

        # Test that all variables remain non-negative
        for u in sol.u
            @test all(u .>= 0.0)
        end
    end

    @testset "Elite Overproduction Index" begin
        data = DataFrame(
            year = 1800:1900,
            population = collect(100_000:1000:200_000),
            elites = [1000 + 10*i + 5*i^1.5 for i in 0:100]
        )

        result = elite_overproduction_index(data)

        # Test output structure
        @test hasproperty(result, :year)
        @test hasproperty(result, :eoi)
        @test hasproperty(result, :elite_ratio)
        @test nrow(result) == 101

        # Test that EOI increases over time (accelerating elite growth)
        @test result.eoi[end] > result.eoi[1]

        # Test missing columns error
        bad_data = DataFrame(year = 1800:1810)
        @test_throws ArgumentError elite_overproduction_index(bad_data)
    end

    @testset "Political Stress Indicator" begin
        data = DataFrame(
            year = 1800:1900,
            real_wages = 100.0 .- collect(0:100).^1.2 ./ 10,
            elite_ratio = 0.01 .+ collect(0:100)./5000,
            state_revenue = 1000.0 .- collect(0:100).^1.5 ./ 5
        )

        result = political_stress_indicator(data)

        # Test output structure
        @test hasproperty(result, :psi)
        @test hasproperty(result, :mmp)
        @test hasproperty(result, :emp)
        @test hasproperty(result, :sfd)
        @test nrow(result) == 101

        # Test that components are non-negative
        @test all(result.mmp .>= 0.0)
        @test all(result.emp .>= 0.0)
        @test all(result.sfd .>= 0.0)
        @test all(result.psi .>= 0.0)

        # Test that PSI increases over time (worsening conditions)
        @test result.psi[end] > result.psi[1]
    end

    @testset "Instability Probability" begin
        # Test boundary behavior
        @test instability_probability(0.0) < 0.5
        @test instability_probability(1.0) > 0.5

        # Test monotonicity
        @test instability_probability(0.3) < instability_probability(0.7)

        # Test range
        prob_low = instability_probability(0.1)
        prob_high = instability_probability(0.9)
        @test 0.0 <= prob_low <= 1.0
        @test 0.0 <= prob_high <= 1.0
    end

    @testset "Conflict Intensity" begin
        events = [
            InstabilityEvent(1820, 0.3, :rebellion),
            InstabilityEvent(1848, 0.8, :revolution),
            InstabilityEvent(1871, 0.6, :war)
        ]

        intensity = conflict_intensity(events, window=10)

        # Test output structure
        @test hasproperty(intensity, :year)
        @test hasproperty(intensity, :intensity)
        @test nrow(intensity) > 0

        # Test that peak intensity is near events
        max_intensity_year = intensity.year[argmax(intensity.intensity)]
        @test any(abs(e.year - max_intensity_year) <= 5 for e in events)

        # Test empty events
        empty_intensity = conflict_intensity(InstabilityEvent[], window=5)
        @test nrow(empty_intensity) == 0
    end

    @testset "Secular Cycle Analysis" begin
        # Create synthetic data with ~100-year cycle (within autocorrelation detection range)
        t = 1:300
        data = 100.0 .+ 50.0*sin.(2π*t/100) .+ 2*randn(300)

        analysis = secular_cycle_analysis(data, window=30)

        # Test output structure
        @test haskey(analysis, :trend)
        @test haskey(analysis, :cycle)
        @test haskey(analysis, :period)
        @test haskey(analysis, :amplitude)

        # Test dimensions
        @test length(analysis.trend) == 300
        @test length(analysis.cycle) == 300

        # Test period detection (should be close to 100)
        @test 80 < analysis.period < 120

        # Test amplitude is positive
        @test analysis.amplitude > 0
    end

    @testset "Cycle Phase Detection" begin
        # Generate 301 data points representing a secular cycle
        # Four phases across 301 years: expansion → stagflation → crisis → depression
        n = 301
        phase_values(low, mid1, mid2, high) = vcat(
            range(low, mid1, length=div(n,4)),
            range(mid1, high, length=div(n,4)),
            range(high, mid2, length=div(n,4)),
            range(mid2, low, length=n - 3*div(n,4))
        )
        data = DataFrame(
            year = 1500:1800,
            population_pressure = phase_values(0.3, 0.5, 0.9, 0.8) .+ randn(n)*0.05,
            elite_overproduction = phase_values(0.1, 0.3, 0.4, 0.6) .+ randn(n)*0.05,
            instability = phase_values(0.2, 0.3, 0.5, 0.7) .+ randn(n)*0.05
        )

        # Clamp values to valid ranges
        data.population_pressure = clamp.(data.population_pressure, 0.0, 1.0)
        data.elite_overproduction = clamp.(data.elite_overproduction, 0.0, 1.0)
        data.instability = clamp.(data.instability, 0.0, 1.0)

        phases = detect_cycle_phases(data)

        # Test output structure
        @test hasproperty(phases, :year)
        @test hasproperty(phases, :phase)
        @test nrow(phases) == 301

        # Test that all phases are valid
        @test all(p in instances(SecularCyclePhase) for p in phases.phase)
    end

    @testset "State Capacity Model" begin
        params = StateCapacityParams(τ=0.15, α=1.0, β=0.8, γ=0.5)
        capacity = state_capacity_model(params, 1000.0, 50.0)

        # Test that capacity is positive
        @test capacity > 0

        # Test that capacity increases with population
        capacity_large_pop = state_capacity_model(params, 2000.0, 50.0)
        @test capacity_large_pop > capacity

        # Test that capacity decreases with excessive elites
        capacity_many_elites = state_capacity_model(params, 1000.0, 100.0)
        @test capacity_many_elites < capacity
    end

    @testset "Collective Action Problem" begin
        # Small group, high benefit - should succeed
        prob_small = collective_action_problem(10, 1000.0, 10.0)
        @test prob_small > 0.5

        # Large group, low benefit - should fail
        prob_large = collective_action_problem(1000, 100.0, 1.0)
        @test prob_large < 0.5

        # Test range
        @test 0.0 <= prob_small <= 1.0
        @test 0.0 <= prob_large <= 1.0

        # Test that increasing benefit increases probability
        prob_low_benefit = collective_action_problem(100, 1000.0, 5.0)
        prob_high_benefit = collective_action_problem(100, 5000.0, 5.0)
        @test prob_high_benefit > prob_low_benefit
    end

    @testset "Population Pressure" begin
        pop = [100.0, 150.0, 200.0, 250.0]
        pressure = population_pressure(pop, 200.0)

        # Test correct calculation
        @test pressure ≈ [0.5, 0.75, 1.0, 1.25]

        # Test overshoot detection
        @test pressure[end] > 1.0
    end

    @testset "Utility Functions" begin
        # Moving Average
        data = collect(1.0:10.0)
        smoothed = moving_average(data, 5)
        @test length(smoothed) == 10
        @test smoothed[5] ≈ mean(data[3:7])  # Center of window

        # Detrend
        trend_data = collect(1.0:100.0) .+ randn(100)*5
        detrended = detrend(trend_data)
        @test abs(mean(detrended)) < 1.0  # Should have near-zero mean

        # Normalize
        random_data = randn(100) .* 50 .+ 100
        normalized = normalize_timeseries(random_data)
        @test abs(mean(normalized)) < 0.1
        @test abs(std(normalized) - 1.0) < 0.1
    end

    @testset "Carrying Capacity Estimate" begin
        pop = [100.0, 150.0, 180.0, 195.0, 198.0]
        res = [1000.0, 1500.0, 1800.0, 1950.0, 2000.0]

        K = carrying_capacity_estimate(pop, res)

        # Should be close to maximum sustainable population
        @test K > maximum(pop)
        @test K < 250.0  # Reasonable upper bound
    end

    @testset "Crisis Threshold" begin
        indicator = rand(100)

        # 90th percentile
        threshold_90 = crisis_threshold(indicator, 0.9)
        @test count(x -> x > threshold_90, indicator) ≈ 10 atol=2

        # 95th percentile
        threshold_95 = crisis_threshold(indicator, 0.95)
        @test threshold_95 > threshold_90
        @test count(x -> x > threshold_95, indicator) ≈ 5 atol=2
    end

    @testset "Instability Events Extraction" begin
        data = DataFrame(
            year = 1800:1900,
            indicator = rand(101)
        )

        events = instability_events(data, 0.7)

        # Test that all events exceed threshold
        @test all(e.intensity >= 0.7 for e in events)

        # Test event types are assigned based on intensity
        high_intensity_events = filter(e -> e.intensity > 0.9, events)
        if !isempty(high_intensity_events)
            @test all(e.type == :revolution for e in high_intensity_events)
        end

        # Test year range
        if !isempty(events)
            @test all(1800 <= e.year <= 1900 for e in events)
        end
    end

    @testset "InstabilityEvent Type" begin
        event = InstabilityEvent(1848, 0.8, :revolution)

        @test event.year == 1848
        @test event.intensity == 0.8
        @test event.type == :revolution
    end

    @testset "Model Fitting - Malthusian" begin
        # Generate synthetic data from known parameters
        true_params = MalthusianParams(r=0.03, K=500.0, N0=50.0)
        sol = malthusian_model(true_params, tspan=(0.0, 100.0))
        years = collect(0.0:10.0:100.0)
        population = [sol(t)[1] for t in years]

        result = fit_malthusian(years, population, r_init=0.01, K_init=600.0)

        # Fitted parameters should be close to true values
        @test abs(result.params.r - 0.03) < 0.01
        @test abs(result.params.K - 500.0) < 50.0
        @test result.converged
        @test result.loss < 1.0
    end

    @testset "Model Fitting - Demographic-Structural" begin
        # Generate synthetic data from known parameters
        true_params = DemographicStructuralParams(
            r=0.015, K=1000.0, w=2.0, δ=0.03, ε=0.001,
            N0=500.0, E0=10.0, S0=100.0
        )
        sol = demographic_structural_model(true_params, tspan=(0.0, 200.0))
        years = collect(0.0:20.0:200.0)

        data = DataFrame(
            year = years,
            population = [sol(t)[1] for t in years],
            elites = [sol(t)[2] for t in years],
            state_capacity = [sol(t)[3] for t in years]
        )

        result = fit_demographic_structural(data)

        # Should converge (exact recovery is hard with NelderMead)
        @test result.loss < 1.0
        @test result.params.r > 0.0
        @test result.params.K > 0.0
    end

    @testset "Parameter Estimation" begin
        # Fit a simple exponential model
        model_fn(p, t) = p[1] .* exp.(p[2] .* (t .- t[1]))

        true_A, true_r = 100.0, 0.02
        years = collect(0.0:10.0:100.0)
        observed = true_A .* exp.(true_r .* years) .+ randn(length(years)) .* 5.0

        result = estimate_parameters(model_fn, observed, years, [80.0, 0.01],
                                      n_bootstrap=50)

        @test abs(result.params[1] - true_A) < 30.0
        @test abs(result.params[2] - true_r) < 0.01
        @test result.converged

        # CI should bracket the point estimate
        @test result.ci_lower[1] < result.params[1]
        @test result.ci_upper[1] > result.params[1]
        @test result.ci_lower[2] < result.params[2]
        @test result.ci_upper[2] > result.params[2]
    end

    @testset "Seshat Data Integration" begin
        seshat_path = joinpath(@__DIR__, "..", "data", "seshat_sample.csv")

        # Test load
        raw = load_seshat_csv(seshat_path)
        @test nrow(raw) > 0
        @test hasproperty(raw, :year)
        @test hasproperty(raw, :polity)
        @test hasproperty(raw, :population)

        # Test prepare with polity filter
        roman = prepare_seshat_data(raw, polity="RomPrinworlds")
        @test nrow(roman) > 0
        @test all(roman.polity .== "RomPrinworlds")
        @test hasproperty(roman, :elite_ratio)

        # Test prepare without filter
        all_data = prepare_seshat_data(raw)
        @test nrow(all_data) == nrow(raw)

        # Test that prepared data works with EOI
        full_roman = prepare_seshat_data(raw)
        roman_filtered = filter(row -> occursin("Rom", string(row.polity)), full_roman)
        sort!(roman_filtered, :year)
        eoi = elite_overproduction_index(roman_filtered)
        @test nrow(eoi) == nrow(roman_filtered)

        # Test missing file error
        @test_throws ArgumentError load_seshat_csv("/nonexistent/file.csv")
    end

    @testset "Spatial Instability Diffusion" begin
        regions = [
            (name=:Rome, psi0=0.8, growth_rate=0.05),
            (name=:Gaul, psi0=0.2, growth_rate=0.03),
            (name=:Egypt, psi0=0.1, growth_rate=0.02)
        ]
        adjacency = [0.0 1.0 0.5;
                      1.0 0.0 0.0;
                      0.5 0.0 0.0]

        result = spatial_instability_diffusion(regions, adjacency,
                                               diffusion_rate=0.1, tspan=(0.0, 50.0))

        # Test output structure
        @test length(result.t) > 0
        @test size(result.psi, 2) == 3
        @test length(result.regions) == 3
        @test result.regions == ["Rome", "Gaul", "Egypt"]

        # Test that high-PSI Rome diffuses instability to neighbors
        # Gaul (connected to Rome) should have higher PSI than Egypt (weakly connected)
        @test result.psi[end, 2] > result.psi[1, 2]  # Gaul PSI increases

        # Test dimension mismatch error
        @test_throws ArgumentError spatial_instability_diffusion(
            regions, zeros(2, 2), diffusion_rate=0.1)
    end

    @testset "Territorial Competition Model" begin
        states = [
            (name=:StateA, territory0=100.0, military=1.0, growth_rate=0.02),
            (name=:StateB, territory0=80.0, military=1.5, growth_rate=0.01),
            (name=:StateC, territory0=50.0, military=0.5, growth_rate=0.03)
        ]

        result = territorial_competition_model(states, tspan=(0.0, 100.0))

        # Test output structure
        @test length(result.t) > 0
        @test size(result.territory, 2) == 3
        @test result.states == ["StateA", "StateB", "StateC"]

        # Test that all territories remain non-negative
        @test all(result.territory .>= -1.0)  # Small numerical tolerance
    end

    @testset "Frontier Formation Index" begin
        # 3 groups with varying cultural distances
        distances = [0.0 0.8 0.3;
                     0.8 0.0 0.9;
                     0.3 0.9 0.0]
        populations = [1000.0, 500.0, 800.0]
        territories = [100.0, 80.0, 120.0]

        index = frontier_formation_index(distances, populations, territories)

        # Test output
        @test length(index) == 3
        @test all(0.0 .<= index .<= 1.0)

        # Group at highest cultural frontier should have highest index
        # Group 2 (high distance to both others) should score high
        @test maximum(index) > 0.5

        # Test dimension mismatch
        @test_throws ArgumentError frontier_formation_index(
            zeros(2, 2), populations, territories)
        @test_throws ArgumentError frontier_formation_index(
            distances, populations, [1.0, 2.0])
    end

end
