# SPDX-License-Identifier: PMPL-1.0-or-later
# SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell

using Test
using Cliometrics
using DataFrames
using Statistics

@testset "Cliometrics.jl Tests" begin
    @testset "Growth Rate Calculations" begin
        # Test geometric growth rates
        data = DataFrame(year=[2000, 2001, 2002], gdp=[100.0, 105.0, 110.25])
        growth = calculate_growth_rates(data, :gdp, method=:geometric)

        @test length(growth) == 2
        @test growth[1] ≈ log(105/100) atol=1e-6
        @test growth[2] ≈ log(110.25/105) atol=1e-6

        # Test arithmetic growth rates
        growth_arith = calculate_growth_rates(data, :gdp, method=:arithmetic)
        @test growth_arith[1] ≈ 0.05 atol=1e-6
        @test growth_arith[2] ≈ 0.05 atol=1e-6
    end

    @testset "Solow Residual" begin
        # Simple test case with known values
        output = [100.0, 105.0, 110.0]
        capital = [300.0, 310.0, 320.0]
        labor = [50.0, 51.0, 52.0]

        tfp = solow_residual(output, capital, labor, alpha=0.3)

        @test length(tfp) == 2
        @test all(isfinite.(tfp))
    end

    @testset "Growth Decomposition" begin
        data = DataFrame(
            year = 2000:2003,
            gdp = [100.0, 105.0, 110.25, 115.76],
            capital = [300.0, 310.0, 320.0, 330.0],
            labor = [50.0, 51.0, 52.0, 53.0]
        )

        decomp = decompose_growth(data, alpha=0.3)

        @test nrow(decomp) == 3
        @test "output_growth" in names(decomp)
        @test "capital_contribution" in names(decomp)
        @test "labor_contribution" in names(decomp)
        @test "tfp_contribution" in names(decomp)

        # Sum of components should equal total growth (approximately)
        for i in 1:nrow(decomp)
            total = decomp.capital_contribution[i] +
                   decomp.labor_contribution[i] +
                   decomp.tfp_contribution[i]
            @test total ≈ decomp.output_growth[i] atol=1e-10
        end
    end

    @testset "Convergence Analysis" begin
        # Create mock cross-country data
        # Poor countries should grow faster (convergence)
        data = DataFrame(
            country = ["A", "B", "C", "D"],
            gdp_1950 = [1000.0, 2000.0, 4000.0, 8000.0],
            growth_rate = [0.05, 0.04, 0.03, 0.02]
        )

        result = convergence_analysis(data, :gdp_1950, :growth_rate)

        @test result.beta < 0  # Should indicate convergence
        @test 0 <= result.r_squared <= 1
        @test result.converging == true
        @test isfinite(result.half_life)
    end

    @testset "Institutional Quality Index" begin
        data = DataFrame(
            country = ["A", "B", "C"],
            rule_of_law = [0.8, 0.5, 0.3],
            property_rights = [0.9, 0.6, 0.4],
            corruption_control = [0.7, 0.5, 0.2]
        )

        indicators = [:rule_of_law, :property_rights, :corruption_control]
        index = institutional_quality_index(data, indicators)

        @test length(index) == 3
        @test all(0 .<= index .<= 1)  # Normalized to [0, 1]
        @test index[1] > index[2] > index[3]  # Should be decreasing
    end

    @testset "Quantify Institutions" begin
        data = DataFrame(
            year = repeat(2000:2004, 2),
            country = repeat(["A", "B"], inner=5),
            rule_of_law = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.78, 0.76, 0.74, 0.72],
            corruption = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.58, 0.55, 0.52, 0.50]
        )

        result = quantify_institutions(data, :country, [:rule_of_law, :corruption])

        @test nrow(result) == 2
        @test "country" in names(result)
        @test "avg_change_rate" in names(result)
        @test "volatility" in names(result)
        @test "direction" in names(result)

        # Country A improving, B deteriorating
        row_a = result[result.country .== "A", :]
        row_b = result[result.country .== "B", :]
        @test row_a.avg_change_rate[1] > 0
        @test row_b.avg_change_rate[1] < 0
        @test row_a.direction[1] == "improving"
        @test row_b.direction[1] == "deteriorating"
    end

    @testset "Historical Series Cleaning" begin
        # Test with missing value in middle
        data = [100.0, 105.0, missing, 115.0, 120.0]
        cleaned = clean_historical_series(data, method=:linear)

        @test length(cleaned) == 5
        @test all(isfinite.(cleaned))
        @test cleaned[3] ≈ 110.0 atol=1e-6  # Linear interpolation

        # Test forward fill
        cleaned_ff = clean_historical_series(data, method=:forward_fill)
        @test cleaned_ff[3] ≈ 105.0
    end

    @testset "Interpolate Missing Years" begin
        # Test basic interpolation
        data = DataFrame(year=[2000, 2002, 2005], gdp=[100.0, 110.0, 130.0])
        result = interpolate_missing_years(data, :gdp)

        @test nrow(result) == 6  # 2000-2005 = 6 years
        @test result.year == 2000:2005
        @test result.gdp[1] ≈ 100.0  # Original value
        @test result.gdp[2] ≈ 105.0 atol=1e-6  # Interpolated
        @test result.gdp[3] ≈ 110.0  # Original value
        @test result.gdp[4] ≈ (110.0 + (130.0-110.0)*1/3) atol=1e-6  # Interpolated

        # Test with single gap
        data2 = DataFrame(year=[2000, 2001, 2003], value=[10.0, 12.0, 16.0])
        result2 = interpolate_missing_years(data2, :value)
        @test nrow(result2) == 4
        @test result2.value[3] ≈ 14.0 atol=1e-6  # Linear interpolation of gap

        # Test edge cases
        data_single = DataFrame(year=[2000], value=[100.0])
        result_single = interpolate_missing_years(data_single, :value)
        @test nrow(result_single) == 1
        @test result_single.value[1] == 100.0
    end

    @testset "Load Historical Data" begin
        # Create temporary test file
        test_file = tempname() * ".csv"
        test_data = DataFrame(year=[1950, 1960, 1970], gdp=[100, 150, 200])
        using CSV
        CSV.write(test_file, test_data)

        # Test basic loading
        loaded = load_historical_data(test_file)
        @test nrow(loaded) == 3
        @test "year" in names(loaded)

        # Test with year filtering
        filtered = load_historical_data(test_file, start_year=1960, end_year=1970)
        @test nrow(filtered) == 2
        @test minimum(filtered.year) == 1960
        @test maximum(filtered.year) == 1970

        # Clean up
        rm(test_file)
    end

    @testset "Compare Historical Trajectories" begin
        data = DataFrame(
            year = repeat(1950:1960, 2),
            region = repeat(["Europe", "Asia"], inner=11),
            gdp_per_capita = [
                # Europe (faster growth)
                1000, 1100, 1210, 1331, 1464, 1610, 1771, 1948, 2143, 2357, 2593,
                # Asia (slower growth)
                500, 525, 551, 579, 608, 638, 670, 703, 738, 775, 814
            ]
        )

        comparison = compare_historical_trajectories(data, ["Europe", "Asia"])

        @test nrow(comparison) == 2
        @test "region" in names(comparison)
        @test "avg_growth" in names(comparison)
        @test comparison[comparison.region .== "Europe", :avg_growth][1] >
              comparison[comparison.region .== "Asia", :avg_growth][1]
    end

    @testset "Counterfactual Scenario" begin
        # Test multiplicative adjustment
        data = DataFrame(year=2000:2004, gdp=[100.0, 110.0, 121.0, 133.1, 146.41])
        result = counterfactual_scenario(data, :gdp, 2002, adjustment=0.9, method=:multiplicative)

        @test "actual" in names(result)
        @test "counterfactual" in names(result)
        @test nrow(result) == 5
        @test result.actual[1] ≈ 100.0
        @test result.counterfactual[1] ≈ 100.0  # Before break, unchanged
        @test result.counterfactual[2] ≈ 110.0  # Year 2001, before break
        @test result.counterfactual[3] ≈ 121.0 * 0.9 atol=1e-6  # Break year adjustment
        # After break year, growth rates applied to counterfactual base
        @test result.counterfactual[4] / result.counterfactual[3] ≈
              result.actual[4] / result.actual[3] atol=1e-6

        # Test additive adjustment
        result_add = counterfactual_scenario(data, :gdp, 2002, adjustment=-10.0, method=:additive)
        @test result_add.counterfactual[3] ≈ 121.0 - 10.0 atol=1e-6

        # Test edge case: break year is first year
        result_first = counterfactual_scenario(data, :gdp, 2000, adjustment=1.1, method=:multiplicative)
        @test result_first.counterfactual[1] ≈ 100.0 * 1.1 atol=1e-6
    end

    @testset "Estimate Treatment Effect" begin
        # Classic DiD setup
        data = DataFrame(
            year = repeat(1990:1999, 2),
            country = repeat(["treated", "control"], inner=10),
            treated = repeat([true, false], inner=10),
            gdp = vcat(
                [100, 102, 104, 106, 108, 115, 120, 125, 130, 135],  # treated: jump at 1995
                [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]   # control: steady
            ) .* 1.0
        )

        result = estimate_treatment_effect(data, :gdp, :treated, 1995)

        @test haskey(result, :treatment_effect)
        @test haskey(result, :pre_treatment_diff)
        @test haskey(result, :post_treatment_diff)
        @test haskey(result, :treated_change)
        @test haskey(result, :control_change)
        @test haskey(result, :se)
        @test haskey(result, :t_statistic)

        # Treatment effect should be positive (treated grew faster post-treatment)
        @test result.treatment_effect > 0
        # Pre-treatment, both groups should be similar
        @test abs(result.pre_treatment_diff) < 1.0
        # Treated group should have larger post-treatment change
        @test result.treated_change > result.control_change

        # Test with no treatment effect (parallel trends)
        data_parallel = DataFrame(
            year = repeat(1990:1999, 2),
            treated = repeat([true, false], inner=10),
            gdp = vcat(
                [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],  # Same growth
                [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]   # Same growth
            ) .* 1.0
        )
        result_parallel = estimate_treatment_effect(data_parallel, :gdp, :treated, 1995)
        @test abs(result_parallel.treatment_effect) < 0.1  # Should be near zero
    end

    @testset "Quantify Institutions with period filter" begin
        data = DataFrame(
            year = repeat(2000:2009, 2),
            country = repeat(["A", "B"], inner=10),
            rule_of_law = vcat(
                collect(0.5:0.05:0.95),  # A improving
                collect(0.8:-0.03:0.53)   # B deteriorating
            )
        )

        # Filter to first 5 years only
        result = quantify_institutions(data, :country, [:rule_of_law]; period=(2000, 2004))
        @test nrow(result) == 2

        # Full range
        result_full = quantify_institutions(data, :country, [:rule_of_law])
        @test nrow(result_full) == 2

        # Country A should be improving in both cases
        row_a = result[result.country .== "A", :]
        @test row_a.direction[1] == "improving"
    end

    @testset "Institutional Quality Index with custom weights" begin
        data = DataFrame(
            country = ["A", "B", "C"],
            ind1 = [1.0, 0.5, 0.0],
            ind2 = [0.0, 0.5, 1.0]
        )

        # Equal weights (default)
        idx_equal = institutional_quality_index(data, [:ind1, :ind2])
        @test length(idx_equal) == 3
        @test idx_equal[2] ≈ 0.5 atol=1e-6  # B is at midpoint for both

        # Custom weights: heavily favour ind1
        idx_weighted = institutional_quality_index(data, [:ind1, :ind2]; weights=[0.9, 0.1])
        @test idx_weighted[1] > idx_weighted[3]  # A strong in ind1 should score higher
    end

    @testset "Clean Historical Series with NaN values" begin
        # NaN in middle
        data_nan = [100.0, NaN, NaN, 130.0, 140.0]
        cleaned = clean_historical_series(data_nan, method=:linear)
        @test all(isfinite.(cleaned))
        @test cleaned[2] ≈ 110.0 atol=1e-6
        @test cleaned[3] ≈ 120.0 atol=1e-6

        # Forward fill with NaN
        cleaned_ff = clean_historical_series(data_nan, method=:forward_fill)
        @test cleaned_ff[2] ≈ 100.0  # forward fill from previous valid
        @test cleaned_ff[3] ≈ 100.0
    end

    @testset "Clean Historical Series error on invalid method" begin
        @test_throws ErrorException clean_historical_series([1.0, 2.0], method=:invalid)
    end

    @testset "Calculate Growth Rates error on invalid method" begin
        data = DataFrame(year=[2000, 2001], gdp=[100.0, 110.0])
        @test_throws ErrorException calculate_growth_rates(data, :gdp, method=:invalid)
    end

    @testset "Interpolate Missing Years preserves other columns" begin
        df = DataFrame(
            year = [2000, 2003, 2005],
            gdp = [100.0, 130.0, 150.0],
            region = ["A", "A", "A"]
        )
        result = interpolate_missing_years(df, :gdp)

        @test nrow(result) == 6  # 2000-2005
        @test "region" in names(result)
        @test result.region[1] == "A"
        # Interpolated years should have missing region
        @test ismissing(result.region[2])  # year 2001 not in original
    end

    @testset "Interpolate Missing Years error cases" begin
        # Missing year column
        df_no_year = DataFrame(value=[1.0, 2.0])
        @test_throws ErrorException interpolate_missing_years(df_no_year, :value)

        # Missing variable column
        df_no_var = DataFrame(year=[2000, 2001])
        @test_throws ErrorException interpolate_missing_years(df_no_var, :nonexistent)
    end

    @testset "Convergence Analysis: non-converging case" begin
        # Countries with parallel growth (no convergence)
        data = DataFrame(
            country = ["A", "B", "C", "D"],
            gdp_1950 = [1000.0, 2000.0, 4000.0, 8000.0],
            growth_rate = [0.03, 0.03, 0.03, 0.03]  # same growth for all
        )

        result = convergence_analysis(data, :gdp_1950, :growth_rate)
        @test result.beta ≈ 0.0 atol=1e-6  # No relationship
        @test result.converging == false
        @test result.half_life == Inf
    end

    @testset "Compare Historical Trajectories: detailed statistics" begin
        data = DataFrame(
            year = repeat(1950:1954, 2),
            region = repeat(["Fast", "Slow"], inner=5),
            gdp_per_capita = vcat(
                [100.0, 110.0, 121.0, 133.1, 146.41],  # 10% growth
                [100.0, 103.0, 106.09, 109.27, 112.55]  # 3% growth
            )
        )

        comparison = compare_historical_trajectories(data, ["Fast", "Slow"])
        @test nrow(comparison) == 2

        fast = comparison[comparison.region .== "Fast", :]
        slow = comparison[comparison.region .== "Slow", :]

        # Check cumulative growth
        @test fast.cumulative_growth[1] > slow.cumulative_growth[1]
        @test fast.cumulative_growth[1] ≈ (146.41 / 100.0 - 1) atol=1e-6

        # std_growth should be finite
        @test isfinite(fast.std_growth[1])
        @test isfinite(slow.std_growth[1])

        # initial and final levels
        @test fast.initial_level[1] ≈ 100.0
        @test fast.final_level[1] ≈ 146.41
    end

    @testset "Counterfactual Scenario error cases" begin
        data = DataFrame(year=2000:2004, gdp=[100.0, 110.0, 121.0, 133.1, 146.41])

        # Invalid break year
        @test_throws ErrorException counterfactual_scenario(data, :gdp, 1999)

        # Invalid method
        @test_throws ErrorException counterfactual_scenario(data, :gdp, 2002, method=:invalid)

        # Missing variable column
        @test_throws ErrorException counterfactual_scenario(data, :nonexistent, 2002)
    end

    @testset "Load Historical Data with capital Year column" begin
        test_file = tempname() * ".csv"
        test_data = DataFrame(Year=[1950, 1960, 1970], gdp=[100, 150, 200])
        CSV.write(test_file, test_data)

        loaded = load_historical_data(test_file)
        @test "year" in names(loaded)  # Should be renamed to lowercase
        @test nrow(loaded) == 3

        rm(test_file)
    end

    @testset "Estimate Treatment Effect error cases" begin
        data = DataFrame(
            year = repeat(1990:1994, 2),
            treated = repeat([true, false], inner=5),
            gdp = collect(1.0:10.0)
        )

        # Missing year column
        data_no_year = select(data, Not(:year))
        @test_throws ErrorException estimate_treatment_effect(data_no_year, :gdp, :treated, 1992)

        # Missing variable column
        @test_throws ErrorException estimate_treatment_effect(data, :nonexistent, :treated, 1992)

        # Missing group column
        @test_throws ErrorException estimate_treatment_effect(data, :gdp, :nonexistent, 1992)
    end

    @testset "Solow Residual: growth accounting identity" begin
        # With alpha=0.3, TFP = g_Y - 0.3*g_K - 0.7*g_L
        output = [100.0, 110.0, 121.0]
        capital = [200.0, 210.0, 220.5]
        labor = [50.0, 52.0, 54.08]

        tfp = solow_residual(output, capital, labor, alpha=0.3)

        # Verify the identity: g_Y = alpha*g_K + (1-alpha)*g_L + TFP
        g_Y = [log(output[i] / output[i-1]) for i in 2:length(output)]
        g_K = [log(capital[i] / capital[i-1]) for i in 2:length(capital)]
        g_L = [log(labor[i] / labor[i-1]) for i in 2:length(labor)]

        for i in 1:length(tfp)
            reconstructed = 0.3 * g_K[i] + 0.7 * g_L[i] + tfp[i]
            @test reconstructed ≈ g_Y[i] atol=1e-10
        end
    end

    @testset "Counterfactual Scenario: additive method preserves changes" begin
        data = DataFrame(year=2000:2004, gdp=[100.0, 110.0, 121.0, 133.1, 146.41])
        result = counterfactual_scenario(data, :gdp, 2002, adjustment=-20.0, method=:additive)

        # Before break year, counterfactual should equal actual
        @test result.counterfactual[1] ≈ result.actual[1]
        @test result.counterfactual[2] ≈ result.actual[2]

        # At break year, shift by adjustment
        @test result.counterfactual[3] ≈ 121.0 - 20.0 atol=1e-6

        # After break year, absolute changes from actual are applied to cf base
        actual_change_3_to_4 = result.actual[4] - result.actual[3]
        @test result.counterfactual[4] ≈ result.counterfactual[3] + actual_change_3_to_4 atol=1e-6
    end
end
