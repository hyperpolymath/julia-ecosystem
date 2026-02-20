# SPDX-License-Identifier: PMPL-1.0-or-later

@testset "Paradoxes" begin
    @testset "continuum_paradox" begin
        dist = Normal(0, 1)
        result = continuum_paradox(dist, 5)

        # Should have 5 sample points
        @test length(result[:points]) == 5

        # All individual probabilities should be zero
        @test all(result[:individual_probs] .== 0.0)

        # But union has probability 1
        @test result[:union_prob] == 1.0

        # Should have explanation
        @test !isempty(result[:explanation])
    end

    @testset "borel_kolmogorov_paradox" begin
        result = borel_kolmogorov_paradox()

        @test haskey(result, :explanation)
        @test haskey(result, :approach_1)
        @test haskey(result, :approach_2)
        @test haskey(result, :resolution)
    end

    @testset "rational_points_paradox" begin
        result = rational_points_paradox((0.0, 1.0), 100)

        @test result[:prob_rational] == 0.0
        @test result[:count_rationals] == Inf
        @test result[:are_rationals_dense] == true
        @test !isempty(result[:explanation])
    end

    @testset "uncountable_union_paradox" begin
        dist = Uniform(0, 1)
        result = uncountable_union_paradox(dist, 10)

        @test all(result[:individual_probs] .== 0.0)
        @test result[:union_prob] == 1.0
        @test result[:countable_additivity] == true
        @test result[:uncountable_additivity] == false
    end

    @testset "almost_sure_vs_sure" begin
        explanation = almost_sure_vs_sure()
        @test typeof(explanation) == String
        @test !isempty(explanation)
    end
end
