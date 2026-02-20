# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

using Test
using Axiology

@testset "Axiology.jl" begin
    @testset "Value Types" begin
        @testset "Fairness construction" begin
            f = Fairness(metric = :demographic_parity, threshold = 0.05)
            @test f.metric == :demographic_parity
            @test f.threshold == 0.05
            @test f.weight == 1.0
        end

        @testset "Welfare construction" begin
            w = Welfare(metric = :utilitarian, weight = 0.5)
            @test w.metric == :utilitarian
            @test w.weight == 0.5
        end

        @testset "Profit construction" begin
            p = Profit(target = 1000.0)
            @test p.target == 1000.0
            @test isempty(p.constraints)
        end

        @testset "Efficiency construction" begin
            e = Efficiency(metric = :computation_time, target = 0.1)
            @test e.metric == :computation_time
            @test e.target == 0.1
        end

        @testset "Safety construction" begin
            s = Safety(invariant = "No harmful actions")
            @test s.invariant == "No harmful actions"
            @test s.critical == true
        end
    end

    @testset "Fairness Metrics" begin
        @testset "Demographic Parity" begin
            predictions = [1, 0, 1, 1, 0, 1, 1, 0]
            protected = [:male, :female, :male, :female, :male, :female, :male, :female]

            disparity = demographic_parity(predictions, protected)
            @test disparity >= 0.0
            @test disparity <= 1.0

            # Perfect parity case
            predictions_equal = [1, 1, 0, 0]
            protected_equal = [:a, :b, :a, :b]
            @test demographic_parity(predictions_equal, protected_equal) == 0.0
        end

        @testset "Disparate Impact" begin
            predictions = [1, 1, 1, 0]
            protected = [:a, :a, :a, :b]

            di = disparate_impact(predictions, protected)
            @test di >= 0.0
            @test di <= 1.0
        end

        @testset "Fairness Satisfaction" begin
            fairness = Fairness(metric = :demographic_parity, threshold = 0.1)

            # Fair case: both groups have similar positive rates
            state_fair = Dict(
                :predictions => [1, 1, 0, 0],  # Group a: 1,0 (50%), Group b: 1,0 (50%)
                :protected => [:a, :b, :a, :b]
            )
            @test satisfy(fairness, state_fair)

            # Unfair case: large disparity between groups
            state_unfair = Dict(
                :predictions => [1, 1, 1, 0],  # Group a: 1,1,1 (100%), Group b: 0 (0%)
                :protected => [:a, :a, :a, :b]
            )
            @test !satisfy(fairness, state_unfair)
        end

        @testset "Equalized Odds" begin
            # Perfect equalized odds: same TPR and FPR across groups
            predictions = [1, 0, 1, 0]
            labels = [1, 0, 1, 0]
            protected = [:a, :b, :a, :b]
            @test equalized_odds(predictions, labels, protected) == 0.0

            # Unequal TPR/FPR
            predictions_unequal = [1, 1, 1, 0]
            labels_unequal = [1, 0, 1, 0]
            protected_unequal = [:a, :a, :b, :b]
            disparity = equalized_odds(predictions_unequal, labels_unequal, protected_unequal)
            @test disparity > 0.0
        end

        @testset "Equal Opportunity" begin
            # Perfect equal opportunity: same TPR across groups
            predictions = [1, 0, 1, 0]
            labels = [1, 0, 1, 0]
            protected = [:a, :b, :a, :b]
            @test equal_opportunity(predictions, labels, protected) == 0.0

            # Unequal TPR
            predictions_unequal = [1, 0, 1, 1]
            labels_unequal = [1, 1, 1, 1]
            protected_unequal = [:a, :a, :b, :b]
            disparity = equal_opportunity(predictions_unequal, labels_unequal, protected_unequal)
            @test disparity > 0.0
        end

        @testset "Individual Fairness" begin
            # Similar individuals with similar predictions (low unfairness)
            predictions = [0.5, 0.5, 0.9, 0.9]
            similarity = [
                1.0 0.9 0.1 0.1;
                0.9 1.0 0.1 0.1;
                0.1 0.1 1.0 0.9;
                0.1 0.1 0.9 1.0
            ]
            fairness_score = individual_fairness(predictions, similarity)
            @test fairness_score ≈ 0.0 atol=0.01

            # Similar individuals with different predictions (high unfairness)
            predictions_unfair = [0.1, 0.9, 0.1, 0.9]
            unfairness_score = individual_fairness(predictions_unfair, similarity)
            @test unfairness_score > 0.5
        end
    end

    @testset "Edge Cases" begin
        @testset "Single group fairness" begin
            @test demographic_parity([1, 0, 1], [:a, :a, :a]) == 0.0
            @test disparate_impact([1, 0, 1], [:a, :a, :a]) == 1.0
        end

        @testset "Empty/small welfare inputs" begin
            @test utilitarian_welfare([]) == 0.0
            @test egalitarian_welfare([10.0]) == 0.0
            @test_throws ErrorException rawlsian_welfare([])
        end

        @testset "Normalization edge cases" begin
            @test normalize_scores([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]
            @test_throws ArgumentError normalize_scores([])
        end

        @testset "Weighted score edge cases" begin
            values = [Fairness(metric=:demographic_parity, threshold=0.1, weight=0.0)]
            state = Dict(:predictions => [1, 0], :protected => [:a, :b])
            @test weighted_score(values, state) == 0.0
        end

        @testset "Invalid inputs" begin
            # Invalid metric should error
            @test_throws ErrorException Fairness(metric=:invalid_metric, threshold=0.1)

            # Empty invariant should error
            @test_throws AssertionError Safety(invariant="", critical=true)
        end
    end

    @testset "Welfare Functions" begin
        @testset "Utilitarian Welfare" begin
            utilities = [10.0, 8.0, 12.0, 7.0]
            welfare = utilitarian_welfare(utilities)
            @test welfare == 37.0
        end

        @testset "Rawlsian Welfare" begin
            utilities = [10.0, 8.0, 12.0, 7.0]
            welfare = rawlsian_welfare(utilities)
            @test welfare == 7.0
        end

        @testset "Egalitarian Welfare" begin
            # Perfect equality
            utilities_equal = [10.0, 10.0, 10.0, 10.0]
            welfare = egalitarian_welfare(utilities_equal)
            @test welfare == 0.0

            # High inequality
            utilities_unequal = [5.0, 10.0, 15.0, 20.0]
            welfare_unequal = egalitarian_welfare(utilities_unequal)
            @test welfare_unequal < 0.0
        end

        @testset "Welfare Satisfaction" begin
            welfare = Welfare(metric = :rawlsian)
            state = Dict(
                :utilities => [8.0, 9.0, 10.0],
                :min_welfare => 7.0
            )
            @test satisfy(welfare, state)

            state_low = Dict(
                :utilities => [5.0, 9.0, 10.0],
                :min_welfare => 7.0
            )
            @test !satisfy(welfare, state_low)
        end
    end

    @testset "Optimization" begin
        @testset "Value Scoring" begin
            fairness = Fairness(metric = :demographic_parity, threshold = 0.1)
            state = Dict(
                :predictions => [1, 0, 1, 0],
                :protected => [:a, :b, :a, :b]
            )

            score = value_score(fairness, state)
            @test score >= 0.0
            @test score <= 1.0
        end

        @testset "Weighted Scoring" begin
            values = [
                Fairness(metric = :demographic_parity, threshold = 0.1, weight = 0.5),
                Welfare(metric = :utilitarian, weight = 0.5)
            ]

            state = Dict(
                :predictions => [1, 0, 1, 0],
                :protected => [:a, :b, :a, :b],
                :utilities => [10.0, 8.0],
                :max_welfare => 20.0
            )

            score = weighted_score(values, state)
            @test score >= 0.0
            @test score <= 1.0
        end

        @testset "Domination" begin
            # Solution A: high welfare, slow
            solution_a = Dict(
                :utilities => [10.0, 8.0],
                :computation_time => 0.2,
                :predictions => [1, 0],
                :protected => [:a, :b],
                :max_welfare => 36.0  # 2 * 18 for normalization
            )

            # Solution B: higher welfare, faster (dominates A)
            solution_b = Dict(
                :utilities => [12.0, 10.0],
                :computation_time => 0.1,
                :predictions => [1, 0],
                :protected => [:a, :b],
                :max_welfare => 36.0
            )

            # Solution C: lower welfare, but TOO slow (misses efficiency target)
            solution_c = Dict(
                :utilities => [6.0, 6.0],
                :computation_time => 0.25,  # Exceeds target of 0.15
                :predictions => [1, 0],
                :protected => [:a, :b],
                :max_welfare => 36.0
            )

            values = [
                Welfare(metric = :utilitarian, weight = 1.0),
                Efficiency(metric = :computation_time, target = 0.15, weight = 1.0)
            ]

            # B dominates A (better on both objectives)
            @test dominated(solution_a, solution_b, values)
            @test !dominated(solution_b, solution_a, values)

            # Neither B nor C dominates the other (tradeoff: B has better welfare, C may have different properties)
            @test !dominated(solution_b, solution_c, values)
            # C is dominated by B (B has better welfare, similar efficiency)
            @test dominated(solution_c, solution_b, values)
        end

        @testset "Pareto Frontier" begin
            solutions = [
                Dict(:utilities => [10.0, 8.0], :computation_time => 0.1,
                     :predictions => [1,0], :protected => [:a,:b], :max_welfare => 20.0),
                Dict(:utilities => [8.0, 8.0], :computation_time => 0.05,
                     :predictions => [1,1], :protected => [:a,:b], :max_welfare => 20.0),
                Dict(:utilities => [12.0, 6.0], :computation_time => 0.15,
                     :predictions => [1,0], :protected => [:a,:b], :max_welfare => 20.0)
            ]

            values = [
                Welfare(metric = :utilitarian, weight = 0.5),
                Efficiency(metric = :computation_time, target = 0.1, weight = 0.5)
            ]

            pareto = pareto_frontier(solutions, values)
            @test !isempty(pareto)
            @test length(pareto) <= length(solutions)
        end
    end

    @testset "Maximize Function" begin
        @testset "Maximize Welfare" begin
            welfare = Welfare(metric = :utilitarian)
            state = Dict(:utilities => [5.0, 6.0, 7.0])
            score = maximize(welfare, state)
            @test score == 18.0
        end

        @testset "Maximize Profit" begin
            profit = Profit(target = 1000.0)
            state = Dict(:profit => 1500.0)
            score = maximize(profit, state)
            @test score == 1500.0
        end

        @testset "Maximize Fairness" begin
            fairness = Fairness(metric = :demographic_parity, threshold = 0.1)
            state = Dict(
                :predictions => [1, 0, 1, 0],
                :protected => [:a, :b, :a, :b]
            )
            score = maximize(fairness, state)
            @test score >= 0.0
            @test score <= 1.0
        end
    end

    @testset "Verify Value" begin
        safety = Safety(invariant = "No harmful actions")

        proof_valid = Dict(:verified => true, :prover => :Lean)
        @test verify_value(safety, proof_valid)

        proof_invalid = Dict(:verified => false)
        @test !verify_value(safety, proof_invalid)
    end

    @testset "Integration Examples" begin
        @testset "Multi-objective ML System" begin
            # Define competing values
            values = [
                Welfare(metric = :utilitarian, weight = 0.4),
                Fairness(metric = :demographic_parity, threshold = 0.08, weight = 0.3),
                Efficiency(metric = :computation_time, target = 0.1, weight = 0.3)
            ]

            # System state - ensure fair predictions
            state = Dict(
                :utilities => [10.0, 8.0, 9.0],
                :max_welfare => 30.0,
                :predictions => [1, 1, 0, 0, 1, 1],  # Groups balanced: a:[1,0,1] b:[1,0,1] = 66% each
                :protected => [:a, :b, :a, :b, :a, :b],
                :computation_time => 0.08
            )

            # Check individual values
            @test satisfy(Welfare(metric = :utilitarian),
                         merge(state, Dict(:min_welfare => 25.0)))
            @test satisfy(Fairness(metric = :demographic_parity, threshold = 0.1),
                         state)
            @test satisfy(Efficiency(metric = :computation_time, target = 0.1),
                         state)

            # Compute weighted score
            score = weighted_score(values, state)
            @test score >= 0.0
            @test score <= 1.0
        end
    end
end
