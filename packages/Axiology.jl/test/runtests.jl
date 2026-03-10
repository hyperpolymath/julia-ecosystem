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

    # ====================================================================
    # Additional coverage for previously untested functions
    # ====================================================================

    @testset "Construction Validation" begin
        @testset "Fairness invalid threshold" begin
            @test_throws AssertionError Fairness(metric=:demographic_parity, threshold=-0.1)
            @test_throws AssertionError Fairness(metric=:demographic_parity, threshold=1.5)
        end

        @testset "Fairness negative weight" begin
            @test_throws AssertionError Fairness(metric=:demographic_parity, weight=-1.0)
        end

        @testset "Welfare invalid metric" begin
            @test_throws AssertionError Welfare(metric=:invalid_metric)
        end

        @testset "Welfare negative weight" begin
            @test_throws AssertionError Welfare(metric=:utilitarian, weight=-0.5)
        end

        @testset "Profit negative weight" begin
            @test_throws AssertionError Profit(weight=-1.0)
        end

        @testset "Efficiency invalid metric" begin
            @test_throws AssertionError Efficiency(metric=:invalid_metric)
        end

        @testset "Efficiency negative weight" begin
            @test_throws AssertionError Efficiency(metric=:pareto, weight=-0.1)
        end

        @testset "Safety negative weight" begin
            @test_throws AssertionError Safety(invariant="test", weight=-1.0)
        end

        @testset "Fairness default values" begin
            f = Fairness()
            @test f.metric == :demographic_parity
            @test f.threshold == 0.05
            @test f.weight == 1.0
            @test isempty(f.protected_attributes)
        end

        @testset "Welfare default values" begin
            w = Welfare()
            @test w.metric == :utilitarian
            @test w.weight == 1.0
        end

        @testset "Profit default values" begin
            p = Profit()
            @test p.target == 0.0
            @test isempty(p.constraints)
            @test p.weight == 1.0
        end

        @testset "Efficiency default values" begin
            e = Efficiency()
            @test e.metric == :pareto
            @test e.target == 1.0
            @test e.weight == 1.0
        end

        @testset "Safety default critical" begin
            s = Safety(invariant="test")
            @test s.critical == true
            @test s.weight == 1.0
        end

        @testset "Safety non-critical" begin
            s = Safety(invariant="soft constraint", critical=false)
            @test s.critical == false
        end

        @testset "Profit with constraints" begin
            fairness_constraint = Fairness(metric=:demographic_parity, threshold=0.1)
            p = Profit(target=500.0, constraints=Value[fairness_constraint], weight=0.8)
            @test length(p.constraints) == 1
            @test p.constraints[1] isa Fairness
        end

        @testset "All Fairness metric variants" begin
            for m in [:demographic_parity, :equalized_odds, :equal_opportunity,
                      :disparate_impact, :individual_fairness]
                f = Fairness(metric=m)
                @test f.metric == m
            end
        end

        @testset "All Welfare metric variants" begin
            for m in [:utilitarian, :rawlsian, :egalitarian]
                w = Welfare(metric=m)
                @test w.metric == m
            end
        end

        @testset "All Efficiency metric variants" begin
            for m in [:pareto, :kaldor_hicks, :computation_time]
                e = Efficiency(metric=m)
                @test e.metric == m
            end
        end
    end

    @testset "Display Methods (show)" begin
        @testset "Fairness show" begin
            f = Fairness(metric=:demographic_parity, threshold=0.05)
            buf = IOBuffer()
            show(buf, f)
            str = String(take!(buf))
            @test occursin("Fairness", str)
            @test occursin("demographic_parity", str)
            @test occursin("0.05", str)
        end

        @testset "Welfare show" begin
            w = Welfare(metric=:rawlsian)
            buf = IOBuffer()
            show(buf, w)
            str = String(take!(buf))
            @test occursin("Welfare", str)
            @test occursin("rawlsian", str)
        end

        @testset "Profit show" begin
            p = Profit(target=1000.0)
            buf = IOBuffer()
            show(buf, p)
            str = String(take!(buf))
            @test occursin("Profit", str)
            @test occursin("1000.0", str)
        end

        @testset "Efficiency show" begin
            e = Efficiency(metric=:computation_time, target=0.5)
            buf = IOBuffer()
            show(buf, e)
            str = String(take!(buf))
            @test occursin("Efficiency", str)
            @test occursin("computation_time", str)
            @test occursin("0.5", str)
        end

        @testset "Safety show" begin
            s = Safety(invariant="No harmful output")
            buf = IOBuffer()
            show(buf, s)
            str = String(take!(buf))
            @test occursin("Safety", str)
            @test occursin("No harmful output", str)
        end
    end

    @testset "Satisfy - Efficiency Metrics" begin
        @testset "Efficiency pareto satisfaction" begin
            e = Efficiency(metric=:pareto)
            state_efficient = Dict(:is_pareto_efficient => true)
            state_not_efficient = Dict(:is_pareto_efficient => false)

            @test satisfy(e, state_efficient)
            @test !satisfy(e, state_not_efficient)
        end

        @testset "Efficiency kaldor_hicks satisfaction" begin
            e = Efficiency(metric=:kaldor_hicks, target=100.0)
            state_ok = Dict(:net_gain => 150.0)
            state_low = Dict(:net_gain => 50.0)

            @test satisfy(e, state_ok)
            @test !satisfy(e, state_low)
        end

        @testset "Efficiency computation_time satisfaction" begin
            e = Efficiency(metric=:computation_time, target=0.5)
            state_fast = Dict(:computation_time => 0.3)
            state_slow = Dict(:computation_time => 0.8)

            @test satisfy(e, state_fast)
            @test !satisfy(e, state_slow)
        end

        @testset "Efficiency missing state errors" begin
            e_pareto = Efficiency(metric=:pareto)
            e_kh = Efficiency(metric=:kaldor_hicks)
            e_ct = Efficiency(metric=:computation_time)

            @test_throws ErrorException satisfy(e_pareto, Dict())
            @test_throws ErrorException satisfy(e_kh, Dict())
            @test_throws ErrorException satisfy(e_ct, Dict())
        end
    end

    @testset "Satisfy - Safety" begin
        @testset "Safety defaults to safe" begin
            s = Safety(invariant="test")
            # With no state keys, defaults to is_safe=true, invariant_holds=true
            @test satisfy(s, Dict())
        end

        @testset "Safety unsafe state" begin
            s = Safety(invariant="test")
            @test !satisfy(s, Dict(:is_safe => false))
            @test !satisfy(s, Dict(:invariant_holds => false))
            @test !satisfy(s, Dict(:is_safe => false, :invariant_holds => false))
        end

        @testset "Safety safe state" begin
            s = Safety(invariant="test")
            @test satisfy(s, Dict(:is_safe => true, :invariant_holds => true))
        end
    end

    @testset "Satisfy - Profit with Constraints" begin
        @testset "Profit meets target with satisfied constraints" begin
            fairness = Fairness(metric=:demographic_parity, threshold=0.5)
            p = Profit(target=100.0, constraints=Value[fairness])

            state = Dict(
                :profit => 150.0,
                :predictions => [1, 1, 0, 0],
                :protected => [:a, :b, :a, :b]
            )
            @test satisfy(p, state)
        end

        @testset "Profit below target" begin
            p = Profit(target=100.0)
            state = Dict(:profit => 50.0)
            @test !satisfy(p, state)
        end

        @testset "Profit missing state" begin
            p = Profit(target=100.0)
            @test_throws ErrorException satisfy(p, Dict())
        end

        @testset "Profit with failing constraint" begin
            # Fairness constraint that will fail (large disparity)
            fairness = Fairness(metric=:demographic_parity, threshold=0.01)
            p = Profit(target=100.0, constraints=Value[fairness])

            state = Dict(
                :profit => 200.0,
                :predictions => [1, 1, 1, 0],  # Group a: 100%, Group b: 0%
                :protected => [:a, :a, :a, :b]
            )
            @test !satisfy(p, state)
        end
    end

    @testset "Satisfy - Welfare Missing State" begin
        w = Welfare(metric=:utilitarian)
        @test_throws ErrorException satisfy(w, Dict())
    end

    @testset "Maximize - Efficiency" begin
        @testset "Maximize computation_time" begin
            e = Efficiency(metric=:computation_time, target=1.0)
            state = Dict(:computation_time => 0.5)
            score = maximize(e, state)
            @test score == -0.5  # Negated for minimization
        end

        @testset "Maximize pareto" begin
            e = Efficiency(metric=:pareto)
            @test maximize(e, Dict(:is_pareto_efficient => true)) == 1.0
            @test maximize(e, Dict(:is_pareto_efficient => false)) == 0.0
        end

        @testset "Maximize kaldor_hicks" begin
            e = Efficiency(metric=:kaldor_hicks)
            state = Dict(:net_gain => 42.0)
            @test maximize(e, state) == 42.0
        end

        @testset "Maximize efficiency missing state" begin
            e_ct = Efficiency(metric=:computation_time)
            e_p = Efficiency(metric=:pareto)
            e_kh = Efficiency(metric=:kaldor_hicks)
            @test_throws ErrorException maximize(e_ct, Dict())
            @test_throws ErrorException maximize(e_p, Dict())
            @test_throws ErrorException maximize(e_kh, Dict())
        end
    end

    @testset "Maximize - Safety" begin
        s = Safety(invariant="test")
        @test maximize(s, Dict(:is_safe => true, :invariant_holds => true)) == 1.0
        @test maximize(s, Dict(:is_safe => false)) == 0.0
        @test maximize(s, Dict(:invariant_holds => false)) == 0.0
    end

    @testset "Maximize - Welfare Variants" begin
        @testset "Maximize rawlsian welfare" begin
            w = Welfare(metric=:rawlsian)
            state = Dict(:utilities => [5.0, 8.0, 12.0])
            @test maximize(w, state) == 5.0
        end

        @testset "Maximize egalitarian welfare" begin
            w = Welfare(metric=:egalitarian)
            state_equal = Dict(:utilities => [10.0, 10.0, 10.0])
            @test maximize(w, state_equal) == 0.0

            state_unequal = Dict(:utilities => [5.0, 15.0])
            @test maximize(w, state_unequal) < 0.0
        end

        @testset "Maximize welfare missing state" begin
            w = Welfare(metric=:utilitarian)
            @test_throws ErrorException maximize(w, Dict())
        end
    end

    @testset "Maximize - Fairness Variants" begin
        @testset "Maximize equalized_odds" begin
            f = Fairness(metric=:equalized_odds, threshold=0.1)
            state = Dict(
                :predictions => [1, 0, 1, 0],
                :labels => [1, 0, 1, 0],
                :protected => [:a, :b, :a, :b]
            )
            score = maximize(f, state)
            @test score >= 0.0
            @test score <= 1.0
        end

        @testset "Maximize disparate_impact" begin
            f = Fairness(metric=:disparate_impact, threshold=0.8)
            state = Dict(
                :predictions => [1, 1, 0, 0],
                :protected => [:a, :b, :a, :b]
            )
            score = maximize(f, state)
            @test score >= 0.0
            @test score <= 1.0
        end

        @testset "Maximize individual_fairness" begin
            f = Fairness(metric=:individual_fairness, threshold=0.1)
            sim = [1.0 0.9; 0.9 1.0]
            state = Dict(
                :predictions => [0.5, 0.5],
                :similarity_matrix => sim
            )
            score = maximize(f, state)
            @test score >= 0.0
            @test score <= 1.0
        end

        @testset "Maximize fairness with missing data returns 0" begin
            f = Fairness(metric=:demographic_parity)
            @test maximize(f, Dict()) == 0.0
        end
    end

    @testset "Value Score - Additional Types" begin
        @testset "value_score for Safety" begin
            s = Safety(invariant="test")
            @test value_score(s, Dict(:is_safe => true, :invariant_holds => true)) == 1.0
            @test value_score(s, Dict(:is_safe => false)) == 0.0
        end

        @testset "value_score for Profit" begin
            p = Profit(target=100.0)
            state = Dict(:profit => 150.0)
            @test value_score(p, state) == 1.5  # 150/100
        end

        @testset "value_score for Efficiency pareto" begin
            e = Efficiency(metric=:pareto)
            @test value_score(e, Dict(:is_pareto_efficient => true)) == 1.0
            @test value_score(e, Dict(:is_pareto_efficient => false)) == 0.0
        end

        @testset "value_score for Efficiency kaldor_hicks" begin
            e = Efficiency(metric=:kaldor_hicks, target=100.0)
            state = Dict(:net_gain => 80.0)
            @test value_score(e, state) == 0.8  # 80/100
        end

        @testset "value_score for Efficiency computation_time" begin
            e = Efficiency(metric=:computation_time, target=1.0)
            state = Dict(:computation_time => 0.3)
            @test value_score(e, state) ≈ 0.7  # 1 - 0.3/1.0
        end

        @testset "value_score for disparate_impact" begin
            f = Fairness(metric=:disparate_impact, threshold=0.8)
            state = Dict(
                :predictions => [1, 1, 0, 0],
                :protected => [:a, :b, :a, :b]
            )
            score = value_score(f, state)
            @test score >= 0.0
            @test score <= 1.0
        end

        @testset "value_score for individual_fairness" begin
            f = Fairness(metric=:individual_fairness, threshold=0.1)
            sim = [1.0 0.9; 0.9 1.0]
            state = Dict(
                :predictions => [0.5, 0.5],
                :similarity_matrix => sim
            )
            score = value_score(f, state)
            @test score >= 0.0
            @test score <= 1.0
        end
    end

    @testset "Verify Value - Generic" begin
        @testset "verify_value for non-Safety value" begin
            f = Fairness(metric=:demographic_parity)
            @test verify_value(f, Dict(:verified => true))
            @test !verify_value(f, Dict(:verified => false))
        end

        @testset "verify_value missing :verified field" begin
            f = Fairness(metric=:demographic_parity)
            @test_throws ErrorException verify_value(f, Dict())
        end

        @testset "verify_value wrong type for :verified" begin
            f = Fairness(metric=:demographic_parity)
            @test_throws ErrorException verify_value(f, Dict(:verified => "yes"))
        end

        @testset "verify_value for Welfare" begin
            w = Welfare(metric=:utilitarian)
            @test verify_value(w, Dict(:verified => true))
        end

        @testset "verify_value for Efficiency" begin
            e = Efficiency(metric=:pareto)
            @test verify_value(e, Dict(:verified => true))
        end

        @testset "verify_value for Profit" begin
            p = Profit(target=100.0)
            @test verify_value(p, Dict(:verified => true))
        end
    end

    @testset "Verify Value - Safety (Critical)" begin
        @testset "Critical safety requires prover" begin
            s = Safety(invariant="test", critical=true)
            # Missing prover should error
            @test_throws ErrorException verify_value(s, Dict(:verified => true))
        end

        @testset "Critical safety with prover and details" begin
            s = Safety(invariant="test", critical=true)
            proof = Dict(:verified => true, :prover => :Lean, :details => "Formally verified")
            @test verify_value(s, proof)
        end

        @testset "Non-critical safety does not require prover" begin
            s = Safety(invariant="test", critical=false)
            @test verify_value(s, Dict(:verified => true))
        end
    end

    @testset "Normalize Scores" begin
        @testset "Basic normalization" begin
            scores = [10.0, 20.0, 30.0]
            normalized = normalize_scores(scores)
            @test normalized[1] ≈ 0.0
            @test normalized[2] ≈ 0.5
            @test normalized[3] ≈ 1.0
        end

        @testset "Single element" begin
            @test normalize_scores([5.0]) == [1.0]
        end

        @testset "Two elements" begin
            normalized = normalize_scores([0.0, 10.0])
            @test normalized[1] ≈ 0.0
            @test normalized[2] ≈ 1.0
        end

        @testset "Integer scores" begin
            normalized = normalize_scores([1, 2, 3, 4, 5])
            @test normalized[1] ≈ 0.0
            @test normalized[end] ≈ 1.0
        end
    end

    @testset "Pareto Frontier - Dict System Input" begin
        @testset "System with :solutions key" begin
            values = [Welfare(metric=:utilitarian, weight=1.0)]
            system = Dict(
                :solutions => [
                    Dict(:utilities => [10.0], :predictions => [1], :protected => [:a]),
                    Dict(:utilities => [5.0], :predictions => [1], :protected => [:a])
                ]
            )
            frontier = pareto_frontier(system, values)
            @test length(frontier) == 1  # Only the better solution survives
        end

        @testset "System without :solutions key" begin
            values = [Welfare(metric=:utilitarian, weight=1.0)]
            system = Dict(:utilities => [10.0])
            frontier = pareto_frontier(system, values)
            @test length(frontier) == 1  # Just evaluates the system itself
        end

        @testset "Empty solutions" begin
            values = [Welfare(metric=:utilitarian, weight=1.0)]
            @test isempty(pareto_frontier(Dict{Symbol,Any}[], values))
        end
    end

    @testset "Satisfy - Fairness with Disparate Impact" begin
        @testset "Disparate impact satisfied" begin
            # DI ratio >= threshold means satisfied
            f = Fairness(metric=:disparate_impact, threshold=0.8)
            state = Dict(
                :predictions => [1, 1, 0, 0],  # 50% each group
                :protected => [:a, :b, :a, :b]
            )
            @test satisfy(f, state)
        end

        @testset "Disparate impact not satisfied" begin
            f = Fairness(metric=:disparate_impact, threshold=0.9)
            state = Dict(
                :predictions => [1, 1, 1, 0],  # Group a: 100%, Group b: 0%
                :protected => [:a, :a, :a, :b]
            )
            @test !satisfy(f, state)
        end
    end

    @testset "Satisfy - Fairness with Individual Fairness" begin
        f = Fairness(metric=:individual_fairness, threshold=0.1)
        sim = [1.0 0.9 0.1; 0.9 1.0 0.1; 0.1 0.1 1.0]

        # Similar individuals, similar predictions
        state_fair = Dict(
            :predictions => [0.5, 0.5, 0.9],
            :similarity_matrix => sim
        )
        @test satisfy(f, state_fair)

        # Similar individuals, different predictions
        state_unfair = Dict(
            :predictions => [0.1, 0.9, 0.5],
            :similarity_matrix => sim
        )
        @test !satisfy(f, state_unfair)
    end

    @testset "Enum Types" begin
        @testset "FairnessMetric instances" begin
            @test demographic_parity_metric isa FairnessMetric
            @test equalized_odds_metric isa FairnessMetric
            @test equal_opportunity_metric isa FairnessMetric
            @test disparate_impact_metric isa FairnessMetric
            @test individual_fairness_metric isa FairnessMetric
            @test length(instances(FairnessMetric)) == 5
        end

        @testset "WelfareMetric instances" begin
            @test utilitarian_metric isa WelfareMetric
            @test rawlsian_metric isa WelfareMetric
            @test egalitarian_metric isa WelfareMetric
            @test length(instances(WelfareMetric)) == 3
        end

        @testset "EfficiencyMetric instances" begin
            @test pareto_metric isa EfficiencyMetric
            @test kaldor_hicks_metric isa EfficiencyMetric
            @test computation_time_metric isa EfficiencyMetric
            @test length(instances(EfficiencyMetric)) == 3
        end
    end

    @testset "Type Hierarchy" begin
        @test Fairness <: Value
        @test Welfare <: Value
        @test Profit <: Value
        @test Efficiency <: Value
        @test Safety <: Value
    end
end
