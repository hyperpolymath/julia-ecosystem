# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Tests for extended paradox demonstrations: construct_cantor_set, banach_tarski_paradox,
vitali_set_paradox, gabriels_horn_paradox, bertrand_paradox, and buffon_needle_problem.
"""

@testset "Extended Paradoxes" begin

    @testset "construct_cantor_set" begin
        # Iteration 0: just [0, 1]
        intervals_0 = construct_cantor_set(0)
        @test length(intervals_0) == 1
        @test intervals_0[1][1] ≈ 0.0
        @test intervals_0[1][2] ≈ 1.0

        # Iteration 1: [0, 1/3] and [2/3, 1]
        intervals_1 = construct_cantor_set(1)
        @test length(intervals_1) == 2
        @test intervals_1[1][1] ≈ 0.0
        @test intervals_1[1][2] ≈ 1.0/3.0
        @test intervals_1[2][1] ≈ 2.0/3.0
        @test intervals_1[2][2] ≈ 1.0

        # Iteration 2: 4 intervals
        intervals_2 = construct_cantor_set(2)
        @test length(intervals_2) == 4

        # Iteration 3: 8 intervals
        intervals_3 = construct_cantor_set(3)
        @test length(intervals_3) == 8

        # General: 2^n intervals after n iterations
        for n in 0:5
            intervals = construct_cantor_set(n)
            @test length(intervals) == 2^n
        end

        # Total length after n iterations is (2/3)^n
        for n in 0:4
            intervals = construct_cantor_set(n)
            total_length = sum(b - a for (a, b) in intervals)
            @test total_length ≈ (2.0/3.0)^n atol=1e-10
        end

        # Each interval has length (1/3)^n
        intervals_4 = construct_cantor_set(4)
        expected_len = (1.0/3.0)^4
        for (a, b) in intervals_4
            @test (b - a) ≈ expected_len atol=1e-10
        end

        # All intervals should be non-overlapping and sorted
        intervals_5 = construct_cantor_set(5)
        for i in 1:length(intervals_5)-1
            @test intervals_5[i][2] <= intervals_5[i+1][1] + 1e-10  # Non-overlapping
        end

        # Negative iterations should throw
        @test_throws AssertionError construct_cantor_set(-1)
    end

    @testset "banach_tarski_paradox" begin
        result = banach_tarski_paradox()

        @test haskey(result, :explanation)
        @test haskey(result, :num_pieces)
        @test haskey(result, :key_ingredients)
        @test haskey(result, :implications)
        @test haskey(result, :resolution)

        @test result[:num_pieces] == 5
        @test !isempty(result[:explanation])

        # Key ingredients should include Axiom of Choice
        @test "Axiom of Choice" in result[:key_ingredients]

        # Should have multiple implications
        @test length(result[:implications]) >= 2
    end

    @testset "vitali_set_paradox" begin
        result = vitali_set_paradox()

        @test haskey(result, :explanation)
        @test haskey(result, :construction_steps)
        @test haskey(result, :contradiction)
        @test haskey(result, :resolution)

        @test !isempty(result[:explanation])
        @test length(result[:construction_steps]) == 4
        @test !isempty(result[:contradiction])
    end

    @testset "gabriels_horn_paradox" begin
        result = gabriels_horn_paradox()

        @test haskey(result, :explanation)
        @test haskey(result, :volume)
        @test haskey(result, :surface_area)
        @test haskey(result, :volume_formula)
        @test haskey(result, :surface_formula)
        @test haskey(result, :paint_paradox)

        # Volume is exactly pi
        @test result[:volume] ≈ pi atol=1e-10

        # Surface area is infinite
        @test result[:surface_area] == Inf

        @test !isempty(result[:explanation])
        @test !isempty(result[:paint_paradox])
    end

    @testset "bertrand_paradox" begin
        result = bertrand_paradox()

        @test haskey(result, :explanation)
        @test haskey(result, :method_1)
        @test haskey(result, :method_2)
        @test haskey(result, :method_3)
        @test haskey(result, :simulated_probs)
        @test haskey(result, :resolution)

        # Theoretical probabilities
        @test result[:method_1][:probability] ≈ 1.0/3.0 atol=1e-10
        @test result[:method_2][:probability] ≈ 1.0/2.0 atol=1e-10
        @test result[:method_3][:probability] ≈ 1.0/4.0 atol=1e-10

        # Simulated probabilities should be in reasonable range of theoretical
        @test abs(result[:simulated_probs][:method_1] - 1.0/3.0) < 0.05
        @test abs(result[:simulated_probs][:method_2] - 1.0/2.0) < 0.05
        @test abs(result[:simulated_probs][:method_3] - 1.0/4.0) < 0.05

        # All three methods give different theoretical answers
        probs = [result[:method_1][:probability],
                 result[:method_2][:probability],
                 result[:method_3][:probability]]
        @test length(unique(probs)) == 3

        @test !isempty(result[:explanation])
    end

    @testset "buffon_needle_problem" begin
        # Default parameters
        result = buffon_needle_problem(1.0, 2.0, n_samples=50000)

        @test haskey(result, :explanation)
        @test haskey(result, :theoretical_probability)
        @test haskey(result, :simulated_probability)
        @test haskey(result, :pi_estimate)
        @test haskey(result, :needle_length)
        @test haskey(result, :line_spacing)
        @test haskey(result, :n_samples)

        # Theoretical probability: 2L/(pi*D) = 2/(2*pi) = 1/pi ≈ 0.3183
        @test result[:theoretical_probability] ≈ 2.0/(pi*2.0) atol=1e-10
        @test result[:needle_length] == 1.0
        @test result[:line_spacing] == 2.0
        @test result[:n_samples] == 50000

        # Pi estimate should be in a reasonable range
        @test abs(result[:pi_estimate] - pi) < 0.3

        # Simulated probability should be close to theoretical
        @test abs(result[:simulated_probability] - result[:theoretical_probability]) < 0.02

        # Different parameters
        result2 = buffon_needle_problem(0.5, 1.0, n_samples=10000)
        @test result2[:theoretical_probability] ≈ 2.0*0.5/(pi*1.0) atol=1e-10

        # Invalid: needle longer than spacing
        @test_throws AssertionError buffon_needle_problem(3.0, 2.0)

        # Invalid: zero/negative lengths
        @test_throws AssertionError buffon_needle_problem(0.0, 2.0)
        @test_throws AssertionError buffon_needle_problem(1.0, 0.0)

        @test !isempty(result[:explanation])
    end
end
