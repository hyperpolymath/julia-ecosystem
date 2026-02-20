# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Comparison conformance tests matching PolyglotFormalisms specification.

These tests correspond exactly to the test cases defined in:
- aggregate-library/specs/comparison/*.md

Each test case matches the PolyglotFormalisms format:
- input: [a, b]
- output: expected_result
- description: "Test case description"
"""

@testset "Comparison Operations" begin
    @testset "less_than" begin
        # Basic test cases
        @test Comparison.less_than(2, 3) == true  # Positive less than positive
        @test Comparison.less_than(5, 5) == false  # Equal values
        @test Comparison.less_than(10, 3) == false  # Greater than
        @test Comparison.less_than(-5, 0) == true  # Negative less than zero
        @test Comparison.less_than(-10, -5) == true  # Negative less than negative
        @test Comparison.less_than(1.5, 2.5) == true  # Decimal numbers
        @test Comparison.less_than(0.0, -0.0) == false  # Signed zeros

        @testset "Transitivity" begin
            # If a < b and b < c, then a < c
            @test Comparison.less_than(1, 2) && Comparison.less_than(2, 3) &&
                  Comparison.less_than(1, 3)
            @test Comparison.less_than(-5, 0) && Comparison.less_than(0, 5) &&
                  Comparison.less_than(-5, 5)
        end

        @testset "Irreflexivity" begin
            # a < a is always false
            @test !Comparison.less_than(5, 5)
            @test !Comparison.less_than(0, 0)
            @test !Comparison.less_than(-3, -3)
        end

        @testset "Asymmetry" begin
            # If a < b, then !(b < a)
            @test Comparison.less_than(2, 5) && !Comparison.less_than(5, 2)
            @test Comparison.less_than(-3, 0) && !Comparison.less_than(0, -3)
        end

        @testset "Edge cases" begin
            @test Comparison.less_than(-Inf, 0) == true
            @test Comparison.less_than(0, Inf) == true
            @test Comparison.less_than(Inf, Inf) == false
            @test !Comparison.less_than(NaN, 0)  # NaN comparisons
            @test !Comparison.less_than(0, NaN)
        end
    end

    @testset "greater_than" begin
        # Basic test cases
        @test Comparison.greater_than(5, 3) == true  # Greater than
        @test Comparison.greater_than(2, 2) == false  # Equal values
        @test Comparison.greater_than(1, 10) == false  # Less than
        @test Comparison.greater_than(0, -5) == true  # Zero greater than negative
        @test Comparison.greater_than(-2, -10) == true  # Negative greater than negative
        @test Comparison.greater_than(3.5, 1.2) == true  # Decimals
        @test Comparison.greater_than(0.0, -0.0) == false  # Signed zeros

        @testset "Transitivity" begin
            @test Comparison.greater_than(5, 3) && Comparison.greater_than(3, 1) &&
                  Comparison.greater_than(5, 1)
        end

        @testset "Irreflexivity" begin
            @test !Comparison.greater_than(7, 7)
            @test !Comparison.greater_than(0, 0)
        end

        @testset "Relation to less_than" begin
            # a > b ⟺ b < a
            @test Comparison.greater_than(5, 2) == Comparison.less_than(2, 5)
            @test Comparison.greater_than(0, -3) == Comparison.less_than(-3, 0)
        end

        @testset "Edge cases" begin
            @test Comparison.greater_than(Inf, 0) == true
            @test Comparison.greater_than(0, -Inf) == true
            @test !Comparison.greater_than(NaN, 0)
        end
    end

    @testset "equal" begin
        # Basic test cases
        @test Comparison.equal(5, 5) == true  # Equal integers
        @test Comparison.equal(3, 7) == false  # Unequal integers
        @test Comparison.equal(0, 0) == true  # Zeros
        @test Comparison.equal(2.5, 2.5) == true  # Equal decimals
        @test Comparison.equal(1.1, 1.2) == false  # Unequal decimals
        @test Comparison.equal(-5, -5) == true  # Equal negatives
        @test Comparison.equal(-0.0, 0.0) == true  # Signed zeros (IEEE 754)

        @testset "Reflexivity" begin
            # a == a for all a
            @test Comparison.equal(5, 5)
            @test Comparison.equal(0, 0)
            @test Comparison.equal(-3, -3)
            @test Comparison.equal(2.5, 2.5)
        end

        @testset "Symmetry" begin
            # If a == b, then b == a
            @test Comparison.equal(5, 7) == Comparison.equal(7, 5)
            @test Comparison.equal(3, 3) == Comparison.equal(3, 3)
        end

        @testset "Transitivity" begin
            # If a == b and b == c, then a == c
            @test Comparison.equal(5, 5) && Comparison.equal(5, 5) &&
                  Comparison.equal(5, 5)
        end

        @testset "Edge cases" begin
            @test !Comparison.equal(NaN, NaN)  # NaN ≠ NaN (IEEE 754)
            @test Comparison.equal(Inf, Inf)
            @test Comparison.equal(-Inf, -Inf)
        end
    end

    @testset "not_equal" begin
        # Basic test cases
        @test Comparison.not_equal(5, 3) == true  # Unequal integers
        @test Comparison.not_equal(7, 7) == false  # Equal integers
        @test Comparison.not_equal(0, 1) == true  # Zero vs non-zero
        @test Comparison.not_equal(-5, -5) == false  # Equal negatives
        @test Comparison.not_equal(2.5, 2.6) == true  # Unequal decimals
        @test Comparison.not_equal(-0.0, 0.0) == false  # Signed zeros

        @testset "Negation of equal" begin
            # a ≠ b ⟺ !(a == b)
            @test Comparison.not_equal(5, 3) == !Comparison.equal(5, 3)
            @test Comparison.not_equal(7, 7) == !Comparison.equal(7, 7)
        end

        @testset "Symmetry" begin
            # If a ≠ b, then b ≠ a
            @test Comparison.not_equal(5, 7) == Comparison.not_equal(7, 5)
        end

        @testset "Edge cases" begin
            @test Comparison.not_equal(NaN, NaN)  # NaN ≠ NaN
            @test !Comparison.not_equal(Inf, Inf)
        end
    end

    @testset "less_equal" begin
        # Basic test cases
        @test Comparison.less_equal(2, 3) == true  # Less than
        @test Comparison.less_equal(5, 5) == true  # Equal
        @test Comparison.less_equal(10, 3) == false  # Greater than
        @test Comparison.less_equal(-5, 0) == true  # Negative ≤ zero
        @test Comparison.less_equal(1.5, 1.5) == true  # Equal decimals
        @test Comparison.less_equal(-0.0, 0.0) == true  # Signed zeros

        @testset "Reflexivity" begin
            # a ≤ a for all a
            @test Comparison.less_equal(5, 5)
            @test Comparison.less_equal(0, 0)
            @test Comparison.less_equal(-3, -3)
        end

        @testset "Transitivity" begin
            @test Comparison.less_equal(1, 2) && Comparison.less_equal(2, 3) &&
                  Comparison.less_equal(1, 3)
        end

        @testset "Antisymmetry" begin
            # If a ≤ b and b ≤ a, then a == b
            @test Comparison.less_equal(5, 5) && Comparison.less_equal(5, 5) &&
                  Comparison.equal(5, 5)
        end

        @testset "Relation to less_than and equal" begin
            # a ≤ b ⟺ (a < b) ∨ (a == b)
            @test Comparison.less_equal(2, 3) ==
                  (Comparison.less_than(2, 3) || Comparison.equal(2, 3))
            @test Comparison.less_equal(5, 5) ==
                  (Comparison.less_than(5, 5) || Comparison.equal(5, 5))
        end

        @testset "Edge cases" begin
            @test Comparison.less_equal(-Inf, 0) == true
            @test !Comparison.less_equal(NaN, 0)
        end
    end

    @testset "greater_equal" begin
        # Basic test cases
        @test Comparison.greater_equal(5, 3) == true  # Greater than
        @test Comparison.greater_equal(7, 7) == true  # Equal
        @test Comparison.greater_equal(2, 10) == false  # Less than
        @test Comparison.greater_equal(0, -5) == true  # Zero ≥ negative
        @test Comparison.greater_equal(3.5, 3.5) == true  # Equal decimals
        @test Comparison.greater_equal(0.0, -0.0) == true  # Signed zeros

        @testset "Reflexivity" begin
            @test Comparison.greater_equal(5, 5)
            @test Comparison.greater_equal(0, 0)
        end

        @testset "Transitivity" begin
            @test Comparison.greater_equal(5, 3) && Comparison.greater_equal(3, 1) &&
                  Comparison.greater_equal(5, 1)
        end

        @testset "Antisymmetry" begin
            @test Comparison.greater_equal(7, 7) && Comparison.greater_equal(7, 7) &&
                  Comparison.equal(7, 7)
        end

        @testset "Relation to greater_than and equal" begin
            @test Comparison.greater_equal(5, 3) ==
                  (Comparison.greater_than(5, 3) || Comparison.equal(5, 3))
        end

        @testset "Relation to less_equal" begin
            # a ≥ b ⟺ b ≤ a
            @test Comparison.greater_equal(5, 2) == Comparison.less_equal(2, 5)
        end

        @testset "Edge cases" begin
            @test Comparison.greater_equal(Inf, 0) == true
            @test !Comparison.greater_equal(NaN, 0)
        end
    end
end
