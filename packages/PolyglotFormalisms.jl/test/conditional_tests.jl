# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Conditional conformance tests matching PolyglotFormalisms specification.

These tests correspond to the test cases defined in:
- aggregate-library/specs/conditional/*.md

Each test case matches the PolyglotFormalisms format:
- input: function arguments
- output: expected_result
- description: "Test case description"

Tests verify both basic functionality and mathematical properties
documented in the Conditional module.
"""

@testset "Conditional Operations" begin
    @testset "if_then_else" begin
        # Basic functionality tests
        @test Conditional.if_then_else(true, 1, 2) == 1
        @test Conditional.if_then_else(false, 1, 2) == 2
        @test Conditional.if_then_else(true, "a", "b") == "a"
        @test Conditional.if_then_else(false, "a", "b") == "b"

        @testset "Idempotence" begin
            @test Conditional.if_then_else(true, 42, 42) == 42
            @test Conditional.if_then_else(false, 42, 42) == 42
        end

        @testset "Negation swap" begin
            @test Conditional.if_then_else(!true, 1, 2) == Conditional.if_then_else(true, 2, 1)
            @test Conditional.if_then_else(!false, 1, 2) == Conditional.if_then_else(false, 2, 1)
        end

        @testset "Type flexibility" begin
            @test Conditional.if_then_else(true, [1, 2], [3, 4]) == [1, 2]
            @test Conditional.if_then_else(false, [1, 2], [3, 4]) == [3, 4]
        end
    end

    @testset "when" begin
        # Basic functionality tests
        @test Conditional.when(true, 42) == Some(42)
        @test Conditional.when(false, 42) === nothing

        @testset "Type checking" begin
            result_true = Conditional.when(true, "hello")
            @test result_true == Some("hello")
            @test result_true isa Some{String}

            result_false = Conditional.when(false, "hello")
            @test result_false === nothing
        end

        @testset "Complement with unless" begin
            # For any fixed predicate, when and unless are complementary
            @test Conditional.when(true, 1) == Some(1)
            @test Conditional.unless(true, 1) === nothing
            @test Conditional.when(false, 1) === nothing
            @test Conditional.unless(false, 1) == Some(1)
        end
    end

    @testset "unless" begin
        # Basic functionality tests
        @test Conditional.unless(false, 42) == Some(42)
        @test Conditional.unless(true, 42) === nothing

        @testset "Inverse of when" begin
            @test Conditional.unless(true, 10) == Conditional.when(!true, 10)
            @test Conditional.unless(false, 10) == Conditional.when(!false, 10)
        end

        @testset "Type checking" begin
            result = Conditional.unless(false, "world")
            @test result == Some("world")
            @test result isa Some{String}
        end
    end

    @testset "coalesce" begin
        # Basic functionality tests
        @test Conditional.coalesce(nothing, nothing, 3, 4) == 3
        @test Conditional.coalesce(1, 2, 3) == 1
        @test Conditional.coalesce(nothing, nothing) === nothing

        @testset "Single value" begin
            @test Conditional.coalesce(42) == 42
            @test Conditional.coalesce(nothing) === nothing
        end

        @testset "First non-nothing" begin
            @test Conditional.coalesce(nothing, "first", "second") == "first"
            @test Conditional.coalesce(nothing, nothing, nothing, 99) == 99
        end

        @testset "Idempotence" begin
            @test Conditional.coalesce(5, 5) == 5
        end
    end

    @testset "clamp_value" begin
        # Basic functionality tests
        @test Conditional.clamp_value(5, 0, 10) == 5
        @test Conditional.clamp_value(-1, 0, 10) == 0
        @test Conditional.clamp_value(15, 0, 10) == 10

        @testset "Boundary values" begin
            @test Conditional.clamp_value(0, 0, 10) == 0
            @test Conditional.clamp_value(10, 0, 10) == 10
        end

        @testset "Idempotence" begin
            @test Conditional.clamp_value(Conditional.clamp_value(15, 0, 10), 0, 10) ==
                  Conditional.clamp_value(15, 0, 10)
            @test Conditional.clamp_value(Conditional.clamp_value(-5, 0, 10), 0, 10) ==
                  Conditional.clamp_value(-5, 0, 10)
        end

        @testset "Monotonicity" begin
            @test Conditional.clamp_value(3, 0, 10) <= Conditional.clamp_value(7, 0, 10)
            @test Conditional.clamp_value(-5, 0, 10) <= Conditional.clamp_value(5, 0, 10)
        end

        @testset "Degenerate range" begin
            @test Conditional.clamp_value(5, 3, 3) == 3
            @test Conditional.clamp_value(0, 3, 3) == 3
        end

        @testset "Floating point" begin
            @test Conditional.clamp_value(1.5, 0.0, 2.0) == 1.5
            @test Conditional.clamp_value(-0.5, 0.0, 1.0) == 0.0
        end

        @testset "Error on lo > hi" begin
            @test_throws ArgumentError Conditional.clamp_value(5, 10, 0)
        end
    end
end
