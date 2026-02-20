# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Arithmetic conformance tests matching CommonLib specification.

These tests correspond exactly to the test cases defined in:
- aggregate-library/specs/arithmetic/*.md

Each test case matches the CommonLib format:
- input: [a, b]
- output: expected_result
- description: "Test case description"
"""

@testset "Arithmetic Operations" begin
    @testset "add" begin
        # Test cases from specs/arithmetic/add.md
        @test Arithmetic.add(2, 3) == 5  # Basic addition of positive integers
        @test Arithmetic.add(-5, 3) == -2  # Addition with negative number
        @test Arithmetic.add(0, 0) == 0  # Addition of zeros
        @test Arithmetic.add(1.5, 2.5) == 4.0  # Addition of decimal numbers
        @test Arithmetic.add(-10, -20) == -30  # Addition of two negative numbers

        # Property tests (when Axiom.jl is integrated, these will be proven at compile time)
        @testset "Commutativity" begin
            @test Arithmetic.add(5, 3) == Arithmetic.add(3, 5)
            @test Arithmetic.add(-2, 7) == Arithmetic.add(7, -2)
        end

        @testset "Associativity" begin
            @test Arithmetic.add(Arithmetic.add(2, 3), 4) == Arithmetic.add(2, Arithmetic.add(3, 4))
            @test Arithmetic.add(Arithmetic.add(1, 2), 3) == Arithmetic.add(1, Arithmetic.add(2, 3))
        end

        @testset "Identity" begin
            @test Arithmetic.add(5, 0) == 5
            @test Arithmetic.add(0, 5) == 5
            @test Arithmetic.add(-3, 0) == -3
        end
    end

    @testset "subtract" begin
        # Test cases from specs/arithmetic/subtract.md
        @test Arithmetic.subtract(10, 3) == 7  # Basic subtraction
        @test Arithmetic.subtract(5, 8) == -3  # Subtraction resulting in negative
        @test Arithmetic.subtract(0, 0) == 0  # Subtraction of zeros
        @test Arithmetic.subtract(10.5, 3.2) ≈ 7.3  # Subtraction of decimals
        @test Arithmetic.subtract(-5, -3) == -2  # Subtraction of two negatives

        @testset "Identity" begin
            @test Arithmetic.subtract(7, 0) == 7
            @test Arithmetic.subtract(-3, 0) == -3
        end

        @testset "Inverse of add" begin
            @test Arithmetic.subtract(10, 3) == Arithmetic.add(10, -3)
            @test Arithmetic.subtract(5, -2) == Arithmetic.add(5, 2)
        end
    end

    @testset "multiply" begin
        # Test cases from specs/arithmetic/multiply.md
        @test Arithmetic.multiply(4, 5) == 20  # Basic multiplication
        @test Arithmetic.multiply(-3, 7) == -21  # Multiplication with negative
        @test Arithmetic.multiply(0, 100) == 0  # Multiplication by zero
        @test Arithmetic.multiply(2.5, 4.0) == 10.0  # Multiplication of decimals
        @test Arithmetic.multiply(-2, -3) == 6  # Multiplication of two negatives

        @testset "Commutativity" begin
            @test Arithmetic.multiply(4, 5) == Arithmetic.multiply(5, 4)
            @test Arithmetic.multiply(-3, 7) == Arithmetic.multiply(7, -3)
        end

        @testset "Associativity" begin
            @test Arithmetic.multiply(Arithmetic.multiply(2, 3), 4) ==
                  Arithmetic.multiply(2, Arithmetic.multiply(3, 4))
        end

        @testset "Identity" begin
            @test Arithmetic.multiply(7, 1) == 7
            @test Arithmetic.multiply(1, 7) == 7
            @test Arithmetic.multiply(-5, 1) == -5
        end

        @testset "Zero element" begin
            @test Arithmetic.multiply(100, 0) == 0
            @test Arithmetic.multiply(0, 100) == 0
            @test Arithmetic.multiply(-5, 0) == 0
        end

        @testset "Distributivity" begin
            a, b, c = 3, 4, 5
            @test Arithmetic.multiply(a, Arithmetic.add(b, c)) ==
                  Arithmetic.add(Arithmetic.multiply(a, b), Arithmetic.multiply(a, c))
        end
    end

    @testset "divide" begin
        # Test cases from specs/arithmetic/divide.md
        @test Arithmetic.divide(10, 2) == 5.0  # Basic division
        @test Arithmetic.divide(7, 2) == 3.5  # Division with remainder
        @test Arithmetic.divide(10.5, 2.0) ≈ 5.25  # Division of decimals
        @test Arithmetic.divide(-10, 2) == -5.0  # Division with negative
        @test Arithmetic.divide(5, -2) == -2.5  # Division by negative

        @testset "Identity" begin
            @test Arithmetic.divide(7, 1) == 7
            @test Arithmetic.divide(-5, 1) == -5
        end

        @testset "Inverse of multiply" begin
            a, b = 10, 2
            @test Arithmetic.multiply(Arithmetic.divide(a, b), b) ≈ a

            a, b = 15.5, 3.1
            @test Arithmetic.multiply(Arithmetic.divide(a, b), b) ≈ a
        end

        @testset "Division by zero" begin
            # In Julia, floating-point division by zero returns Inf (IEEE 754)
            # This is implementation-specific behavior per CommonLib spec
            @test isinf(Arithmetic.divide(5, 0))
            @test isinf(Arithmetic.divide(5.0, 0.0))
            @test Arithmetic.divide(5, 0) == Inf
        end
    end

    @testset "modulo" begin
        # Test cases from specs/arithmetic/modulo.md
        @test Arithmetic.modulo(10, 3) == 1  # Basic modulo
        @test Arithmetic.modulo(15, 4) == 3  # Modulo with larger remainder
        @test Arithmetic.modulo(7, 7) == 0  # Modulo with equal numbers
        @test Arithmetic.modulo(0, 5) == 0  # Modulo of zero
        @test Arithmetic.modulo(10, 1) == 0  # Modulo by one

        @testset "Range constraint" begin
            # For positive divisors, result is in [0, divisor)
            @test 0 <= Arithmetic.modulo(10, 3) < 3
            @test 0 <= Arithmetic.modulo(15, 7) < 7
            @test 0 <= Arithmetic.modulo(100, 11) < 11
        end

        @testset "Division relation" begin
            # a == (a ÷ b) * b + (a mod b)
            a, b = 10, 3
            @test a == Arithmetic.add(
                Arithmetic.multiply(div(a, b), b),
                Arithmetic.modulo(a, b)
            )

            a, b = 17, 5
            @test a == Arithmetic.add(
                Arithmetic.multiply(div(a, b), b),
                Arithmetic.modulo(a, b)
            )
        end

        @testset "Modulo by zero" begin
            @test_throws DivideError Arithmetic.modulo(5, 0)
        end
    end
end
