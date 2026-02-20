# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Logical conformance tests matching PolyglotFormalisms specification.

These tests correspond exactly to the test cases defined in:
- aggregate-library/specs/logical/*.md

Each test case matches the PolyglotFormalisms format:
- input: [a, b] or [a]
- output: expected_result
- description: "Test case description"
"""

@testset "Logical Operations" begin
    @testset "and" begin
        # Truth table test cases
        @test Logical.and(true, true) == true
        @test Logical.and(true, false) == false
        @test Logical.and(false, true) == false
        @test Logical.and(false, false) == false

        @testset "Commutativity" begin
            # a ∧ b == b ∧ a
            @test Logical.and(true, false) == Logical.and(false, true)
            @test Logical.and(false, false) == Logical.and(false, false)
            @test Logical.and(true, true) == Logical.and(true, true)
        end

        @testset "Associativity" begin
            # (a ∧ b) ∧ c == a ∧ (b ∧ c)
            @test Logical.and(Logical.and(true, true), false) ==
                  Logical.and(true, Logical.and(true, false))
            @test Logical.and(Logical.and(true, false), true) ==
                  Logical.and(true, Logical.and(false, true))
        end

        @testset "Identity" begin
            # a ∧ true == a
            @test Logical.and(true, true) == true
            @test Logical.and(false, true) == false
        end

        @testset "Annihilator" begin
            # a ∧ false == false
            @test Logical.and(true, false) == false
            @test Logical.and(false, false) == false
        end

        @testset "Idempotence" begin
            # a ∧ a == a
            @test Logical.and(true, true) == true
            @test Logical.and(false, false) == false
        end

        @testset "Absorption" begin
            # a ∧ (a ∨ b) == a
            @test Logical.and(true, Logical.or(true, false)) == true
            @test Logical.and(false, Logical.or(false, true)) == false
        end

        @testset "Distributivity" begin
            # a ∧ (b ∨ c) == (a ∧ b) ∨ (a ∧ c)
            a, b, c = true, false, true
            @test Logical.and(a, Logical.or(b, c)) ==
                  Logical.or(Logical.and(a, b), Logical.and(a, c))
        end

        @testset "De Morgan's law" begin
            # ¬(a ∧ b) == (¬a) ∨ (¬b)
            @test Logical.not(Logical.and(true, false)) ==
                  Logical.or(Logical.not(true), Logical.not(false))
            @test Logical.not(Logical.and(false, false)) ==
                  Logical.or(Logical.not(false), Logical.not(false))
        end
    end

    @testset "or" begin
        # Truth table test cases
        @test Logical.or(true, true) == true
        @test Logical.or(true, false) == true
        @test Logical.or(false, true) == true
        @test Logical.or(false, false) == false

        @testset "Commutativity" begin
            # a ∨ b == b ∨ a
            @test Logical.or(true, false) == Logical.or(false, true)
            @test Logical.or(false, false) == Logical.or(false, false)
        end

        @testset "Associativity" begin
            # (a ∨ b) ∨ c == a ∨ (b ∨ c)
            @test Logical.or(Logical.or(true, false), false) ==
                  Logical.or(true, Logical.or(false, false))
        end

        @testset "Identity" begin
            # a ∨ false == a
            @test Logical.or(true, false) == true
            @test Logical.or(false, false) == false
        end

        @testset "Annihilator" begin
            # a ∨ true == true
            @test Logical.or(true, true) == true
            @test Logical.or(false, true) == true
        end

        @testset "Idempotence" begin
            # a ∨ a == a
            @test Logical.or(true, true) == true
            @test Logical.or(false, false) == false
        end

        @testset "Absorption" begin
            # a ∨ (a ∧ b) == a
            @test Logical.or(true, Logical.and(true, false)) == true
            @test Logical.or(false, Logical.and(false, true)) == false
        end

        @testset "Distributivity" begin
            # a ∨ (b ∧ c) == (a ∨ b) ∧ (a ∨ c)
            a, b, c = false, true, false
            @test Logical.or(a, Logical.and(b, c)) ==
                  Logical.and(Logical.or(a, b), Logical.or(a, c))
        end

        @testset "De Morgan's law" begin
            # ¬(a ∨ b) == (¬a) ∧ (¬b)
            @test Logical.not(Logical.or(true, false)) ==
                  Logical.and(Logical.not(true), Logical.not(false))
            @test Logical.not(Logical.or(false, false)) ==
                  Logical.and(Logical.not(false), Logical.not(false))
        end
    end

    @testset "not" begin
        # Truth table test cases
        @test Logical.not(true) == false
        @test Logical.not(false) == true

        @testset "Involution (double negation)" begin
            # ¬(¬a) == a
            @test Logical.not(Logical.not(true)) == true
            @test Logical.not(Logical.not(false)) == false
        end

        @testset "Excluded middle" begin
            # a ∨ (¬a) == true
            @test Logical.or(true, Logical.not(true)) == true
            @test Logical.or(false, Logical.not(false)) == true
        end

        @testset "Non-contradiction" begin
            # a ∧ (¬a) == false
            @test Logical.and(true, Logical.not(true)) == false
            @test Logical.and(false, Logical.not(false)) == false
        end

        @testset "De Morgan's laws" begin
            # ¬(a ∧ b) == (¬a) ∨ (¬b)
            @test Logical.not(Logical.and(true, false)) ==
                  Logical.or(Logical.not(true), Logical.not(false))

            # ¬(a ∨ b) == (¬a) ∧ (¬b)
            @test Logical.not(Logical.or(true, false)) ==
                  Logical.and(Logical.not(true), Logical.not(false))
        end
    end
end
