# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Tests for previously untested exports: relevance_score, display methods,
probability(DiscreteZeroProbEvent), handles_zero_prob_event for extended types,
and handles_zero_prob_events with single event dispatch.
"""

@testset "Coverage Gaps" begin

    # ========================================================================
    # Display methods
    # ========================================================================
    @testset "Display Methods" begin
        @testset "ContinuousZeroProbEvent show" begin
            event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
            buf = IOBuffer()
            show(buf, event)
            str = String(take!(buf))
            @test occursin("ContinuousZeroProbEvent", str)
            @test occursin("Float64", str)
            @test occursin("point=0.0", str)
            @test occursin("measure=:density", str)
        end

        @testset "ContinuousZeroProbEvent show with different types" begin
            event_int = ContinuousZeroProbEvent(Uniform(0, 10), 5, :hausdorff)
            buf = IOBuffer()
            show(buf, event_int)
            str = String(take!(buf))
            @test occursin("ContinuousZeroProbEvent", str)
            @test occursin("point=5", str)
        end

        @testset "DiscreteZeroProbEvent show" begin
            event = DiscreteZeroProbEvent(Geometric(0.5), -1)
            buf = IOBuffer()
            show(buf, event)
            str = String(take!(buf))
            @test occursin("DiscreteZeroProbEvent", str)
            @test occursin("point=-1", str)
        end

        @testset "AlmostSureEvent show" begin
            exception = ContinuousZeroProbEvent(Normal(0, 1), 0.0)
            event = AlmostSureEvent(exception, "Sample is irrational")
            buf = IOBuffer()
            show(buf, event)
            str = String(take!(buf))
            @test occursin("AlmostSureEvent", str)
            @test occursin("Sample is irrational", str)
        end

        @testset "SureEvent show" begin
            event = SureEvent("X is a real number")
            buf = IOBuffer()
            show(buf, event)
            str = String(take!(buf))
            @test occursin("SureEvent", str)
            @test occursin("X is a real number", str)
        end

        @testset "TailRiskEvent show" begin
            evt = TailRiskEvent(Normal(0, 1), 3.0, 740.0, 3.37)
            buf = IOBuffer()
            show(buf, evt)
            str = String(take!(buf))
            @test occursin("TailRiskEvent", str)
            @test occursin("threshold=3.0", str)
            @test occursin("return_period=740.0", str)
        end

        @testset "QuantumMeasurementEvent show" begin
            state = ComplexF64[1/sqrt(2), 1/sqrt(2)]
            basis = [ComplexF64[1, 0], ComplexF64[0, 1]]
            event = QuantumMeasurementEvent(state, basis, 1)
            buf = IOBuffer()
            show(buf, event)
            str = String(take!(buf))
            @test occursin("QuantumMeasurementEvent", str)
            @test occursin("dim=2", str)
            @test occursin("outcome=1", str)
        end

        @testset "InsuranceCatastropheEvent show" begin
            evt = InsuranceCatastropheEvent(Pareto(2.0, 1e6), 5e7, 200.0, 1e9)
            buf = IOBuffer()
            show(buf, evt)
            str = String(take!(buf))
            @test occursin("InsuranceCatastropheEvent", str)
            @test occursin("return_period=200.0", str)
        end
    end

    # ========================================================================
    # probability(DiscreteZeroProbEvent)
    # ========================================================================
    @testset "probability(DiscreteZeroProbEvent)" begin
        # Point outside support has zero probability
        dist = Geometric(0.5)
        event = DiscreteZeroProbEvent(dist, -1)
        @test probability(event) == 0.0
    end

    # ========================================================================
    # relevance_score
    # ========================================================================
    @testset "relevance_score" begin
        dist = Normal(100, 10)

        @testset "black_swan application" begin
            event = ContinuousZeroProbEvent(dist, 100.0, :density)
            score = relevance_score(event, :black_swan)
            @test score > 0.0
            @test score isa Float64

            # Center should have higher relevance than tail for black_swan
            event_tail = ContinuousZeroProbEvent(dist, 140.0, :density)
            score_tail = relevance_score(event_tail, :black_swan)
            @test score > score_tail
        end

        @testset "betting application" begin
            event = ContinuousZeroProbEvent(dist, 100.0, :density)
            score = relevance_score(event, :betting)
            @test score > 0.0
            # Betting relevance equals density ratio
            @test score ≈ density_ratio(event)
        end

        @testset "decision_theory application" begin
            event = ContinuousZeroProbEvent(dist, 100.0, :density)
            score = relevance_score(event, :decision_theory)
            @test score > 0.0
            # Decision theory uses epsilon_neighborhood with eps=0.05
            expected = epsilon_neighborhood(event, 0.05)
            @test score ≈ expected
        end

        @testset "unknown application errors" begin
            event = ContinuousZeroProbEvent(dist, 100.0, :density)
            @test_throws ErrorException relevance_score(event, :unknown_app)
        end
    end

    # ========================================================================
    # handles_zero_prob_event for extended types
    # ========================================================================
    @testset "handles_zero_prob_event - Extended Types" begin
        # Simple model that always works
        model_ok = x -> x * 2.0

        @testset "TailRiskEvent handling" begin
            evt = TailRiskEvent(Normal(0, 1), 2.0, 44.0, 2.37)
            @test handles_zero_prob_event(model_ok, evt) == true
        end

        @testset "TailRiskEvent model crash" begin
            evt = TailRiskEvent(Normal(0, 1), 2.0, 44.0, 2.37)
            model_crash = x -> x > 2.5 ? error("crash") : x
            @test handles_zero_prob_event(model_crash, evt) == false
        end

        @testset "TailRiskEvent model returns nothing" begin
            evt = TailRiskEvent(Normal(0, 1), 2.0, 44.0, 2.37)
            model_nothing = x -> nothing
            @test handles_zero_prob_event(model_nothing, evt) == false
        end

        @testset "QuantumMeasurementEvent handling" begin
            state = ComplexF64[1/sqrt(2), 1/sqrt(2)]
            basis = [ComplexF64[1, 0], ComplexF64[0, 1]]
            event = QuantumMeasurementEvent(state, basis, 1)
            @test handles_zero_prob_event(model_ok, event) == true
        end

        @testset "QuantumMeasurementEvent model crash" begin
            state = ComplexF64[1/sqrt(2), 1/sqrt(2)]
            basis = [ComplexF64[1, 0], ComplexF64[0, 1]]
            event = QuantumMeasurementEvent(state, basis, 1)
            model_crash = x -> error("crash")
            @test handles_zero_prob_event(model_crash, event) == false
        end

        @testset "QuantumMeasurementEvent model returns nothing" begin
            state = ComplexF64[1/sqrt(2), 1/sqrt(2)]
            basis = [ComplexF64[1, 0], ComplexF64[0, 1]]
            event = QuantumMeasurementEvent(state, basis, 1)
            model_nothing = x -> nothing
            @test handles_zero_prob_event(model_nothing, event) == false
        end

        @testset "InsuranceCatastropheEvent handling" begin
            evt = InsuranceCatastropheEvent(Pareto(2.0, 1e3), 5e3, 50.0, 1e6)
            @test handles_zero_prob_event(model_ok, evt) == true
        end

        @testset "InsuranceCatastropheEvent model crash" begin
            evt = InsuranceCatastropheEvent(Pareto(2.0, 1e3), 5e3, 50.0, 1e6)
            model_crash = x -> error("crash")
            @test handles_zero_prob_event(model_crash, evt) == false
        end

        @testset "DiscreteZeroProbEvent handling" begin
            dist = Geometric(0.5)
            event = DiscreteZeroProbEvent(dist, -1)
            @test handles_zero_prob_event(model_ok, event) == true
        end

        @testset "DiscreteZeroProbEvent model crash" begin
            dist = Geometric(0.5)
            event = DiscreteZeroProbEvent(dist, -1)
            model_crash = x -> error("crash")
            @test handles_zero_prob_event(model_crash, event) == false
        end

        @testset "DiscreteZeroProbEvent model returns nothing" begin
            dist = Geometric(0.5)
            event = DiscreteZeroProbEvent(dist, -1)
            model_nothing = x -> nothing
            @test handles_zero_prob_event(model_nothing, event) == false
        end
    end

    # ========================================================================
    # handles_zero_prob_events with single event (non-vector dispatch)
    # ========================================================================
    @testset "handles_zero_prob_events - Single Event Dispatch" begin
        model_ok = x -> x
        event = ContinuousZeroProbEvent(Normal(0, 1), 0.0)

        @test handles_zero_prob_events(model_ok, event) == true

        model_crash = x -> abs(x) < 0.01 ? error("crash") : x
        @test handles_zero_prob_events(model_crash, event) == false
    end

    # ========================================================================
    # epsilon_neighborhood edge cases
    # ========================================================================
    @testset "epsilon_neighborhood edge cases" begin
        @testset "Non-zero point" begin
            dist = Normal(5, 2)
            event = ContinuousZeroProbEvent(dist, 5.0, :epsilon)
            prob = epsilon_neighborhood(event, 0.5)
            expected = cdf(dist, 5.5) - cdf(dist, 4.5)
            @test prob ≈ expected
        end

        @testset "Negative epsilon errors" begin
            event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :epsilon)
            @test_throws AssertionError epsilon_neighborhood(event, -0.1)
        end

        @testset "Zero epsilon errors" begin
            event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :epsilon)
            @test_throws AssertionError epsilon_neighborhood(event, 0.0)
        end
    end

    # ========================================================================
    # ContinuousZeroProbEvent with different relevance measures
    # ========================================================================
    @testset "ContinuousZeroProbEvent relevance measures" begin
        dist = Normal(0, 1)

        @testset "density measure" begin
            event = ContinuousZeroProbEvent(dist, 0.0, :density)
            @test event.relevance_measure == :density
            @test relevance(event) ≈ pdf(dist, 0.0)
        end

        @testset "hausdorff measure" begin
            event = ContinuousZeroProbEvent(dist, 0.0, :hausdorff)
            @test event.relevance_measure == :hausdorff
            @test relevance(event) == 1.0
        end

        @testset "epsilon measure" begin
            event = ContinuousZeroProbEvent(dist, 0.0, :epsilon)
            @test event.relevance_measure == :epsilon
            @test relevance(event) > 0.0
        end
    end

    # ========================================================================
    # BlackSwanEvent and MarketCrashEvent additional coverage
    # ========================================================================
    @testset "MarketCrashEvent severity levels" begin
        @testset "catastrophic severity" begin
            crash = MarketCrashEvent(severity=:catastrophic)
            @test crash.threshold == -0.5
        end

        @testset "high severity" begin
            crash = MarketCrashEvent(severity=:high)
            @test crash.threshold == -0.3
        end

        @testset "moderate severity" begin
            crash = MarketCrashEvent(severity=:moderate)
            @test crash.threshold == -0.1
        end

        @testset "custom loss threshold" begin
            crash = MarketCrashEvent(loss_threshold=500_000)
            @test impact_severity(crash, crash.threshold - 0.01) == 500_000
            @test impact_severity(crash, 0.0) == 0
        end
    end

    @testset "BettingEdgeCase expected_value methods" begin
        bet = BettingEdgeCase(Normal(50, 5), 50.0, 500.0, 2.0)

        @testset "epsilon method" begin
            ev = expected_value(bet, method=:epsilon, ε=0.5)
            @test ev isa Float64
        end

        @testset "density method" begin
            ev = expected_value(bet, method=:density, ε=0.1)
            @test ev isa Float64
        end

        @testset "unknown method errors" begin
            @test_throws ErrorException expected_value(bet, method=:unknown)
        end
    end

    # ========================================================================
    # Type hierarchy checks
    # ========================================================================
    @testset "Type Hierarchy" begin
        @test ContinuousZeroProbEvent{Float64} <: ZeroProbEvent
        @test DiscreteZeroProbEvent{Int} <: ZeroProbEvent
        @test TailRiskEvent{Normal{Float64}} <: ZeroProbEvent
        @test QuantumMeasurementEvent <: ZeroProbEvent
        @test InsuranceCatastropheEvent{Pareto{Float64}} <: ZeroProbEvent
        @test BettingEdgeCase{Float64} <: ZeroProbEvent
        # BlackSwanEvent is NOT a ZeroProbEvent subtype (it's parametric standalone)
        @test !(BlackSwanEvent{Float64} <: ZeroProbEvent)
    end
end
