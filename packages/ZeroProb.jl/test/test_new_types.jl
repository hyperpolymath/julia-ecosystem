# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Tests for the extended type system: TailRiskEvent, QuantumMeasurementEvent,
and InsuranceCatastropheEvent.
"""

@testset "Extended Types" begin

    @testset "TailRiskEvent" begin
        # Basic construction
        evt = TailRiskEvent(Normal(0, 1), 3.0, 740.0, 3.37)
        @test evt.distribution == Normal(0, 1)
        @test evt.threshold == 3.0
        @test evt.return_period == 740.0
        @test evt.expected_shortfall == 3.37

        # Type hierarchy
        @test evt isa ZeroProbEvent
        @test evt isa TailRiskEvent{Normal{Float64}}

        # Probability: exceedance P(X > 3) for N(0,1) ≈ 0.00135
        p = probability(evt)
        @test p > 0.0
        @test p < 0.01
        @test p ≈ 1.0 - cdf(Normal(0, 1), 3.0) atol=1e-10

        # Different distributions
        evt_uniform = TailRiskEvent(Uniform(0, 1), 0.99, 100.0, 0.995)
        @test probability(evt_uniform) ≈ 0.01 atol=1e-10

        # Display
        io = IOBuffer()
        show(io, evt)
        str = String(take!(io))
        @test occursin("TailRiskEvent", str)
        @test occursin("threshold=3.0", str)
    end

    @testset "QuantumMeasurementEvent" begin
        # Equal superposition qubit
        state = ComplexF64[1/sqrt(2), 1/sqrt(2)]
        basis = [ComplexF64[1, 0], ComplexF64[0, 1]]  # |0>, |1>

        event_0 = QuantumMeasurementEvent(state, basis, 1)
        event_1 = QuantumMeasurementEvent(state, basis, 2)

        # Type hierarchy
        @test event_0 isa ZeroProbEvent
        @test event_0 isa QuantumMeasurementEvent

        # Born rule: P(|0>) = |<0|psi>|^2 = 0.5
        @test probability(event_0) ≈ 0.5 atol=1e-10
        @test probability(event_1) ≈ 0.5 atol=1e-10

        # Total probability should sum to 1
        total_prob = probability(event_0) + probability(event_1)
        @test total_prob ≈ 1.0 atol=1e-10

        # Asymmetric state
        state_asym = ComplexF64[sqrt(0.3), sqrt(0.7)]
        event_asym_0 = QuantumMeasurementEvent(state_asym, basis, 1)
        event_asym_1 = QuantumMeasurementEvent(state_asym, basis, 2)
        @test probability(event_asym_0) ≈ 0.3 atol=1e-10
        @test probability(event_asym_1) ≈ 0.7 atol=1e-10

        # Invalid outcome index
        @test_throws AssertionError QuantumMeasurementEvent(state, basis, 0)
        @test_throws AssertionError QuantumMeasurementEvent(state, basis, 3)

        # Dimension mismatch
        bad_state = ComplexF64[1, 0, 0]
        @test_throws AssertionError QuantumMeasurementEvent(bad_state, basis, 1)

        # Display
        io = IOBuffer()
        show(io, event_0)
        str = String(take!(io))
        @test occursin("QuantumMeasurementEvent", str)
        @test occursin("dim=2", str)
    end

    @testset "InsuranceCatastropheEvent" begin
        # Pareto-distributed losses
        cat_event = InsuranceCatastropheEvent(
            Pareto(2.0, 1e6),
            5e7,
            200.0,
            1e9
        )

        @test cat_event.loss_threshold == 5e7
        @test cat_event.return_period_years == 200.0
        @test cat_event.max_probable_loss == 1e9

        # Type hierarchy
        @test cat_event isa ZeroProbEvent
        @test cat_event isa InsuranceCatastropheEvent

        # Probability: annual exceedance ≈ 1 - exp(-1/200) ≈ 0.004988
        p = probability(cat_event)
        @test p > 0.0
        @test p < 0.01
        @test p ≈ 1.0 - exp(-1.0/200.0) atol=1e-10

        # Short return period = higher probability
        freq_event = InsuranceCatastropheEvent(Pareto(2.0, 1e6), 5e7, 10.0, 1e9)
        @test probability(freq_event) > probability(cat_event)

        # Display
        io = IOBuffer()
        show(io, cat_event)
        str = String(take!(io))
        @test occursin("InsuranceCatastropheEvent", str)
        @test occursin("return_period=200.0", str)
    end
end
