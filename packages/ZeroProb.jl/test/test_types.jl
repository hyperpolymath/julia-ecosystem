# SPDX-License-Identifier: PMPL-1.0-or-later

@testset "Type System" begin
    @testset "ContinuousZeroProbEvent" begin
        dist = Normal(0, 1)
        event = ContinuousZeroProbEvent(dist, 0.0, :density)

        @test event.distribution == dist
        @test event.point == 0.0
        @test event.relevance_measure == :density

        # Test invalid measure
        @test_throws AssertionError ContinuousZeroProbEvent(dist, 0.0, :invalid)
    end

    @testset "DiscreteZeroProbEvent" begin
        # Point outside support should work
        dist = Geometric(0.5)
        event = DiscreteZeroProbEvent(dist, -1)

        @test event.point == -1
    end

    @testset "AlmostSureEvent" begin
        dist = Normal(0, 1)
        exception = ContinuousZeroProbEvent(dist, 0.0)
        event = AlmostSureEvent(exception, "Test property")

        @test event.description == "Test property"
        @test typeof(event.exception_set) <: ZeroProbEvent
    end

    @testset "SureEvent" begin
        event = SureEvent("X ∈ ℝ")
        @test event.description == "X ∈ ℝ"
    end
end
