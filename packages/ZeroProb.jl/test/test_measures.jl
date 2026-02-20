# SPDX-License-Identifier: PMPL-1.0-or-later

@testset "Measures" begin
    @testset "probability" begin
        dist = Normal(0, 1)
        event = ContinuousZeroProbEvent(dist, 0.0)

        @test probability(event) == 0.0
    end

    @testset "density_ratio" begin
        dist = Normal(0, 1)
        center = ContinuousZeroProbEvent(dist, 0.0, :density)
        tail = ContinuousZeroProbEvent(dist, 3.0, :density)

        # Center should have higher density than tail
        @test density_ratio(center) > density_ratio(tail)
        @test density_ratio(center) ≈ pdf(dist, 0.0)
    end

    @testset "hausdorff_measure" begin
        dist = Normal(0, 1)
        event = ContinuousZeroProbEvent(dist, 0.0, :hausdorff)

        # 0-dimensional measure of a point is 1
        @test hausdorff_measure(event, 0) == 1.0

        # 1-dimensional measure of a point is 0
        @test hausdorff_measure(event, 1) == 0.0
    end

    @testset "epsilon_neighborhood" begin
        dist = Normal(0, 1)
        event = ContinuousZeroProbEvent(dist, 0.0, :epsilon)

        # P(|X - 0| < 0.1)
        ε = 0.1
        prob = epsilon_neighborhood(event, ε)

        # Should equal cdf(0.1) - cdf(-0.1)
        expected = cdf(dist, ε) - cdf(dist, -ε)
        @test prob ≈ expected

        # Larger ε should give larger probability
        prob_large = epsilon_neighborhood(event, 0.5)
        @test prob_large > prob
    end

    @testset "relevance" begin
        dist = Normal(0, 1)

        # Test each measure type
        event_density = ContinuousZeroProbEvent(dist, 0.0, :density)
        @test relevance(event_density) ≈ pdf(dist, 0.0)

        event_hausdorff = ContinuousZeroProbEvent(dist, 0.0, :hausdorff)
        @test relevance(event_hausdorff, dimension=0) == 1.0

        event_epsilon = ContinuousZeroProbEvent(dist, 0.0, :epsilon)
        @test relevance(event_epsilon, ε=0.1) > 0.0
    end
end
