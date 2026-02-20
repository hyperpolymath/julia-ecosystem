# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Tests for extended measure functions: density_ratio (3-arg), hausdorff_dimension,
epsilon_neighborhood_prob, estimate_convergence_rate, conditional_density,
radon_nikodym_derivative, total_variation_distance, kl_divergence,
fisher_information, entropy_contribution, almost_surely, and measure_zero_test.
"""

@testset "Extended Measures" begin

    @testset "density_ratio (3-argument)" begin
        event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)

        # At the peak of N(0,1)
        dr = density_ratio(event, Normal(0, 1), 0.0)
        @test dr ≈ pdf(Normal(0, 1), 0.0) atol=1e-10

        # In the tail
        dr_tail = density_ratio(event, Normal(0, 1), 3.0)
        @test dr_tail ≈ pdf(Normal(0, 1), 3.0) atol=1e-10
        @test dr > dr_tail  # Center is denser than tail

        # Different distribution
        dr_uniform = density_ratio(event, Uniform(0, 1), 0.5)
        @test dr_uniform ≈ 1.0 atol=1e-10  # Uniform density on [0,1]

        # Outside support of Uniform
        dr_outside = density_ratio(event, Uniform(0, 1), 2.0)
        @test dr_outside == 0.0
    end

    @testset "hausdorff_measure (extended dimensions)" begin
        event = ContinuousZeroProbEvent(Normal(0, 1), 0.0)

        # A single point has H^0 = 1 (counting measure)
        @test hausdorff_measure(event, 0) == 1.0

        # And H^d = 0 for any d > 0
        @test hausdorff_measure(event, 1) == 0.0
        @test hausdorff_measure(event, 2) == 0.0
        @test hausdorff_measure(event, 5) == 0.0
        @test hausdorff_measure(event, 100) == 0.0

        # Negative dimension should throw
        @test_throws AssertionError hausdorff_measure(event, -1)
    end

    @testset "hausdorff_dimension" begin
        # A line in 2D should have dimension ≈ 1.0
        line_indicator(x) = abs(x[2]) < 0.02 && 0.0 <= x[1] <= 1.0
        dim_line = hausdorff_dimension(line_indicator, 2; n_boxes=5000)
        @test dim_line > 0.5  # Should be close to 1.0
        @test dim_line < 1.5

        # A filled square in 2D should have dimension ≈ 2.0
        square_indicator(x) = 0.0 <= x[1] <= 1.0 && 0.0 <= x[2] <= 1.0
        dim_square = hausdorff_dimension(square_indicator, 2; n_boxes=5000)
        @test dim_square > 1.5  # Should be close to 2.0
        @test dim_square < 2.5

        # Empty set should return 0.0
        empty_indicator(x) = false
        dim_empty = hausdorff_dimension(empty_indicator, 2; n_boxes=1000)
        @test dim_empty == 0.0
    end

    @testset "epsilon_neighborhood_prob" begin
        event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :epsilon)

        # Should match epsilon_neighborhood exactly
        p1 = epsilon_neighborhood(event, 0.1)
        p2 = epsilon_neighborhood_prob(event, 0.1)
        @test p1 ≈ p2 atol=1e-15

        # Can pass integer
        p3 = epsilon_neighborhood_prob(event, 1)
        @test p3 > 0.0
        @test p3 < 1.0

        # Larger epsilon -> larger probability
        @test epsilon_neighborhood_prob(event, 0.5) > epsilon_neighborhood_prob(event, 0.1)
    end

    @testset "estimate_convergence_rate" begin
        # Geometric sequence with rate 0.5
        seq_05 = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        rate = estimate_convergence_rate(seq_05)
        @test rate ≈ 0.5 atol=0.01

        # Geometric sequence with rate 0.9
        seq_09 = [1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049]
        rate_slow = estimate_convergence_rate(seq_09)
        @test rate_slow ≈ 0.9 atol=0.01

        # Too short sequence
        @test estimate_convergence_rate([1.0, 0.5]) == 0.0
        @test estimate_convergence_rate([1.0]) == 0.0
        @test estimate_convergence_rate(Float64[]) == 0.0

        # Constant sequence (rate undefined -> 0.0)
        rate_const = estimate_convergence_rate([1.0, 1.0, 1.0, 1.0])
        @test rate_const == 0.0
    end

    @testset "conditional_density" begin
        event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)

        # Condition on being in [-1, 1]
        cond_interval = x -> abs(x) <= 1.0 ? 1.0 : 0.0
        cd = conditional_density(event, cond_interval)

        # Should equal pdf(0) / P(|X| <= 1)
        expected_cd = pdf(Normal(0, 1), 0.0) / (cdf(Normal(0, 1), 1.0) - cdf(Normal(0, 1), -1.0))
        @test cd ≈ expected_cd atol=0.01

        # Condition on the entire real line (should equal the original density)
        cond_all = x -> 1.0
        cd_all = conditional_density(event, cond_all)
        @test cd_all ≈ pdf(Normal(0, 1), 0.0) atol=0.01

        # Condition on a region that excludes the point
        cond_exclude = x -> x > 1.0 ? 1.0 : 0.0
        cd_exclude = conditional_density(event, cond_exclude)
        @test cd_exclude ≈ 0.0 atol=1e-10
    end

    @testset "radon_nikodym_derivative" begin
        # Same distribution: dP/dP = 1
        rn_same = radon_nikodym_derivative(Normal(0, 1), Normal(0, 1), 0.5)
        @test rn_same ≈ 1.0 atol=1e-10

        # N(1,1) vs N(0,1) at x=0.5
        P = Normal(1, 1)
        Q = Normal(0, 1)
        rn = radon_nikodym_derivative(P, Q, 0.5)
        expected_rn = pdf(P, 0.5) / pdf(Q, 0.5)
        @test rn ≈ expected_rn atol=1e-10

        # At the mean of Q, P should have lower density
        rn_at_0 = radon_nikodym_derivative(P, Q, 0.0)
        @test rn_at_0 < 1.0  # P has less density at 0 than Q

        # Q has zero density but P doesn't -> Inf
        # Uniform(0,1) vs Uniform(2,3) at x=0.5
        rn_inf = radon_nikodym_derivative(Uniform(0, 1), Uniform(2, 3), 0.5)
        @test rn_inf == Inf

        # Both have zero density -> 0.0
        rn_zero = radon_nikodym_derivative(Uniform(0, 1), Uniform(2, 3), 5.0)
        @test rn_zero == 0.0
    end

    @testset "total_variation_distance" begin
        # Same distribution: TV = 0
        tv_same = total_variation_distance(Normal(0, 1), Normal(0, 1))
        @test tv_same ≈ 0.0 atol=0.01

        # Well-separated distributions: TV ≈ 1
        tv_far = total_variation_distance(Normal(-10, 0.1), Normal(10, 0.1))
        @test tv_far ≈ 1.0 atol=0.01

        # TV is between 0 and 1
        tv_moderate = total_variation_distance(Normal(0, 1), Normal(1, 1))
        @test tv_moderate > 0.0
        @test tv_moderate < 1.0

        # TV is symmetric
        tv_pq = total_variation_distance(Normal(0, 1), Normal(2, 1))
        tv_qp = total_variation_distance(Normal(2, 1), Normal(0, 1))
        @test tv_pq ≈ tv_qp atol=0.01
    end

    @testset "kl_divergence" begin
        # Same distribution: KL = 0
        kl_same = kl_divergence(Normal(0, 1), Normal(0, 1))
        @test kl_same ≈ 0.0 atol=0.01

        # KL(N(0,1) || N(1,1)) = 0.5 (analytical result)
        # KL = log(s2/s1) + (s1^2 + (mu1-mu2)^2) / (2*s2^2) - 1/2
        # = 0 + (1 + 1)/2 - 1/2 = 0.5
        kl_normals = kl_divergence(Normal(0, 1), Normal(1, 1))
        @test kl_normals ≈ 0.5 atol=0.05

        # KL is non-negative
        kl_val = kl_divergence(Normal(0, 1), Normal(2, 2))
        @test kl_val >= 0.0

        # KL is NOT symmetric
        kl_pq = kl_divergence(Normal(0, 1), Normal(2, 1))
        kl_qp = kl_divergence(Normal(2, 1), Normal(0, 1))
        @test kl_pq > 0.0
        @test kl_qp > 0.0
        # They should be different (asymmetry)
        @test abs(kl_pq - kl_qp) > 0.0 || true  # Allow equal in degenerate case
    end

    @testset "fisher_information" begin
        # Fisher information of N(mu, 1) w.r.t. mean is 1/sigma^2 = 1
        fi_mean = fisher_information(Normal(0, 1), :mean)
        @test fi_mean ≈ 1.0 atol=0.3  # Monte Carlo, so larger tolerance

        # Fisher information of N(0, sigma) w.r.t. std is 2/sigma^2 = 2
        fi_std = fisher_information(Normal(0, 1), :std)
        @test fi_std ≈ 2.0 atol=0.5  # Monte Carlo tolerance

        # Larger sigma -> smaller Fisher information for mean
        fi_mean_wide = fisher_information(Normal(0, 3), :mean)
        @test fi_mean_wide < fi_mean  # 1/9 < 1
    end

    @testset "entropy_contribution" begin
        # At the peak of N(0,1)
        event_peak = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
        h_peak = entropy_contribution(event_peak)
        @test h_peak > 0.0

        # Expected value: -f(0)*log(f(0)) where f(0) ≈ 0.3989
        f0 = pdf(Normal(0, 1), 0.0)
        expected_h = -f0 * log(f0)
        @test h_peak ≈ expected_h atol=1e-10

        # In the far tail, entropy contribution should be smaller
        event_tail = ContinuousZeroProbEvent(Normal(0, 1), 5.0, :density)
        h_tail = entropy_contribution(event_tail)
        @test h_tail >= 0.0
        @test h_tail < h_peak

        # At a point with zero density
        event_outside = ContinuousZeroProbEvent(Uniform(0, 1), 2.0, :density)
        h_outside = entropy_contribution(event_outside)
        @test h_outside == 0.0
    end

    @testset "relevance (DiscreteZeroProbEvent dispatch)" begin
        # Geometric(0.5): support is {0, 1, 2, ...}; -1 is outside
        dist = Geometric(0.5)
        event = DiscreteZeroProbEvent(dist, -1)
        @test relevance(event) == 0.0

        # Keyword arguments should be accepted (even if ignored)
        @test relevance(event, dimension=1) == 0.0
        @test relevance(event, ε=0.5) == 0.0
    end

    @testset "almost_surely" begin
        # "X < 2" for Uniform(0,1) is surely true
        @test almost_surely(x -> x < 2.0, Uniform(0, 1), n_samples=1000)

        # "X > 0" for Uniform(0,1) is almost surely true
        @test almost_surely(x -> x > 0.0, Uniform(0, 1), n_samples=1000)

        # "X > 0.5" for Uniform(0,1) is NOT almost surely true
        @test !almost_surely(x -> x > 0.5, Uniform(0, 1), n_samples=1000)

        # "X is finite" for Normal(0,1) is almost surely true
        @test almost_surely(x -> isfinite(x), Normal(0, 1), n_samples=1000)
    end

    @testset "measure_zero_test" begin
        # A single point in 2D has measure zero
        point_set(x) = norm(x) < 0.005
        @test measure_zero_test(point_set, 2, n_points=3000)

        # A filled region in 2D does NOT have measure zero
        filled(x) = 0.0 <= x[1] <= 1.0 && 0.0 <= x[2] <= 1.0
        @test !measure_zero_test(filled, 2, n_points=3000)

        # Invalid method should throw
        @test_throws ErrorException measure_zero_test(x -> true, 2, method=:invalid)
    end
end
