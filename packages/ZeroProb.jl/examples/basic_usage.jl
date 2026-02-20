# SPDX-License-Identifier: PMPL-1.0-or-later

"""
Basic usage example for ZeroProb.jl

Demonstrates:
1. Zero-probability continuous events
2. Relevance vs. probability
3. The continuum paradox
4. Black swan event modeling
"""

using ZeroProb
using Distributions

println("=== Zero-Probability Events ===\n")

# Example 1: Continuous zero-probability event
dist = Normal(100, 10)
event = ContinuousZeroProbEvent(dist, 100.0)

println("Event: X = 100 where X ~ Normal(100, 10)")
println("Probability P(X = 100): ", probability(event))
println("Relevance (PDF value): ", relevance(event))
println("Density ratio: ", density_ratio(event))
println()

# Example 2: Continuum paradox
println("=== The Continuum Paradox ===\n")
result = continuum_paradox(Normal(0, 1), 5)
println(result[:explanation])
println()

# Example 3: Black swan event
println("=== Black Swan Modeling ===\n")
crash = MarketCrashEvent(loss_threshold=1_000_000)
println("Market crash event (loss > \$1M)")
println("Expected impact: \$", expected_impact(crash))
println("Severity at -0.6: ", impact_severity(crash, -0.6))
println()

# Example 4: Discrete zero-probability event
println("=== Discrete Zero-Probability ===\n")
# Geometric distribution: support is {0, 1, 2, ...}, so -1 is outside support
discrete_dist = Geometric(0.5)
discrete_event = DiscreteZeroProbEvent(discrete_dist, -1)
println("Event: point -1 outside support of Geometric(0.5)")
println("Probability: ", probability(discrete_event))
println("Relevance: ", relevance(discrete_event))
println()

# Example 5: Tail risk event
println("=== Tail Risk Event ===\n")
tail_event = TailRiskEvent(Normal(0, 1), 3.0, 740.0, 3.37)
println("Tail risk event: 3-sigma exceedance")
println("Exceedance probability: ", probability(tail_event))
println()

# Example 6: Cantor set
println("=== Cantor Set Construction ===\n")
intervals = construct_cantor_set(3)
println("After 3 iterations: ", length(intervals), " intervals")
total_len = sum(b - a for (a, b) in intervals)
println("Total length: ", total_len, " (approaches 0 as iterations increase)")
println()

# Example 7: Buffon's needle
println("=== Buffon's Needle ===\n")
buffon = buffon_needle_problem(1.0, 2.0, n_samples=50000)
println("Pi estimate from needle drops: ", buffon[:pi_estimate])
println()

println("All examples complete")
