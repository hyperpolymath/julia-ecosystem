# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
    ZeroProb

A Julia library for handling zero-probability events in continuous probability spaces.

Zero-probability events are events with P(E) = 0 that can still occur in continuous
distributions. This library provides:

- Alternative relevance measures (density ratios, Hausdorff measures, ε-neighborhoods)
- Pedagogical tools for understanding paradoxes (Borel-Kolmogorov, continuum paradox)
- Applications to black swan events, betting systems, and rare-event analysis
- Integration with Distributions.jl and visualization tools

# Core Concepts

## Almost Surely vs. Surely

- **Surely**: A property that holds for all sample points without exception
- **Almost Surely (a.s.)**: A property that holds except on a zero-probability set

## The Continuum Paradox

The unit interval [0,1] can be decomposed as a union of disjoint zero-probability
points, yet P([0,1]) = 1, not 0. This is resolved through measure theory and
countable vs. uncountable unions.

# Examples

```julia
using ZeroProb, Distributions

# Define a zero-probability event
dist = Normal(100, 10)
event = ContinuousZeroProbEvent(dist, 100.0)

# P(X = 100) = 0, but it has relevance
@assert probability(event) == 0.0
@assert relevance(event) > 0.0  # Uses PDF value

# Visualize the paradox
plot_zero_probability(event)

# Black swan application
crash = MarketCrashEvent(loss_threshold = 1_000_000)
@ensure handles_black_swan(trading_model, crash)
```

# Integration with Axiom.jl

```julia
using Axiom, ZeroProb

@axiom RobustTradingModel begin
    input :: MarketData
    output :: TradingDecision

    # Verify handling of zero-probability but critical events
    @ensure handles_zero_prob_events(output, MarketCrashEvent())
    @prove ∀x. relevance(x) > critical_threshold → model_responds(x)
end
```
"""
module ZeroProb

using Distributions
using StatsBase
using LinearAlgebra

# Core types
export ZeroProbEvent, ContinuousZeroProbEvent, DiscreteZeroProbEvent,
       AlmostSureEvent, SureEvent

# Extended types (Phase 2)
export TailRiskEvent, QuantumMeasurementEvent, InsuranceCatastropheEvent

# Measures
export probability, relevance, density_ratio, hausdorff_measure,
       epsilon_neighborhood, relevance_score

# Extended measures (Phase 1 + Phase 3 + Phase 5)
export hausdorff_dimension, estimate_convergence_rate,
       epsilon_neighborhood_prob
export conditional_density, radon_nikodym_derivative
export total_variation_distance, kl_divergence
export fisher_information, entropy_contribution
export almost_surely, measure_zero_test

# Paradoxes
export continuum_paradox, borel_kolmogorov_paradox,
       rational_points_paradox, uncountable_union_paradox,
       almost_sure_vs_sure

# Extended paradoxes (Phase 1 + Phase 4)
export construct_cantor_set
export banach_tarski_paradox, vitali_set_paradox
export gabriels_horn_paradox, bertrand_paradox, buffon_needle_problem

# Applications
export BlackSwanEvent, MarketCrashEvent, BettingEdgeCase,
       impact_severity, expected_impact, expected_value,
       handles_black_swan, handles_zero_prob_events, handles_zero_prob_event

# Visualization
export plot_zero_probability, plot_continuum_paradox,
       plot_density_vs_probability, plot_epsilon_neighborhood,
       plot_black_swan_impact

# Include submodules
include("types.jl")
include("measures.jl")
include("paradoxes.jl")
include("applications.jl")
include("visualization.jl")

end # module ZeroProb
