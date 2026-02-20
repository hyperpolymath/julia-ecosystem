# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Core type system for zero-probability events.

Defines the hierarchy of events and their properties in continuous and discrete
probability spaces.
"""

# Abstract base type
"""
    ZeroProbEvent

Abstract base type for all zero-probability events.

A zero-probability event is an event E where P(E) = 0, but which can still occur
in the sample space. This is common in continuous distributions where individual
points have measure zero.
"""
abstract type ZeroProbEvent end

"""
    ContinuousZeroProbEvent{T<:Real} <: ZeroProbEvent

A zero-probability event in a continuous distribution.

# Fields
- `distribution::Distribution`: The probability distribution
- `point::T`: The specific point with P(X = point) = 0
- `relevance_measure::Symbol`: Which measure to use for relevance (:density, :hausdorff, :epsilon)

# Examples

```julia
# Exact value in a normal distribution
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
@assert probability(event) == 0.0
@assert relevance(event) ≈ pdf(Normal(0, 1), 0.0)

# Hitting exactly £100 in a gambling scenario
gambling_event = ContinuousZeroProbEvent(Normal(100, 10), 100.0, :density)
```
"""
struct ContinuousZeroProbEvent{T<:Real} <: ZeroProbEvent
    distribution::Distribution
    point::T
    relevance_measure::Symbol

    function ContinuousZeroProbEvent{T}(dist::Distribution, point::T,
                                        measure::Symbol=:density) where T<:Real
        @assert measure in [:density, :hausdorff, :epsilon] "Invalid relevance measure"
        new{T}(dist, point, measure)
    end
end

# Convenience constructor
ContinuousZeroProbEvent(dist::Distribution, point::T, measure::Symbol=:density) where T<:Real =
    ContinuousZeroProbEvent{T}(dist, point, measure)

"""
    DiscreteZeroProbEvent{T} <: ZeroProbEvent

A zero-probability event in a discrete distribution.

In truly discrete distributions, zero-probability events don't typically manifest
unless they are outside the support. This type is provided for completeness and
edge cases.

# Fields
- `distribution::Distribution`: The discrete distribution
- `point::T`: The point outside the support
"""
struct DiscreteZeroProbEvent{T} <: ZeroProbEvent
    distribution::Distribution
    point::T

    function DiscreteZeroProbEvent{T}(dist::Distribution, point::T) where T
        # Verify it's actually zero-probability
        if hasmethod(pdf, (typeof(dist), typeof(point)))
            p = pdf(dist, point)
            @assert p == 0.0 "Point $point has non-zero probability $p in discrete distribution"
        end
        new{T}(dist, point)
    end
end

DiscreteZeroProbEvent(dist::Distribution, point::T) where T =
    DiscreteZeroProbEvent{T}(dist, point)

"""
    AlmostSureEvent{T<:ZeroProbEvent}

An event that is "almost sure" - it has probability 1, but there exists a
zero-probability set where it doesn't hold.

This captures the distinction between "P(E) = 1" and "E is certain".

# Fields
- `event::ZeroProbEvent`: The zero-probability exception set
- `description::String`: What property holds almost surely

# Examples

```julia
# Hitting ANY point in a continuous distribution is almost sure
hitting_something = AlmostSureEvent(
    ContinuousZeroProbEvent(Normal(0, 1), NaN),  # No specific point
    "Sample will take some value"
)

# A random real in [0,1] is almost surely irrational
irrational = AlmostSureEvent(
    ContinuousZeroProbEvent(Uniform(0, 1), NaN),
    "Sample is irrational (rationals have measure zero)"
)
```
"""
struct AlmostSureEvent{T<:ZeroProbEvent}
    exception_set::T
    description::String
end

"""
    SureEvent

An event that holds with absolute certainty - no exceptions, not even on
zero-probability sets.

This is the stronger form of certainty, distinct from "almost sure".
"""
struct SureEvent
    description::String
end

# Display methods
function Base.show(io::IO, e::ContinuousZeroProbEvent{T}) where T
    print(io, "ContinuousZeroProbEvent{$T}(")
    print(io, "$(typeof(e.distribution)), point=$(e.point), measure=:$(e.relevance_measure))")
end

function Base.show(io::IO, e::DiscreteZeroProbEvent{T}) where T
    print(io, "DiscreteZeroProbEvent{$T}($(typeof(e.distribution)), point=$(e.point))")
end

function Base.show(io::IO, e::AlmostSureEvent)
    print(io, "AlmostSureEvent: \"$(e.description)\"")
end

function Base.show(io::IO, e::SureEvent)
    print(io, "SureEvent: \"$(e.description)\"")
end

# ============================================================================
# Phase 2: Extended Event Types
# ============================================================================

"""
    TailRiskEvent{D<:Distribution} <: ZeroProbEvent

Represents extreme tail risk events from Extreme Value Theory (EVT).
Uses Generalized Extreme Value (GEV) or Generalized Pareto Distribution (GPD)
concepts to model events in the far tail of a distribution.

In Extreme Value Theory, the distribution of maxima (or minima) of a large
sample converges to the GEV distribution. The tail index determines whether
the tail is bounded (Weibull), exponential (Gumbel), or heavy (Frechet).

# Fields
- `distribution::D`: The underlying probability distribution
- `threshold::Real`: The extreme value threshold beyond which the event is considered extreme
- `return_period::Real`: Expected return period in time units (e.g., a 100-year flood has return_period=100)
- `expected_shortfall::Real`: Expected loss given exceedance, also known as CVaR (Conditional Value at Risk)

# Mathematical Background

For a distribution F(x), the exceedance probability is P(X > threshold) = 1 - F(threshold).
The return period T relates to exceedance probability by: T = 1 / P(X > threshold).
The expected shortfall (CVaR at level alpha) is: ES_alpha = E[X | X > VaR_alpha].

# Examples

```julia
# A 3-sigma event in a standard normal distribution
# Return period: ~740 observations for a 3-sigma exceedance
# Expected shortfall: ~3.37 (expected value given X > 3)
evt = TailRiskEvent(Normal(0, 1), 3.0, 740.0, 3.37)

# A financial tail risk event
market = Normal(0.001, 0.02)
tail_event = TailRiskEvent(market, -0.06, 250.0, -0.08)
```
"""
struct TailRiskEvent{D<:Distribution} <: ZeroProbEvent
    distribution::D
    threshold::Real
    return_period::Real
    expected_shortfall::Real
end

"""
    QuantumMeasurementEvent <: ZeroProbEvent

Represents a quantum measurement event governed by the Born rule.

In quantum mechanics, the probability of measuring a particular outcome is given
by the Born rule: P(outcome_i) = |<basis_i | state>|^2. When the state vector is
in a superposition of measurement basis states, each individual outcome has a
well-defined probability, but the act of measurement "collapses" the superposition.

For continuous observables (e.g., position), the probability of measuring an exact
value is zero, analogous to classical continuous distributions. This type captures
the discrete measurement case where outcome probabilities are computable via the
Born rule.

# Fields
- `state_vector::Vector{ComplexF64}`: The quantum state vector (normalized, |psi> in Dirac notation)
- `measurement_basis::Vector{Vector{ComplexF64}}`: Orthonormal measurement basis vectors
- `outcome_index::Int`: Which measurement outcome this event represents (1-indexed)

# Mathematical Background

Given state |psi> and measurement basis {|e_i>}, the Born rule gives:
  P(outcome_i) = |<e_i | psi>|^2

The state after measurement collapses to |e_i> (projection postulate).

# Examples

```julia
# Qubit in equal superposition, measured in computational basis
state = ComplexF64[1/sqrt(2), 1/sqrt(2)]
basis = [ComplexF64[1, 0], ComplexF64[0, 1]]  # |0>, |1>
event = QuantumMeasurementEvent(state, basis, 1)  # P(|0>) = 0.5
```
"""
struct QuantumMeasurementEvent <: ZeroProbEvent
    state_vector::Vector{ComplexF64}
    measurement_basis::Vector{Vector{ComplexF64}}
    outcome_index::Int

    function QuantumMeasurementEvent(state::Vector{ComplexF64},
                                      basis::Vector{Vector{ComplexF64}},
                                      index::Int)
        @assert index >= 1 && index <= length(basis) "Outcome index must be within basis range"
        @assert length(state) == length(basis[1]) "State and basis vectors must have same dimension"
        new(state, basis, index)
    end
end

"""
    InsuranceCatastropheEvent{D<:Distribution} <: ZeroProbEvent

Represents an actuarial catastrophe event used in insurance and reinsurance modelling.

Catastrophe events (natural disasters, pandemics, large-scale infrastructure failures)
are characterised by very low frequency but extremely high severity. Insurance and
reinsurance companies model these using Extreme Value Theory, Poisson processes for
arrival times, and heavy-tailed severity distributions (Pareto, Lognormal).

The return period (in years) indicates how rare the event is, while the maximum
probable loss (MPL) represents the worst-case loss scenario used for capital adequacy.

# Fields
- `distribution::D`: The loss severity distribution (typically heavy-tailed)
- `loss_threshold::Real`: The loss level beyond which the event is considered catastrophic
- `return_period_years::Real`: Expected time between occurrences (e.g., 100-year flood)
- `max_probable_loss::Real`: Maximum probable loss used for capital requirements

# Mathematical Background

For a Poisson process with rate lambda, the probability of at least one event in time T is:
  P(N >= 1) = 1 - exp(-lambda * T)
where lambda = 1 / return_period_years.

The loss given occurrence follows the severity distribution, and the catastrophe
is triggered when loss exceeds loss_threshold.

# Examples

```julia
# A 200-year flood event with Pareto-distributed losses
flood = InsuranceCatastropheEvent(
    Pareto(2.0, 1e6),   # Pareto severity with alpha=2, scale=1M
    5e7,                  # Catastrophic if loss > 50M
    200.0,                # 200-year return period
    1e9                   # Max probable loss: 1 billion
)
```
"""
struct InsuranceCatastropheEvent{D<:Distribution} <: ZeroProbEvent
    distribution::D
    loss_threshold::Real
    return_period_years::Real
    max_probable_loss::Real
end

# Display methods for new types

function Base.show(io::IO, e::TailRiskEvent)
    print(io, "TailRiskEvent($(typeof(e.distribution)), threshold=$(e.threshold), ",
          "return_period=$(e.return_period))")
end

function Base.show(io::IO, e::QuantumMeasurementEvent)
    dim = length(e.state_vector)
    print(io, "QuantumMeasurementEvent(dim=$dim, outcome=$(e.outcome_index))")
end

function Base.show(io::IO, e::InsuranceCatastropheEvent)
    print(io, "InsuranceCatastropheEvent($(typeof(e.distribution)), ",
          "threshold=$(e.loss_threshold), return_period=$(e.return_period_years)yr)")
end
