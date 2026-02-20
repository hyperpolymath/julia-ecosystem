# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Practical applications of zero-probability event handling.

Focus areas:
- Black swan events (finance, risk management)
- Betting systems (edge cases, exact values)
- Decision theory under uncertainty
- Rare event simulation
"""

"""
    BlackSwanEvent{T<:Real}

A zero-probability event that, while unlikely, would have catastrophic consequences.

Named after Nassim Taleb's concept: events that are:
1. Extreme outliers
2. Carry massive impact
3. Rationalized in hindsight

# Fields
- `distribution::Distribution`: The assumed distribution
- `threshold::T`: The catastrophic threshold
- `impact::Function`: Impact function (loss, damage, etc.)

# Examples

```julia
# Market crash: losing > £1M
returns = Normal(0.001, 0.02)  # Daily returns: 0.1% mean, 2% volatility
crash = BlackSwanEvent(
    returns,
    -0.5,  # -50% loss
    x -> x < -0.5 ? 1_000_000 : 0  # £1M loss if crash
)

# Even though P(crash) ≈ 0, we must plan for it
@assert probability(crash) ≈ 0.0
@assert impact_severity(crash) == 1_000_000
```
"""
struct BlackSwanEvent{T<:Real}
    distribution::Distribution
    threshold::T
    impact::Function

    function BlackSwanEvent{T}(dist::Distribution, threshold::T,
                               impact::Function) where T<:Real
        new{T}(dist, threshold, impact)
    end
end

BlackSwanEvent(dist::Distribution, threshold::T, impact::Function) where T<:Real =
    BlackSwanEvent{T}(dist, threshold, impact)

"""
    MarketCrashEvent

Convenience constructor for market crash black swan events.

# Examples

```julia
crash = MarketCrashEvent(loss_threshold = 1_000_000, severity = :catastrophic)
```
"""
function MarketCrashEvent(;
                          loss_threshold::Real=1_000_000,
                          mean_return::Real=0.001,
                          volatility::Real=0.02,
                          severity::Symbol=:high)
    dist = Normal(mean_return, volatility)

    # Threshold: what % loss triggers catastrophe?
    threshold = if severity == :catastrophic
        -0.5  # -50%
    elseif severity == :high
        -0.3  # -30%
    else
        -0.1  # -10%
    end

    impact = x -> x <= threshold ? loss_threshold : 0

    return BlackSwanEvent(dist, threshold, impact)
end

"""
    probability(event::BlackSwanEvent) -> Float64

Compute the (tiny) probability of a black swan event.

Even though this is technically non-zero (unlike point events), it's often
so small that traditional risk models ignore it. This is the mistake ZeroProb.jl
helps avoid.
"""
function probability(event::BlackSwanEvent)
    # P(X ≤ threshold)
    return cdf(event.distribution, event.threshold)
end

"""
    impact_severity(event::BlackSwanEvent, x::Real) -> Real

Compute the impact if the event occurs at value x.
"""
function impact_severity(event::BlackSwanEvent, x::Real)
    return event.impact(x)
end

"""
    expected_impact(event::BlackSwanEvent, samples::Int=10000) -> Float64

Compute the expected impact of a black swan event via Monte Carlo simulation.

Even though P(event) ≈ 0, the expected impact can be significant if the
consequence is catastrophic.

# Examples

```julia
crash = MarketCrashEvent(loss_threshold = 1_000_000)
exp_impact = expected_impact(crash, 10000)
# Might be small, but non-zero and worth considering
```
"""
function expected_impact(event::BlackSwanEvent, samples::Int=10000)
    # Monte Carlo: E[impact(X)] = ∫ impact(x) p(x) dx
    xs = rand(event.distribution, samples)
    impacts = [impact_severity(event, x) for x in xs]
    return mean(impacts)
end

"""
    BettingEdgeCase{T<:Real}

A betting scenario where hitting an exact value (zero-probability) provides an edge.

# Fields
- `distribution::Distribution`: Probability distribution of outcomes
- `bet_value::T`: The exact value bet on
- `payout::Float64`: Payout if exact value hit
- `cost::Float64`: Cost to place bet

# Examples

```julia
# Betting on exactly £100 in a roulette-like game
game = Normal(100, 10)
bet = BettingEdgeCase(
    game,
    100.0,  # Bet on exactly £100
    1000.0,  # Pays 1000x if hit
    1.0      # Costs £1 to play
)

# Mathematically, P(exact hit) = 0
# But with relevance measures, we can quantify the edge
@assert probability(bet) == 0.0
@assert relevance(ContinuousZeroProbEvent(game, 100.0, :density)) > 0.0
```
"""
struct BettingEdgeCase{T<:Real} <: ZeroProbEvent
    distribution::Distribution
    bet_value::T
    payout::Float64
    cost::Float64
end

"""
    probability(bet::BettingEdgeCase) -> Float64

Return the probability of hitting the exact bet value (always 0 in continuous case).
"""
function probability(bet::BettingEdgeCase)
    return 0.0  # Exact hit in continuous distribution
end

"""
    expected_value(bet::BettingEdgeCase; method::Symbol=:epsilon, ε::Float64=0.01) -> Float64

Compute the expected value of a bet using an approximation method.

Since the exact hit has P = 0, we use an ε-neighborhood approximation:
EV ≈ P(|X - bet_value| < ε) × payout - cost

# Methods
- `:epsilon` - Use ε-neighborhood probability
- `:density` - Use density ratio as pseudo-probability

# Examples

```julia
bet = BettingEdgeCase(Normal(100, 10), 100.0, 1000.0, 1.0)
ev = expected_value(bet, method=:epsilon, ε=0.1)
println("Expected value: £\$ev")
```
"""
function expected_value(bet::BettingEdgeCase; method::Symbol=:epsilon, ε::Float64=0.01)
    event = ContinuousZeroProbEvent(bet.distribution, bet.bet_value, method)

    if method == :epsilon
        # P(within ε of bet_value) × payout - cost
        p = epsilon_neighborhood(event, ε)
        return p * bet.payout - bet.cost
    elseif method == :density
        # Use density as pseudo-probability (heuristic)
        d = density_ratio(event)
        # Normalize by some scale factor (domain-specific)
        pseudo_p = d * ε
        return pseudo_p * bet.payout - bet.cost
    else
        error("Unknown method: $method")
    end
end

"""
    handles_black_swan(model, event::BlackSwanEvent) -> Bool

Check if a model (e.g., risk management system) properly handles a black swan event.

This is a placeholder for integration with Axiom.jl's verification system.
In practice, you'd verify:
- Model doesn't crash on extreme inputs
- Model outputs reasonable responses
- Model's confidence calibration accounts for tail events

# Examples

```julia
# Pseudocode - would integrate with Axiom.jl
@axiom RobustRiskModel begin
    input :: MarketData
    output :: RiskAssessment

    crash = MarketCrashEvent(severity = :catastrophic)
    @ensure handles_black_swan(output, crash)
end
```
"""
function handles_black_swan(model, event::BlackSwanEvent; num_samples::Int=100)
    # Use inverse transform sampling to efficiently generate samples from the tail
    tail_prob = cdf(event.distribution, event.threshold)
    if tail_prob == 0
        @warn "The probability of the tail is zero. No samples can be generated."
        return true # Or false, depending on desired behavior for impossible tails
    end

    for _ in 1:num_samples
        # Sample a probability from [0, tail_prob]
        u = rand() * tail_prob
        # Convert to a value using the quantile function
        sample = quantile(event.distribution, u)

        try
            result = model(sample)
            if result === nothing
                @warn "Model returned `nothing` for sample $sample"
                return false
            end
        catch e
            @error "Model crashed on tail sample $sample" exception=(e, catch_backtrace())
            return false
        end
    end

    return true
end

"""
    handles_zero_prob_events(model, events::Vector{ZeroProbEvent}) -> Bool

Verify a model handles a suite of zero-probability edge cases.

# Examples

```julia
events = [
    ContinuousZeroProbEvent(Normal(0,1), 0.0),
    MarketCrashEvent(),
    BettingEdgeCase(Uniform(0,100), 50.0, 1000.0, 1.0)
]

@assert handles_zero_prob_events(my_model, events)
```
"""
function handles_zero_prob_events(model, events::Vector{<:ZeroProbEvent})
    # Placeholder for verification
    return all(e -> handles_zero_prob_event(model, e), events)
end

function handles_zero_prob_events(model, event::ZeroProbEvent)
    return handles_zero_prob_event(model, event)
end

"""
    handles_zero_prob_event(model, event::ContinuousZeroProbEvent; num_samples, ε) -> Bool

Test whether a model properly handles a continuous zero-probability event by generating
samples near the event's point and verifying the model doesn't crash or return nothing.

Generates samples from the event's distribution that fall within an ε-neighborhood
of the event's point, then passes them to the model.
"""
function handles_zero_prob_event(model, event::ContinuousZeroProbEvent;
                                  num_samples::Int=100, ε::Float64=0.01)
    value = event.point

    samples_in_neighborhood = 0
    max_tries = num_samples * 1000

    for _ in 1:max_tries
        sample = rand(event.distribution)
        if abs(sample - value) < ε
            samples_in_neighborhood += 1
            try
                result = model(sample)
                if result === nothing
                    @warn "Model returned `nothing` for sample $sample"
                    return false
                end
            catch e
                @error "Model crashed on sample $sample" exception=(e, catch_backtrace())
                return false
            end
        end
        if samples_in_neighborhood >= num_samples
            break
        end
    end

    if samples_in_neighborhood < num_samples
        @warn "Could not generate enough samples in the ε-neighborhood after $max_tries tries."
    end

    return true
end

"""
    handles_zero_prob_event(model, event::BettingEdgeCase; num_samples, ε) -> Bool

Test whether a model properly handles a betting edge case by generating samples near
the bet value and verifying the model doesn't crash or return nothing.
"""
function handles_zero_prob_event(model, event::BettingEdgeCase;
                                  num_samples::Int=100, ε::Float64=0.01)
    value = event.bet_value

    samples_in_neighborhood = 0
    max_tries = num_samples * 1000

    for _ in 1:max_tries
        sample = rand(event.distribution)
        if abs(sample - value) < ε
            samples_in_neighborhood += 1
            try
                result = model(sample)
                if result === nothing
                    @warn "Model returned `nothing` for sample $sample"
                    return false
                end
            catch e
                @error "Model crashed on sample $sample" exception=(e, catch_backtrace())
                return false
            end
        end
        if samples_in_neighborhood >= num_samples
            break
        end
    end

    if samples_in_neighborhood < num_samples
        @warn "Could not generate enough samples in the ε-neighborhood after $max_tries tries."
    end

    return true
end

"""
    handles_zero_prob_event(model, event::BlackSwanEvent; num_samples, ε) -> Bool

Test whether a model properly handles a black swan event by delegating to `handles_black_swan`.
"""
function handles_zero_prob_event(model, event::BlackSwanEvent;
                                  num_samples::Int=100, ε::Float64=0.01)
    return handles_black_swan(model, event; num_samples=num_samples)
end

"""
    handles_zero_prob_event(model, event::TailRiskEvent; num_samples, ε) -> Bool

Test whether a model properly handles a tail risk event by generating samples from
the extreme tail of the distribution and verifying the model doesn't crash.
"""
function handles_zero_prob_event(model, event::TailRiskEvent;
                                  num_samples::Int=100, ε::Float64=0.01)
    dist = event.distribution
    threshold = event.threshold

    # Generate samples beyond the threshold using inverse transform sampling
    tail_prob = 1.0 - cdf(dist, threshold)
    if tail_prob < 1e-300
        @warn "Tail probability is effectively zero; cannot generate tail samples."
        return true
    end

    for _ in 1:num_samples
        # Sample uniformly from the tail region [threshold, inf)
        u = cdf(dist, threshold) + rand() * tail_prob
        sample = quantile(dist, min(u, 1.0 - 1e-15))

        try
            result = model(sample)
            if result === nothing
                @warn "Model returned `nothing` for tail sample $sample"
                return false
            end
        catch e
            @error "Model crashed on tail sample $sample" exception=(e, catch_backtrace())
            return false
        end
    end

    return true
end

"""
    handles_zero_prob_event(model, event::QuantumMeasurementEvent; num_samples, ε) -> Bool

Test whether a model properly handles a quantum measurement event.

Passes the outcome index (as a Float64) to the model for testing.
"""
function handles_zero_prob_event(model, event::QuantumMeasurementEvent;
                                  num_samples::Int=100, ε::Float64=0.01)
    try
        result = model(Float64(event.outcome_index))
        if result === nothing
            @warn "Model returned `nothing` for quantum measurement outcome $(event.outcome_index)"
            return false
        end
    catch e
        @error "Model crashed on quantum measurement outcome $(event.outcome_index)" exception=(e, catch_backtrace())
        return false
    end
    return true
end

"""
    handles_zero_prob_event(model, event::InsuranceCatastropheEvent; num_samples, ε) -> Bool

Test whether a model properly handles an insurance catastrophe event by generating
loss samples that exceed the catastrophe threshold.
"""
function handles_zero_prob_event(model, event::InsuranceCatastropheEvent;
                                  num_samples::Int=100, ε::Float64=0.01)
    dist = event.distribution
    threshold = event.loss_threshold

    tail_prob = 1.0 - cdf(dist, threshold)
    if tail_prob < 1e-300
        @warn "Catastrophe tail probability is effectively zero."
        return true
    end

    for _ in 1:num_samples
        u = cdf(dist, threshold) + rand() * tail_prob
        sample = quantile(dist, min(u, 1.0 - 1e-15))

        try
            result = model(sample)
            if result === nothing
                @warn "Model returned `nothing` for catastrophe loss sample $sample"
                return false
            end
        catch e
            @error "Model crashed on catastrophe loss sample $sample" exception=(e, catch_backtrace())
            return false
        end
    end

    return true
end

"""
    handles_zero_prob_event(model, event::DiscreteZeroProbEvent; num_samples, ε) -> Bool

Test whether a model properly handles a discrete zero-probability event.

Passes the event's point value to the model directly since discrete events
have a specific identifiable point.
"""
function handles_zero_prob_event(model, event::DiscreteZeroProbEvent;
                                  num_samples::Int=100, ε::Float64=0.01)
    try
        result = model(event.point)
        if result === nothing
            @warn "Model returned `nothing` for discrete event point $(event.point)"
            return false
        end
    catch e
        @error "Model crashed on discrete event point $(event.point)" exception=(e, catch_backtrace())
        return false
    end
    return true
end
