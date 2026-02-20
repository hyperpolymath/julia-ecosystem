# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Alternative measures for quantifying the "relevance" of zero-probability events.

While P(E) = 0 for individual points in continuous distributions, these events
can still have significance that we capture through alternative measures.

This module provides:
- Core probability and density measures
- Hausdorff dimension estimation via box-counting
- Epsilon-neighborhood probability computation
- Information-theoretic measures (KL divergence, Fisher information, entropy)
- Measure-theoretic tools (Radon-Nikodym derivatives, total variation distance)
- Convergence rate estimation for numerical sequences
- Monte Carlo verification of almost-sure properties
- Measure-zero testing via box-counting methods
"""

using LinearAlgebra

# ============================================================================
# Core Probability Measures
# ============================================================================

"""
    probability(event::ZeroProbEvent) -> Float64

Return the actual probability of the event (always 0.0 for true zero-probability events).

This is the baseline measure - the event has zero probability in the classical sense.
"""
function probability(event::ContinuousZeroProbEvent{T}) where T
    return 0.0  # By definition
end

function probability(event::DiscreteZeroProbEvent{T}) where T
    return pdf(event.distribution, event.point)
end

"""
    probability(event::TailRiskEvent) -> Float64

Compute the exceedance probability P(X > threshold) for a tail risk event.

For a TailRiskEvent, this returns the probability of exceeding the extreme
value threshold, which is typically very small but non-zero.

# Returns
The exceedance probability 1 - F(threshold), where F is the CDF of the
underlying distribution.
"""
function probability(event::TailRiskEvent)
    return 1.0 - cdf(event.distribution, event.threshold)
end

"""
    probability(event::QuantumMeasurementEvent) -> Float64

Compute the measurement probability using the Born rule.

The Born rule states that the probability of obtaining outcome i when measuring
state |psi> in basis {|e_i>} is P(i) = |<e_i | psi>|^2.

# Returns
The Born rule probability for the specified outcome index.
"""
function probability(event::QuantumMeasurementEvent)
    basis_vector = event.measurement_basis[event.outcome_index]
    # Born rule: P = |<basis|state>|^2
    amplitude = dot(basis_vector, event.state_vector)
    return abs2(amplitude)
end

"""
    probability(event::InsuranceCatastropheEvent) -> Float64

Compute the annual exceedance probability for a catastrophe event.

Uses the Poisson process relationship: the probability of at least one event
in one year is P = 1 - exp(-1/return_period_years). For large return periods,
this approximates to 1/return_period_years.

# Returns
The annual probability of the catastrophe occurring.
"""
function probability(event::InsuranceCatastropheEvent)
    # Poisson process: P(at least one event in 1 year) = 1 - exp(-lambda)
    # where lambda = 1 / return_period_years
    lambda = 1.0 / event.return_period_years
    return 1.0 - exp(-lambda)
end

# ============================================================================
# Density Measures
# ============================================================================

"""
    density_ratio(event::ContinuousZeroProbEvent{T}) -> Float64

Use the probability density function (PDF) value as a relevance measure.

While P(X = x) = 0 in continuous distributions, pdf(X, x) > 0 represents the
"density" of probability mass near that point. This is useful for comparing
the relative importance of different zero-probability events.

# Examples

```julia
# Normal distribution - center has highest density
dist = Normal(0, 1)
center = ContinuousZeroProbEvent(dist, 0.0)
tail = ContinuousZeroProbEvent(dist, 3.0)

@assert density_ratio(center) > density_ratio(tail)  # Center is "more relevant"
```
"""
function density_ratio(event::ContinuousZeroProbEvent{T}) where T
    return pdf(event.distribution, event.point)
end

"""
    density_ratio(event::ZeroProbEvent, dist::Distribution, point::Real) -> Float64

Compute the density ratio at a specific point for a given event and distribution.
This 3-argument form allows computing density ratios without creating a new event object.

Returns the probability density function value at `point` for `dist`, providing
a measure of how "relevant" this zero-probability point is relative to others.

The density ratio is defined as f(point) where f is the PDF of `dist`. This
value is always non-negative and integrates to 1 over the support. Higher
density indicates the point lies in a region with more concentrated probability
mass.

# Arguments
- `event::ZeroProbEvent`: The zero-probability event context (used for dispatch)
- `dist::Distribution`: The probability distribution to evaluate
- `point::Real`: The point at which to evaluate the density

# Returns
A `Float64` representing the PDF value at `point`.

# Examples

```julia
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
density_ratio(event, Normal(0, 1), 0.0)  # ≈ 0.3989 (peak of standard normal)
density_ratio(event, Normal(0, 1), 2.0)  # ≈ 0.0540 (in the tail)
```
"""
function density_ratio(event::ZeroProbEvent, dist::Distribution, point::Real)
    return pdf(dist, point)
end

# ============================================================================
# Hausdorff Measures
# ============================================================================

"""
    hausdorff_measure(event::ContinuousZeroProbEvent{T}, dimension::Int=0) -> Float64

Compute the Hausdorff measure of the event (a single point in the ambient space).

For a single point in n-dimensional space, the d-dimensional Hausdorff measure is:
- d = 0: H^0({x}) = 1 (the counting measure; a single point has count 1)
- d > 0: H^d({x}) = 0 (a point has zero length, area, volume, etc.)

More generally, for a set S embedded in n-dimensional space:
- If d < dim(S): H^d(S) = Inf (the measure "overflows")
- If d = dim(S): H^d(S) is finite and positive (the "right" dimension)
- If d > dim(S): H^d(S) = 0 (the measure "underflows")

For a single point, dim(S) = 0, so the above specialises to:
- dimension == 0: returns 1.0 (finite positive)
- dimension > 0: returns 0.0 (point is too small for higher-dimensional measure)

# Arguments
- `event`: The zero-probability event (representing a single point)
- `dimension`: The Hausdorff dimension d at which to compute the measure (default 0)

# Examples

```julia
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0)
@assert hausdorff_measure(event, 0) == 1.0  # Point has unit 0-dimensional measure
@assert hausdorff_measure(event, 1) == 0.0  # But zero 1-dimensional measure
@assert hausdorff_measure(event, 5) == 0.0  # Zero in any higher dimension
```
"""
function hausdorff_measure(event::ContinuousZeroProbEvent{T}, dimension::Int=0) where T
    @assert dimension >= 0 "Hausdorff dimension must be non-negative"
    # For a single point, the intrinsic dimension is 0.
    # H^0({x}) = 1 (counting measure), H^d({x}) = 0 for d > 0.
    if dimension == 0
        return 1.0
    else
        return 0.0
    end
end

"""
    hausdorff_dimension(set_indicator::Function, dim::Int;
                        n_boxes::Int=1000, scales::Vector{Float64}=Float64[]) -> Float64

Estimate the Hausdorff (box-counting) dimension of a set via the box-counting method.

The box-counting dimension (also called Minkowski-Sausage dimension) is defined as:
  d_box = lim_{epsilon -> 0} log(N(epsilon)) / log(1/epsilon)
where N(epsilon) is the number of boxes of side length epsilon needed to cover the set.

This function estimates d_box by performing a linear regression of log(N(epsilon))
against log(1/epsilon) over a range of scales.

# Arguments
- `set_indicator::Function`: A function `f(x::Vector{Float64}) -> Bool` that returns
  `true` if the point x is in the set and `false` otherwise.
- `dim::Int`: The ambient dimension of the space (e.g., 2 for a subset of R^2).
- `n_boxes::Int=1000`: Number of random sample points per scale for estimating box counts.
  Higher values give more accurate estimates but take longer.
- `scales::Vector{Float64}`: Box side lengths to use. If empty, defaults to a
  logarithmically-spaced range from 0.01 to 1.0 with 20 points.

# Returns
A `Float64` representing the estimated box-counting dimension.

# Mathematical Background

For a fractal set like the Cantor set (d ≈ 0.631), the Koch curve (d ≈ 1.262),
or the Sierpinski triangle (d ≈ 1.585), the box-counting dimension captures
the scaling relationship between resolution and complexity.

# Examples

```julia
# Estimate dimension of a line segment in 2D (should be ≈ 1.0)
line_indicator(x) = abs(x[2]) < 0.01 && 0.0 <= x[1] <= 1.0
hausdorff_dimension(line_indicator, 2)  # ≈ 1.0

# Estimate dimension of a filled square in 2D (should be ≈ 2.0)
square_indicator(x) = 0.0 <= x[1] <= 1.0 && 0.0 <= x[2] <= 1.0
hausdorff_dimension(square_indicator, 2)  # ≈ 2.0
```
"""
function hausdorff_dimension(set_indicator::Function, dim::Int;
                              n_boxes::Int=1000,
                              scales::Vector{Float64}=Float64[])
    @assert dim >= 1 "Ambient dimension must be at least 1"

    # Default scales: logarithmically spaced from 0.01 to 1.0
    if isempty(scales)
        scales = exp.(range(log(0.01), log(1.0), length=20))
    end

    log_inv_scales = Float64[]
    log_counts = Float64[]

    for eps in scales
        # Count occupied boxes at this scale
        # Use random sampling to probe the set within [0,1]^dim
        occupied = Set{NTuple{length(1:dim), Int}}()

        for _ in 1:n_boxes
            # Random point in [0, 1]^dim
            x = rand(dim)
            if set_indicator(x)
                # Determine which box this point falls in
                box_indices = ntuple(i -> floor(Int, x[i] / eps), dim)
                push!(occupied, box_indices)
            end
        end

        count = length(occupied)
        if count > 0
            push!(log_inv_scales, log(1.0 / eps))
            push!(log_counts, log(Float64(count)))
        end
    end

    if length(log_inv_scales) < 2
        return 0.0  # Not enough data to estimate dimension
    end

    # Linear regression: log(N) = d * log(1/eps) + c
    # Using least squares: d = (n * sum(xy) - sum(x)*sum(y)) / (n * sum(x^2) - sum(x)^2)
    n = length(log_inv_scales)
    sx = sum(log_inv_scales)
    sy = sum(log_counts)
    sxy = sum(log_inv_scales .* log_counts)
    sx2 = sum(log_inv_scales .^ 2)

    denominator = n * sx2 - sx^2
    if abs(denominator) < 1e-15
        return 0.0
    end

    slope = (n * sxy - sx * sy) / denominator
    return slope
end

# ============================================================================
# Epsilon-Neighborhood Measures
# ============================================================================

"""
    epsilon_neighborhood(event::ContinuousZeroProbEvent{T}, ε::Float64) -> Float64

Compute P(|X - x| < ε) - the probability of being within ε of the zero-probability point.

This is a practical measure: while P(X = x) = 0, we can ask "what's the probability
of getting close enough?" This is useful in applications where approximate equality
matters (betting, auctions, physical measurements).

# Examples

```julia
dist = Normal(0, 1)
event = ContinuousZeroProbEvent(dist, 0.0)

# Probability of being within 0.1 of 0
prob_near = epsilon_neighborhood(event, 0.1)
@assert prob_near ≈ cdf(dist, 0.1) - cdf(dist, -0.1)
```
"""
function epsilon_neighborhood(event::ContinuousZeroProbEvent{T}, ε::Float64) where T
    @assert ε > 0.0 "ε must be positive"

    dist = event.distribution
    x = event.point

    # P(|X - x| < ε) = P(x - ε < X < x + ε) = cdf(x + ε) - cdf(x - ε)
    return cdf(dist, x + ε) - cdf(dist, x - ε)
end

"""
    epsilon_neighborhood_prob(event::ContinuousZeroProbEvent, eps::Real) -> Float64

Convenience wrapper that computes P(|X - x| < eps) for a given epsilon value.

This is an alias for `epsilon_neighborhood` with a `Real`-typed epsilon parameter,
allowing integer and other numeric types to be passed without explicit conversion.

# Arguments
- `event::ContinuousZeroProbEvent`: The zero-probability event with associated distribution and point
- `eps::Real`: The radius of the epsilon-neighborhood (must be positive)

# Returns
The probability P(|X - x| < eps) as a `Float64`.

# Examples

```julia
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :epsilon)

# These are equivalent:
p1 = epsilon_neighborhood(event, 0.1)
p2 = epsilon_neighborhood_prob(event, 0.1)
@assert p1 ≈ p2

# Can pass integers directly
p3 = epsilon_neighborhood_prob(event, 1)  # eps = 1
```
"""
function epsilon_neighborhood_prob(event::ContinuousZeroProbEvent, eps::Real)
    return epsilon_neighborhood(event, Float64(eps))
end

# ============================================================================
# Relevance Measures
# ============================================================================

"""
    relevance(event::ContinuousZeroProbEvent; kwargs...) -> Float64

Compute a relevance score for a continuous zero-probability event using the configured measure.

This dispatches to the appropriate measure based on the event's `relevance_measure` field:
- `:density` -> `density_ratio(event)`
- `:hausdorff` -> `hausdorff_measure(event, dimension)`
- `:epsilon` -> `epsilon_neighborhood(event, epsilon)`

# Keyword Arguments
- `dimension::Int=0`: Hausdorff dimension (for :hausdorff measure)
- `ε::Float64=0.01`: Neighborhood radius (for :epsilon measure)

# Examples

```julia
# Default: density ratio
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
@assert relevance(event) ≈ pdf(Normal(0, 1), 0.0)

# Hausdorff measure
event_h = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :hausdorff)
@assert relevance(event_h) == 1.0

# Epsilon neighborhood
event_e = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :epsilon)
@assert relevance(event_e, ε=0.1) > 0.0
```
"""
function relevance(event::ContinuousZeroProbEvent{T};
                   dimension::Int=0, ε::Float64=0.01) where T
    measure = event.relevance_measure

    if measure == :density
        return density_ratio(event)
    elseif measure == :hausdorff
        return hausdorff_measure(event, dimension)
    elseif measure == :epsilon
        return epsilon_neighborhood(event, ε)
    else
        error("Unknown relevance measure: $measure")
    end
end

"""
    relevance(event::DiscreteZeroProbEvent; kwargs...) -> Float64

Compute the relevance of a discrete zero-probability event.

For discrete distributions, the relevance is defined as the probability mass function
(PMF) value at the event's point. Since the event has zero probability by construction
(the point is outside the distribution's support), this will return 0.0.

This dispatch ensures that `relevance` can be called uniformly on all `ZeroProbEvent`
subtypes without a `MethodError`.

# Keyword Arguments
Accepted but ignored (provided for API consistency with `ContinuousZeroProbEvent`):
- `dimension::Int=0`
- `ε::Float64=0.01`

# Returns
The PMF value at the event's point (0.0 for a true zero-probability event).

# Examples

```julia
dist = Geometric(0.5)
event = DiscreteZeroProbEvent(dist, -1)  # -1 is outside support
@assert relevance(event) == 0.0
```
"""
function relevance(event::DiscreteZeroProbEvent{T};
                   dimension::Int=0, ε::Float64=0.01) where T
    return pdf(event.distribution, event.point)
end

"""
    relevance_score(event::ZeroProbEvent, application::Symbol) -> Float64

Compute an application-specific relevance score.

Different applications care about different aspects of zero-probability events:
- `:black_swan` - How catastrophic would this event be?
- `:betting` - How much edge would this give in a betting system?
- `:decision_theory` - How should this influence decisions under uncertainty?

# Examples

```julia
crash = MarketCrashEvent(loss_threshold = 1_000_000)
score = relevance_score(crash, :black_swan)
# Returns high score because even P=0 events matter when catastrophic
```
"""
function relevance_score(event::ContinuousZeroProbEvent{T}, application::Symbol) where T
    if application == :black_swan
        # For black swans, even tiny probabilities of extreme events matter
        # Use a combination of density and tail behavior
        density = density_ratio(event)
        tail_weight = 1.0 / (1.0 + abs(event.point))  # Penalize extreme tails
        return density * tail_weight
    elseif application == :betting
        # For betting, we care about the density near the bet point
        return density_ratio(event)
    elseif application == :decision_theory
        # General decision-theoretic relevance
        return epsilon_neighborhood(event, 0.05)  # 5% neighborhood
    else
        error("Unknown application: $application")
    end
end

# ============================================================================
# Convergence Analysis
# ============================================================================

"""
    estimate_convergence_rate(sequence::Vector{<:Real}) -> Float64

Estimate the convergence rate of a numerical sequence by computing successive
ratios of absolute differences.

For a convergent sequence {a_n} converging to limit L, the convergence rate r
is estimated from the ratio |a_{n+1} - a_n| / |a_n - a_{n-1}|. If the sequence
converges geometrically (linearly in optimisation terminology), this ratio
converges to r where |a_n - L| ~ C * r^n.

The function computes these ratios for all consecutive triples in the sequence
and returns the median ratio as a robust estimate of the convergence rate.

# Arguments
- `sequence::Vector{<:Real}`: A numerical sequence of at least 3 elements.

# Returns
A `Float64` representing the estimated geometric convergence rate.
Returns 0.0 if the sequence has fewer than 3 elements or if differences are zero.

A rate of:
- r < 1: Sequence is converging (smaller r = faster convergence)
- r = 1: Sequence is converging sub-geometrically (e.g., algebraically)
- r > 1: Sequence is diverging

# Examples

```julia
# Geometric sequence converging to 0 with rate 0.5
seq = [1.0, 0.5, 0.25, 0.125, 0.0625]
rate = estimate_convergence_rate(seq)
@assert rate ≈ 0.5 atol=0.01

# Slower convergence
slow_seq = [1.0, 0.9, 0.81, 0.729, 0.6561]
rate_slow = estimate_convergence_rate(slow_seq)
@assert rate_slow ≈ 0.9 atol=0.01
```
"""
function estimate_convergence_rate(sequence::Vector{<:Real})
    n = length(sequence)
    if n < 3
        return 0.0
    end

    ratios = Float64[]
    for i in 3:n
        diff_prev = abs(sequence[i-1] - sequence[i-2])
        diff_curr = abs(sequence[i] - sequence[i-1])

        if diff_prev > 1e-15  # Avoid division by zero
            push!(ratios, diff_curr / diff_prev)
        end
    end

    if isempty(ratios)
        return 0.0
    end

    # Return median ratio as a robust estimate
    sort!(ratios)
    mid = div(length(ratios) + 1, 2)
    return ratios[mid]
end

# ============================================================================
# Information-Theoretic Measures
# ============================================================================

"""
    conditional_density(event::ContinuousZeroProbEvent, condition::Function) -> Float64

Compute the conditional density at the event's point, resolving the Borel-Kolmogorov
paradox by using the limiting density ratio approach.

The Borel-Kolmogorov paradox shows that conditioning on a zero-probability event is
ambiguous. This function resolves the ambiguity by computing the conditional density
as the ratio of the joint density to the marginal density at the conditioning set.

Specifically, for a conditioning function `condition` that maps a real value to a
non-negative weight, the conditional density is:

  f(x | condition) = f(x) * condition(x) / integral(f(t) * condition(t) dt)

When `condition` is the indicator of an interval [a,b], this reduces to the
standard conditional density on an interval.

# Arguments
- `event::ContinuousZeroProbEvent`: The zero-probability event providing distribution and point
- `condition::Function`: A function `g(x) -> Real` returning a non-negative weight.
  Common choices: indicator functions for intervals, Gaussian kernels for soft conditioning.

# Returns
The conditional density value at event.point as a `Float64`.
Returns 0.0 if the normalising integral is zero.

# Examples

```julia
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
# Condition on being in [-1, 1]
cond = x -> abs(x) <= 1.0 ? 1.0 : 0.0
cd = conditional_density(event, cond)
# cd ≈ pdf(Normal(0,1), 0.0) / (cdf(Normal(0,1), 1.0) - cdf(Normal(0,1), -1.0))
```
"""
function conditional_density(event::ContinuousZeroProbEvent, condition::Function)
    dist = event.distribution
    x = event.point

    # Numerator: f(x) * condition(x)
    numerator = pdf(dist, x) * condition(x)

    if numerator == 0.0
        return 0.0
    end

    # Denominator: numerical integration of f(t) * condition(t) over support
    # Use simple quadrature over a wide range
    if isa(dist, Normal)
        mu, sigma = params(dist)
        lower = mu - 6 * sigma
        upper = mu + 6 * sigma
    else
        lower = quantile(dist, 0.0001)
        upper = quantile(dist, 0.9999)
    end

    # Simpson's rule with 1000 points
    n_points = 1000
    h = (upper - lower) / n_points
    integral = 0.0
    for i in 0:n_points
        t = lower + i * h
        weight = if i == 0 || i == n_points
            1.0
        elseif i % 2 == 1
            4.0
        else
            2.0
        end
        integral += weight * pdf(dist, t) * condition(t)
    end
    integral *= h / 3.0

    if integral < 1e-15
        return 0.0
    end

    return numerator / integral
end

"""
    radon_nikodym_derivative(P::Distribution, Q::Distribution, point::Real) -> Float64

Compute the Radon-Nikodym derivative dP/dQ evaluated at a specific point.

The Radon-Nikodym derivative (also known as the likelihood ratio) is the ratio
of the density of P to the density of Q:

  (dP/dQ)(x) = f_P(x) / f_Q(x)

This exists when P is absolutely continuous with respect to Q (i.e., Q(A) = 0
implies P(A) = 0 for all measurable sets A). It is fundamental in:
- Change of measure (Girsanov theorem in stochastic calculus)
- Importance sampling
- Hypothesis testing (Neyman-Pearson lemma)
- Risk-neutral pricing in finance

# Arguments
- `P::Distribution`: The numerator distribution (measure P)
- `Q::Distribution`: The denominator distribution (measure Q, the reference measure)
- `point::Real`: The point at which to evaluate the derivative

# Returns
The Radon-Nikodym derivative dP/dQ at the given point as a `Float64`.
Returns `Inf` if Q has zero density but P has positive density at the point.
Returns 0.0 if both have zero density.

# Examples

```julia
# Change of measure from N(1,1) to N(0,1)
P = Normal(1, 1)
Q = Normal(0, 1)
rn = radon_nikodym_derivative(P, Q, 0.5)
# rn = pdf(N(1,1), 0.5) / pdf(N(0,1), 0.5)
```
"""
function radon_nikodym_derivative(P::Distribution, Q::Distribution, point::Real)
    f_P = pdf(P, point)
    f_Q = pdf(Q, point)

    if f_Q < 1e-300
        # Q has (effectively) zero density at this point
        if f_P < 1e-300
            return 0.0  # Both zero: 0/0 convention
        else
            return Inf  # P has mass where Q doesn't: P not abs. continuous w.r.t. Q
        end
    end

    return f_P / f_Q
end

"""
    total_variation_distance(P::Distribution, Q::Distribution;
                              n_points::Int=10000) -> Float64

Estimate the total variation distance between two distributions.

The total variation distance is defined as:
  TV(P, Q) = sup_A |P(A) - Q(A)| = (1/2) * integral |f_P(x) - f_Q(x)| dx

where the supremum is over all measurable sets A. The integral form is used
for distributions with densities.

This function estimates TV(P, Q) numerically using quadrature over a wide range.

# Properties
- 0 <= TV(P, Q) <= 1
- TV(P, Q) = 0 iff P = Q
- TV(P, Q) = 1 iff P and Q have disjoint supports

# Arguments
- `P::Distribution`: First distribution
- `Q::Distribution`: Second distribution
- `n_points::Int=10000`: Number of quadrature points for numerical integration

# Returns
The estimated total variation distance as a `Float64`.

# Examples

```julia
# Same distribution: TV = 0
tv = total_variation_distance(Normal(0, 1), Normal(0, 1))
@assert tv ≈ 0.0 atol=0.01

# Well-separated: TV close to 1
tv = total_variation_distance(Normal(-10, 0.1), Normal(10, 0.1))
@assert tv ≈ 1.0 atol=0.01
```
"""
function total_variation_distance(P::Distribution, Q::Distribution;
                                   n_points::Int=10000)
    # Determine integration range from both distributions
    lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
    upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))

    # Trapezoidal rule integration of |f_P - f_Q|
    h = (upper - lower) / n_points
    integral = 0.0
    for i in 0:n_points
        x = lower + i * h
        diff = abs(pdf(P, x) - pdf(Q, x))
        weight = (i == 0 || i == n_points) ? 0.5 : 1.0
        integral += weight * diff
    end
    integral *= h

    # TV = (1/2) * integral |f_P - f_Q|
    return 0.5 * integral
end

"""
    kl_divergence(P::Distribution, Q::Distribution; n_points::Int=10000) -> Float64

Compute the Kullback-Leibler divergence D_KL(P || Q) between two distributions.

The KL divergence is defined as:
  D_KL(P || Q) = integral f_P(x) * log(f_P(x) / f_Q(x)) dx

It measures the "information loss" when Q is used to approximate P. Note that
KL divergence is NOT symmetric: D_KL(P || Q) != D_KL(Q || P) in general.

# Properties
- D_KL(P || Q) >= 0 (Gibbs' inequality)
- D_KL(P || Q) = 0 iff P = Q almost everywhere
- D_KL(P || Q) = Inf if supp(P) is not subset of supp(Q)
- Not a true metric (not symmetric, doesn't satisfy triangle inequality)

# Arguments
- `P::Distribution`: The "true" distribution
- `Q::Distribution`: The "approximating" distribution
- `n_points::Int=10000`: Number of quadrature points

# Returns
The KL divergence as a `Float64`. Returns `Inf` if P has support where Q does not.

# Examples

```julia
# KL divergence between two normals
kl = kl_divergence(Normal(0, 1), Normal(1, 1))
# For N(mu1,s1) vs N(mu2,s2): KL = log(s2/s1) + (s1^2 + (mu1-mu2)^2)/(2*s2^2) - 1/2
# Here: KL = 0 + (1 + 1)/2 - 1/2 = 0.5
@assert kl ≈ 0.5 atol=0.05
```
"""
function kl_divergence(P::Distribution, Q::Distribution; n_points::Int=10000)
    # Integration range based on P's support (where f_P > 0)
    lower = quantile(P, 0.00001)
    upper = quantile(P, 0.99999)

    h = (upper - lower) / n_points
    integral = 0.0

    for i in 0:n_points
        x = lower + i * h
        fp = pdf(P, x)
        fq = pdf(Q, x)

        if fp > 1e-300
            if fq < 1e-300
                return Inf  # P has mass where Q doesn't
            end
            weight = (i == 0 || i == n_points) ? 0.5 : 1.0
            integral += weight * fp * log(fp / fq)
        end
    end
    integral *= h

    return max(integral, 0.0)  # Clamp to non-negative (numerical errors can cause small negatives)
end

"""
    fisher_information(dist::Distribution, param::Symbol; delta::Float64=1e-5) -> Float64

Estimate the Fisher information of a distribution with respect to a parameter.

The Fisher information I(theta) measures the amount of information that an observable
random variable X carries about an unknown parameter theta of the distribution:

  I(theta) = E[(d/dtheta log f(X; theta))^2]
           = -E[d^2/dtheta^2 log f(X; theta)]

This function estimates Fisher information numerically using Monte Carlo sampling
and finite-difference approximation of the score function.

The Fisher information is fundamental to:
- Cramer-Rao bound: Var(estimator) >= 1/I(theta)
- Maximum likelihood estimation asymptotics
- Jeffreys prior in Bayesian statistics
- Natural gradient in information geometry

# Arguments
- `dist::Distribution`: The distribution (parameterised at current parameter values)
- `param::Symbol`: Which parameter to compute Fisher information for.
  Supported: `:mean` (location parameter), `:std` (scale parameter)
- `delta::Float64=1e-5`: Step size for finite difference approximation

# Returns
The estimated Fisher information as a `Float64`.

# Examples

```julia
# Fisher information of Normal(0, sigma) for sigma at sigma=1 is 2.0
# (I_sigma = 2/sigma^2)
fi = fisher_information(Normal(0, 1), :std)
@assert fi ≈ 2.0 atol=0.3  # Monte Carlo estimate, so allow some tolerance
```
"""
function fisher_information(dist::Distribution, param::Symbol; delta::Float64=1e-5)
    n_samples = 5000

    # Generate samples from the distribution
    samples = rand(dist, n_samples)

    # Compute score function via finite differences
    scores = Float64[]

    for x in samples
        # Create perturbed distributions
        if param == :mean
            if isa(dist, Normal)
                mu, sigma = params(dist)
                dist_plus = Normal(mu + delta, sigma)
                dist_minus = Normal(mu - delta, sigma)
            else
                @warn "Fisher information for :mean only supported for Normal distribution"
                return NaN
            end
        elseif param == :std
            if isa(dist, Normal)
                mu, sigma = params(dist)
                dist_plus = Normal(mu, sigma + delta)
                dist_minus = Normal(mu, sigma - delta)
            else
                @warn "Fisher information for :std only supported for Normal distribution"
                return NaN
            end
        else
            error("Unknown parameter: $param. Supported: :mean, :std")
        end

        # Score = d/dtheta log f(x; theta) ≈ (log f(x; theta+delta) - log f(x; theta-delta)) / (2*delta)
        lp_plus = logpdf(dist_plus, x)
        lp_minus = logpdf(dist_minus, x)
        score = (lp_plus - lp_minus) / (2.0 * delta)
        push!(scores, score)
    end

    # Fisher information = E[score^2]
    return mean(scores .^ 2)
end

"""
    entropy_contribution(event::ContinuousZeroProbEvent) -> Float64

Compute the Shannon entropy contribution of a zero-probability event.

While a single point has zero probability and thus zero direct entropy contribution,
the local entropy density (negative log-density weighted by density) provides a
meaningful measure of information content at that point.

The local entropy density at point x is defined as:
  h(x) = -f(x) * log(f(x))

where f(x) is the PDF. This quantity is the integrand of the differential entropy:
  H(X) = -integral f(x) log(f(x)) dx

Points with moderate density contribute the most entropy (the function -t*log(t) is
maximised at t = 1/e), while points with very low or very high density contribute less.

# Arguments
- `event::ContinuousZeroProbEvent`: The event to evaluate

# Returns
The local entropy density h(x) = -f(x) * log(f(x)) as a `Float64`.
Returns 0.0 if the density at the point is zero.

# Examples

```julia
# At the peak of N(0,1), f(0) ≈ 0.3989
# h(0) = -0.3989 * log(0.3989) ≈ 0.3665
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
h = entropy_contribution(event)
@assert h > 0.0
```
"""
function entropy_contribution(event::ContinuousZeroProbEvent)
    f = pdf(event.distribution, event.point)
    if f <= 0.0
        return 0.0
    end
    # Local entropy density: -f(x) * log(f(x))
    return -f * log(f)
end

# ============================================================================
# Almost-Sure Verification
# ============================================================================

"""
    almost_surely(predicate::Function, distribution::Distribution;
                  n_samples::Int=10000) -> Bool

Verify via Monte Carlo sampling that a predicate holds "almost surely" (with probability 1)
for the given distribution.

A property holds almost surely (a.s.) if P({omega : predicate(X(omega)) = false}) = 0.
Since we cannot verify this exactly, this function draws `n_samples` from the distribution
and checks whether the predicate holds for all of them.

If the predicate fails on any sample, it is NOT almost surely true (though a single
failure in a Monte Carlo test is a strong signal). If it passes all samples, we have
high confidence (but not certainty) that it holds a.s.

# Arguments
- `predicate::Function`: A function `f(x) -> Bool` to test on each sample
- `distribution::Distribution`: The distribution to sample from
- `n_samples::Int=10000`: Number of Monte Carlo samples to draw

# Returns
`true` if the predicate holds for all `n_samples` samples, `false` otherwise.

# Examples

```julia
# "A random real in [0,1] is less than 2" -- almost surely (and surely) true
@assert almost_surely(x -> x < 2.0, Uniform(0, 1))

# "A random real in [0,1] is irrational" -- almost surely true
# (but we can't really test irrationality of Float64 values)
@assert almost_surely(x -> x > 0.0, Uniform(0, 1), n_samples=10000)
```
"""
function almost_surely(predicate::Function, distribution::Distribution;
                        n_samples::Int=10000)
    samples = rand(distribution, n_samples)
    return all(predicate, samples)
end

# ============================================================================
# Measure-Zero Testing
# ============================================================================

"""
    measure_zero_test(set_indicator::Function, dim::Int;
                      method::Symbol=:box_counting,
                      n_points::Int=10000) -> Bool

Test whether a set has measure zero (Lebesgue measure) in the given ambient dimension.

A set S has Lebesgue measure zero in R^d if, for any epsilon > 0, S can be covered
by a countable collection of d-dimensional boxes whose total volume is less than epsilon.

This function uses the box-counting method: if the estimated Hausdorff dimension
of the set is strictly less than the ambient dimension, the set has measure zero.

# Arguments
- `set_indicator::Function`: A function `f(x::Vector{Float64}) -> Bool` that returns
  `true` if x is in the set.
- `dim::Int`: The ambient dimension of the space.
- `method::Symbol=:box_counting`: The method to use. Currently only `:box_counting` is supported.
- `n_points::Int=10000`: Number of sample points for the box-counting estimate.

# Returns
`true` if the set appears to have measure zero, `false` otherwise.

# Examples

```julia
# A single point in R^2 has measure zero
point_set(x) = norm(x) < 0.001
@assert measure_zero_test(point_set, 2)

# A filled square in R^2 does NOT have measure zero
square(x) = 0.0 <= x[1] <= 1.0 && 0.0 <= x[2] <= 1.0
@assert !measure_zero_test(square, 2)
```
"""
function measure_zero_test(set_indicator::Function, dim::Int;
                            method::Symbol=:box_counting,
                            n_points::Int=10000)
    if method != :box_counting
        error("Unknown method: $method. Only :box_counting is currently supported.")
    end

    @assert dim >= 1 "Ambient dimension must be at least 1"

    # Estimate the Hausdorff (box-counting) dimension
    estimated_dim = hausdorff_dimension(set_indicator, dim; n_boxes=n_points)

    # If the estimated dimension is strictly less than the ambient dimension,
    # the set has Lebesgue measure zero in that ambient space.
    # Use a threshold to account for numerical estimation error.
    return estimated_dim < (dim - 0.3)
end
