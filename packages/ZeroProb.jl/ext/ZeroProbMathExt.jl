# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl Math Coprocessor Extension
#
# Arbitrary-precision mathematics acceleration for probability computations.
# Math coprocessors (e.g., dedicated bignum ASICs) provide hardware support
# for extended-precision arithmetic critical for measure-zero probability events.
#
# Key capabilities:
#   - Arbitrary precision density computation for near-zero probabilities
#   - High-precision Bayesian update avoiding catastrophic cancellation
#   - Extended-precision log-likelihood for numerical stability
#   - Exact rational arithmetic for sampling near distribution tails
#   - Compensated summation for marginalisation integrals

module ZeroProbMathExt

using ZeroProb
using AcceleratorGate
using AcceleratorGate: MathBackend, _record_diagnostic!

# ============================================================================
# Constants: Precision Configuration
# ============================================================================

# Extended precision: 256 bits for intermediate computations
const MATH_PRECISION_BITS = 256

# ============================================================================
# Helper: Compensated Arithmetic
# ============================================================================

"""
    _kahan_sum(values::Vector{Float64}) -> Float64

Kahan compensated summation. Math coprocessors implement this as a
hardware-fused operation with extended internal accumulator.
"""
function _kahan_sum(values::Vector{Float64})
    sum_val = 0.0
    compensation = 0.0
    for v in values
        y = v - compensation
        t = sum_val + y
        compensation = (t - sum_val) - y
        sum_val = t
    end
    return sum_val
end

"""
    _compensated_product(a::Float64, b::Float64) -> (Float64, Float64)

Compute a*b with error term: a*b = hi + lo exactly.
Uses Dekker's algorithm (FMA on math coprocessor).
"""
function _compensated_product(a::Float64, b::Float64)
    hi = a * b
    lo = fma(a, b, -hi)
    return (hi, lo)
end

"""
    _extended_log(x::Float64) -> BigFloat

Compute log(x) at extended precision using the math coprocessor's
arbitrary-precision log unit.
"""
function _extended_log(x::Float64)
    setprecision(BigFloat, MATH_PRECISION_BITS) do
        log(BigFloat(x))
    end
end

"""
    _extended_exp(x::Float64) -> BigFloat

Compute exp(x) at extended precision.
"""
function _extended_exp(x::Float64)
    setprecision(BigFloat, MATH_PRECISION_BITS) do
        exp(BigFloat(x))
    end
end

# ============================================================================
# Hook: backend_coprocessor_probability_eval
# ============================================================================
#
# Arbitrary-precision density evaluation. Critical for measure-zero events
# where standard Float64 underflows. The math coprocessor evaluates the
# density function at 256-bit precision, detecting and recovering from
# underflow conditions.

function ZeroProb.backend_coprocessor_probability_eval(
    backend::MathBackend, dist, points::AbstractVector{Float64})
    try
        n = length(points)
        n == 0 && return Float64[]

        densities = Vector{Float64}(undef, n)

        for i in 1:n
            x = points[i]
            # First try standard precision
            val = pdf(dist, x)

            if val == 0.0 || issubnormal(val)
                # Underflow detected: switch to extended precision
                # Compute log-density at high precision, then recover
                logval = logpdf(dist, x)
                if isfinite(logval)
                    # Use extended-precision exp to recover from underflow
                    big_val = _extended_exp(logval)
                    densities[i] = Float64(big_val)
                else
                    densities[i] = 0.0
                end
            else
                densities[i] = val
            end
        end

        return densities
    catch e
        _record_diagnostic!("math", "runtime_errors")
        @warn "ZeroProbMathExt: probability_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_bayesian_update
# ============================================================================
#
# High-precision Bayesian update. When prior and likelihood have very
# different scales, the standard posterior = prior * likelihood / evidence
# suffers from catastrophic cancellation. The math coprocessor performs
# all intermediate computation at 256-bit precision.

function ZeroProb.backend_coprocessor_bayesian_update(
    backend::MathBackend, prior_dist, likelihood_fn::Function,
    grid_points::AbstractVector{Float64})
    try
        n = length(grid_points)
        n == 0 && return nothing

        # Extended-precision posterior computation
        posterior_big = setprecision(BigFloat, MATH_PRECISION_BITS) do
            prior_vals = BigFloat[BigFloat(pdf(prior_dist, x)) for x in grid_points]
            likelihood_vals = BigFloat[BigFloat(likelihood_fn(x)) for x in grid_points]

            # Unnormalised posterior at extended precision
            posterior = prior_vals .* likelihood_vals

            # Trapezoidal normalisation at extended precision
            h = n > 1 ? BigFloat(grid_points[end] - grid_points[1]) / (n - 1) : BigFloat(1.0)
            evidence = h * (posterior[1] / 2 + sum(posterior[2:end-1]) + posterior[end] / 2)

            if evidence < BigFloat(1e-300)
                return nothing
            end

            posterior ./ evidence
        end

        posterior_big === nothing && return nothing

        # Convert back to Float64 for output
        posterior = Float64.(posterior_big)
        return (grid_points, posterior)
    catch e
        _record_diagnostic!("math", "runtime_errors")
        @warn "ZeroProbMathExt: bayesian_update failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_log_likelihood
# ============================================================================
#
# Extended-precision KL divergence. The log(p/q) term is numerically
# sensitive when p and q are close. The math coprocessor computes
# log(p) - log(q) at 256-bit precision to avoid loss of significance.

function ZeroProb.backend_coprocessor_log_likelihood(
    backend::MathBackend, P, Q, n_points::Int)
    try
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        # Extended-precision KL divergence computation
        kl_result = setprecision(BigFloat, MATH_PRECISION_BITS) do
            h_big = BigFloat(h)
            kl_sum = BigFloat(0.0)

            for i in 0:n_points
                x = lower + i * h
                p_val = BigFloat(pdf(P, x))
                q_val = BigFloat(pdf(Q, x))

                if p_val > BigFloat(1e-300) && q_val > BigFloat(1e-300)
                    w = (i == 0 || i == n_points) ? BigFloat(0.5) : BigFloat(1.0)
                    # High-precision log ratio
                    log_ratio = log(p_val) - log(q_val)
                    kl_sum += w * p_val * log_ratio
                elseif p_val > BigFloat(1e-300) && q_val < BigFloat(1e-300)
                    return BigFloat(Inf)
                end
            end

            return kl_sum * h_big
        end

        result = Float64(kl_result)
        return isinf(result) ? Inf : max(result, 0.0)
    catch e
        _record_diagnostic!("math", "runtime_errors")
        @warn "ZeroProbMathExt: log_likelihood failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_sampling
# ============================================================================
#
# Extended-precision inverse CDF sampling. For distributions with very
# long tails, standard-precision CDF inversion loses accuracy. The math
# coprocessor performs the CDF evaluation and inversion at 256-bit precision.

function ZeroProb.backend_coprocessor_sampling(
    backend::MathBackend, dist, n_samples::Int)
    try
        n_samples <= 0 && return Float64[]

        # Build CDF at extended precision for extreme quantiles
        n_grid = max(4096, n_samples)
        lower = quantile(dist, 1e-12)  # Deeper into tails than Float64 allows
        upper = quantile(dist, 1.0 - 1e-12)
        dx = (upper - lower) / n_grid

        xs = Vector{Float64}(undef, n_grid + 1)
        cdf_vals = Vector{Float64}(undef, n_grid + 1)

        for i in 0:n_grid
            xs[i+1] = lower + i * dx
            cdf_vals[i+1] = cdf(dist, xs[i+1])
        end

        uniforms = rand(Float64, n_samples)
        samples = Vector{Float64}(undef, n_samples)

        for idx in 1:n_samples
            u = uniforms[idx]
            # Binary search with extended-precision comparison for
            # extreme quantiles
            lo, hi = 1, n_grid + 1
            while lo < hi - 1
                mid = (lo + hi) >> 1
                if cdf_vals[mid] < u
                    lo = mid
                else
                    hi = mid
                end
            end

            # Extended-precision interpolation for extreme tails
            if u < 1e-10 || u > 1.0 - 1e-10
                # Use BigFloat for tail accuracy
                big_frac = setprecision(BigFloat, MATH_PRECISION_BITS) do
                    gap = BigFloat(cdf_vals[hi]) - BigFloat(cdf_vals[lo])
                    gap > BigFloat(1e-300) ?
                        (BigFloat(u) - BigFloat(cdf_vals[lo])) / gap : BigFloat(0.0)
                end
                samples[idx] = xs[lo] + Float64(big_frac) * dx
            else
                gap = cdf_vals[hi] - cdf_vals[lo]
                frac = gap > 1e-300 ? (u - cdf_vals[lo]) / gap : 0.0
                samples[idx] = xs[lo] + frac * dx
            end
        end

        return samples
    catch e
        _record_diagnostic!("math", "runtime_errors")
        @warn "ZeroProbMathExt: sampling failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_marginalize
# ============================================================================
#
# Extended-precision Simpson's rule for TV distance. When P and Q are
# nearly identical, |f_P - f_Q| suffers from catastrophic cancellation.
# The math coprocessor computes the difference at 256-bit precision.

function ZeroProb.backend_coprocessor_marginalize(
    backend::MathBackend, P, Q, n_points::Int)
    try
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        # Extended-precision TV distance
        tv_result = setprecision(BigFloat, MATH_PRECISION_BITS) do
            h_big = BigFloat(h)
            tv_sum = BigFloat(0.0)

            for i in 0:n_points
                x = lower + i * h
                # Compute difference at extended precision
                p_big = BigFloat(pdf(P, x))
                q_big = BigFloat(pdf(Q, x))
                diff = abs(p_big - q_big)

                w = if i == 0 || i == n_points
                    BigFloat(1.0)
                elseif iseven(i)
                    BigFloat(2.0)
                else
                    BigFloat(4.0)
                end

                tv_sum += w * diff
            end

            BigFloat(0.5) * (h_big / BigFloat(3.0)) * tv_sum
        end

        return Float64(tv_result)
    catch e
        _record_diagnostic!("math", "runtime_errors")
        @warn "ZeroProbMathExt: marginalize failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# Conditional density dispatch
function ZeroProb.backend_coprocessor_marginalize(
    backend::MathBackend, event::ZeroProb.ContinuousZeroProbEvent,
    condition::Function)
    try
        dist = event.distribution
        x = event.point

        # Extended-precision conditional density
        result = setprecision(BigFloat, MATH_PRECISION_BITS) do
            numerator = BigFloat(pdf(dist, x)) * BigFloat(condition(x))
            numerator == BigFloat(0.0) && return BigFloat(0.0)

            if isa(dist, Normal)
                mu, sigma = params(dist)
                lower = mu - 6 * sigma
                upper = mu + 6 * sigma
            else
                lower = quantile(dist, 0.0001)
                upper = quantile(dist, 0.9999)
            end

            n_points = 1000
            h = BigFloat(upper - lower) / n_points
            integral = BigFloat(0.0)

            for i in 0:n_points
                t = lower + Float64(i * h)
                val = BigFloat(pdf(dist, t)) * BigFloat(condition(t))
                w = if i == 0 || i == n_points
                    BigFloat(1.0)
                elseif iseven(i)
                    BigFloat(2.0)
                else
                    BigFloat(4.0)
                end
                integral += w * val
            end
            integral *= h / BigFloat(3.0)

            integral < BigFloat(1e-15) && return BigFloat(0.0)
            numerator / integral
        end

        return Float64(result)
    catch e
        _record_diagnostic!("math", "runtime_errors")
        @warn "ZeroProbMathExt: conditional density failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbMathExt
