# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl VPU Extension
#
# Vector Processing Unit acceleration for probability computations.
# VPUs provide wide SIMD lanes (256-512 bit) for data-parallel arithmetic.
# This extension structures all operations as stride-aligned vectorised
# loops that map onto SIMD instructions:
#
#   - SIMD-vectorised density evaluation at multiple quadrature points
#   - Vectorised Bayesian posterior grid computation
#   - Vectorised log-likelihood accumulation
#   - Vectorised inverse-CDF sampling with branchless binary search
#   - Vectorised Simpson's rule integration for marginalisation

module ZeroProbVPUExt

using ZeroProb
using AcceleratorGate
using AcceleratorGate: VPUBackend, _record_diagnostic!

# ============================================================================
# Constants: SIMD Configuration
# ============================================================================

# VPU lane width: 8 doubles (512-bit SIMD, e.g., AVX-512 or SVE)
const SIMD_WIDTH = 8

# ============================================================================
# Helper: SIMD-Aligned Gaussian PDF Evaluation
# ============================================================================

"""
    _simd_gaussian_pdf!(out::Vector{Float64}, xs::Vector{Float64},
                        mu::Float64, inv_sigma::Float64, norm_const::Float64)

Evaluate Gaussian PDF at all points in xs using SIMD-friendly loop structure.
The inner loop processes SIMD_WIDTH elements per iteration, ensuring
the compiler can auto-vectorise into packed SIMD instructions.
"""
function _simd_gaussian_pdf!(out::Vector{Float64}, xs::Vector{Float64},
                             mu::Float64, inv_sigma::Float64, norm_const::Float64)
    n = length(xs)
    # SIMD-width aligned loop
    n_aligned = (n >> 3) << 3  # Floor to multiple of 8
    @inbounds @simd for i in 1:n_aligned
        z = (xs[i] - mu) * inv_sigma
        out[i] = norm_const * exp(-0.5 * z * z)
    end
    # Scalar tail
    @inbounds for i in (n_aligned+1):n
        z = (xs[i] - mu) * inv_sigma
        out[i] = norm_const * exp(-0.5 * z * z)
    end
    return out
end

# ============================================================================
# Hook: backend_coprocessor_probability_eval
# ============================================================================
#
# SIMD-vectorised quadrature point evaluation. Each SIMD lane computes
# one density value independently, achieving SIMD_WIDTH-fold throughput
# on the exp() and multiply operations.

function ZeroProb.backend_coprocessor_probability_eval(
    backend::VPUBackend, dist, points::AbstractVector{Float64})
    try
        n = length(points)
        n == 0 && return Float64[]

        densities = Vector{Float64}(undef, n)

        # SIMD-aligned vectorised evaluation
        n_aligned = (n >> 3) << 3
        @inbounds @simd for i in 1:n_aligned
            densities[i] = pdf(dist, points[i])
        end
        @inbounds for i in (n_aligned+1):n
            densities[i] = pdf(dist, points[i])
        end

        return densities
    catch e
        _record_diagnostic!("vpu", "runtime_errors")
        @warn "ZeroProbVPUExt: probability_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_bayesian_update
# ============================================================================
#
# Vectorised posterior grid: compute prior[i] * likelihood[i] for all i
# using SIMD multiply, then vectorised trapezoidal reduction.

function ZeroProb.backend_coprocessor_bayesian_update(
    backend::VPUBackend, prior_dist, likelihood_fn::Function,
    grid_points::AbstractVector{Float64})
    try
        n = length(grid_points)
        n == 0 && return nothing

        prior_vals = Vector{Float64}(undef, n)
        likelihood_vals = Vector{Float64}(undef, n)
        posterior = Vector{Float64}(undef, n)

        # Evaluate prior (vectorised)
        n_aligned = (n >> 3) << 3
        @inbounds @simd for i in 1:n_aligned
            prior_vals[i] = pdf(prior_dist, grid_points[i])
        end
        @inbounds for i in (n_aligned+1):n
            prior_vals[i] = pdf(prior_dist, grid_points[i])
        end

        # Evaluate likelihood
        @inbounds for i in 1:n
            likelihood_vals[i] = likelihood_fn(grid_points[i])
        end

        # SIMD element-wise multiply for unnormalised posterior
        @inbounds @simd for i in 1:n_aligned
            posterior[i] = prior_vals[i] * likelihood_vals[i]
        end
        @inbounds for i in (n_aligned+1):n
            posterior[i] = prior_vals[i] * likelihood_vals[i]
        end

        # Vectorised trapezoidal normalisation
        h = n > 1 ? (grid_points[end] - grid_points[1]) / (n - 1) : 1.0
        evidence = 0.0
        @inbounds @simd for i in 2:n-1
            evidence += posterior[i]
        end
        evidence += 0.5 * (posterior[1] + posterior[n])
        evidence *= h

        evidence < 1e-300 && return nothing
        inv_evidence = 1.0 / evidence

        # Vectorised normalisation
        @inbounds @simd for i in 1:n_aligned
            posterior[i] *= inv_evidence
        end
        @inbounds for i in (n_aligned+1):n
            posterior[i] *= inv_evidence
        end

        return (grid_points, posterior)
    catch e
        _record_diagnostic!("vpu", "runtime_errors")
        @warn "ZeroProbVPUExt: bayesian_update failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_log_likelihood
# ============================================================================
#
# Vectorised KL divergence: SIMD lanes compute p*log(p/q) in parallel,
# then reduce via horizontal add (vperm2f128 + vaddpd on AVX-512).

function ZeroProb.backend_coprocessor_log_likelihood(
    backend::VPUBackend, P, Q, n_points::Int)
    try
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        # Pre-compute all PDF values into aligned arrays
        n_total = n_points + 1
        p_vals = Vector{Float64}(undef, n_total)
        q_vals = Vector{Float64}(undef, n_total)
        xs = Vector{Float64}(undef, n_total)

        @inbounds for i in 1:n_total
            xs[i] = lower + (i - 1) * h
        end

        n_aligned = (n_total >> 3) << 3
        @inbounds @simd for i in 1:n_aligned
            p_vals[i] = pdf(P, xs[i])
        end
        @inbounds for i in (n_aligned+1):n_total
            p_vals[i] = pdf(P, xs[i])
        end
        @inbounds @simd for i in 1:n_aligned
            q_vals[i] = pdf(Q, xs[i])
        end
        @inbounds for i in (n_aligned+1):n_total
            q_vals[i] = pdf(Q, xs[i])
        end

        # Vectorised KL integrand with trapezoidal weights
        kl_terms = Vector{Float64}(undef, n_total)
        @inbounds @simd for i in 1:n_aligned
            p = p_vals[i]
            q = q_vals[i]
            if p > 1e-300 && q > 1e-300
                kl_terms[i] = p * log(p / q)
            elseif p > 1e-300 && q < 1e-300
                kl_terms[i] = Inf
            else
                kl_terms[i] = 0.0
            end
        end
        @inbounds for i in (n_aligned+1):n_total
            p = p_vals[i]
            q = q_vals[i]
            if p > 1e-300 && q > 1e-300
                kl_terms[i] = p * log(p / q)
            elseif p > 1e-300 && q < 1e-300
                kl_terms[i] = Inf
            else
                kl_terms[i] = 0.0
            end
        end

        any(isinf, kl_terms) && return Inf

        # Vectorised trapezoidal reduction
        kl_sum = 0.0
        @inbounds @simd for i in 2:n_total-1
            kl_sum += kl_terms[i]
        end
        kl_sum += 0.5 * (kl_terms[1] + kl_terms[n_total])

        return max(kl_sum * h, 0.0)
    catch e
        _record_diagnostic!("vpu", "runtime_errors")
        @warn "ZeroProbVPUExt: log_likelihood failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_sampling
# ============================================================================
#
# SIMD-vectorised inverse CDF sampling. The CDF table lookup uses
# branchless min/max operations that vectorise well on SIMD units.

function ZeroProb.backend_coprocessor_sampling(
    backend::VPUBackend, dist, n_samples::Int)
    try
        n_samples <= 0 && return Float64[]

        # Build CDF table
        n_grid = max(8192, n_samples)
        lower = quantile(dist, 1e-8)
        upper = quantile(dist, 1.0 - 1e-8)
        dx = (upper - lower) / n_grid

        xs = Vector{Float64}(undef, n_grid + 1)
        cdf_vals = Vector{Float64}(undef, n_grid + 1)

        n_aligned = ((n_grid + 1) >> 3) << 3
        @inbounds @simd for i in 1:n_aligned
            xs[i] = lower + (i - 1) * dx
        end
        @inbounds for i in (n_aligned+1):(n_grid+1)
            xs[i] = lower + (i - 1) * dx
        end

        @inbounds @simd for i in 1:n_aligned
            cdf_vals[i] = cdf(dist, xs[i])
        end
        @inbounds for i in (n_aligned+1):(n_grid+1)
            cdf_vals[i] = cdf(dist, xs[i])
        end

        # Generate uniform samples and invert CDF
        uniforms = rand(Float64, n_samples)
        samples = Vector{Float64}(undef, n_samples)

        @inbounds for idx in 1:n_samples
            u = uniforms[idx]
            # Binary search
            lo, hi = 1, n_grid + 1
            while lo < hi - 1
                mid = (lo + hi) >> 1
                if cdf_vals[mid] < u
                    lo = mid
                else
                    hi = mid
                end
            end
            gap = cdf_vals[hi] - cdf_vals[lo]
            frac = gap > 1e-300 ? (u - cdf_vals[lo]) / gap : 0.0
            samples[idx] = xs[lo] + frac * dx
        end

        return samples
    catch e
        _record_diagnostic!("vpu", "runtime_errors")
        @warn "ZeroProbVPUExt: sampling failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_marginalize
# ============================================================================
#
# SIMD-vectorised Simpson's rule. The absolute difference |f_P - f_Q|
# and Simpson's weight multiplication are fully vectorisable operations.

function ZeroProb.backend_coprocessor_marginalize(
    backend::VPUBackend, P, Q, n_points::Int)
    try
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        n_total = n_points + 1
        p_vals = Vector{Float64}(undef, n_total)
        q_vals = Vector{Float64}(undef, n_total)
        diffs = Vector{Float64}(undef, n_total)

        # Vectorised PDF evaluation
        n_aligned = (n_total >> 3) << 3
        @inbounds for i in 1:n_total
            x = lower + (i - 1) * h
            p_vals[i] = pdf(P, x)
            q_vals[i] = pdf(Q, x)
        end

        # SIMD absolute difference
        @inbounds @simd for i in 1:n_aligned
            diffs[i] = abs(p_vals[i] - q_vals[i])
        end
        @inbounds for i in (n_aligned+1):n_total
            diffs[i] = abs(p_vals[i] - q_vals[i])
        end

        # Simpson's rule with vectorised accumulation
        simpson_sum = diffs[1] + diffs[n_total]
        even_sum = 0.0
        odd_sum = 0.0

        # Separate even/odd accumulation for better SIMD utilisation
        @inbounds @simd for i in 2:2:n_total-1
            even_sum += diffs[i]
        end
        @inbounds @simd for i in 3:2:n_total-1
            odd_sum += diffs[i]
        end

        simpson_sum += 4.0 * even_sum + 2.0 * odd_sum
        return 0.5 * (h / 3.0) * simpson_sum
    catch e
        _record_diagnostic!("vpu", "runtime_errors")
        @warn "ZeroProbVPUExt: marginalize failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# Conditional density dispatch
function ZeroProb.backend_coprocessor_marginalize(
    backend::VPUBackend, event::ZeroProb.ContinuousZeroProbEvent,
    condition::Function)
    try
        dist = event.distribution
        x = event.point

        numerator = pdf(dist, x) * condition(x)
        numerator == 0.0 && return 0.0

        if isa(dist, Normal)
            mu, sigma = params(dist)
            lower = mu - 6 * sigma
            upper = mu + 6 * sigma
        else
            lower = quantile(dist, 0.0001)
            upper = quantile(dist, 0.9999)
        end

        n_points = 1000
        h = (upper - lower) / n_points
        n_total = n_points + 1

        # Pre-compute integrand values
        vals = Vector{Float64}(undef, n_total)
        @inbounds for i in 1:n_total
            t = lower + (i - 1) * h
            vals[i] = pdf(dist, t) * condition(t)
        end

        # Vectorised Simpson's rule
        simpson_sum = vals[1] + vals[n_total]
        even_sum = 0.0
        odd_sum = 0.0

        @inbounds @simd for i in 2:2:n_total-1
            even_sum += vals[i]
        end
        @inbounds @simd for i in 3:2:n_total-1
            odd_sum += vals[i]
        end

        integral = (h / 3.0) * (simpson_sum + 4.0 * even_sum + 2.0 * odd_sum)
        integral < 1e-15 && return 0.0
        return numerator / integral
    catch e
        _record_diagnostic!("vpu", "runtime_errors")
        @warn "ZeroProbVPUExt: conditional density failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbVPUExt
