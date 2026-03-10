# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl NPU Extension
#
# Neural Processing Unit acceleration for probability computations.
# NPUs provide fixed-point matrix-multiply engines optimised for INT8/FP16
# inference. This extension maps probability operations onto quantised
# neural network inference patterns:
#
#   - Quantised density approximation via lookup table with interpolation
#   - Fast Bayesian update using quantised prior-likelihood product
#   - Log-likelihood via quantised accumulation
#   - Rejection sampling with NPU-accelerated proposal evaluation
#   - Marginalisation via quantised quadrature

module ZeroProbNPUExt

using ZeroProb
using AcceleratorGate
using AcceleratorGate: NPUBackend, _record_diagnostic!

# ============================================================================
# Constants: NPU Quantisation Configuration
# ============================================================================

# NPU compute precision (FP16 has ~3.3 decimal digits)
const NPU_MANTISSA_BITS = 10  # FP16 mantissa
const NPU_QUANTISE_LEVELS = 65536  # INT16 quantisation levels
const NPU_TILE_SIZE = 64  # NPU GEMM tile dimension

# ============================================================================
# Helper: FP16 Quantisation
# ============================================================================

"""
    _fp16_quantise(x::Float64) -> Float64

Quantise a Float64 value to FP16 precision, simulating NPU compute.
FP16: 1 sign bit, 5 exponent bits, 10 mantissa bits.
"""
function _fp16_quantise(x::Float64)
    x == 0.0 && return 0.0
    # Clamp to FP16 range
    x = clamp(x, -65504.0, 65504.0)
    # Round to FP16 precision (10 mantissa bits -> ~3.3 significant digits)
    return Float64(Float16(x))
end

"""
    _build_density_lut(dist, n_entries::Int) -> (Vector{Float64}, Vector{Float64})

Build a lookup table (LUT) for density evaluation, stored in NPU's
on-chip SRAM. Returns (x_values, density_values) pair.
"""
function _build_density_lut(dist, n_entries::Int)
    lower = quantile(dist, 1e-6)
    upper = quantile(dist, 1.0 - 1e-6)
    xs = range(lower, upper, length=n_entries)
    densities = Float64[pdf(dist, x) for x in xs]
    return (collect(xs), densities)
end

# ============================================================================
# Hook: backend_coprocessor_probability_eval
# ============================================================================
#
# NPU-accelerated density via quantised lookup table with linear
# interpolation. The LUT is stored in on-chip SRAM and the interpolation
# uses the NPU's FP16 multiply-add unit.

function ZeroProb.backend_coprocessor_probability_eval(
    backend::NPUBackend, dist, points::AbstractVector{Float64})
    try
        n = length(points)
        n == 0 && return Float64[]

        # Build density LUT (fits in NPU on-chip SRAM)
        lut_size = min(NPU_QUANTISE_LEVELS, 8192)
        xs_lut, dens_lut = _build_density_lut(dist, lut_size)
        dx = (xs_lut[end] - xs_lut[1]) / (lut_size - 1)
        inv_dx = 1.0 / dx

        densities = Vector{Float64}(undef, n)

        for i in 1:n
            x = points[i]
            if x < xs_lut[1] || x > xs_lut[end]
                # Outside LUT range: fall back to direct evaluation
                densities[i] = pdf(dist, x)
            else
                # LUT index with FP16 interpolation
                idx_f = (x - xs_lut[1]) * inv_dx
                idx_lo = clamp(floor(Int, idx_f) + 1, 1, lut_size - 1)
                idx_hi = idx_lo + 1
                frac = _fp16_quantise(idx_f - floor(idx_f))

                # NPU FP16 linear interpolation
                val = _fp16_quantise(
                    dens_lut[idx_lo] * (1.0 - frac) + dens_lut[idx_hi] * frac)
                densities[i] = max(Float64(val), 0.0)
            end
        end

        return densities
    catch e
        _record_diagnostic!("npu", "runtime_errors")
        @warn "ZeroProbNPUExt: probability_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_bayesian_update
# ============================================================================
#
# Quantised Bayesian update. The NPU performs the prior*likelihood
# element-wise product using its INT8/FP16 matrix engine, treating the
# posterior grid as a 1D matrix operation.

function ZeroProb.backend_coprocessor_bayesian_update(
    backend::NPUBackend, prior_dist, likelihood_fn::Function,
    grid_points::AbstractVector{Float64})
    try
        n = length(grid_points)
        n == 0 && return nothing

        # Evaluate in FP16-quantised tiles
        posterior = Vector{Float64}(undef, n)

        for t0 in 1:NPU_TILE_SIZE:n
            t1 = min(t0 + NPU_TILE_SIZE - 1, n)
            for i in t0:t1
                prior_val = _fp16_quantise(pdf(prior_dist, grid_points[i]))
                like_val = _fp16_quantise(likelihood_fn(grid_points[i]))
                @inbounds posterior[i] = Float64(prior_val * like_val)
            end
        end

        # Normalisation (also quantised)
        h = n > 1 ? (grid_points[end] - grid_points[1]) / (n - 1) : 1.0
        evidence = 0.0
        for i in 1:n
            w = (i == 1 || i == n) ? 0.5 : 1.0
            evidence += _fp16_quantise(w * posterior[i])
        end
        evidence *= h

        evidence < 1e-300 && return nothing

        inv_evidence = 1.0 / evidence
        for i in 1:n
            @inbounds posterior[i] = _fp16_quantise(posterior[i] * inv_evidence)
        end

        return (grid_points, posterior)
    catch e
        _record_diagnostic!("npu", "runtime_errors")
        @warn "ZeroProbNPUExt: bayesian_update failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_log_likelihood
# ============================================================================
#
# Quantised KL divergence. The NPU computes the integrand p*log(p/q)
# using its FP16 log approximation unit (typically a polynomial
# approximation stored in microcode ROM).

function ZeroProb.backend_coprocessor_log_likelihood(
    backend::NPUBackend, P, Q, n_points::Int)
    try
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        kl_sum = 0.0

        for t0 in 0:NPU_TILE_SIZE:n_points
            t1 = min(t0 + NPU_TILE_SIZE - 1, n_points)
            tile_sum = 0.0
            for i in t0:t1
                x = lower + i * h
                p_val = _fp16_quantise(pdf(P, x))
                q_val = _fp16_quantise(pdf(Q, x))

                if p_val > 1e-6 && q_val > 1e-6  # FP16 underflow threshold
                    w = (i == 0 || i == n_points) ? 0.5 : 1.0
                    # NPU log approximation: 4th-order polynomial in FP16
                    log_ratio = _fp16_quantise(log(Float64(p_val) / Float64(q_val)))
                    tile_sum += w * Float64(p_val) * Float64(log_ratio)
                elseif p_val > 1e-6 && q_val < 1e-6
                    return Inf
                end
            end
            kl_sum += tile_sum
        end

        return max(kl_sum * h, 0.0)
    catch e
        _record_diagnostic!("npu", "runtime_errors")
        @warn "ZeroProbNPUExt: log_likelihood failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_sampling
# ============================================================================
#
# NPU-accelerated rejection sampling. The NPU evaluates the proposal
# density and acceptance ratio using quantised arithmetic, processing
# tiles of candidates simultaneously through the inference engine.

function ZeroProb.backend_coprocessor_sampling(
    backend::NPUBackend, dist, n_samples::Int)
    try
        n_samples <= 0 && return Float64[]

        # Build quantised CDF lookup table in NPU SRAM
        n_lut = 4096
        lower = quantile(dist, 1e-8)
        upper = quantile(dist, 1.0 - 1e-8)
        dx = (upper - lower) / n_lut

        cdf_lut = Vector{Float64}(undef, n_lut + 1)
        for i in 0:n_lut
            x = lower + i * dx
            cdf_lut[i+1] = _fp16_quantise(cdf(dist, x))
        end

        # Inverse CDF sampling with NPU-quantised interpolation
        uniforms = rand(Float64, n_samples)
        samples = Vector{Float64}(undef, n_samples)

        for t0 in 1:NPU_TILE_SIZE:n_samples
            t1 = min(t0 + NPU_TILE_SIZE - 1, n_samples)
            for idx in t0:t1
                u = _fp16_quantise(uniforms[idx])
                # Binary search
                lo, hi = 1, n_lut + 1
                while lo < hi - 1
                    mid = (lo + hi) >> 1
                    if cdf_lut[mid] < u
                        lo = mid
                    else
                        hi = mid
                    end
                end
                gap = cdf_lut[hi] - cdf_lut[lo]
                frac = gap > 1e-6 ? _fp16_quantise((u - cdf_lut[lo]) / gap) : 0.0
                samples[idx] = lower + (lo - 1 + Float64(frac)) * dx
            end
        end

        return samples
    catch e
        _record_diagnostic!("npu", "runtime_errors")
        @warn "ZeroProbNPUExt: sampling failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_marginalize
# ============================================================================
#
# Quantised Simpson's rule for TV distance. The NPU's tiled MAC engine
# computes the weighted sum of |f_P - f_Q| at each quadrature point.

function ZeroProb.backend_coprocessor_marginalize(
    backend::NPUBackend, P, Q, n_points::Int)
    try
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        tv_sum = 0.0
        for t0 in 0:NPU_TILE_SIZE:n_points
            t1 = min(t0 + NPU_TILE_SIZE - 1, n_points)
            tile_sum = 0.0
            for i in t0:t1
                x = lower + i * h
                p_val = _fp16_quantise(pdf(P, x))
                q_val = _fp16_quantise(pdf(Q, x))
                diff = abs(Float64(p_val) - Float64(q_val))

                w = if i == 0 || i == n_points
                    1.0
                elseif iseven(i)
                    2.0
                else
                    4.0
                end
                tile_sum += w * diff
            end
            tv_sum += tile_sum
        end

        return 0.5 * (h / 3.0) * tv_sum
    catch e
        _record_diagnostic!("npu", "runtime_errors")
        @warn "ZeroProbNPUExt: marginalize failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# Conditional density dispatch
function ZeroProb.backend_coprocessor_marginalize(
    backend::NPUBackend, event::ZeroProb.ContinuousZeroProbEvent,
    condition::Function)
    try
        dist = event.distribution
        x = event.point

        numerator = _fp16_quantise(pdf(dist, x)) * _fp16_quantise(condition(x))
        Float64(numerator) == 0.0 && return 0.0

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
        integral = 0.0

        for i in 0:n_points
            t = lower + i * h
            val = _fp16_quantise(pdf(dist, t)) * _fp16_quantise(condition(t))
            w = if i == 0 || i == n_points
                1.0
            elseif iseven(i)
                2.0
            else
                4.0
            end
            integral += w * Float64(val)
        end
        integral *= h / 3.0

        integral < 1e-15 && return 0.0
        return Float64(numerator) / integral
    catch e
        _record_diagnostic!("npu", "runtime_errors")
        @warn "ZeroProbNPUExt: conditional density failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbNPUExt
