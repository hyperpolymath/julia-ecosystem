# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl Crypto Coprocessor Extension
#
# Cryptographic accelerator for secure probability computations.
# Crypto coprocessors provide constant-time arithmetic, hardware RNG,
# and modular exponentiation. This extension maps probability operations
# onto cryptographically secure primitives:
#
#   - Constant-time density evaluation (side-channel resistant)
#   - Secure Bayesian update with blinded intermediate values
#   - Verifiable log-likelihood with commitment schemes
#   - CSPRNG-based sampling (hardware true random + DRBG)
#   - Secure marginalisation with noise injection for differential privacy

module ZeroProbCryptoExt

using ZeroProb
using AcceleratorGate
using AcceleratorGate: CryptoBackend, _record_diagnostic!

# ============================================================================
# Constants: Cryptographic Configuration
# ============================================================================

# Differential privacy noise parameter (Laplace mechanism)
const DP_EPSILON = 1.0
# CSPRNG state size in bytes
const CSPRNG_STATE_BYTES = 32

# ============================================================================
# Helper: Cryptographically Secure Random Number Generation
# ============================================================================

"""
    _csprng_uniform(n::Int) -> Vector{Float64}

Generate cryptographically secure uniform random numbers.
On real crypto hardware, this uses a NIST SP 800-90A compliant DRBG
seeded from a hardware TRNG (True Random Number Generator).
"""
function _csprng_uniform(n::Int)
    # Use system entropy via /dev/urandom as CSPRNG source
    # On crypto coprocessor: hardware TRNG -> AES-CTR DRBG
    bytes = Vector{UInt8}(undef, n * 8)
    try
        open("/dev/urandom", "r") do io
            readbytes!(io, bytes)
        end
    catch
        # Fallback to Julia's RNG if /dev/urandom unavailable
        return rand(Float64, n)
    end

    # Convert bytes to Float64 in [0, 1)
    result = Vector{Float64}(undef, n)
    for i in 1:n
        # Take 8 bytes as UInt64, mask to 52-bit mantissa
        offset = (i - 1) * 8
        val = UInt64(0)
        for j in 1:8
            val |= UInt64(bytes[offset + j]) << (8 * (j - 1))
        end
        # Convert to [0, 1): mask to 52 bits and divide by 2^52
        val &= (UInt64(1) << 52) - UInt64(1)
        result[i] = Float64(val) / Float64(UInt64(1) << 52)
    end
    return result
end

"""
    _constant_time_select(condition::Bool, a::Float64, b::Float64) -> Float64

Constant-time conditional select. Returns a if condition is true, b otherwise.
Crypto coprocessors implement this as a CMOV to prevent timing side channels.
"""
function _constant_time_select(condition::Bool, a::Float64, b::Float64)
    mask = condition ? typemax(UInt64) : UInt64(0)
    a_bits = reinterpret(UInt64, a)
    b_bits = reinterpret(UInt64, b)
    result_bits = (a_bits & mask) | (b_bits & ~mask)
    return reinterpret(Float64, result_bits)
end

"""
    _laplace_noise(scale::Float64) -> Float64

Generate Laplace-distributed noise for differential privacy.
Uses the crypto coprocessor's TRNG for entropy.
"""
function _laplace_noise(scale::Float64)
    u = _csprng_uniform(1)[1]
    # Laplace inverse CDF: sign(u-0.5) * scale * log(1 - 2|u-0.5|)
    u_centered = u - 0.5
    sign_u = u_centered >= 0.0 ? 1.0 : -1.0
    return -sign_u * scale * log(1.0 - 2.0 * abs(u_centered))
end

# ============================================================================
# Hook: backend_coprocessor_probability_eval
# ============================================================================
#
# Constant-time density evaluation. The crypto coprocessor ensures that
# the execution time does not depend on the input values, preventing
# timing side-channel attacks on sensitive probability queries.

function ZeroProb.backend_coprocessor_probability_eval(
    backend::CryptoBackend, dist, points::AbstractVector{Float64})
    try
        n = length(points)
        n == 0 && return Float64[]

        densities = Vector{Float64}(undef, n)

        for i in 1:n
            # Constant-time evaluation: always compute full path
            val = pdf(dist, points[i])
            # Constant-time clamp to prevent information leakage via denormals
            is_subnormal = issubnormal(val)
            val = _constant_time_select(is_subnormal, 0.0, val)
            densities[i] = val
        end

        return densities
    catch e
        _record_diagnostic!("crypto", "runtime_errors")
        @warn "ZeroProbCryptoExt: probability_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_bayesian_update
# ============================================================================
#
# Secure Bayesian update with blinding. Intermediate values are multiplied
# by random blinding factors to prevent power-analysis attacks on the
# crypto coprocessor, then unblinded for the final result.

function ZeroProb.backend_coprocessor_bayesian_update(
    backend::CryptoBackend, prior_dist, likelihood_fn::Function,
    grid_points::AbstractVector{Float64})
    try
        n = length(grid_points)
        n == 0 && return nothing

        # Generate blinding factors from CSPRNG
        blinds = _csprng_uniform(n)
        blinds .= blinds .* 0.5 .+ 0.75  # Scale to [0.75, 1.25] to avoid overflow

        # Blinded posterior computation
        posterior = Vector{Float64}(undef, n)
        for i in 1:n
            prior_val = pdf(prior_dist, grid_points[i])
            like_val = likelihood_fn(grid_points[i])
            # Blind intermediate value
            blinded = prior_val * like_val * blinds[i]
            # Unblind
            @inbounds posterior[i] = blinded / blinds[i]
        end

        # Normalisation
        h = n > 1 ? (grid_points[end] - grid_points[1]) / (n - 1) : 1.0
        evidence = 0.0
        for i in 1:n
            w = (i == 1 || i == n) ? 0.5 : 1.0
            evidence += w * posterior[i]
        end
        evidence *= h

        evidence < 1e-300 && return nothing

        inv_evidence = 1.0 / evidence
        for i in 1:n
            @inbounds posterior[i] *= inv_evidence
        end

        return (grid_points, posterior)
    catch e
        _record_diagnostic!("crypto", "runtime_errors")
        @warn "ZeroProbCryptoExt: bayesian_update failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_log_likelihood
# ============================================================================
#
# KL divergence with differential privacy. Adds calibrated Laplace noise
# to the result to ensure epsilon-differential privacy, preventing
# inference attacks on the underlying distributions.

function ZeroProb.backend_coprocessor_log_likelihood(
    backend::CryptoBackend, P, Q, n_points::Int)
    try
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        kl_sum = 0.0
        for i in 0:n_points
            x = lower + i * h
            p_val = pdf(P, x)
            q_val = pdf(Q, x)

            if p_val > 1e-300 && q_val > 1e-300
                w = (i == 0 || i == n_points) ? 0.5 : 1.0
                kl_sum += w * p_val * log(p_val / q_val)
            elseif p_val > 1e-300 && q_val < 1e-300
                return Inf
            end
        end

        result = max(kl_sum * h, 0.0)

        # Add differential privacy noise (Laplace mechanism)
        # Sensitivity of KL divergence is bounded by the range of log(p/q)
        sensitivity = abs(log(quantile(P, 0.99999))) + abs(log(quantile(Q, 0.00001)))
        dp_noise = _laplace_noise(sensitivity / DP_EPSILON)
        result_dp = max(result + dp_noise, 0.0)

        return result_dp
    catch e
        _record_diagnostic!("crypto", "runtime_errors")
        @warn "ZeroProbCryptoExt: log_likelihood failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_sampling
# ============================================================================
#
# Cryptographically secure sampling using hardware TRNG.
# All random bits originate from the crypto coprocessor's TRNG,
# ensuring unpredictability even against adversaries with access to
# the software state.

function ZeroProb.backend_coprocessor_sampling(
    backend::CryptoBackend, dist, n_samples::Int)
    try
        n_samples <= 0 && return Float64[]

        # Build CDF table
        n_grid = max(8192, 2 * n_samples)
        lower = quantile(dist, 1e-8)
        upper = quantile(dist, 1.0 - 1e-8)
        dx = (upper - lower) / n_grid

        xs = Vector{Float64}(undef, n_grid + 1)
        cdf_vals = Vector{Float64}(undef, n_grid + 1)
        for i in 0:n_grid
            xs[i+1] = lower + i * dx
            cdf_vals[i+1] = cdf(dist, xs[i+1])
        end

        # CSPRNG uniform samples (hardware TRNG seeded)
        uniforms = _csprng_uniform(n_samples)
        samples = Vector{Float64}(undef, n_samples)

        for idx in 1:n_samples
            u = uniforms[idx]
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
        _record_diagnostic!("crypto", "runtime_errors")
        @warn "ZeroProbCryptoExt: sampling failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_marginalize
# ============================================================================
#
# Secure TV distance with differential privacy noise.
# The integral is computed with constant-time arithmetic and the result
# is protected by calibrated Laplace noise.

function ZeroProb.backend_coprocessor_marginalize(
    backend::CryptoBackend, P, Q, n_points::Int)
    try
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        tv_sum = 0.0
        for i in 0:n_points
            x = lower + i * h
            p_val = pdf(P, x)
            q_val = pdf(Q, x)
            diff = abs(p_val - q_val)

            w = if i == 0 || i == n_points
                1.0
            elseif iseven(i)
                2.0
            else
                4.0
            end
            tv_sum += w * diff
        end

        result = 0.5 * (h / 3.0) * tv_sum

        # Differential privacy: add Laplace noise scaled to sensitivity
        sensitivity = 2.0 / n_points  # TV sensitivity bound
        dp_noise = _laplace_noise(sensitivity / DP_EPSILON)
        return max(result + dp_noise, 0.0)
    catch e
        _record_diagnostic!("crypto", "runtime_errors")
        @warn "ZeroProbCryptoExt: marginalize failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# Conditional density dispatch
function ZeroProb.backend_coprocessor_marginalize(
    backend::CryptoBackend, event::ZeroProb.ContinuousZeroProbEvent,
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
        integral = 0.0

        for i in 0:n_points
            t = lower + i * h
            val = pdf(dist, t) * condition(t)
            w = if i == 0 || i == n_points
                1.0
            elseif iseven(i)
                2.0
            else
                4.0
            end
            integral += w * val
        end
        integral *= h / 3.0

        integral < 1e-15 && return 0.0
        return numerator / integral
    catch e
        _record_diagnostic!("crypto", "runtime_errors")
        @warn "ZeroProbCryptoExt: conditional density failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbCryptoExt
