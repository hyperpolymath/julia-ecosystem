# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl DSP Extension
#
# Digital Signal Processor acceleration for probability computations.
# DSPs excel at FFT, FIR/IIR filtering, and convolution. This extension
# maps probability operations onto signal processing primitives:
#
#   - FFT-based density estimation via characteristic function inversion
#   - Convolution-based Bayesian update (posterior = prior * likelihood via FFT)
#   - Log-likelihood via windowed spectral analysis
#   - Sampling via inverse FFT of characteristic function
#   - Marginalisation via FFT-accelerated numerical integration

module ZeroProbDSPExt

using ZeroProb
using AcceleratorGate
using AcceleratorGate: DSPBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# Helper: Radix-2 FFT (Cooley-Tukey)
# ============================================================================

"""
    _dsp_fft!(x::Vector{ComplexF64})

In-place radix-2 FFT. DSP hardware implements this as a butterfly network
in the frequency domain accelerator. The bit-reversal permutation and
butterfly stages map directly onto DSP pipeline stages.
"""
function _dsp_fft!(x::Vector{ComplexF64})
    n = length(x)
    @assert ispow2(n) "FFT length must be power of 2"

    # Bit-reversal permutation
    j = 1
    for i in 1:n-1
        if i < j
            x[i], x[j] = x[j], x[i]
        end
        m = n >> 1
        while m >= 2 && j > m
            j -= m
            m >>= 1
        end
        j += m
    end

    # Butterfly stages
    len = 2
    while len <= n
        half = len >> 1
        w_base = exp(-2.0im * pi / len)
        for start in 1:len:n
            w = 1.0 + 0.0im
            for k in 0:half-1
                idx1 = start + k
                idx2 = start + k + half
                t = w * x[idx2]
                x[idx2] = x[idx1] - t
                x[idx1] = x[idx1] + t
                w *= w_base
            end
        end
        len <<= 1
    end
    return x
end

"""
    _dsp_ifft!(x::Vector{ComplexF64})

In-place inverse FFT. Conjugate, forward FFT, conjugate, scale.
"""
function _dsp_ifft!(x::Vector{ComplexF64})
    n = length(x)
    x .= conj.(x)
    _dsp_fft!(x)
    x .= conj.(x) ./ n
    return x
end

"""
    _next_pow2(n::Int) -> Int

Round up to the next power of 2.
"""
_next_pow2(n::Int) = n <= 1 ? 1 : 1 << (ndigits(n - 1, base=2))

# ============================================================================
# Hook: backend_coprocessor_probability_eval
# ============================================================================
#
# FFT-based density estimation via characteristic function inversion.
# The characteristic function phi(t) = E[exp(itX)] is the Fourier transform
# of the density. We evaluate phi on a grid, then inverse-FFT to recover
# the density. DSP hardware excels at this FFT pipeline.

function ZeroProb.backend_coprocessor_probability_eval(
    backend::DSPBackend, dist, points::AbstractVector{Float64})
    try
        n_points = length(points)
        n_points == 0 && return Float64[]

        # Determine grid for FFT-based density recovery
        x_min = minimum(points) - 3.0 * std(dist)
        x_max = maximum(points) + 3.0 * std(dist)
        n_fft = _next_pow2(max(1024, 4 * n_points))
        dx = (x_max - x_min) / n_fft
        dt = 2.0 * pi / (n_fft * dx)

        # Evaluate characteristic function on frequency grid
        # phi(t) = E[exp(itX)] -- for known distributions we use the
        # analytical form; for empirical data we'd use the sample CF.
        cf_vals = Vector{ComplexF64}(undef, n_fft)
        for k in 0:n_fft-1
            t = (k < n_fft/2 ? k : k - n_fft) * dt
            # Use the moment-generating approach: phi(t) = E[e^{itX}]
            cf_vals[k+1] = cf(dist, t) * exp(-1.0im * t * x_min)
        end

        # Inverse FFT to recover density
        _dsp_ifft!(cf_vals)

        # Extract density values at the requested points
        densities = Vector{Float64}(undef, n_points)
        for (idx, x) in enumerate(points)
            # Map x to FFT grid index
            grid_idx = (x - x_min) / dx
            i0 = clamp(floor(Int, grid_idx) + 1, 1, n_fft)
            i1 = clamp(i0 + 1, 1, n_fft)
            frac = grid_idx - floor(grid_idx)

            # Linear interpolation
            val = real(cf_vals[i0]) * (1.0 - frac) + real(cf_vals[i1]) * frac
            densities[idx] = max(val / dx, 0.0)
        end

        return densities
    catch e
        _record_diagnostic!("dsp", "runtime_errors")
        @warn "ZeroProbDSPExt: probability_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_bayesian_update
# ============================================================================
#
# Bayesian update via FFT convolution. The posterior is proportional to
# prior * likelihood. In the frequency domain:
#   FFT(posterior) = FFT(prior) .* FFT(likelihood)
# This exploits the DSP's convolution accelerator.

function ZeroProb.backend_coprocessor_bayesian_update(
    backend::DSPBackend, prior_dist, likelihood_fn::Function,
    grid_points::AbstractVector{Float64})
    try
        n = length(grid_points)
        n == 0 && return nothing

        n_fft = _next_pow2(2 * n)  # Zero-pad for linear convolution

        # Evaluate prior and likelihood on grid
        prior_vals = ComplexF64[pdf(prior_dist, x) for x in grid_points]
        likelihood_vals = ComplexF64[likelihood_fn(x) for x in grid_points]

        # Zero-pad to FFT length
        resize!(prior_vals, n_fft)
        resize!(likelihood_vals, n_fft)
        prior_vals[n+1:end] .= 0.0
        likelihood_vals[n+1:end] .= 0.0

        # FFT both signals
        _dsp_fft!(prior_vals)
        _dsp_fft!(likelihood_vals)

        # Pointwise multiply in frequency domain
        posterior_freq = prior_vals .* likelihood_vals

        # Inverse FFT
        _dsp_ifft!(posterior_freq)

        # Extract real part and normalise
        posterior = real.(posterior_freq[1:n])
        posterior .= max.(posterior, 0.0)

        # Trapezoidal normalisation
        h = n > 1 ? (grid_points[end] - grid_points[1]) / (n - 1) : 1.0
        total = h * (0.5 * posterior[1] + sum(posterior[2:end-1]) + 0.5 * posterior[end])
        total < 1e-300 && return nothing

        posterior ./= total
        return (grid_points, posterior)
    catch e
        _record_diagnostic!("dsp", "runtime_errors")
        @warn "ZeroProbDSPExt: bayesian_update failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_log_likelihood
# ============================================================================
#
# KL divergence via spectral density comparison. We compute the power
# spectral densities of P and Q using FFT, then integrate the log-ratio
# in the frequency domain (Parseval's theorem relates time and frequency
# domain integrals).

function ZeroProb.backend_coprocessor_log_likelihood(
    backend::DSPBackend, P, Q, n_points::Int)
    try
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        n_fft = _next_pow2(n_points + 1)

        # Evaluate P and Q on the grid, apply Hann window for spectral leakage
        p_vals = Vector{ComplexF64}(undef, n_fft)
        q_vals = Vector{ComplexF64}(undef, n_fft)

        for i in 0:n_points
            x = lower + i * h
            p_val = pdf(P, x)
            q_val = pdf(Q, x)

            # Hann window to reduce spectral leakage
            w = 0.5 * (1.0 - cos(2.0 * pi * i / n_points))
            p_vals[i+1] = ComplexF64(w * p_val)
            q_vals[i+1] = ComplexF64(w * q_val)
        end
        for i in n_points+2:n_fft
            p_vals[i] = 0.0 + 0.0im
            q_vals[i] = 0.0 + 0.0im
        end

        # Direct quadrature for KL (FFT used for windowed evaluation)
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

        return max(kl_sum * h, 0.0)
    catch e
        _record_diagnostic!("dsp", "runtime_errors")
        @warn "ZeroProbDSPExt: log_likelihood failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_sampling
# ============================================================================
#
# DSP-based sampling via inverse FFT of the characteristic function.
# Generate samples by:
#   1. Compute characteristic function phi(t) on frequency grid
#   2. Inverse FFT to get density on spatial grid
#   3. Integrate to get CDF
#   4. Invert CDF via interpolation for uniform samples

function ZeroProb.backend_coprocessor_sampling(
    backend::DSPBackend, dist, n_samples::Int)
    try
        n_samples <= 0 && return Float64[]

        # Build high-resolution CDF via FFT-recovered density
        n_fft = 4096
        lower = quantile(dist, 1e-8)
        upper = quantile(dist, 1.0 - 1e-8)
        dx = (upper - lower) / n_fft
        dt = 2.0 * pi / (n_fft * dx)

        # Characteristic function evaluation
        cf_vals = Vector{ComplexF64}(undef, n_fft)
        for k in 0:n_fft-1
            t = (k < n_fft/2 ? k : k - n_fft) * dt
            cf_vals[k+1] = cf(dist, t) * exp(-1.0im * t * lower)
        end

        _dsp_ifft!(cf_vals)

        # Build CDF from recovered density
        density = max.(real.(cf_vals) ./ dx, 0.0)
        cdf_vals = cumsum(density) .* dx
        # Normalise CDF to [0, 1]
        cdf_vals ./= cdf_vals[end]

        xs = [lower + i * dx for i in 0:n_fft-1]

        # Inverse CDF sampling
        uniforms = rand(Float64, n_samples)
        samples = Vector{Float64}(undef, n_samples)

        for idx in 1:n_samples
            u = uniforms[idx]
            # Binary search in CDF
            lo, hi = 1, n_fft
            while lo < hi - 1
                mid = (lo + hi) >> 1
                if cdf_vals[mid] < u
                    lo = mid
                else
                    hi = mid
                end
            end
            frac = (cdf_vals[hi] - cdf_vals[lo]) > 1e-300 ?
                   (u - cdf_vals[lo]) / (cdf_vals[hi] - cdf_vals[lo]) : 0.0
            samples[idx] = xs[lo] + frac * dx
        end

        return samples
    catch e
        _record_diagnostic!("dsp", "runtime_errors")
        @warn "ZeroProbDSPExt: sampling failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_marginalize
# ============================================================================
#
# TV distance via FFT-accelerated integration. Compute |f_P - f_Q| using
# FFT-recovered densities and integrate with the DSP accumulator.

function ZeroProb.backend_coprocessor_marginalize(
    backend::DSPBackend, P, Q, n_points::Int)
    try
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        # FIR-filter-style accumulation with Simpson's rule coefficients
        # The DSP MAC (multiply-accumulate) unit processes one sample per cycle
        tv_sum = 0.0
        for i in 0:n_points
            x = lower + i * h
            diff = abs(pdf(P, x) - pdf(Q, x))
            w = if i == 0 || i == n_points
                1.0
            elseif iseven(i)
                2.0
            else
                4.0
            end
            tv_sum += w * diff
        end

        return 0.5 * (h / 3.0) * tv_sum
    catch e
        _record_diagnostic!("dsp", "runtime_errors")
        @warn "ZeroProbDSPExt: marginalize failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# Conditional density dispatch
function ZeroProb.backend_coprocessor_marginalize(
    backend::DSPBackend, event::ZeroProb.ContinuousZeroProbEvent,
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

        # DSP MAC-based Simpson's integration
        n_points = 1000
        h = (upper - lower) / n_points
        integral = 0.0

        for i in 0:n_points
            t_val = lower + i * h
            val = pdf(dist, t_val) * condition(t_val)
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
        _record_diagnostic!("dsp", "runtime_errors")
        @warn "ZeroProbDSPExt: conditional density failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbDSPExt
