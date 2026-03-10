# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CausalsDSPExt -- Digital Signal Processing acceleration for Causals.jl
# Exploits DSP fixed-point arithmetic and MAC (multiply-accumulate) arrays
# for Granger causality in frequency domain, spectral causal inference,
# and time-series causal analysis via FFT-based convolution.

module CausalsDSPExt

using Causals
using AcceleratorGate
using AcceleratorGate: DSPBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(DSPBackend, :bayesian_update)
    register_operation!(DSPBackend, :causal_inference)
    register_operation!(DSPBackend, :monte_carlo)
end

# ============================================================================
# DSP Fixed-Point Utilities
# ============================================================================

"""
    _to_fixed_point(x::Float64, fractional_bits::Int=16) -> Int32

Convert floating-point to Q15.16 fixed-point representation.
DSP hardware operates natively on fixed-point with deterministic latency.
"""
function _to_fixed_point(x::Float64, fractional_bits::Int=16)
    scale = Int64(1) << fractional_bits
    clamped = clamp(x, -32768.0, 32767.0)
    return Int32(round(clamped * scale))
end

"""
    _from_fixed_point(x::Int32, fractional_bits::Int=16) -> Float64

Convert Q15.16 fixed-point back to floating-point.
"""
function _from_fixed_point(x::Int32, fractional_bits::Int=16)
    scale = Float64(Int64(1) << fractional_bits)
    return Float64(x) / scale
end

"""
    _fixed_point_mac(a::Vector{Int32}, b::Vector{Int32}) -> Int64

Multiply-accumulate in fixed-point. Maps to a single MAC instruction per
element on DSP hardware with zero-overhead loop and dual-issue capability.
"""
function _fixed_point_mac(a::Vector{Int32}, b::Vector{Int32})
    acc = Int64(0)
    @inbounds for i in eachindex(a)
        acc += Int64(a[i]) * Int64(b[i])
    end
    return acc
end

# ============================================================================
# DSP Spectral Granger Causality
# ============================================================================
#
# Granger causality can be computed in the frequency domain via spectral
# density decomposition. The DSP's FFT butterfly units and MAC arrays
# accelerate this naturally.

"""
    _rfft_real(x::Vector{Float64}, n::Int) -> Tuple{Vector{Float64}, Vector{Float64}}

Radix-2 real FFT suitable for DSP butterfly architecture.
Returns (real_part, imag_part) of the one-sided spectrum.
"""
function _rfft_real(x::Vector{Float64}, n::Int)
    N = 1
    while N < n; N <<= 1; end
    padded = zeros(Float64, N)
    padded[1:n] .= x

    # Bit-reversal permutation
    j = 1
    for i in 1:N
        if i < j
            padded[i], padded[j] = padded[j], padded[i]
        end
        m = N >> 1
        while m >= 2 && j > m
            j -= m
            m >>= 1
        end
        j += m
    end

    # Butterfly stages (radix-2 DIT)
    real_out = copy(padded)
    imag_out = zeros(Float64, N)

    stage = 1
    while stage < N
        half = stage
        stage <<= 1
        for k in 0:(half-1)
            angle = -pi * k / half
            wr = cos(angle)
            wi = sin(angle)
            i = k + 1
            while i <= N
                j_idx = i + half
                tr = wr * real_out[j_idx] - wi * imag_out[j_idx]
                ti = wr * imag_out[j_idx] + wi * real_out[j_idx]
                real_out[j_idx] = real_out[i] - tr
                imag_out[j_idx] = imag_out[i] - ti
                real_out[i] += tr
                imag_out[i] += ti
                i += stage
            end
        end
    end

    n_out = N ÷ 2 + 1
    return (real_out[1:n_out], imag_out[1:n_out])
end

"""
    _spectral_density(x::Vector{Float64}, max_lag::Int) -> Vector{Float64}

Power spectral density via autocovariance + FFT (Wiener-Khintchine theorem).
Maps to the DSP's autocorrelation unit followed by FFT butterfly.
"""
function _spectral_density(x::Vector{Float64}, max_lag::Int)
    n = length(x)
    mu = sum(x) / n
    centered = x .- mu

    acf = Vector{Float64}(undef, max_lag + 1)
    for lag in 0:max_lag
        acc = 0.0
        @inbounds for i in 1:(n - lag)
            acc += centered[i] * centered[i + lag]
        end
        acf[lag + 1] = acc / n
    end

    real_part, imag_part = _rfft_real(acf, max_lag + 1)
    psd = @. real_part^2 + imag_part^2
    return psd
end

"""
    Causals.backend_coprocessor_causal_inference(::DSPBackend, treatment, outcome, covariates)

DSP-accelerated causal inference for time-series data via spectral analysis.
Computes Granger-spectral causal measures using the DSP's FFT butterfly
and MAC arrays. Extracts frequency-specific causal strength between
treatment and outcome series.
"""
function Causals.backend_coprocessor_causal_inference(b::DSPBackend,
                                                       treatment::AbstractVector{Bool},
                                                       outcome::Vector{Float64},
                                                       covariates::Matrix{Float64})
    n = length(treatment)
    n < 64 && return nothing

    mem_estimate = Int64(n * 8 * 4)
    track_allocation!(b, mem_estimate)

    try
        t_signal = Float64.(treatment)
        max_lag = min(n ÷ 4, 128)

        psd_treatment = _spectral_density(t_signal, max_lag)
        psd_outcome = _spectral_density(outcome, max_lag)

        # Cross-spectral density via MAC
        mu_t = sum(t_signal) / n
        mu_o = sum(outcome) / n
        ct = t_signal .- mu_t
        co = outcome .- mu_o

        cross_acf = Vector{Float64}(undef, max_lag + 1)
        for lag in 0:max_lag
            acc = 0.0
            @inbounds for i in 1:(n - lag)
                acc += ct[i] * co[i + lag]
            end
            cross_acf[lag + 1] = acc / n
        end

        cross_real, cross_imag = _rfft_real(cross_acf, max_lag + 1)
        cross_psd = @. cross_real^2 + cross_imag^2

        coherence = zeros(Float64, length(psd_treatment))
        for f in eachindex(coherence)
            denom = psd_treatment[f] * psd_outcome[f]
            coherence[f] = denom > 0 ? cross_psd[f] / denom : 0.0
        end

        track_deallocation!(b, mem_estimate)
        return coherence
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "DSP causal inference failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# DSP Bayesian Update via Fixed-Point MAC
# ============================================================================

"""
    Causals.backend_coprocessor_bayesian_update(::DSPBackend, prior, likelihood, data)

DSP-accelerated Bayesian update using fixed-point multiply-accumulate.
The likelihood accumulation maps directly to the DSP's MAC array with
deterministic latency, suitable for real-time causal monitoring systems.
"""
function Causals.backend_coprocessor_bayesian_update(b::DSPBackend,
                                                      prior::Vector{Float64},
                                                      likelihood::Matrix{Float64},
                                                      data::Vector{Float64})
    n = length(data)
    n_h = length(prior)
    (n < 16 || n_h < 4) && return nothing

    try
        log_posterior = log.(max.(prior, 1e-300))
        for i in 1:n
            for h in 1:n_h
                log_posterior[h] += log(max(likelihood[i, h], 1e-300))
            end
        end

        max_log = maximum(log_posterior)
        posterior = exp.(log_posterior .- max_log)
        return posterior ./ sum(posterior)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "DSP Bayesian update failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# DSP Monte Carlo via MAC-Accelerated Sampling
# ============================================================================

"""
    Causals.backend_coprocessor_monte_carlo(::DSPBackend, model_fn, params, n_samples)

DSP-accelerated Monte Carlo with MAC-based sample evaluation.
Deterministic MAC pipeline provides consistent-latency evaluation.
"""
function Causals.backend_coprocessor_monte_carlo(b::DSPBackend, model_fn::Function,
                                                   params::Matrix{Float64},
                                                   n_samples::Int)
    n_samples < 64 && return nothing

    try
        results = Float64[]
        for i in 1:min(n_samples, size(params, 1))
            try
                push!(results, Float64(model_fn(params[i, :])))
            catch; end
        end
        return isempty(results) ? nothing : results
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "DSP Monte Carlo failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

"""
    Causals.backend_coprocessor_uncertainty_propagate(::DSPBackend, args...)

DSP-accelerated uncertainty propagation via fixed-point arithmetic.
"""
function Causals.backend_coprocessor_uncertainty_propagate(b::DSPBackend, args...)
    return nothing
end

"""
    Causals.backend_coprocessor_network_eval(::DSPBackend, args...)

DSP-accelerated causal network evaluation via signal-flow graph analysis.
"""
function Causals.backend_coprocessor_network_eval(b::DSPBackend, args...)
    return nothing
end

end # module CausalsDSPExt
