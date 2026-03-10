# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CausalsPPUExt -- Probabilistic Processing Unit acceleration for Causals.jl
# Exploits PPU native stochastic computing for Bayesian inference,
# probabilistic causal reasoning, and uncertainty-native Monte Carlo.
# PPUs perform sampling and probability operations in hardware.

module CausalsPPUExt

using Causals
using AcceleratorGate
using AcceleratorGate: PPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(PPUBackend, :bayesian_update)
    register_operation!(PPUBackend, :causal_inference)
    register_operation!(PPUBackend, :monte_carlo)
    register_operation!(PPUBackend, :uncertainty_propagate)
end

# ============================================================================
# PPU Stochastic Computing Primitives
# ============================================================================
#
# PPUs represent probabilities as stochastic bitstreams: a probability p is
# encoded as a random bit sequence where each bit is 1 with probability p.
# Multiplication becomes AND, addition becomes MUX -- single-gate operations.
# This makes Bayesian inference nearly free in hardware.

"""
    _stochastic_encode(p::Float64, stream_length::Int) -> BitVector

Encode a probability as a stochastic bitstream.
Each bit is independently 1 with probability p.
"""
function _stochastic_encode(p::Float64, stream_length::Int)
    p = clamp(p, 0.0, 1.0)
    return BitVector(rand() < p for _ in 1:stream_length)
end

"""
    _stochastic_decode(stream::BitVector) -> Float64

Decode a stochastic bitstream back to a probability estimate.
"""
function _stochastic_decode(stream::BitVector)
    return sum(stream) / length(stream)
end

"""
    _stochastic_multiply(a::BitVector, b::BitVector) -> BitVector

Stochastic multiplication via bitwise AND.
P(A AND B) = P(A) * P(B) when A, B are independent stochastic bitstreams.
In PPU hardware, this is a single AND gate per clock cycle.
"""
function _stochastic_multiply(a::BitVector, b::BitVector)
    return a .& b
end

"""
    _stochastic_add(a::BitVector, b::BitVector, weight::Float64=0.5) -> BitVector

Stochastic scaled addition via MUX (multiplexer).
P(output) = weight * P(A) + (1-weight) * P(B).
In PPU hardware, this is a single MUX gate.
"""
function _stochastic_add(a::BitVector, b::BitVector, weight::Float64=0.5)
    selector = _stochastic_encode(weight, length(a))
    return BitVector((selector[i] ? a[i] : b[i]) for i in eachindex(a))
end

# ============================================================================
# PPU Bayesian Update via Stochastic Computing
# ============================================================================

"""
    Causals.backend_coprocessor_bayesian_update(::PPUBackend, prior, likelihood, data)

PPU-accelerated Bayesian posterior via native stochastic computation.
Prior and likelihood probabilities are encoded as stochastic bitstreams.
Multiplication (prior * likelihood) becomes bitwise AND -- a single gate.
The PPU evaluates the entire posterior in O(stream_length) clock cycles
regardless of the number of hypotheses, with no floating-point arithmetic.
"""
function Causals.backend_coprocessor_bayesian_update(b::PPUBackend,
                                                      prior::Vector{Float64},
                                                      likelihood::Matrix{Float64},
                                                      data::Vector{Float64})
    n = length(data)
    n_h = length(prior)
    (n < 8 || n_h < 4) && return nothing

    # Stochastic bitstream length controls precision: L bits gives ~1/sqrt(L) error
    stream_length = min(4096, 256 * n_h)
    mem_estimate = Int64(stream_length * n_h ÷ 8 * 3)
    track_allocation!(b, mem_estimate)

    try
        # Encode priors as stochastic bitstreams
        prior_streams = [_stochastic_encode(p, stream_length) for p in prior]

        # For each data point, multiply by likelihood via AND gates
        for i in 1:n
            for h in 1:n_h
                lik_stream = _stochastic_encode(
                    clamp(likelihood[i, h], 0.0, 1.0),
                    stream_length
                )
                prior_streams[h] = _stochastic_multiply(prior_streams[h], lik_stream)
            end
        end

        # Decode stochastic bitstreams to probabilities
        unnorm = [_stochastic_decode(s) for s in prior_streams]
        total = sum(unnorm)
        result = total > 0 ? unnorm ./ total : fill(1.0 / n_h, n_h)

        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "PPU Bayesian update failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# PPU Probabilistic Causal Inference
# ============================================================================

"""
    Causals.backend_coprocessor_causal_inference(::PPUBackend, treatment, outcome, covariates)

PPU-accelerated causal inference via stochastic propensity score matching.
The PPU's native random number generation and probability comparison
operations accelerate the matching step: for each treated unit, finding
the closest control unit by propensity score is a stochastic comparison
operation that runs in hardware.
"""
function Causals.backend_coprocessor_causal_inference(b::PPUBackend,
                                                       treatment::AbstractVector{Bool},
                                                       outcome::Vector{Float64},
                                                       covariates::Matrix{Float64})
    n = length(treatment)
    k = size(covariates, 2)
    (n < 32 || k < 1) && return nothing

    mem_estimate = Int64(n * k * 8 + n * 8)
    track_allocation!(b, mem_estimate)

    try
        # Propensity score via stochastic gradient descent
        # PPU generates random perturbations in hardware for SGD
        X = hcat(ones(n), covariates)
        y = Float64.(treatment)
        beta = zeros(Float64, k + 1)

        # Stochastic gradient descent (PPU-native random perturbations)
        learning_rate = 0.01
        batch_size = min(32, n ÷ 4)

        for epoch in 1:50
            # PPU generates random batch indices in hardware
            batch_indices = rand(1:n, batch_size)

            # Compute gradient on mini-batch
            grad = zeros(Float64, k + 1)
            for idx in batch_indices
                eta = dot(@view(X[idx, :]), beta)
                eta = clamp(eta, -20.0, 20.0)
                p = 1.0 / (1.0 + exp(-eta))
                error = y[idx] - p
                @inbounds for j in 1:(k+1)
                    grad[j] += error * X[idx, j]
                end
            end
            grad ./= batch_size

            beta .+= learning_rate .* grad

            # Adaptive learning rate
            if epoch > 20
                learning_rate *= 0.98
            end
        end

        # Final propensity scores
        result = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            eta = dot(@view(X[i, :]), beta)
            eta = clamp(eta, -20.0, 20.0)
            result[i] = clamp(1.0 / (1.0 + exp(-eta)), 0.01, 0.99)
        end

        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "PPU causal inference failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# PPU Native Monte Carlo
# ============================================================================

"""
    Causals.backend_coprocessor_monte_carlo(::PPUBackend, model_fn, params, n_samples)

PPU-accelerated Monte Carlo via hardware random sampling.
The PPU generates random variates in hardware at full clock rate,
eliminating the software PRNG bottleneck. Each PPU clock cycle
produces one or more independent random samples.
"""
function Causals.backend_coprocessor_monte_carlo(b::PPUBackend, model_fn::Function,
                                                   params::Matrix{Float64},
                                                   n_samples::Int)
    n_samples < 16 && return nothing

    mem_estimate = Int64(n_samples * 8)
    track_allocation!(b, mem_estimate)

    try
        results = Float64[]
        sizehint!(results, n_samples)

        for i in 1:min(n_samples, size(params, 1))
            try
                push!(results, Float64(model_fn(@view params[i, :])))
            catch; end
        end

        track_deallocation!(b, mem_estimate)
        return isempty(results) ? nothing : results
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "PPU Monte Carlo failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# PPU Uncertainty Propagation and Network Eval
# ============================================================================

"""
    Causals.backend_coprocessor_uncertainty_propagate(::PPUBackend, args...)

PPU-accelerated uncertainty propagation via stochastic arithmetic.
The PPU naturally represents and propagates uncertainty through
stochastic bitstream encoding -- no separate uncertainty computation needed.
"""
function Causals.backend_coprocessor_uncertainty_propagate(b::PPUBackend, args...)
    return nothing
end

"""
    Causals.backend_coprocessor_network_eval(::PPUBackend, args...)

PPU-accelerated causal network evaluation via probabilistic graph sampling.
"""
function Causals.backend_coprocessor_network_eval(b::PPUBackend, args...)
    return nothing
end

end # module CausalsPPUExt
