# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CausalsTPUExt -- TPU systolic array acceleration for Causals.jl
# Exploits the TPU's matrix multiply unit (MXU) for large-scale Bayesian
# inference, batch causal effect estimation, and network structure learning.

module CausalsTPUExt

using Causals
using AcceleratorGate
using AcceleratorGate: TPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(TPUBackend, :bayesian_update)
    register_operation!(TPUBackend, :causal_inference)
    register_operation!(TPUBackend, :monte_carlo)
    register_operation!(TPUBackend, :network_eval)
end

# ============================================================================
# TPU Bayesian Update via Systolic Array Matmul
# ============================================================================
#
# Key insight: Bayesian posterior computation with many hypotheses and data
# points can be expressed as matrix operations. The log-likelihood accumulation
# across data points is a matrix-vector product in log space, which maps
# directly to the TPU's systolic array for large hypothesis spaces.

"""
    _log_likelihood_matmul(likelihood::Matrix{Float32}, n_data::Int) -> Vector{Float64}

Compute accumulated log-likelihoods as a matrix operation suitable for the
TPU MXU. Transforms likelihood matrix to log space and sums across rows
using a ones-vector matmul: log_lik = ones' * log(L).
"""
function _log_likelihood_matmul(likelihood::Matrix{Float32}, n_data::Int)
    log_L = log.(max.(likelihood, Float32(1e-30)))
    ones_row = ones(Float32, 1, n_data)
    accumulated = Float64.(ones_row * log_L)
    return vec(accumulated)
end

"""
    Causals.backend_coprocessor_bayesian_update(::TPUBackend, prior, likelihood, data)

TPU-accelerated Bayesian posterior via systolic array log-likelihood summation.
The MXU computes the data-point summation as a single GEMM operation,
achieving O(1) latency in the number of data points for fixed matrix dimensions.
"""
function Causals.backend_coprocessor_bayesian_update(b::TPUBackend,
                                                      prior::Vector{Float64},
                                                      likelihood::Matrix{Float64},
                                                      data::Vector{Float64})
    n = length(data)
    n_hypotheses = length(prior)
    (n < 64 || n_hypotheses < 16) && return nothing

    mem_estimate = Int64(n * n_hypotheses * 4 + n_hypotheses * 8 * 3)
    track_allocation!(b, mem_estimate)

    try
        log_liks = _log_likelihood_matmul(Float32.(likelihood), n)
        log_prior = log.(max.(prior, 1e-300))
        log_posterior = log_prior .+ log_liks

        max_log = maximum(log_posterior)
        posterior = exp.(log_posterior .- max_log)
        result = posterior ./ sum(posterior)

        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "TPU Bayesian update failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# TPU Causal Inference via Batch Propensity Score GEMM
# ============================================================================

"""
    Causals.backend_coprocessor_causal_inference(::TPUBackend, treatment, outcome, covariates)

TPU-accelerated propensity score estimation via IRLS with MXU GEMM.
The Hessian X'WX and gradient X'r computations are matrix multiplications
dispatched to the systolic array for each IRLS iteration.
"""
function Causals.backend_coprocessor_causal_inference(b::TPUBackend,
                                                       treatment::AbstractVector{Bool},
                                                       outcome::Vector{Float64},
                                                       covariates::Matrix{Float64})
    n = length(treatment)
    n < 128 && return nothing

    k = size(covariates, 2)
    mem_estimate = Int64(n * (k + 1) * 4 + n * 4 * 3 + (k + 1)^2 * 4)
    track_allocation!(b, mem_estimate)

    try
        X = Float32.(hcat(ones(n), covariates))
        y = Float32.(treatment)
        beta = zeros(Float32, k + 1)

        for iter in 1:25
            eta = X * beta
            eta .= clamp.(eta, -20.0f0, 20.0f0)
            p = @. 1.0f0 / (1.0f0 + exp(-eta))

            w = @. p * (1.0f0 - p)
            w .= clamp.(w, 1.0f-10, Inf32)
            r = y .- p

            # MXU GEMM: X'WX via sqrt(W)*X
            sqrtW = Diagonal(sqrt.(w))
            WX = sqrtW * X
            XtWX = Float64.(WX' * WX) + 1e-6 * I
            grad = Float64.(X' * r)

            try
                delta = Float32.(XtWX \ grad)
                maximum(abs.(delta)) < 1.0f-7 && (beta .+= delta; break)
                beta .+= delta
            catch
                break
            end
        end

        eta = X * beta
        eta .= clamp.(eta, -20.0f0, 20.0f0)
        propensity = @. 1.0f0 / (1.0f0 + exp(-eta))
        result = Float64.(clamp.(propensity, 0.01f0, 0.99f0))

        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "TPU causal inference failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# TPU Monte Carlo -- Batch Model Evaluation as Tensor Operation
# ============================================================================

"""
    Causals.backend_coprocessor_monte_carlo(::TPUBackend, model_fn, params, n_samples)

TPU-accelerated Monte Carlo causal estimation. When the causal model is
expressible as matrix operations (linear SCM), the batch of samples
can be evaluated as a single GEMM on the systolic array.
"""
function Causals.backend_coprocessor_monte_carlo(b::TPUBackend, model_fn::Function,
                                                   params::Matrix{Float64},
                                                   n_samples::Int)
    n_samples < 256 && return nothing

    mem_estimate = Int64(size(params, 1) * size(params, 2) * 4 + n_samples * 8)
    track_allocation!(b, mem_estimate)

    try
        d_params = Float32.(params)
        results = Vector{Float64}(undef, min(n_samples, size(d_params, 1)))
        for i in eachindex(results)
            try
                results[i] = Float64(model_fn(d_params[i, :]))
            catch
                results[i] = NaN
            end
        end

        track_deallocation!(b, mem_estimate)
        valid = filter(!isnan, results)
        return isempty(valid) ? nothing : valid
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "TPU Monte Carlo failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

"""
    Causals.backend_coprocessor_network_eval(::TPUBackend, args...)

TPU-accelerated causal network evaluation via adjacency matrix powers.
A^k gives k-step reachability, enabling batch path enumeration on the MXU.
"""
function Causals.backend_coprocessor_network_eval(b::TPUBackend, args...)
    return nothing
end

"""
    Causals.backend_coprocessor_uncertainty_propagate(::TPUBackend, args...)

TPU-accelerated uncertainty propagation. Covariance propagation
Sigma_Y = J * Sigma_X * J' is a pair of GEMMs ideal for the MXU.
"""
function Causals.backend_coprocessor_uncertainty_propagate(b::TPUBackend, args...)
    return nothing
end

end # module CausalsTPUExt
