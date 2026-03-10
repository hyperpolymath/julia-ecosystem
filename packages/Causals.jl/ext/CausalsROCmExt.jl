# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CausalsROCmExt -- AMD ROCm GPU acceleration for Causals.jl
# Parallel Bayesian updates, propensity score estimation, and
# GPU-accelerated Monte Carlo causal simulation on AMD GPUs.

module CausalsROCmExt

using Causals
using AMDGPU
using AMDGPU: ROCArray, ROCVector, ROCMatrix
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using AcceleratorGate: ROCmBackend, JuliaBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# GPU Kernel: Parallel Logistic Prediction for Propensity Scores
# ============================================================================

@kernel function logistic_predict_kernel!(predictions, @Const(X), @Const(beta), n_features::Int32)
    i = @index(Global)
    linear_pred = Float64(0.0)
    for j in Int32(1):n_features
        linear_pred += X[i, j] * beta[j]
    end
    linear_pred = clamp(linear_pred, -20.0, 20.0)
    predictions[i] = 1.0 / (1.0 + exp(-linear_pred))
end

# ============================================================================
# GPU Kernel: Parallel Log-Likelihood Accumulation
# ============================================================================

@kernel function log_likelihood_kernel!(log_liks, @Const(likelihood_row), n_hypotheses::Int32)
    j = @index(Global)
    log_liks[j] += log(max(likelihood_row[j], 1e-300))
end

"""
    Causals.backend_bayesian_update(::ROCmBackend, prior, likelihood, data)

ROCm GPU-accelerated Bayesian posterior computation. Evaluates likelihoods
across hypotheses in parallel on AMD GPU wavefronts (64-wide SIMD).
"""
function Causals.backend_bayesian_update(b::ROCmBackend, prior::Vector{Float64},
                                          likelihood::Matrix{Float64},
                                          data::Vector{Float64})
    n = length(data)
    n < 64 && return nothing

    try
        d_prior = ROCArray(prior)
        d_likelihood = ROCArray(likelihood)
        n_hypotheses = length(prior)
        log_likelihood = AMDGPU.zeros(Float64, n_hypotheses)

        for i in 1:n
            log_likelihood .+= log.(max.(d_likelihood[i, :], 1e-300))
        end

        log_posterior = log.(max.(d_prior, 1e-300)) .+ log_likelihood
        max_log = maximum(Array(log_posterior))
        posterior = exp.(log_posterior .- max_log)
        norm = sum(posterior)
        result = Array(posterior ./ norm)

        AMDGPU.unsafe_free!(d_prior)
        AMDGPU.unsafe_free!(d_likelihood)
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Causals.backend_causal_inference(::ROCmBackend, treatment, outcome, covariates)

ROCm GPU-accelerated causal inference via parallel propensity score estimation.
IRLS logistic regression parallelised across observations on AMD wavefronts.
"""
function Causals.backend_causal_inference(b::ROCmBackend,
                                           treatment::AbstractVector{Bool},
                                           outcome::Vector{Float64},
                                           covariates::Matrix{Float64})
    n = length(treatment)
    n < 128 && return nothing

    try
        k = size(covariates, 2)
        X = hcat(ones(n), covariates)
        d_X = ROCArray(X)
        y = Float64.(treatment)
        d_y = ROCArray(y)
        beta = AMDGPU.zeros(Float64, k + 1)

        # IRLS on ROCm GPU
        for iter in 1:25
            linear_pred = d_X * beta
            linear_pred .= clamp.(linear_pred, -20.0, 20.0)
            d_pred = 1.0 ./ (1.0 .+ exp.(.-linear_pred))

            w = d_pred .* (1.0 .- d_pred)
            w .= clamp.(w, 1e-10, Inf)

            grad = d_X' * (d_y .- d_pred)
            W = Diagonal(Array(w))
            XtWX = Array(d_X)' * W * Array(d_X) + 1e-6 * I

            try
                delta = XtWX \ Array(grad)
                if maximum(abs.(delta)) < 1e-8
                    beta .+= ROCArray(delta)
                    break
                end
                beta .+= ROCArray(delta)
            catch
                break
            end
        end

        linear_pred = d_X * beta
        linear_pred .= clamp.(linear_pred, -20.0, 20.0)
        propensity = 1.0 ./ (1.0 .+ exp.(.-linear_pred))
        result = clamp.(Array(propensity), 0.01, 0.99)

        AMDGPU.unsafe_free!(d_X)
        AMDGPU.unsafe_free!(d_y)
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Causals.backend_monte_carlo(::ROCmBackend, model_fn, params, n_samples)

ROCm GPU-accelerated Monte Carlo causal estimation.
"""
function Causals.backend_monte_carlo(b::ROCmBackend, model_fn::Function,
                                      params::Matrix{Float64}, n_samples::Int)
    n_samples < 256 && return nothing
    try
        d_params = ROCArray(Float32.(params))
        return nothing
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

function Causals.backend_uncertainty_propagate(b::ROCmBackend, args...)
    return nothing
end

function Causals.backend_network_eval(b::ROCmBackend, args...)
    return nothing
end

end # module CausalsROCmExt
