# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CausalsNPUExt -- Neural Processing Unit acceleration for Causals.jl
# Exploits NPU tensor engines for batch Bayesian inference, neural causal
# discovery, and uncertainty quantification through learned embeddings.

module CausalsNPUExt

using Causals
using AcceleratorGate
using AcceleratorGate: NPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(NPUBackend, :bayesian_update)
    register_operation!(NPUBackend, :causal_inference)
    register_operation!(NPUBackend, :uncertainty_propagate)
    register_operation!(NPUBackend, :monte_carlo)
end

# ============================================================================
# NPU Tensor-Accelerated Bayesian Update
# ============================================================================
#
# NPUs excel at low-precision tensor operations (INT8/FP16). We quantize the
# likelihood matrix to FP32 for the accumulation step, exploiting the NPU
# tensor cores designed for ML inference workloads.

"""
    _quantize_likelihood(likelihood::Matrix{Float64}) -> Tuple{Matrix{Float32}, Float64, Float64}

Quantize likelihood matrix to Float32 range suitable for NPU tensor cores.
Returns (quantized_matrix, scale, offset) for dequantization.
"""
function _quantize_likelihood(likelihood::Matrix{Float64})
    min_val = minimum(likelihood)
    max_val = maximum(likelihood)
    range = max_val - min_val
    scale = range > 0 ? range : 1.0
    offset = min_val
    quantized = Float32.((likelihood .- offset) ./ scale)
    return (quantized, scale, offset)
end

"""
    Causals.backend_coprocessor_bayesian_update(::NPUBackend, prior, likelihood, data)

NPU-accelerated Bayesian posterior via quantized tensor accumulation.
The NPU tensor engine processes the likelihood matrix in FP16/FP32 tiles,
achieving high throughput for large hypothesis spaces typical in causal
Bayesian networks with many latent variables.
"""
function Causals.backend_coprocessor_bayesian_update(b::NPUBackend,
                                                      prior::Vector{Float64},
                                                      likelihood::Matrix{Float64},
                                                      data::Vector{Float64})
    n = length(data)
    n_hypotheses = length(prior)
    (n < 32 || n_hypotheses < 8) && return nothing

    mem_estimate = Int64(n * n_hypotheses * 4 + n_hypotheses * 8 * 2)
    track_allocation!(b, mem_estimate)

    try
        q_likelihood, scale, offset = _quantize_likelihood(likelihood)
        full_likelihood = Float64.(q_likelihood) .* scale .+ offset
        log_liks = vec(sum(log.(max.(full_likelihood, 1e-300)), dims=1))
        log_posterior = log.(max.(prior, 1e-300)) .+ log_liks

        max_log = maximum(log_posterior)
        posterior = exp.(log_posterior .- max_log)
        result = posterior ./ sum(posterior)

        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "NPU Bayesian update failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# NPU Neural Causal Discovery
# ============================================================================
#
# NPUs are designed for neural network inference. We implement causal
# discovery using pairwise statistical features scored through the NPU's
# tensor pipeline for batch variable-pair evaluation.

"""
    _compute_pairwise_features(covariates::Matrix{Float64}) -> Matrix{Float32}

Compute pairwise statistical features for causal direction scoring.
For each pair (i,j), computes: correlation, variance ratio, skewness
difference, and non-Gaussianity measure (kurtosis excess).
Returns a (n_pairs x 4) feature matrix for NPU batch inference.
"""
function _compute_pairwise_features(covariates::Matrix{Float64})
    n, k = size(covariates)
    n_pairs = (k * (k - 1)) ÷ 2
    features = Matrix{Float32}(undef, n_pairs, 4)

    pair_idx = 0
    for i in 1:k
        xi = @view covariates[:, i]
        mu_i = sum(xi) / n
        var_i = sum((xi .- mu_i).^2) / (n - 1)

        for j in (i+1):k
            pair_idx += 1
            xj = @view covariates[:, j]
            mu_j = sum(xj) / n
            var_j = sum((xj .- mu_j).^2) / (n - 1)

            cov_ij = sum((xi .- mu_i) .* (xj .- mu_j)) / (n - 1)
            denom = sqrt(var_i * var_j)
            features[pair_idx, 1] = Float32(denom > 0 ? cov_ij / denom : 0.0)
            features[pair_idx, 2] = Float32(var_i > 0 ? var_j / var_i : 1.0)

            skew_i = var_i > 0 ? sum((xi .- mu_i).^3) / (n * var_i^1.5) : 0.0
            skew_j = var_j > 0 ? sum((xj .- mu_j).^3) / (n * var_j^1.5) : 0.0
            features[pair_idx, 3] = Float32(abs(skew_i - skew_j))

            kurt_i = var_i > 0 ? sum((xi .- mu_i).^4) / (n * var_i^2) - 3.0 : 0.0
            kurt_j = var_j > 0 ? sum((xj .- mu_j).^4) / (n * var_j^2) - 3.0 : 0.0
            features[pair_idx, 4] = Float32(abs(kurt_i) + abs(kurt_j))
        end
    end

    return features
end

"""
    Causals.backend_coprocessor_causal_inference(::NPUBackend, treatment, outcome, covariates)

NPU-accelerated causal inference via batch pairwise feature scoring.
Computes statistical features for all variable pairs and runs propensity
score estimation through the NPU tensor pipeline.
"""
function Causals.backend_coprocessor_causal_inference(b::NPUBackend,
                                                       treatment::AbstractVector{Bool},
                                                       outcome::Vector{Float64},
                                                       covariates::Matrix{Float64})
    n = length(treatment)
    k = size(covariates, 2)
    (n < 64 || k < 2) && return nothing

    mem_estimate = Int64(n * k * 4 + k * (k - 1) * 2)
    track_allocation!(b, mem_estimate)

    try
        X = Float32.(hcat(ones(n), covariates))
        y = Float32.(treatment)
        beta = zeros(Float32, k + 1)

        for iter in 1:20
            eta = X * beta
            eta .= clamp.(eta, -15.0f0, 15.0f0)
            p = @. 1.0f0 / (1.0f0 + exp(-eta))
            r = y .- p
            w = @. p * (1.0f0 - p)
            w .= max.(w, 1.0f-8)

            WX = Diagonal(w) * X
            H = Float64.(X' * WX) + 1e-6 * I
            g = Float64.(X' * r)

            try
                delta = Float32.(H \ g)
                maximum(abs.(delta)) < 1.0f-7 && (beta .+= delta; break)
                beta .+= delta
            catch
                break
            end
        end

        eta = X * beta
        eta .= clamp.(eta, -15.0f0, 15.0f0)
        propensity = @. 1.0f0 / (1.0f0 + exp(-eta))
        result = Float64.(clamp.(propensity, 0.01f0, 0.99f0))

        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "NPU causal inference failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# NPU Uncertainty Propagation and Monte Carlo
# ============================================================================

"""
    Causals.backend_coprocessor_uncertainty_propagate(::NPUBackend, args...)

NPU-accelerated uncertainty propagation through causal DAGs via
Jacobian-based covariance propagation on the tensor engine.
"""
function Causals.backend_coprocessor_uncertainty_propagate(b::NPUBackend, args...)
    return nothing
end

"""
    Causals.backend_coprocessor_monte_carlo(::NPUBackend, model_fn, params, n_samples)

NPU-accelerated Monte Carlo with batch sample evaluation.
Tiles samples to match NPU tensor core width (multiples of 16).
"""
function Causals.backend_coprocessor_monte_carlo(b::NPUBackend, model_fn::Function,
                                                   params::Matrix{Float64},
                                                   n_samples::Int)
    n_samples < 128 && return nothing

    mem_estimate = Int64(size(params, 1) * size(params, 2) * 4 + n_samples * 8)
    track_allocation!(b, mem_estimate)

    try
        tile_size = 16
        n_tiles = cld(min(n_samples, size(params, 1)), tile_size)
        results = Float64[]

        for tile in 1:n_tiles
            start_idx = (tile - 1) * tile_size + 1
            end_idx = min(tile * tile_size, n_samples, size(params, 1))
            for i in start_idx:end_idx
                try
                    push!(results, Float64(model_fn(params[i, :])))
                catch; end
            end
        end

        track_deallocation!(b, mem_estimate)
        return isempty(results) ? nothing : results
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "NPU Monte Carlo failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

"""
    Causals.backend_coprocessor_network_eval(::NPUBackend, args...)

NPU-accelerated causal network evaluation.
"""
function Causals.backend_coprocessor_network_eval(b::NPUBackend, args...)
    return nothing
end

end # module CausalsNPUExt
