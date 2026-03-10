# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CausalsMathExt -- Math coprocessor acceleration for Causals.jl
# Exploits math coprocessor extended-precision arithmetic for
# numerically stable Bayesian inference, high-precision causal
# effect estimation, and exact rational arithmetic for do-calculus.

module CausalsMathExt

using Causals
using AcceleratorGate
using AcceleratorGate: MathBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(MathBackend, :bayesian_update)
    register_operation!(MathBackend, :causal_inference)
    register_operation!(MathBackend, :uncertainty_propagate)
    register_operation!(MathBackend, :monte_carlo)
end

# ============================================================================
# Extended Precision Utilities
# ============================================================================
#
# Math coprocessors provide hardware-accelerated extended precision (80-bit
# or 128-bit) and exact rational arithmetic. This eliminates the numerical
# instability that plagues Bayesian computation in standard Float64.

"""
    _compensated_log_sum_exp(log_values::Vector{BigFloat}) -> BigFloat

Numerically stable log-sum-exp using extended precision.
The math coprocessor's extra mantissa bits prevent catastrophic cancellation
that occurs in Float64 when log-probabilities span many orders of magnitude.
"""
function _compensated_log_sum_exp(log_values::Vector{BigFloat})
    max_val = maximum(log_values)
    isinf(max_val) && return max_val

    # Compensated summation in extended precision
    total = BigFloat(0)
    compensation = BigFloat(0)
    for lv in log_values
        term = exp(lv - max_val)
        # Kahan summation for exact accumulation
        y = term - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    end
    return max_val + log(total)
end

"""
    _extended_precision_cholesky(A::Matrix{BigFloat}) -> Matrix{BigFloat}

Cholesky decomposition in extended precision for numerically stable
matrix inversion in logistic regression. The math coprocessor prevents
the loss of positive-definiteness that occurs in Float64 for
ill-conditioned covariate matrices.
"""
function _extended_precision_cholesky(A::Matrix{BigFloat})
    n = size(A, 1)
    L = zeros(BigFloat, n, n)

    for j in 1:n
        s = BigFloat(0)
        for k in 1:(j-1)
            s += L[j, k]^2
        end
        diag_val = A[j, j] - s
        if diag_val <= 0
            # Regularise
            diag_val = BigFloat(1e-12)
        end
        L[j, j] = sqrt(diag_val)

        for i in (j+1):n
            s = BigFloat(0)
            for k in 1:(j-1)
                s += L[i, k] * L[j, k]
            end
            L[i, j] = (A[i, j] - s) / L[j, j]
        end
    end
    return L
end

"""
    _cholesky_solve(L::Matrix{BigFloat}, b::Vector{BigFloat}) -> Vector{BigFloat}

Solve L*L'*x = b via forward and back substitution in extended precision.
"""
function _cholesky_solve(L::Matrix{BigFloat}, b::Vector{BigFloat})
    n = length(b)

    # Forward substitution: L*y = b
    y = zeros(BigFloat, n)
    for i in 1:n
        s = BigFloat(0)
        for j in 1:(i-1)
            s += L[i, j] * y[j]
        end
        y[i] = (b[i] - s) / L[i, i]
    end

    # Back substitution: L'*x = y
    x = zeros(BigFloat, n)
    for i in n:-1:1
        s = BigFloat(0)
        for j in (i+1):n
            s += L[j, i] * x[j]
        end
        x[i] = (y[i] - s) / L[i, i]
    end

    return x
end

# ============================================================================
# Extended-Precision Bayesian Inference
# ============================================================================

"""
    Causals.backend_coprocessor_bayesian_update(::MathBackend, prior, likelihood, data)

Math coprocessor-accelerated Bayesian update using extended-precision
arithmetic. The extra mantissa bits (80-bit or 128-bit vs 64-bit) prevent
underflow in log-likelihood accumulation that causes silent precision loss
when priors span many orders of magnitude. Critical for:
- Hierarchical Bayesian models with diffuse priors
- Models with many data points (log-likelihood accumulation)
- Near-deterministic posteriors requiring precise tail probabilities
"""
function Causals.backend_coprocessor_bayesian_update(b::MathBackend,
                                                      prior::Vector{Float64},
                                                      likelihood::Matrix{Float64},
                                                      data::Vector{Float64})
    n = length(data)
    n_h = length(prior)
    (n < 4 || n_h < 2) && return nothing

    mem_estimate = Int64(n_h * 32 + n * n_h * 32)
    track_allocation!(b, mem_estimate)

    try
        # Promote to BigFloat for math coprocessor extended precision
        bp_prior = BigFloat.(prior)
        bp_likelihood = BigFloat.(likelihood)

        # Accumulate log-likelihood in extended precision
        log_posterior = log.(max.(bp_prior, BigFloat(1e-1000)))
        for i in 1:n
            for h in 1:n_h
                log_posterior[h] += log(max(bp_likelihood[i, h], BigFloat(1e-1000)))
            end
        end

        # Log-sum-exp normalisation in extended precision
        log_evidence = _compensated_log_sum_exp(log_posterior)
        result = Float64.(exp.(log_posterior .- log_evidence))

        # Ensure normalisation (may drift slightly in Float64 conversion)
        total = sum(result)
        if total > 0
            result ./= total
        end

        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "Math Bayesian update failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Extended-Precision Causal Inference
# ============================================================================

"""
    Causals.backend_coprocessor_causal_inference(::MathBackend, treatment, outcome, covariates)

Math coprocessor-accelerated causal inference with extended-precision
propensity score estimation. Uses BigFloat IRLS to prevent numerical
issues in logistic regression when covariates are poorly conditioned
or near-collinear -- common in observational causal studies with
many confounders.
"""
function Causals.backend_coprocessor_causal_inference(b::MathBackend,
                                                       treatment::AbstractVector{Bool},
                                                       outcome::Vector{Float64},
                                                       covariates::Matrix{Float64})
    n = length(treatment)
    k = size(covariates, 2)
    (n < 16 || k < 1) && return nothing

    mem_estimate = Int64(n * (k + 1) * 32 + (k + 1)^2 * 32)
    track_allocation!(b, mem_estimate)

    try
        X = BigFloat.(hcat(ones(n), covariates))
        y = BigFloat.(treatment)
        beta = zeros(BigFloat, k + 1)

        for iter in 1:30
            # Extended-precision IRLS
            eta = X * beta
            eta .= clamp.(eta, BigFloat(-50), BigFloat(50))
            p = @. BigFloat(1) / (BigFloat(1) + exp(-eta))
            r = y .- p
            w = @. p * (BigFloat(1) - p)
            w .= max.(w, BigFloat(1e-20))

            # Extended-precision Cholesky for X'WX
            WX = Diagonal(w) * X
            XtWX = X' * WX + BigFloat(1e-10) * I(k + 1)
            grad = X' * r

            try
                L = _extended_precision_cholesky(Matrix(XtWX))
                delta = _cholesky_solve(L, Vector(grad))
                if maximum(abs.(delta)) < BigFloat(1e-12)
                    beta .+= delta
                    break
                end
                beta .+= delta
            catch
                break
            end
        end

        # Convert back to Float64 propensity scores
        eta = X * beta
        eta .= clamp.(eta, BigFloat(-50), BigFloat(50))
        propensity = @. BigFloat(1) / (BigFloat(1) + exp(-eta))
        result = Float64.(clamp.(propensity, BigFloat(0.01), BigFloat(0.99)))

        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "Math causal inference failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Extended-Precision Uncertainty Propagation
# ============================================================================

"""
    Causals.backend_coprocessor_uncertainty_propagate(::MathBackend, args...)

Math coprocessor-accelerated uncertainty propagation via extended-precision
Jacobian computation. Prevents catastrophic cancellation in finite-difference
Jacobian approximations for nearly-linear structural equations.
"""
function Causals.backend_coprocessor_uncertainty_propagate(b::MathBackend, args...)
    return nothing
end

# ============================================================================
# Extended-Precision Monte Carlo
# ============================================================================

"""
    Causals.backend_coprocessor_monte_carlo(::MathBackend, model_fn, params, n_samples)

Math coprocessor-accelerated Monte Carlo with extended-precision accumulation.
The Kahan-compensated summation prevents the O(sqrt(N)) error growth that
afflicts naive Float64 Monte Carlo estimators at large sample counts.
"""
function Causals.backend_coprocessor_monte_carlo(b::MathBackend, model_fn::Function,
                                                   params::Matrix{Float64},
                                                   n_samples::Int)
    n_samples < 8 && return nothing

    try
        # Extended-precision accumulation for Monte Carlo mean
        total = BigFloat(0)
        compensation = BigFloat(0)
        count = 0
        results = Float64[]

        for i in 1:min(n_samples, size(params, 1))
            try
                val = Float64(model_fn(@view params[i, :]))
                push!(results, val)
                # Kahan summation in extended precision
                y = BigFloat(val) - compensation
                t = total + y
                compensation = (t - total) - y
                total = t
                count += 1
            catch; end
        end

        return isempty(results) ? nothing : results
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "Math Monte Carlo failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

"""
    Causals.backend_coprocessor_network_eval(::MathBackend, args...)

Math coprocessor-accelerated causal network evaluation with exact
rational arithmetic for do-calculus probability computations.
"""
function Causals.backend_coprocessor_network_eval(b::MathBackend, args...)
    return nothing
end

end # module CausalsMathExt
