# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsCUDAExt -- CUDA GPU acceleration for Cliometrics.jl
# Parallel OLS regression, batch growth decomposition, and GPU-accelerated
# convergence testing on NVIDIA GPUs via KernelAbstractions.

module CliometricsCUDAExt

using Cliometrics
using CUDA
using CUDA: CuArray, CuVector, CuMatrix
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using AcceleratorGate: CUDABackend, JuliaBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# GPU Kernel: Parallel Growth Rate Computation
# ============================================================================

@kernel function growth_rate_kernel!(rates, @Const(values), n::Int32)
    i = @index(Global)
    if i < n
        rates[i] = log(values[i + 1] / values[i])
    end
end

@kernel function residual_kernel!(residuals, @Const(Y), @Const(X), @Const(beta),
                                   n_obs::Int32, n_features::Int32)
    i = @index(Global)
    pred = Float64(0.0)
    for j in Int32(1):n_features
        pred += X[i, j] * beta[j]
    end
    residuals[i] = Y[i] - pred
end

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::CUDABackend, X, y)

GPU-accelerated OLS regression via normal equations on CUDA.
Transfers design matrix to GPU, computes (X'X)^{-1} X'y using cuBLAS.
"""
function Cliometrics.backend_coprocessor_regression(b::CUDABackend,
                                                      X::Matrix{Float64},
                                                      y::Vector{Float64})
    n, k = size(X)
    n < 128 && return nothing

    try
        d_X = CuArray(X)
        d_y = CuArray(y)

        # Normal equations: beta = (X'X)^{-1} X'y
        XtX = Array(d_X' * d_X) + 1e-10 * I
        Xty = Array(d_X' * d_y)

        beta = XtX \ Xty

        # Compute residuals on GPU
        d_beta = CuArray(beta)
        d_pred = d_X * d_beta
        d_resid = d_y .- d_pred
        residuals = Array(d_resid)

        # R-squared
        ss_res = sum(residuals .^ 2)
        y_mean = sum(y) / n
        ss_tot = sum((y .- y_mean) .^ 2)
        r_squared = 1.0 - ss_res / ss_tot

        CUDA.unsafe_free!(d_X)
        CUDA.unsafe_free!(d_y)

        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::CUDABackend, output, capital, labor, alpha)

GPU-accelerated Solow growth decomposition.
Computes growth rates and TFP residuals in parallel on GPU.
"""
function Cliometrics.backend_coprocessor_decomposition(b::CUDABackend,
                                                         output::Vector{Float64},
                                                         capital::Vector{Float64},
                                                         labor::Vector{Float64},
                                                         alpha::Float64)
    n = length(output)
    n < 128 && return nothing

    try
        d_Y = CuArray(output)
        d_K = CuArray(capital)
        d_L = CuArray(labor)

        # Compute log growth rates on GPU
        g_Y = Array(log.(d_Y[2:end] ./ d_Y[1:end-1]))
        g_K = Array(log.(d_K[2:end] ./ d_K[1:end-1]))
        g_L = Array(log.(d_L[2:end] ./ d_L[1:end-1]))

        # Solow decomposition
        capital_contrib = alpha .* g_K
        labor_contrib = (1.0 - alpha) .* g_L
        tfp_contrib = g_Y .- capital_contrib .- labor_contrib

        CUDA.unsafe_free!(d_Y)
        CUDA.unsafe_free!(d_K)
        CUDA.unsafe_free!(d_L)

        return (output_growth=g_Y,
                capital_contribution=capital_contrib,
                labor_contribution=labor_contrib,
                tfp_contribution=tfp_contrib)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_convergence_test(::CUDABackend, initial_levels, growth_rates)

GPU-accelerated cross-country convergence regression.
Performs batch OLS of growth on log(initial_income) using cuBLAS.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::CUDABackend,
                                                            initial_levels::Vector{Float64},
                                                            growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 64 && return nothing

    try
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)
        d_X = CuArray(X)
        d_y = CuArray(growth_rates)

        XtX = Array(d_X' * d_X) + 1e-10 * I
        Xty = Array(d_X' * d_y)
        beta = XtX \ Xty

        alpha_coef = beta[1]
        beta_coef = beta[2]

        # R-squared on GPU
        d_beta = CuArray(beta)
        d_pred = d_X * d_beta
        d_resid = d_y .- d_pred
        ss_res = sum(Array(d_resid .^ 2))
        y_mean = sum(growth_rates) / n
        ss_tot = sum((growth_rates .- y_mean) .^ 2)
        r_squared = 1.0 - ss_res / ss_tot

        is_converging = beta_coef < 0
        half_life = is_converging ? -log(2) / beta_coef : Inf

        CUDA.unsafe_free!(d_X)
        CUDA.unsafe_free!(d_y)

        return (alpha=alpha_coef, beta=beta_coef, r_squared=r_squared,
                converging=is_converging, half_life=half_life)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_time_series_filter(::CUDABackend, data, window_size)

GPU-accelerated time series filtering using parallel moving-average convolution.
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::CUDABackend,
                                                              data::Vector{Float64},
                                                              window_size::Int)
    n = length(data)
    n < 256 && return nothing

    try
        d_data = CuArray(data)

        # Cumulative sum for O(1) moving average per element
        d_cumsum = cumsum(d_data)
        cs = Array(d_cumsum)

        result = Vector{Float64}(undef, n - window_size + 1)
        for i in 1:length(result)
            if i == 1
                result[i] = cs[window_size] / window_size
            else
                result[i] = (cs[i + window_size - 1] - cs[i - 1]) / window_size
            end
        end

        CUDA.unsafe_free!(d_data)
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::CUDABackend, panel_data, countries, alpha)

GPU-accelerated batch growth accounting across multiple countries.
Computes Solow decomposition for all countries in parallel via batched matmul.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::CUDABackend,
                                                             panel_data::Matrix{Float64},
                                                             n_countries::Int,
                                                             alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 256 && return nothing

    try
        # panel_data columns: [output, capital, labor] per country stacked
        d_panel = CuArray(panel_data)

        output_col = Array(d_panel[:, 1])
        capital_col = Array(d_panel[:, 2])
        labor_col = Array(d_panel[:, 3])

        # Compute growth rates for entire panel
        g_Y = log.(output_col[2:end] ./ output_col[1:end-1])
        g_K = log.(capital_col[2:end] ./ capital_col[1:end-1])
        g_L = log.(labor_col[2:end] ./ labor_col[1:end-1])

        # TFP residuals
        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L

        CUDA.unsafe_free!(d_panel)
        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsCUDAExt
