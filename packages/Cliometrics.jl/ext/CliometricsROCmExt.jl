# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsROCmExt -- AMD ROCm GPU acceleration for Cliometrics.jl
# Parallel econometric computations on AMD GPUs.

module CliometricsROCmExt

using Cliometrics
using AMDGPU
using AMDGPU: ROCArray
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using AcceleratorGate: ROCmBackend, JuliaBackend, _record_diagnostic!
using LinearAlgebra

"""
    Cliometrics.backend_coprocessor_regression(::ROCmBackend, X, y)

ROCm GPU-accelerated OLS regression via normal equations.
"""
function Cliometrics.backend_coprocessor_regression(b::ROCmBackend,
                                                      X::Matrix{Float64},
                                                      y::Vector{Float64})
    n, k = size(X)
    n < 128 && return nothing

    try
        d_X = ROCArray(X)
        d_y = ROCArray(y)

        XtX = Array(d_X' * d_X) + 1e-10 * I
        Xty = Array(d_X' * d_y)
        beta = XtX \ Xty

        d_pred = d_X * ROCArray(beta)
        residuals = Array(d_y .- d_pred)

        ss_res = sum(residuals .^ 2)
        y_mean = sum(y) / n
        ss_tot = sum((y .- y_mean) .^ 2)
        r_squared = 1.0 - ss_res / ss_tot

        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::ROCmBackend, output, capital, labor, alpha)

ROCm-accelerated Solow growth decomposition on AMD GPUs.
"""
function Cliometrics.backend_coprocessor_decomposition(b::ROCmBackend,
                                                         output::Vector{Float64},
                                                         capital::Vector{Float64},
                                                         labor::Vector{Float64},
                                                         alpha::Float64)
    n = length(output)
    n < 128 && return nothing

    try
        d_Y = ROCArray(output)
        d_K = ROCArray(capital)
        d_L = ROCArray(labor)

        g_Y = Array(log.(d_Y[2:end] ./ d_Y[1:end-1]))
        g_K = Array(log.(d_K[2:end] ./ d_K[1:end-1]))
        g_L = Array(log.(d_L[2:end] ./ d_L[1:end-1]))

        capital_contrib = alpha .* g_K
        labor_contrib = (1.0 - alpha) .* g_L
        tfp_contrib = g_Y .- capital_contrib .- labor_contrib

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
    Cliometrics.backend_coprocessor_convergence_test(::ROCmBackend, initial_levels, growth_rates)

ROCm GPU-accelerated convergence regression.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::ROCmBackend,
                                                            initial_levels::Vector{Float64},
                                                            growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 64 && return nothing

    try
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)
        d_X = ROCArray(X)
        d_y = ROCArray(growth_rates)

        XtX = Array(d_X' * d_X) + 1e-10 * I
        Xty = Array(d_X' * d_y)
        beta = XtX \ Xty

        d_pred = d_X * ROCArray(beta)
        ss_res = sum(Array((d_y .- d_pred) .^ 2))
        y_mean = sum(growth_rates) / n
        ss_tot = sum((growth_rates .- y_mean) .^ 2)
        r_squared = 1.0 - ss_res / ss_tot

        is_converging = beta[2] < 0
        return (alpha=beta[1], beta=beta[2], r_squared=r_squared,
                converging=is_converging,
                half_life=is_converging ? -log(2) / beta[2] : Inf)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

function Cliometrics.backend_coprocessor_time_series_filter(b::ROCmBackend, args...)
    return nothing
end

function Cliometrics.backend_coprocessor_growth_accounting(b::ROCmBackend, args...)
    return nothing
end

end # module CliometricsROCmExt
