# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsMetalExt -- Apple Metal GPU acceleration for Cliometrics.jl
# Parallel econometric computations on Apple Silicon using Float32 precision.

module CliometricsMetalExt

using Cliometrics
using Metal
using Metal: MtlArray
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using AcceleratorGate: MetalBackend, JuliaBackend, _record_diagnostic!
using LinearAlgebra

"""
    Cliometrics.backend_coprocessor_regression(::MetalBackend, X, y)

Metal GPU-accelerated OLS regression in Float32 precision for Apple Silicon.
"""
function Cliometrics.backend_coprocessor_regression(b::MetalBackend,
                                                      X::Matrix{Float64},
                                                      y::Vector{Float64})
    n, k = size(X)
    n < 128 && return nothing

    try
        X32 = Float32.(X)
        y32 = Float32.(y)
        d_X = MtlArray(X32)
        d_y = MtlArray(y32)

        XtX = Array(d_X' * d_X) + 1f-8 * I
        Xty = Array(d_X' * d_y)
        beta = XtX \ Xty

        d_pred = d_X * MtlArray(beta)
        residuals = Float64.(Array(d_y .- d_pred))

        ss_res = sum(residuals .^ 2)
        y_mean = sum(y) / n
        ss_tot = sum((y .- y_mean) .^ 2)
        r_squared = 1.0 - ss_res / ss_tot

        return (coefficients=Float64.(beta), residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::MetalBackend, output, capital, labor, alpha)

Metal GPU-accelerated Solow growth decomposition in Float32.
"""
function Cliometrics.backend_coprocessor_decomposition(b::MetalBackend,
                                                         output::Vector{Float64},
                                                         capital::Vector{Float64},
                                                         labor::Vector{Float64},
                                                         alpha::Float64)
    n = length(output)
    n < 128 && return nothing

    try
        d_Y = MtlArray(Float32.(output))
        d_K = MtlArray(Float32.(capital))
        d_L = MtlArray(Float32.(labor))

        g_Y = Float64.(Array(log.(d_Y[2:end] ./ d_Y[1:end-1])))
        g_K = Float64.(Array(log.(d_K[2:end] ./ d_K[1:end-1])))
        g_L = Float64.(Array(log.(d_L[2:end] ./ d_L[1:end-1])))

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
    Cliometrics.backend_coprocessor_convergence_test(::MetalBackend, initial_levels, growth_rates)

Metal GPU-accelerated convergence regression in Float32.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::MetalBackend,
                                                            initial_levels::Vector{Float64},
                                                            growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 64 && return nothing

    try
        log_initial = Float32.(log.(initial_levels))
        X = hcat(ones(Float32, n), log_initial)
        d_X = MtlArray(X)
        d_y = MtlArray(Float32.(growth_rates))

        XtX = Array(d_X' * d_X) + 1f-8 * I
        Xty = Array(d_X' * d_y)
        beta = XtX \ Xty

        d_pred = d_X * MtlArray(beta)
        ss_res = sum(Array((d_y .- d_pred) .^ 2))
        y_mean = sum(growth_rates) / n
        ss_tot = sum((growth_rates .- y_mean) .^ 2)
        r_squared = Float64(1.0 - ss_res / ss_tot)

        beta_coef = Float64(beta[2])
        is_converging = beta_coef < 0
        return (alpha=Float64(beta[1]), beta=beta_coef, r_squared=r_squared,
                converging=is_converging,
                half_life=is_converging ? -log(2) / beta_coef : Inf)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

function Cliometrics.backend_coprocessor_time_series_filter(b::MetalBackend, args...)
    return nothing
end

function Cliometrics.backend_coprocessor_growth_accounting(b::MetalBackend, args...)
    return nothing
end

end # module CliometricsMetalExt
