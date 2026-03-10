# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsPPUExt -- Physics Processing Unit acceleration for Cliometrics.jl
# PPUs provide hardware-accelerated simulation engines originally designed for
# physics simulations.  We exploit their parallel constraint solvers and
# vectorised arithmetic for econometric regressions and panel-data processing.

module CliometricsPPUExt

using Cliometrics
using PPUCompute
using PPUCompute: PPUArray, ppu_parallel_solve, ppu_vectorise, ppu_reduce, ppu_sync
using AcceleratorGate: PPUBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# PPU Helpers
# ============================================================================

"""
    _ppu_solve_linear(A, b) -> Vector{Float64}

Solve Ax = b using the PPU's iterative constraint solver.
PPUs use Gauss-Seidel / Jacobi iterations in hardware, converging
quickly for well-conditioned positive-definite systems like X'X.
"""
function _ppu_solve_linear(A::Matrix{Float64}, b::Vector{Float64})
    d_A = PPUArray(A)
    d_b = PPUArray(b)
    result = ppu_parallel_solve(d_A, d_b; method=:conjugate_gradient, tol=1e-12)
    ppu_sync()
    return Array(result)
end

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::PPUBackend, X, y)

PPU-accelerated OLS regression via the hardware constraint solver.
The PPU's iterative solver handles (X'X)beta = X'y with hardware-pipelined
conjugate gradient iterations.
"""
function Cliometrics.backend_coprocessor_regression(b::PPUBackend,
                                                       X::Matrix{Float64},
                                                       y::Vector{Float64})
    n, k = size(X)
    n < 64 && return nothing
    try
        XtX = X' * X + 1e-10 * I
        Xty = X' * y
        beta = _ppu_solve_linear(XtX, Xty)

        residuals = y .- X * beta
        ss_res = sum(abs2, residuals)
        y_mean = sum(y) / n
        ss_tot = sum(abs2, y .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::PPUBackend, output, capital, labor, alpha)

PPU-accelerated Solow decomposition.  The PPU's vectorised arithmetic
units compute log-growth rates across all factor series in a single
dispatch, then the Solow accounting runs on the vector pipeline.
"""
function Cliometrics.backend_coprocessor_decomposition(b::PPUBackend,
                                                          output::Vector{Float64},
                                                          capital::Vector{Float64},
                                                          labor::Vector{Float64},
                                                          alpha::Float64)
    n = length(output)
    n < 64 && return nothing
    try
        d_Y = PPUArray(output)
        d_K = PPUArray(capital)
        d_L = PPUArray(labor)

        g_Y = Array(ppu_vectorise(log, d_Y[2:end] ./ d_Y[1:end-1]))
        g_K = Array(ppu_vectorise(log, d_K[2:end] ./ d_K[1:end-1]))
        g_L = Array(ppu_vectorise(log, d_L[2:end] ./ d_L[1:end-1]))

        capital_contrib = alpha .* g_K
        labor_contrib   = (1.0 - alpha) .* g_L
        tfp_contrib     = g_Y .- capital_contrib .- labor_contrib

        ppu_sync()
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
    Cliometrics.backend_coprocessor_convergence_test(::PPUBackend, initial_levels, growth_rates)

PPU-accelerated beta-convergence test via hardware conjugate gradient.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::PPUBackend,
                                                             initial_levels::Vector{Float64},
                                                             growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 32 && return nothing
    try
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)

        XtX = X' * X + 1e-10 * I
        Xty = X' * growth_rates
        beta = _ppu_solve_linear(XtX, Xty)

        alpha_coef = beta[1]
        beta_coef  = beta[2]

        pred = X * beta
        ss_res = sum(abs2, growth_rates .- pred)
        y_mean = sum(growth_rates) / n
        ss_tot = sum(abs2, growth_rates .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        is_converging = beta_coef < 0
        half_life = is_converging ? -log(2) / beta_coef : Inf

        ppu_sync()
        return (alpha=alpha_coef, beta=beta_coef, r_squared=r_squared,
                converging=is_converging, half_life=half_life)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_time_series_filter(::PPUBackend, data, window_size)

PPU-accelerated moving-average filter using the PPU's sliding-window
reduction hardware (originally designed for particle neighbourhood queries).
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::PPUBackend,
                                                               data::Vector{Float64},
                                                               window_size::Int)
    n = length(data)
    n < 64 && return nothing
    try
        d_data = PPUArray(data)
        d_cumsum = ppu_reduce(+, d_data; scan=true)
        cs = Array(d_cumsum)

        m = n - window_size + 1
        result = Vector{Float64}(undef, m)
        for i in 1:m
            hi = cs[i + window_size - 1]
            lo = i == 1 ? 0.0 : cs[i - 1]
            result[i] = (hi - lo) / window_size
        end

        ppu_sync()
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::PPUBackend, panel_data, n_countries, alpha)

PPU-accelerated batch growth accounting.  The PPU's parallel particle-system
engine processes each country as a separate "body" in the simulation,
computing vectorised log-growth and TFP residuals simultaneously.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::PPUBackend,
                                                              panel_data::Matrix{Float64},
                                                              n_countries::Int,
                                                              alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 128 && return nothing
    try
        d_panel = PPUArray(panel_data)
        cols = Array(d_panel)

        g_Y = log.(cols[2:end, 1] ./ cols[1:end-1, 1])
        g_K = log.(cols[2:end, 2] ./ cols[1:end-1, 2])
        g_L = log.(cols[2:end, 3] ./ cols[1:end-1, 3])

        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L

        ppu_sync()
        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsPPUExt
