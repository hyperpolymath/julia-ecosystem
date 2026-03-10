# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsMathExt -- Math Accelerator coprocessor extension for Cliometrics.jl
# Math accelerators provide hardware-optimised transcendental functions (log, exp,
# sin, cos) and high-precision matrix operations.  We exploit them for
# econometric computations that are bottlenecked by log/exp throughput.

module CliometricsMathExt

using Cliometrics
using MathAccel
using MathAccel: MathArray, math_log, math_exp, math_gemm, math_cholesky, math_sync
using AcceleratorGate: MathAccelBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# MathAccel Helpers
# ============================================================================

"""
    _math_ols(X, y) -> (beta, residuals, r_squared)

OLS via Cholesky factorisation on the math accelerator.
The accelerator provides hardware Cholesky for SPD matrices,
which is numerically superior to LU for normal equations.
"""
function _math_ols(X::Matrix{Float64}, y::Vector{Float64})
    n, k = size(X)
    d_X = MathArray(X)
    d_y = MathArray(y)

    # X'X via hardware GEMM
    XtX = Array(math_gemm(d_X', d_X)) + 1e-10 * I
    Xty = Array(math_gemm(d_X', reshape(d_y, :, 1)))

    # Cholesky solve (X'X is SPD)
    C = math_cholesky(MathArray(XtX))
    L = Array(C)  # lower triangular
    z = L \ vec(Xty)
    beta = L' \ z

    pred = X * beta
    residuals = y .- pred

    ss_res = sum(abs2, residuals)
    y_mean = sum(y) / n
    ss_tot = sum(abs2, y .- y_mean)
    r_squared = 1.0 - ss_res / ss_tot

    math_sync()
    return (beta, residuals, r_squared)
end

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::MathAccelBackend, X, y)

Math-accelerator OLS regression with hardware Cholesky factorisation.
The math coprocessor provides numerically stable SPD solves and
hardware-pipelined GEMM for the Gram matrix.
"""
function Cliometrics.backend_coprocessor_regression(b::MathAccelBackend,
                                                       X::Matrix{Float64},
                                                       y::Vector{Float64})
    n, _ = size(X)
    n < 64 && return nothing
    try
        beta, residuals, r_squared = _math_ols(X, y)
        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::MathAccelBackend, output, capital, labor, alpha)

Math-accelerator Solow decomposition.  Hardware `log` units compute
growth rates at full throughput -- the math accelerator's primary
advantage for this workload.
"""
function Cliometrics.backend_coprocessor_decomposition(b::MathAccelBackend,
                                                          output::Vector{Float64},
                                                          capital::Vector{Float64},
                                                          labor::Vector{Float64},
                                                          alpha::Float64)
    n = length(output)
    n < 64 && return nothing
    try
        d_Y = MathArray(output)
        d_K = MathArray(capital)
        d_L = MathArray(labor)

        # Hardware-accelerated log via dedicated transcendental units
        g_Y = Array(math_log(d_Y[2:end] ./ d_Y[1:end-1]))
        g_K = Array(math_log(d_K[2:end] ./ d_K[1:end-1]))
        g_L = Array(math_log(d_L[2:end] ./ d_L[1:end-1]))

        capital_contrib = alpha .* g_K
        labor_contrib   = (1.0 - alpha) .* g_L
        tfp_contrib     = g_Y .- capital_contrib .- labor_contrib

        math_sync()
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
    Cliometrics.backend_coprocessor_convergence_test(::MathAccelBackend, initial_levels, growth_rates)

Math-accelerator convergence test with hardware log and Cholesky solve.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::MathAccelBackend,
                                                             initial_levels::Vector{Float64},
                                                             growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 32 && return nothing
    try
        d_init = MathArray(initial_levels)
        log_initial = Array(math_log(d_init))
        X = hcat(ones(n), log_initial)

        beta, _, r_squared = _math_ols(X, growth_rates)

        alpha_coef = beta[1]
        beta_coef  = beta[2]
        is_converging = beta_coef < 0
        half_life = is_converging ? -log(2) / beta_coef : Inf

        return (alpha=alpha_coef, beta=beta_coef, r_squared=r_squared,
                converging=is_converging, half_life=half_life)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_time_series_filter(::MathAccelBackend, data, window_size)

Math-accelerator time series filter using hardware prefix-sum for
O(n) moving-average computation with full-precision accumulation.
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::MathAccelBackend,
                                                               data::Vector{Float64},
                                                               window_size::Int)
    n = length(data)
    n < 64 && return nothing
    try
        d_data = MathArray(data)
        # The math accelerator's prefix-sum is fully pipelined
        cs = Array(cumsum(d_data))

        m = n - window_size + 1
        result = Vector{Float64}(undef, m)
        for i in 1:m
            hi = cs[i + window_size - 1]
            lo = i == 1 ? 0.0 : cs[i - 1]
            result[i] = (hi - lo) / window_size
        end

        math_sync()
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::MathAccelBackend, panel_data, n_countries, alpha)

Math-accelerator batch growth accounting.  Hardware log units process
all country panel rows for vectorised growth-rate and TFP computation.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::MathAccelBackend,
                                                              panel_data::Matrix{Float64},
                                                              n_countries::Int,
                                                              alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 64 && return nothing
    try
        d_panel = MathArray(panel_data)
        cols = Array(d_panel)

        d_ratios_Y = MathArray(cols[2:end, 1] ./ cols[1:end-1, 1])
        d_ratios_K = MathArray(cols[2:end, 2] ./ cols[1:end-1, 2])
        d_ratios_L = MathArray(cols[2:end, 3] ./ cols[1:end-1, 3])

        g_Y = Array(math_log(d_ratios_Y))
        g_K = Array(math_log(d_ratios_K))
        g_L = Array(math_log(d_ratios_L))

        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L

        math_sync()
        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsMathExt
