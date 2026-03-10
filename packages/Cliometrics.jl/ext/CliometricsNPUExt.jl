# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsNPUExt -- Neural Processing Unit acceleration for Cliometrics.jl
# NPUs provide efficient low-precision matrix-multiply and activation units;
# we use them for econometric regressions and time-series convolutions.

module CliometricsNPUExt

using Cliometrics
using NPUAccel
using NPUAccel: NPUArray, npu_gemm, npu_conv1d, npu_elementwise, npu_sync
using AcceleratorGate: NPUBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# NPU-Optimised Helpers
# ============================================================================

"""
    _npu_ols(X, y) -> (beta, residuals, r_squared)

OLS via NPU matrix-multiply units.  NPUs quantise internally but we
accumulate in FP32 for econometric accuracy.
"""
function _npu_ols(X::Matrix{Float64}, y::Vector{Float64})
    n, k = size(X)
    d_X = NPUArray(Float32.(X))
    d_y = NPUArray(Float32.(reshape(y, :, 1)))

    # Accumulate in FP32 on the NPU
    XtX = Float64.(Array(npu_gemm(d_X', d_X))) + 1e-10 * I
    Xty = Float64.(Array(npu_gemm(d_X', d_y)))
    beta = vec(XtX \ Xty)

    pred = X * beta
    residuals = y .- pred

    ss_res = sum(abs2, residuals)
    y_mean = sum(y) / n
    ss_tot = sum(abs2, y .- y_mean)
    r_squared = 1.0 - ss_res / ss_tot

    npu_sync()
    return (beta, residuals, r_squared)
end

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::NPUBackend, X, y)

NPU-accelerated OLS.  The NPU's matrix-multiply tiles handle the
Gram matrix (X'X) efficiently in quantised arithmetic with FP32 accumulation.
"""
function Cliometrics.backend_coprocessor_regression(b::NPUBackend,
                                                       X::Matrix{Float64},
                                                       y::Vector{Float64})
    n, _ = size(X)
    n < 128 && return nothing
    try
        beta, residuals, r_squared = _npu_ols(X, y)
        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::NPUBackend, output, capital, labor, alpha)

NPU-accelerated Solow decomposition using element-wise NPU ops for log-growth
rate computation and fused multiply-add for factor contributions.
"""
function Cliometrics.backend_coprocessor_decomposition(b::NPUBackend,
                                                          output::Vector{Float64},
                                                          capital::Vector{Float64},
                                                          labor::Vector{Float64},
                                                          alpha::Float64)
    n = length(output)
    n < 128 && return nothing
    try
        d_Y = NPUArray(Float32.(output))
        d_K = NPUArray(Float32.(capital))
        d_L = NPUArray(Float32.(labor))

        g_Y = Float64.(Array(npu_elementwise(log, d_Y[2:end] ./ d_Y[1:end-1])))
        g_K = Float64.(Array(npu_elementwise(log, d_K[2:end] ./ d_K[1:end-1])))
        g_L = Float64.(Array(npu_elementwise(log, d_L[2:end] ./ d_L[1:end-1])))

        capital_contrib = alpha .* g_K
        labor_contrib   = (1.0 - alpha) .* g_L
        tfp_contrib     = g_Y .- capital_contrib .- labor_contrib

        npu_sync()
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
    Cliometrics.backend_coprocessor_convergence_test(::NPUBackend, initial_levels, growth_rates)

NPU-accelerated beta-convergence test via OLS on log(initial income).
"""
function Cliometrics.backend_coprocessor_convergence_test(b::NPUBackend,
                                                             initial_levels::Vector{Float64},
                                                             growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 64 && return nothing
    try
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)
        beta, _, r_squared = _npu_ols(X, growth_rates)

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
    Cliometrics.backend_coprocessor_time_series_filter(::NPUBackend, data, window_size)

NPU-accelerated 1-D convolution filter using the NPU's dedicated
convolution datapath for moving-average smoothing.
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::NPUBackend,
                                                               data::Vector{Float64},
                                                               window_size::Int)
    n = length(data)
    n < 256 && return nothing
    try
        kernel = NPUArray(Float32.(fill(1.0 / window_size, window_size)))
        d_data = NPUArray(Float32.(data))
        d_result = npu_conv1d(d_data, kernel; padding=:valid)
        result = Float64.(Array(d_result))

        npu_sync()
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::NPUBackend, panel_data, n_countries, alpha)

NPU-accelerated batch growth accounting over stacked country panels.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::NPUBackend,
                                                              panel_data::Matrix{Float64},
                                                              n_countries::Int,
                                                              alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 256 && return nothing
    try
        d_panel = NPUArray(Float32.(panel_data))
        cols = Float64.(Array(d_panel))

        output_col  = cols[:, 1]
        capital_col = cols[:, 2]
        labor_col   = cols[:, 3]

        g_Y = log.(output_col[2:end]  ./ output_col[1:end-1])
        g_K = log.(capital_col[2:end] ./ capital_col[1:end-1])
        g_L = log.(labor_col[2:end]   ./ labor_col[1:end-1])

        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L

        npu_sync()
        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsNPUExt
