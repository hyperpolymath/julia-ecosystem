# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsVPUExt -- Vision Processing Unit acceleration for Cliometrics.jl
# VPUs provide wide SIMD vector pipelines originally designed for image
# processing; we repurpose them for bulk element-wise econometric operations,
# vectorised log-growth computation, and sliding-window convolutions.

module CliometricsVPUExt

using Cliometrics
using VPURuntime
using VPURuntime: VPUVector, vpu_map, vpu_reduce, vpu_sliding_window, vpu_sync
using AcceleratorGate: VPUBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::VPUBackend, X, y)

VPU-accelerated OLS.  The VPU's wide SIMD lanes compute the Gram matrix
row dot-products in parallel; the solve is performed on the host.
"""
function Cliometrics.backend_coprocessor_regression(b::VPUBackend,
                                                       X::Matrix{Float64},
                                                       y::Vector{Float64})
    n, k = size(X)
    n < 128 && return nothing
    try
        # Use VPU vectorised dot products for X'X
        XtX = Matrix{Float64}(undef, k, k)
        for j in 1:k
            col_j = VPUVector(X[:, j])
            for i in j:k
                col_i = VPUVector(X[:, i])
                dot_val = vpu_reduce(+, col_i .* col_j)
                XtX[i, j] = dot_val
                XtX[j, i] = dot_val
            end
        end
        XtX += 1e-10 * I

        Xty = Vector{Float64}(undef, k)
        d_y = VPUVector(y)
        for j in 1:k
            col_j = VPUVector(X[:, j])
            Xty[j] = vpu_reduce(+, col_j .* d_y)
        end

        beta = XtX \ Xty
        residuals = y .- X * beta

        ss_res = sum(abs2, residuals)
        y_mean = sum(y) / n
        ss_tot = sum(abs2, y .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        vpu_sync()
        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::VPUBackend, output, capital, labor, alpha)

VPU-accelerated Solow decomposition.  The VPU's SIMD lanes compute
element-wise log ratios across all three factor series in parallel.
"""
function Cliometrics.backend_coprocessor_decomposition(b::VPUBackend,
                                                          output::Vector{Float64},
                                                          capital::Vector{Float64},
                                                          labor::Vector{Float64},
                                                          alpha::Float64)
    n = length(output)
    n < 64 && return nothing
    try
        d_Y = VPUVector(output)
        d_K = VPUVector(capital)
        d_L = VPUVector(labor)

        g_Y = Array(vpu_map(log, d_Y[2:end] ./ d_Y[1:end-1]))
        g_K = Array(vpu_map(log, d_K[2:end] ./ d_K[1:end-1]))
        g_L = Array(vpu_map(log, d_L[2:end] ./ d_L[1:end-1]))

        capital_contrib = alpha .* g_K
        labor_contrib   = (1.0 - alpha) .* g_L
        tfp_contrib     = g_Y .- capital_contrib .- labor_contrib

        vpu_sync()
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
    Cliometrics.backend_coprocessor_convergence_test(::VPUBackend, initial_levels, growth_rates)

VPU-accelerated convergence regression using SIMD dot-product lanes.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::VPUBackend,
                                                             initial_levels::Vector{Float64},
                                                             growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 64 && return nothing
    try
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)

        # VPU dot products for 2x2 Gram matrix
        d_log  = VPUVector(log_initial)
        d_y    = VPUVector(growth_rates)

        XtX = [Float64(n)                       vpu_reduce(+, d_log);
               vpu_reduce(+, d_log)             vpu_reduce(+, d_log .* d_log)] + 1e-10 * I
        Xty = [vpu_reduce(+, d_y); vpu_reduce(+, d_log .* d_y)]

        beta = XtX \ Xty
        alpha_coef = beta[1]
        beta_coef  = beta[2]

        pred = X * beta
        ss_res = sum(abs2, growth_rates .- pred)
        y_mean = sum(growth_rates) / n
        ss_tot = sum(abs2, growth_rates .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        is_converging = beta_coef < 0
        half_life = is_converging ? -log(2) / beta_coef : Inf

        vpu_sync()
        return (alpha=alpha_coef, beta=beta_coef, r_squared=r_squared,
                converging=is_converging, half_life=half_life)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_time_series_filter(::VPUBackend, data, window_size)

VPU-accelerated sliding-window filter.  The VPU's native sliding-window
instruction computes the moving average using its image-convolution hardware.
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::VPUBackend,
                                                               data::Vector{Float64},
                                                               window_size::Int)
    n = length(data)
    n < 64 && return nothing
    try
        d_data = VPUVector(data)
        d_sums = vpu_sliding_window(+, d_data, window_size)
        result = Array(d_sums) ./ window_size

        vpu_sync()
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::VPUBackend, panel_data, n_countries, alpha)

VPU-accelerated batch growth accounting.  SIMD lanes process all country
rows for vectorised log-growth and TFP residual computation.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::VPUBackend,
                                                              panel_data::Matrix{Float64},
                                                              n_countries::Int,
                                                              alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 128 && return nothing
    try
        d_Y = VPUVector(panel_data[:, 1])
        d_K = VPUVector(panel_data[:, 2])
        d_L = VPUVector(panel_data[:, 3])

        g_Y = Array(vpu_map(log, d_Y[2:end] ./ d_Y[1:end-1]))
        g_K = Array(vpu_map(log, d_K[2:end] ./ d_K[1:end-1]))
        g_L = Array(vpu_map(log, d_L[2:end] ./ d_L[1:end-1]))

        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L

        vpu_sync()
        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsVPUExt
