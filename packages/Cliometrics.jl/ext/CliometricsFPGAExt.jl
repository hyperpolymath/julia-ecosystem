# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsFPGAExt -- FPGA acceleration for Cliometrics.jl
# FPGAs offer custom-pipeline parallelism with deterministic latency.
# We synthesise fixed-point datapaths for regression and time-series
# filtering where the FPGA's reconfigurable fabric excels.

module CliometricsFPGAExt

using Cliometrics
using FPGASynthesis
using FPGASynthesis: FPGABuffer, fpga_submit, fpga_await, fpga_pipeline, fpga_sync
using AcceleratorGate: FPGABackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# FPGA Pipeline Definitions
# ============================================================================

# Pre-defined bitstream handles for econometric kernels
const PIPELINE_OLS       = :cliometrics_ols_fp64
const PIPELINE_GROWTHLOG = :cliometrics_log_growth
const PIPELINE_FIR       = :cliometrics_fir_filter
const PIPELINE_SOLOW     = :cliometrics_solow_decomp

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::FPGABackend, X, y)

FPGA-accelerated OLS via a synthesised fixed-point pipeline.
The FPGA fabric computes X'X and X'y in a single-pass streaming pipeline
with deterministic latency proportional to n.
"""
function Cliometrics.backend_coprocessor_regression(b::FPGABackend,
                                                       X::Matrix{Float64},
                                                       y::Vector{Float64})
    n, k = size(X)
    n < 64 && return nothing
    try
        buf_X = FPGABuffer(X)
        buf_y = FPGABuffer(y)

        handle = fpga_submit(PIPELINE_OLS, buf_X, buf_y; n_features=k)
        raw = fpga_await(handle)

        beta      = raw.coefficients::Vector{Float64}
        residuals = y .- X * beta

        ss_res = sum(abs2, residuals)
        y_mean = sum(y) / n
        ss_tot = sum(abs2, y .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        fpga_sync()
        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::FPGABackend, output, capital, labor, alpha)

FPGA-accelerated Solow decomposition via pipelined log-growth computation.
The FPGA computes log(v[i+1]/v[i]) in a streaming fashion for all three
factor series simultaneously.
"""
function Cliometrics.backend_coprocessor_decomposition(b::FPGABackend,
                                                          output::Vector{Float64},
                                                          capital::Vector{Float64},
                                                          labor::Vector{Float64},
                                                          alpha::Float64)
    n = length(output)
    n < 64 && return nothing
    try
        buf_Y = FPGABuffer(output)
        buf_K = FPGABuffer(capital)
        buf_L = FPGABuffer(labor)

        handle = fpga_submit(PIPELINE_SOLOW, buf_Y, buf_K, buf_L; alpha=alpha)
        raw = fpga_await(handle)

        g_Y = raw.output_growth::Vector{Float64}
        capital_contrib = raw.capital_contribution::Vector{Float64}
        labor_contrib   = raw.labor_contribution::Vector{Float64}
        tfp_contrib     = g_Y .- capital_contrib .- labor_contrib

        fpga_sync()
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
    Cliometrics.backend_coprocessor_convergence_test(::FPGABackend, initial_levels, growth_rates)

FPGA-accelerated beta-convergence regression via pipelined OLS.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::FPGABackend,
                                                             initial_levels::Vector{Float64},
                                                             growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 32 && return nothing
    try
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)

        buf_X = FPGABuffer(X)
        buf_y = FPGABuffer(growth_rates)

        handle = fpga_submit(PIPELINE_OLS, buf_X, buf_y; n_features=2)
        raw = fpga_await(handle)

        beta = raw.coefficients::Vector{Float64}
        alpha_coef = beta[1]
        beta_coef  = beta[2]

        pred = X * beta
        ss_res = sum(abs2, growth_rates .- pred)
        y_mean = sum(growth_rates) / n
        ss_tot = sum(abs2, growth_rates .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        is_converging = beta_coef < 0
        half_life = is_converging ? -log(2) / beta_coef : Inf

        fpga_sync()
        return (alpha=alpha_coef, beta=beta_coef, r_squared=r_squared,
                converging=is_converging, half_life=half_life)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_time_series_filter(::FPGABackend, data, window_size)

FPGA-native FIR filter.  The reconfigurable fabric implements a
hardware FIR tap-chain with window_size coefficients, streaming data
through at one sample per clock cycle.
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::FPGABackend,
                                                               data::Vector{Float64},
                                                               window_size::Int)
    n = length(data)
    n < 32 && return nothing
    try
        coeffs = fill(1.0 / window_size, window_size)
        buf_data   = FPGABuffer(data)
        buf_coeffs = FPGABuffer(coeffs)

        handle = fpga_submit(PIPELINE_FIR, buf_data, buf_coeffs; padding=:valid)
        raw = fpga_await(handle)

        fpga_sync()
        return raw.filtered::Vector{Float64}
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::FPGABackend, panel_data, n_countries, alpha)

FPGA-accelerated batch growth accounting.  The pipeline processes all
country rows in a single streaming pass through the log-growth datapath.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::FPGABackend,
                                                              panel_data::Matrix{Float64},
                                                              n_countries::Int,
                                                              alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 64 && return nothing
    try
        buf_panel = FPGABuffer(panel_data)

        handle = fpga_submit(PIPELINE_SOLOW, buf_panel; alpha=alpha, batch=true)
        raw = fpga_await(handle)

        g_Y = raw.growth_rates::Vector{Float64}
        tfp = raw.tfp_residuals::Vector{Float64}

        fpga_sync()
        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsFPGAExt
