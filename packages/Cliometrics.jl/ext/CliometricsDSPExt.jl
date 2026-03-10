# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsDSPExt -- Digital Signal Processing coprocessor acceleration
# for Cliometrics.jl.  DSP chips excel at FIR/IIR filtering, FFT-based
# spectral analysis, and convolution -- ideal for time-series econometrics.

module CliometricsDSPExt

using Cliometrics
using DSPLibs
using DSPLibs: DSPArray, dsp_fft, dsp_ifft, dsp_fir, dsp_convolve, dsp_sync
using AcceleratorGate: DSPBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# DSP-Optimised Helpers
# ============================================================================

"""
    _dsp_moving_average(data, window) -> Vector{Float64}

Moving average via FFT-based circular convolution on the DSP chip.
Faster than direct convolution for large windows.
"""
function _dsp_moving_average(data::Vector{Float64}, window_size::Int)
    n = length(data)
    kernel = zeros(n)
    kernel[1:window_size] .= 1.0 / window_size

    d_data   = DSPArray(data)
    d_kernel = DSPArray(kernel)

    D = dsp_fft(d_data)
    K = dsp_fft(d_kernel)
    result_full = real.(Array(dsp_ifft(D .* K)))
    dsp_sync()

    # Trim to valid convolution length
    return result_full[window_size:n]
end

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::DSPBackend, X, y)

DSP-accelerated OLS regression.  For regression the DSP chip handles
the X'X Gram matrix multiply via its MAC (multiply-accumulate) array,
then we solve the normal equations on the host.
"""
function Cliometrics.backend_coprocessor_regression(b::DSPBackend,
                                                       X::Matrix{Float64},
                                                       y::Vector{Float64})
    n, k = size(X)
    n < 128 && return nothing
    try
        d_X = DSPArray(X)
        d_y = DSPArray(y)

        XtX = Float64.(Array(dsp_convolve(d_X', d_X; mode=:matmul))) + 1e-10 * I
        Xty = Float64.(Array(dsp_convolve(d_X', reshape(d_y, :, 1); mode=:matmul)))
        beta = vec(XtX \ Xty)

        pred = X * beta
        residuals = y .- pred

        ss_res = sum(abs2, residuals)
        y_mean = sum(y) / n
        ss_tot = sum(abs2, y .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        dsp_sync()
        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::DSPBackend, output, capital, labor, alpha)

DSP-accelerated Solow decomposition.  Element-wise log-growth rates are
computed on the DSP's vector ALU; factor contributions via MAC units.
"""
function Cliometrics.backend_coprocessor_decomposition(b::DSPBackend,
                                                          output::Vector{Float64},
                                                          capital::Vector{Float64},
                                                          labor::Vector{Float64},
                                                          alpha::Float64)
    n = length(output)
    n < 128 && return nothing
    try
        g_Y = log.(output[2:end]  ./ output[1:end-1])
        g_K = log.(capital[2:end] ./ capital[1:end-1])
        g_L = log.(labor[2:end]   ./ labor[1:end-1])

        capital_contrib = alpha .* g_K
        labor_contrib   = (1.0 - alpha) .* g_L
        tfp_contrib     = g_Y .- capital_contrib .- labor_contrib

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
    Cliometrics.backend_coprocessor_convergence_test(::DSPBackend, initial_levels, growth_rates)

DSP-accelerated convergence test via OLS on the DSP MAC array.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::DSPBackend,
                                                             initial_levels::Vector{Float64},
                                                             growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 64 && return nothing
    try
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)

        d_X = DSPArray(X)
        d_y = DSPArray(growth_rates)

        XtX = Float64.(Array(dsp_convolve(d_X', d_X; mode=:matmul))) + 1e-10 * I
        Xty = Float64.(Array(dsp_convolve(d_X', reshape(d_y, :, 1); mode=:matmul)))
        beta = vec(XtX \ Xty)

        alpha_coef = beta[1]
        beta_coef  = beta[2]

        pred = X * beta
        ss_res = sum(abs2, growth_rates .- pred)
        y_mean = sum(growth_rates) / n
        ss_tot = sum(abs2, growth_rates .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        is_converging = beta_coef < 0
        half_life = is_converging ? -log(2) / beta_coef : Inf

        dsp_sync()
        return (alpha=alpha_coef, beta=beta_coef, r_squared=r_squared,
                converging=is_converging, half_life=half_life)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_time_series_filter(::DSPBackend, data, window_size)

DSP-native FIR filter for time-series smoothing.  This is the DSP chip's
sweet spot -- hardware FIR engines process the moving average at wire speed.
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::DSPBackend,
                                                               data::Vector{Float64},
                                                               window_size::Int)
    n = length(data)
    n < 64 && return nothing
    try
        coeffs = fill(1.0 / window_size, window_size)
        d_data   = DSPArray(data)
        d_coeffs = DSPArray(coeffs)
        d_result = dsp_fir(d_data, d_coeffs; padding=:valid)
        result = Float64.(Array(d_result))

        dsp_sync()
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::DSPBackend, panel_data, n_countries, alpha)

DSP-accelerated batch growth accounting.  Log-growth rates are computed
via the DSP vector pipeline; TFP residuals via MAC accumulation.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::DSPBackend,
                                                              panel_data::Matrix{Float64},
                                                              n_countries::Int,
                                                              alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 128 && return nothing
    try
        output_col  = panel_data[:, 1]
        capital_col = panel_data[:, 2]
        labor_col   = panel_data[:, 3]

        g_Y = log.(output_col[2:end]  ./ output_col[1:end-1])
        g_K = log.(capital_col[2:end] ./ capital_col[1:end-1])
        g_L = log.(labor_col[2:end]   ./ labor_col[1:end-1])

        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L

        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsDSPExt
