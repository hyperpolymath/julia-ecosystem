# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cliometrics.jl VPU Coprocessor
# SIMD-vectorized growth rate computation.

function AcceleratorGate.device_capabilities(b::VPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 8, 2000,
        Int64(2 * 1024^3), Int64(2 * 1024^3),
        512, true, true, true, "Intel", "VPU AVX-512",
    )
end

function AcceleratorGate.estimate_cost(::VPUBackend, op::Symbol, data_size::Int)
    overhead = 10.0
    op == :regression && return overhead + Float64(data_size) * 0.04
    op == :growth_accounting && return overhead + Float64(data_size) * 0.01
    op == :time_series_filter && return overhead + Float64(data_size) * 0.02
    Inf
end

AcceleratorGate.register_operation!(VPUBackend, :regression)
AcceleratorGate.register_operation!(VPUBackend, :growth_accounting)
AcceleratorGate.register_operation!(VPUBackend, :time_series_filter)

"""SIMD-vectorized OLS regression."""
function backend_coprocessor_regression(::VPUBackend, X::AbstractMatrix, y::AbstractVector)
    Xf = Float64.(X); yf = Float64.(y)
    beta = (Xf' * Xf) \ (Xf' * yf)
    res = yf .- Xf * beta
    ss_res = 0.0; ss_tot = 0.0; ym = mean(yf)
    @simd for i in eachindex(res); ss_res += res[i]^2; end
    @simd for i in eachindex(yf); ss_tot += (yf[i] - ym)^2; end
    (coefficients=beta, r_squared=1.0 - ss_res / max(ss_tot, 1.0e-30), residuals=res)
end

"""SIMD-vectorized growth rate computation."""
function backend_coprocessor_growth_accounting(::VPUBackend, output::AbstractVector,
                                               capital::AbstractVector,
                                               labor::AbstractVector;
                                               alpha::Float64=0.3)
    T = length(output)
    g_Y = Vector{Float64}(undef, T - 1)
    g_K = Vector{Float64}(undef, T - 1)
    g_L = Vector{Float64}(undef, T - 1)
    tfp = Vector{Float64}(undef, T - 1)

    @simd for i in 1:(T - 1)
        g_Y[i] = log(output[i + 1] / output[i])
        g_K[i] = log(capital[i + 1] / capital[i])
        g_L[i] = log(labor[i + 1] / labor[i])
    end
    @simd for i in 1:(T - 1)
        tfp[i] = g_Y[i] - alpha * g_K[i] - (1 - alpha) * g_L[i]
    end
    (output_growth=g_Y, capital_contribution=alpha .* g_K,
     labor_contribution=(1 - alpha) .* g_L, tfp=tfp)
end

"""SIMD-vectorized moving average filter."""
function backend_coprocessor_time_series_filter(::VPUBackend, data::AbstractVector;
                                                 window::Int=5)
    T = length(data)
    result = Vector{Float64}(undef, T)
    half = div(window, 2)
    for t in 1:T
        lo = max(1, t - half); hi = min(T, t + half)
        s = 0.0
        @simd for i in lo:hi; s += data[i]; end
        result[t] = s / (hi - lo + 1)
    end
    result
end
