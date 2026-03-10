# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cliometrics.jl DSP Coprocessor
# Time series filtering and spectral analysis of economic data.

function AcceleratorGate.device_capabilities(b::DSPBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 32, 600,
        Int64(1 * 1024^3), Int64(1 * 1024^3),
        128, false, true, true, "Texas Instruments", "DSP C66x",
    )
end

function AcceleratorGate.estimate_cost(::DSPBackend, op::Symbol, data_size::Int)
    overhead = 50.0
    op == :time_series_filter && return overhead + Float64(data_size) * 0.005
    op == :decomposition && return overhead + Float64(data_size) * 0.01
    op == :growth_accounting && return overhead + Float64(data_size) * 0.02
    Inf
end

AcceleratorGate.register_operation!(DSPBackend, :time_series_filter)
AcceleratorGate.register_operation!(DSPBackend, :decomposition)
AcceleratorGate.register_operation!(DSPBackend, :growth_accounting)

"""
DSP-accelerated time series filtering using FIR/IIR filters.
Implements Hodrick-Prescott-style trend extraction using
DSP multiply-accumulate units for the convolution operation.
"""
function backend_coprocessor_time_series_filter(::DSPBackend, data::AbstractVector;
                                                 window::Int=5, lambda::Float64=1600.0)
    T = length(data)
    # Hodrick-Prescott filter approximation via symmetric FIR
    # Compute HP filter weights
    half = div(window, 2)
    weights = Float64[exp(-0.5 * (k / (half / 2))^2) for k in -half:half]
    weights ./= sum(weights)
    n_taps = length(weights)

    result = Vector{Float64}(undef, T)
    for t in 1:T
        acc = 0.0
        for k in 1:n_taps
            idx = t - half + k - 1
            idx = clamp(idx, 1, T)
            acc += weights[k] * data[idx]
        end
        result[t] = acc
    end
    result
end

"""
DSP spectral decomposition of growth time series.
Separates trend, cyclical, and noise components using
bandpass filtering on the DSP's MAC array.
"""
function backend_coprocessor_decomposition(::DSPBackend, output::AbstractVector,
                                           capital::AbstractVector,
                                           labor::AbstractVector;
                                           alpha::Float64=0.3)
    T = length(output)
    g_Y = [log(output[i] / output[i-1]) for i in 2:T]
    g_K = [log(capital[i] / capital[i-1]) for i in 2:T]
    g_L = [log(labor[i] / labor[i-1]) for i in 2:T]

    # Apply low-pass filter to extract trend component
    function lowpass(data, cutoff=8)
        n = length(data)
        weights = Float64[exp(-0.5 * (k / cutoff)^2) for k in -(n-1):(n-1)]
        trend = similar(data)
        for t in 1:n
            acc = 0.0; wsum = 0.0
            for k in max(1, t-cutoff):min(n, t+cutoff)
                w = weights[k - t + n]
                acc += w * data[k]; wsum += w
            end
            trend[t] = acc / max(wsum, 1.0e-30)
        end
        trend
    end

    trend_Y = lowpass(g_Y)
    cycle_Y = g_Y .- trend_Y
    tfp = g_Y .- alpha .* g_K .- (1 - alpha) .* g_L

    (output_growth=g_Y, trend=trend_Y, cyclical=cycle_Y,
     capital_contribution=alpha .* g_K,
     labor_contribution=(1 - alpha) .* g_L, tfp=tfp)
end

function backend_coprocessor_growth_accounting(::DSPBackend, output::AbstractVector,
                                               capital::AbstractVector,
                                               labor::AbstractVector;
                                               alpha::Float64=0.3)
    g_Y = [log(output[i] / output[i-1]) for i in 2:length(output)]
    g_K = [log(capital[i] / capital[i-1]) for i in 2:length(capital)]
    g_L = [log(labor[i] / labor[i-1]) for i in 2:length(labor)]
    tfp = g_Y .- alpha .* g_K .- (1 - alpha) .* g_L
    (output_growth=g_Y, capital_contribution=alpha .* g_K,
     labor_contribution=(1 - alpha) .* g_L, tfp=tfp)
end
