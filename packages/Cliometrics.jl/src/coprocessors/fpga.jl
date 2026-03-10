# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cliometrics.jl FPGA Coprocessor
# Streaming time series processing using FPGA pipeline architecture.

function AcceleratorGate.device_capabilities(b::FPGABackend)
    AcceleratorGate.DeviceCapabilities(
        b, 1000, 200,
        Int64(8 * 1024^3), Int64(6 * 1024^3),
        64, false, false, true, "Intel", "FPGA Stratix",
    )
end

function AcceleratorGate.estimate_cost(::FPGABackend, op::Symbol, data_size::Int)
    setup = 5000.0
    op == :time_series_filter && return setup + Float64(data_size) * 0.0005
    op == :growth_accounting && return setup + Float64(data_size) * 0.001
    Inf
end

AcceleratorGate.register_operation!(FPGABackend, :time_series_filter)
AcceleratorGate.register_operation!(FPGABackend, :growth_accounting)

"""
Streaming time series filter using FPGA pipeline. Models a hardware FIR filter
with configurable taps, processing one sample per clock cycle.
"""
function backend_coprocessor_time_series_filter(::FPGABackend, data::AbstractVector;
                                                 window::Int=5, taps::Union{Nothing, Vector{Float64}}=nothing)
    T = length(data)
    if taps === nothing
        # Uniform FIR filter
        taps = fill(1.0 / window, window)
    end
    n_taps = length(taps)
    result = zeros(Float64, T)

    # Streaming pipeline: circular buffer simulating FPGA shift register
    buffer = zeros(Float64, n_taps)
    buf_idx = 1

    for t in 1:T
        buffer[buf_idx] = Float64(data[t])
        # FIR convolution (single clock cycle in hardware)
        acc = 0.0
        for k in 1:n_taps
            idx = mod1(buf_idx - k + 1, n_taps)
            acc += taps[k] * buffer[idx]
        end
        result[t] = acc
        buf_idx = mod1(buf_idx + 1, n_taps)
    end
    result
end

"""
Streaming growth accounting: processes time series in a single pass
(FPGA pipeline computes log ratios and weighted sums in one cycle).
"""
function backend_coprocessor_growth_accounting(::FPGABackend, output::AbstractVector,
                                               capital::AbstractVector,
                                               labor::AbstractVector;
                                               alpha::Float64=0.3)
    T = length(output)
    g_Y = Vector{Float64}(undef, T - 1)
    cap_c = Vector{Float64}(undef, T - 1)
    lab_c = Vector{Float64}(undef, T - 1)
    tfp = Vector{Float64}(undef, T - 1)

    # Single-pass pipeline: each iteration is one clock cycle
    for t in 2:T
        gy = log(output[t] / output[t-1])
        gk = log(capital[t] / capital[t-1])
        gl = log(labor[t] / labor[t-1])
        i = t - 1
        g_Y[i] = gy
        cap_c[i] = alpha * gk
        lab_c[i] = (1 - alpha) * gl
        tfp[i] = gy - alpha * gk - (1 - alpha) * gl
    end
    (output_growth=g_Y, capital_contribution=cap_c,
     labor_contribution=lab_c, tfp=tfp)
end
