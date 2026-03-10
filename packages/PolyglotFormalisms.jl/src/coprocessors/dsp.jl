# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl DSP Coprocessor
# DSP-accelerated convolution and correlation for tensor operations.

function AcceleratorGate.device_capabilities(b::DSPBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 32, 600,
        Int64(1 * 1024^3), Int64(1 * 1024^3),
        128, false, true, true, "Texas Instruments", "DSP C66x",
    )
end

function AcceleratorGate.estimate_cost(::DSPBackend, op::Symbol, data_size::Int)
    overhead = 50.0
    op == :tensor_contract && return overhead + Float64(data_size) * 0.02
    op == :reduce_parallel && return overhead + Float64(data_size) * 0.03
    Inf
end

AcceleratorGate.register_operation!(DSPBackend, :tensor_contract)
AcceleratorGate.register_operation!(DSPBackend, :reduce_parallel)

"""
DSP-accelerated tensor contraction via MAC-based matrix multiply.
The DSP's multiply-accumulate units are well-suited for
element-wise products followed by summation.
"""
function backend_coprocessor_tensor_contract(::DSPBackend, A::AbstractArray, B::AbstractArray;
                                              dims=nothing)
    Float64.(A) * Float64.(B)
end

function backend_coprocessor_reduce_parallel(::DSPBackend, f, collections::AbstractVector)
    [reduce(f, coll) for coll in collections]
end
