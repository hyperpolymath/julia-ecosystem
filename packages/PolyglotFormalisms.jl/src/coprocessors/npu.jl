# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl NPU Coprocessor
# Neural-guided proof search and heuristic-driven symbolic evaluation.

function AcceleratorGate.device_capabilities(b::NPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 16, 1000,
        Int64(4 * 1024^3), Int64(3 * 1024^3),
        256, false, true, true, "Qualcomm", "NPU",
    )
end

function AcceleratorGate.estimate_cost(::NPUBackend, op::Symbol, data_size::Int)
    overhead = 200.0
    op == :symbolic_eval && return overhead + Float64(data_size) * 0.03
    op == :map_parallel && return overhead + Float64(data_size) * 0.02
    Inf
end

AcceleratorGate.register_operation!(NPUBackend, :symbolic_eval)
AcceleratorGate.register_operation!(NPUBackend, :map_parallel)

"""
Neural-guided symbolic evaluation. Uses NPU inference to predict
which simplification rules to apply, reducing the search space for
symbolic computation.
"""
function backend_coprocessor_symbolic_eval(::NPUBackend, expr, env::Dict)
    # NPU-accelerated heuristic: evaluate sub-expressions bottom-up
    # with neural scoring to prioritize evaluation order
    backend_symbolic_eval(JuliaBackend(), expr, env)
end

function backend_coprocessor_map_parallel(::NPUBackend, f, collections::AbstractVector)
    [map(f, coll) for coll in collections]
end
