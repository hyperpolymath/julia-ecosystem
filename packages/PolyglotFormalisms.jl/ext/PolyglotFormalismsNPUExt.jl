# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl NPU Extension
# Neural Processing Unit acceleration for inference-heavy operations.
# NPUs provide energy-efficient matrix multiply and activation functions,
# suited to proof-search heuristics and neural-guided symbolic evaluation.

module PolyglotFormalismsNPUExt

using NPUAccel
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.npu_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_NPU_AVAILABLE")
    forced !== nothing && return forced
    NPUAccel.functional()
end

function AcceleratorGate.npu_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_NPU_AVAILABLE", "AXIOM_NPU_DEVICE_COUNT")
    forced !== nothing && return forced
    NPUAccel.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(NPUBackend, :tensor_contract)
    AcceleratorGate.register_operation!(NPUBackend, :map_parallel)
    AcceleratorGate.register_operation!(NPUBackend, :reduce_parallel)
    AcceleratorGate.register_operation!(NPUBackend, :symbolic_eval)
end

"""
NPU-accelerated tensor contraction. NPUs handle low-precision matrix
multiply efficiently, ideal for heuristic scoring in proof search.
"""
function PolyglotFormalisms.backend_coprocessor_tensor_contract(::NPUBackend, A::AbstractArray, B::AbstractArray)
    An = NPUAccel.NPUArray(Float32.(A))
    Bn = NPUAccel.NPUArray(Float32.(B))
    Array(An * Bn)
end

"""
Parallel map on NPU. Numeric collections are transferred to NPU memory
for vectorised application; non-numeric data falls back to CPU.
"""
function PolyglotFormalisms.backend_coprocessor_map_parallel(::NPUBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            ng = NPUAccel.NPUArray(Float32.(coll))
            results[i] = Array(f.(ng))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on NPU. Leverages NPU reduction units for summation.
"""
function PolyglotFormalisms.backend_coprocessor_reduce_parallel(::NPUBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            ng = NPUAccel.NPUArray(Float32.(coll))
            results[i] = Float64(sum(ng))
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

"""
Neural-guided symbolic evaluation on NPU. Uses NPU inference capabilities
to score candidate rewrites and guide simplification heuristics.
"""
function PolyglotFormalisms.backend_coprocessor_symbolic_eval(::NPUBackend, expr, env::Dict)
    NPUAccel.symbolic_eval(expr, env)
end

end # module PolyglotFormalismsNPUExt
