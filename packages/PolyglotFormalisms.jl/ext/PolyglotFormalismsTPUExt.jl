# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl TPU Extension
# Tensor Processing Unit acceleration for large-scale tensor contraction
# and parallel map/reduce on Google TPU hardware via TPUCompute.jl.

module PolyglotFormalismsTPUExt

using TPUCompute
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.tpu_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_TPU_AVAILABLE")
    forced !== nothing && return forced
    TPUCompute.functional()
end

function AcceleratorGate.tpu_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_TPU_AVAILABLE", "AXIOM_TPU_DEVICE_COUNT")
    forced !== nothing && return forced
    TPUCompute.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(TPUBackend, :tensor_contract)
    AcceleratorGate.register_operation!(TPUBackend, :map_parallel)
    AcceleratorGate.register_operation!(TPUBackend, :reduce_parallel)
    AcceleratorGate.register_operation!(TPUBackend, :fold_parallel)
    AcceleratorGate.register_operation!(TPUBackend, :symbolic_eval)
end

"""
TPU-accelerated tensor contraction using XLA-style matrix multiply.
TPUs excel at large, regular tensor operations with their systolic array architecture.
"""
function PolyglotFormalisms.backend_coprocessor_tensor_contract(::TPUBackend, A::AbstractArray, B::AbstractArray)
    At = TPUCompute.TPUArray(Float32.(A))
    Bt = TPUCompute.TPUArray(Float32.(B))
    Array(At * Bt)
end

"""
Parallel map across collections on TPU. Numeric data is transferred to
TPU memory for vectorised broadcast; non-numeric falls back to CPU map.
"""
function PolyglotFormalisms.backend_coprocessor_map_parallel(::TPUBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            tg = TPUCompute.TPUArray(Float32.(coll))
            results[i] = Array(f.(tg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on TPU. Exploits TPU reduction hardware for summation;
other reduction operators fall back to CPU.
"""
function PolyglotFormalisms.backend_coprocessor_reduce_parallel(::TPUBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            tg = TPUCompute.TPUArray(Float32.(coll))
            results[i] = Float64(sum(tg))
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

"""
Parallel fold on TPU. Accumulates via TPU reduction when the operator
and data types permit device-side execution.
"""
function PolyglotFormalisms.backend_coprocessor_fold_parallel(::TPUBackend, f, init, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            tg = TPUCompute.TPUArray(Float32.(coll))
            results[i] = Float64(sum(tg)) + Float64(init)
        else
            results[i] = foldl(f, coll; init=init)
        end
    end
    results
end

"""
Symbolic evaluation on TPU. Delegates expression tree evaluation to
the TPU's XLA compiler for optimised execution of arithmetic sub-expressions.
"""
function PolyglotFormalisms.backend_coprocessor_symbolic_eval(::TPUBackend, expr, env::Dict)
    # TPU-accelerated symbolic evaluation: compile arithmetic sub-expressions
    # through XLA and evaluate on-device, falling back to CPU for
    # non-arithmetic nodes.
    TPUCompute.symbolic_eval(expr, env)
end

end # module PolyglotFormalismsTPUExt
