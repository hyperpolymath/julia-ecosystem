# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl VPU Extension
# Vector Processing Unit acceleration via VPURuntime.jl.
# VPUs provide wide SIMD datapaths for data-parallel operations,
# suited to batch evaluation of proof obligations and parallel rewriting.

module PolyglotFormalismsVPUExt

using VPURuntime
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.vpu_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_VPU_AVAILABLE")
    forced !== nothing && return forced
    VPURuntime.functional()
end

function AcceleratorGate.vpu_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_VPU_AVAILABLE", "AXIOM_VPU_DEVICE_COUNT")
    forced !== nothing && return forced
    VPURuntime.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(VPUBackend, :map_parallel)
    AcceleratorGate.register_operation!(VPUBackend, :reduce_parallel)
    AcceleratorGate.register_operation!(VPUBackend, :fold_parallel)
    AcceleratorGate.register_operation!(VPUBackend, :tensor_contract)
end

"""
VPU-accelerated tensor contraction. Uses wide vector lanes for
block-wise matrix multiply with high throughput.
"""
function PolyglotFormalisms.backend_coprocessor_tensor_contract(::VPUBackend, A::AbstractArray, B::AbstractArray)
    Av = VPURuntime.VPUArray(Float32.(A))
    Bv = VPURuntime.VPUArray(Float32.(B))
    Array(Av * Bv)
end

"""
Parallel map on VPU. Exploits SIMD lanes for element-wise function
application across numeric collections.
"""
function PolyglotFormalisms.backend_coprocessor_map_parallel(::VPUBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            vg = VPURuntime.VPUArray(Float32.(coll))
            results[i] = Array(f.(vg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on VPU. Uses horizontal SIMD reduction instructions
for fast summation across vector lanes.
"""
function PolyglotFormalisms.backend_coprocessor_reduce_parallel(::VPUBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            vg = VPURuntime.VPUArray(Float32.(coll))
            results[i] = Float64(sum(vg))
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

"""
Parallel fold on VPU. Streams data through vector accumulator registers
for efficient sequential reduction with initial value.
"""
function PolyglotFormalisms.backend_coprocessor_fold_parallel(::VPUBackend, f, init, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            vg = VPURuntime.VPUArray(Float32.(coll))
            results[i] = Float64(sum(vg)) + Float64(init)
        else
            results[i] = foldl(f, coll; init=init)
        end
    end
    results
end

end # module PolyglotFormalismsVPUExt
