# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl CUDA Extension
# GPU-accelerated parallel proof search and tensor contraction on NVIDIA GPUs.

module PolyglotFormalismsCUDAExt

using CUDA
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.cuda_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_CUDA_AVAILABLE")
    forced !== nothing && return forced
    CUDA.functional()
end

function AcceleratorGate.cuda_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_CUDA_AVAILABLE", "AXIOM_CUDA_DEVICE_COUNT")
    forced !== nothing && return forced
    CUDA.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(CUDABackend, :tensor_contract)
    AcceleratorGate.register_operation!(CUDABackend, :map_parallel)
    AcceleratorGate.register_operation!(CUDABackend, :reduce_parallel)
end

"""
GPU-accelerated tensor contraction via cuBLAS GEMM.
"""
function PolyglotFormalisms.backend_tensor_contract(::CUDABackend, A::AbstractArray, B::AbstractArray)
    Ag = CuArray(Float32.(A)); Bg = CuArray(Float32.(B))
    Array(Ag * Bg)
end

"""
Parallel map on GPU: transfers collections to device memory,
applies function via broadcast, transfers results back.
"""
function PolyglotFormalisms.backend_map_parallel(::CUDABackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            cg = CuArray(Float32.(coll))
            results[i] = Array(f.(cg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on GPU.
"""
function PolyglotFormalisms.backend_reduce_parallel(::CUDABackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            cg = CuArray(Float32.(coll))
            results[i] = Float64(sum(cg))
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

end # module PolyglotFormalismsCUDAExt
