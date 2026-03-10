# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl ROCm Extension
# GPU-accelerated operations on AMD GPUs.

module PolyglotFormalismsROCmExt

using AMDGPU
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.rocm_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_ROCM_AVAILABLE")
    forced !== nothing && return forced
    AMDGPU.functional()
end

function __init__()
    AcceleratorGate.register_operation!(ROCmBackend, :tensor_contract)
    AcceleratorGate.register_operation!(ROCmBackend, :map_parallel)
end

function PolyglotFormalisms.backend_tensor_contract(::ROCmBackend, A::AbstractArray, B::AbstractArray)
    Ag = ROCArray(Float32.(A)); Bg = ROCArray(Float32.(B))
    Array(Ag * Bg)
end

function PolyglotFormalisms.backend_map_parallel(::ROCmBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            cg = ROCArray(Float32.(coll))
            results[i] = Array(f.(cg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

end # module PolyglotFormalismsROCmExt
