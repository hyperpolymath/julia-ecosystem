# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl Metal Extension
# GPU-accelerated operations on Apple Silicon.

module PolyglotFormalismsMetalExt

using Metal
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.metal_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_METAL_AVAILABLE")
    forced !== nothing && return forced
    Metal.functional()
end

function __init__()
    AcceleratorGate.register_operation!(MetalBackend, :tensor_contract)
    AcceleratorGate.register_operation!(MetalBackend, :map_parallel)
end

function PolyglotFormalisms.backend_tensor_contract(::MetalBackend, A::AbstractArray, B::AbstractArray)
    Ag = MtlArray(Float32.(A)); Bg = MtlArray(Float32.(B))
    Array(Ag * Bg)
end

function PolyglotFormalisms.backend_map_parallel(::MetalBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            cg = MtlArray(Float32.(coll))
            results[i] = Array(f.(cg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

end # module PolyglotFormalismsMetalExt
