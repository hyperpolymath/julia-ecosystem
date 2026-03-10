# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl PPU Extension
# Physics Processing Unit acceleration via PPUCompute.jl.
# PPUs provide hardware-accelerated physics simulation primitives,
# enabling fast evaluation of continuous-domain proof obligations
# and constraint-based geometric reasoning.

module PolyglotFormalismsPPUExt

using PPUCompute
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.ppu_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_PPU_AVAILABLE")
    forced !== nothing && return forced
    PPUCompute.functional()
end

function AcceleratorGate.ppu_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_PPU_AVAILABLE", "AXIOM_PPU_DEVICE_COUNT")
    forced !== nothing && return forced
    PPUCompute.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(PPUBackend, :tensor_contract)
    AcceleratorGate.register_operation!(PPUBackend, :map_parallel)
    AcceleratorGate.register_operation!(PPUBackend, :reduce_parallel)
    AcceleratorGate.register_operation!(PPUBackend, :symbolic_eval)
end

"""
PPU-accelerated tensor contraction. Uses physics engine matrix pipelines
for fast contraction of tensors arising from geometric constraints.
"""
function PolyglotFormalisms.backend_coprocessor_tensor_contract(::PPUBackend, A::AbstractArray, B::AbstractArray)
    Ap = PPUCompute.PPUArray(Float32.(A))
    Bp = PPUCompute.PPUArray(Float32.(B))
    Array(Ap * Bp)
end

"""
Parallel map on PPU. Applies functions across collections using the PPU's
parallel simulation lanes for element-wise computation.
"""
function PolyglotFormalisms.backend_coprocessor_map_parallel(::PPUBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            pg = PPUCompute.PPUArray(Float32.(coll))
            results[i] = Array(f.(pg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on PPU. Exploits physics engine accumulator hardware
for efficient summation of constraint residuals.
"""
function PolyglotFormalisms.backend_coprocessor_reduce_parallel(::PPUBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            pg = PPUCompute.PPUArray(Float32.(coll))
            results[i] = Float64(sum(pg))
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

"""
Physics-aware symbolic evaluation on PPU. Evaluates continuous-domain
expressions using the PPU's native physics simulation primitives for
geometric, kinematic, and dynamic sub-expressions.
"""
function PolyglotFormalisms.backend_coprocessor_symbolic_eval(::PPUBackend, expr, env::Dict)
    PPUCompute.symbolic_eval(expr, env)
end

end # module PolyglotFormalismsPPUExt
