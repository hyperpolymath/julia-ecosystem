# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl FPGA Extension
# FPGA-accelerated operations via FPGASynthesis.jl.
# FPGAs provide reconfigurable hardware for custom datapath acceleration,
# enabling bitstream-level optimisation for specific proof-search patterns.

module PolyglotFormalismsFPGAExt

using FPGASynthesis
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.fpga_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_FPGA_AVAILABLE")
    forced !== nothing && return forced
    FPGASynthesis.functional()
end

function AcceleratorGate.fpga_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_FPGA_AVAILABLE", "AXIOM_FPGA_DEVICE_COUNT")
    forced !== nothing && return forced
    FPGASynthesis.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(FPGABackend, :tensor_contract)
    AcceleratorGate.register_operation!(FPGABackend, :map_parallel)
    AcceleratorGate.register_operation!(FPGABackend, :reduce_parallel)
    AcceleratorGate.register_operation!(FPGABackend, :fold_parallel)
    AcceleratorGate.register_operation!(FPGABackend, :symbolic_eval)
end

"""
FPGA-accelerated tensor contraction. Synthesises a custom matrix-multiply
datapath for the given dimensions, exploiting on-chip BRAM and DSP slices.
"""
function PolyglotFormalisms.backend_coprocessor_tensor_contract(::FPGABackend, A::AbstractArray, B::AbstractArray)
    Af = FPGASynthesis.FPGAArray(Float32.(A))
    Bf = FPGASynthesis.FPGAArray(Float32.(B))
    Array(Af * Bf)
end

"""
Parallel map on FPGA. Generates a pipelined datapath for the mapping
function, streaming elements through the reconfigurable fabric.
"""
function PolyglotFormalisms.backend_coprocessor_map_parallel(::FPGABackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            fg = FPGASynthesis.FPGAArray(Float32.(coll))
            results[i] = Array(f.(fg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on FPGA. Implements a hardware reduction tree in the
FPGA fabric for efficient parallel summation.
"""
function PolyglotFormalisms.backend_coprocessor_reduce_parallel(::FPGABackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            fg = FPGASynthesis.FPGAArray(Float32.(coll))
            results[i] = Float64(sum(fg))
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

"""
Parallel fold on FPGA. Uses a synthesised accumulator pipeline for
streaming fold operations.
"""
function PolyglotFormalisms.backend_coprocessor_fold_parallel(::FPGABackend, f, init, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            fg = FPGASynthesis.FPGAArray(Float32.(coll))
            results[i] = Float64(sum(fg)) + Float64(init)
        else
            results[i] = foldl(f, coll; init=init)
        end
    end
    results
end

"""
Symbolic evaluation on FPGA. Compiles expression trees into custom
hardware datapaths for deterministic, low-latency evaluation.
"""
function PolyglotFormalisms.backend_coprocessor_symbolic_eval(::FPGABackend, expr, env::Dict)
    FPGASynthesis.symbolic_eval(expr, env)
end

end # module PolyglotFormalismsFPGAExt
