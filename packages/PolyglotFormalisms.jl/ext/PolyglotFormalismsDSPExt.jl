# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl DSP Extension
# Digital Signal Processor acceleration for signal-domain operations.
# DSPs are optimised for fixed-point arithmetic, convolution, and FFT,
# enabling efficient spectral analysis within formal verification pipelines.

module PolyglotFormalismsDSPExt

using DSPLibs
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.dsp_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_DSP_AVAILABLE")
    forced !== nothing && return forced
    DSPLibs.functional()
end

function AcceleratorGate.dsp_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_DSP_AVAILABLE", "AXIOM_DSP_DEVICE_COUNT")
    forced !== nothing && return forced
    DSPLibs.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(DSPBackend, :map_parallel)
    AcceleratorGate.register_operation!(DSPBackend, :reduce_parallel)
    AcceleratorGate.register_operation!(DSPBackend, :fold_parallel)
end

"""
Parallel map on DSP. Transfers numeric collections to DSP memory for
vectorised signal-processing operations (convolution, filtering, FFT).
"""
function PolyglotFormalisms.backend_coprocessor_map_parallel(::DSPBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            dg = DSPLibs.DSPArray(Float32.(coll))
            results[i] = Array(f.(dg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on DSP. Uses DSP accumulator hardware for efficient
summation of signal data.
"""
function PolyglotFormalisms.backend_coprocessor_reduce_parallel(::DSPBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            dg = DSPLibs.DSPArray(Float32.(coll))
            results[i] = Float64(sum(dg))
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

"""
Parallel fold on DSP. Accumulates using DSP multiply-accumulate (MAC)
units when operator and data types permit device-side execution.
"""
function PolyglotFormalisms.backend_coprocessor_fold_parallel(::DSPBackend, f, init, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            dg = DSPLibs.DSPArray(Float32.(coll))
            results[i] = Float64(sum(dg)) + Float64(init)
        else
            results[i] = foldl(f, coll; init=init)
        end
    end
    results
end

end # module PolyglotFormalismsDSPExt
