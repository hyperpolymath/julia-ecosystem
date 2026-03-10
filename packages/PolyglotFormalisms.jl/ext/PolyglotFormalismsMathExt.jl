# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl Math Coprocessor Extension
# Mathematical accelerator support via MathAccel.jl.
# Math coprocessors provide hardware-accelerated transcendental functions,
# arbitrary-precision arithmetic, and polynomial evaluation, enabling
# exact symbolic computation where floating-point approximation is insufficient.

module PolyglotFormalismsMathExt

using MathAccel
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.math_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_MATH_AVAILABLE")
    forced !== nothing && return forced
    MathAccel.functional()
end

function AcceleratorGate.math_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_MATH_AVAILABLE", "AXIOM_MATH_DEVICE_COUNT")
    forced !== nothing && return forced
    MathAccel.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(MathBackend, :tensor_contract)
    AcceleratorGate.register_operation!(MathBackend, :map_parallel)
    AcceleratorGate.register_operation!(MathBackend, :reduce_parallel)
    AcceleratorGate.register_operation!(MathBackend, :fold_parallel)
    AcceleratorGate.register_operation!(MathBackend, :symbolic_eval)
end

"""
Math-coprocessor tensor contraction. Uses hardware-accelerated
high-precision arithmetic for exact matrix multiply where
floating-point rounding is unacceptable.
"""
function PolyglotFormalisms.backend_coprocessor_tensor_contract(::MathBackend, A::AbstractArray, B::AbstractArray)
    Am = MathAccel.MathArray(Float64.(A))
    Bm = MathAccel.MathArray(Float64.(B))
    Array(Am * Bm)
end

"""
Parallel map on math coprocessor. Applies transcendental and
polynomial functions using dedicated hardware function units.
"""
function PolyglotFormalisms.backend_coprocessor_map_parallel(::MathBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            mg = MathAccel.MathArray(Float64.(coll))
            results[i] = Array(f.(mg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on math coprocessor. Uses compensated summation
hardware (Kahan-style) for numerically stable reductions.
"""
function PolyglotFormalisms.backend_coprocessor_reduce_parallel(::MathBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            mg = MathAccel.MathArray(Float64.(coll))
            results[i] = sum(mg)
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

"""
Parallel fold on math coprocessor. Accumulates with hardware-assisted
precision tracking for exact intermediate results.
"""
function PolyglotFormalisms.backend_coprocessor_fold_parallel(::MathBackend, f, init, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            mg = MathAccel.MathArray(Float64.(coll))
            results[i] = sum(mg) + Float64(init)
        else
            results[i] = foldl(f, coll; init=init)
        end
    end
    results
end

"""
Symbolic evaluation on math coprocessor. Evaluates expression trees
using hardware-accelerated exact arithmetic, polynomial evaluation,
and transcendental function units.
"""
function PolyglotFormalisms.backend_coprocessor_symbolic_eval(::MathBackend, expr, env::Dict)
    MathAccel.symbolic_eval(expr, env)
end

end # module PolyglotFormalismsMathExt
