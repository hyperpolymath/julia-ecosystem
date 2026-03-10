# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl Crypto Accelerator Extension
# Cryptographic coprocessor support via CryptoAccel.jl.
# Crypto accelerators provide hardware-backed homomorphic encryption,
# zero-knowledge proof generation, and verifiable computation primitives,
# enabling privacy-preserving formal verification workflows.

module PolyglotFormalismsCryptoExt

using CryptoAccel
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.crypto_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_CRYPTO_AVAILABLE")
    forced !== nothing && return forced
    CryptoAccel.functional()
end

function AcceleratorGate.crypto_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_CRYPTO_AVAILABLE", "AXIOM_CRYPTO_DEVICE_COUNT")
    forced !== nothing && return forced
    CryptoAccel.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(CryptoBackend, :tensor_contract)
    AcceleratorGate.register_operation!(CryptoBackend, :map_parallel)
    AcceleratorGate.register_operation!(CryptoBackend, :reduce_parallel)
    AcceleratorGate.register_operation!(CryptoBackend, :fold_parallel)
    AcceleratorGate.register_operation!(CryptoBackend, :symbolic_eval)
end

"""
Crypto-accelerated tensor contraction. Performs matrix multiply over
homomorphically encrypted data using hardware-accelerated lattice
operations (NTT, polynomial multiply).
"""
function PolyglotFormalisms.backend_coprocessor_tensor_contract(::CryptoBackend, A::AbstractArray, B::AbstractArray)
    Ac = CryptoAccel.CryptoArray(Float64.(A))
    Bc = CryptoAccel.CryptoArray(Float64.(B))
    Array(Ac * Bc)
end

"""
Parallel map on crypto coprocessor. Applies functions to encrypted
collections without decryption, leveraging FHE evaluation keys.
"""
function PolyglotFormalisms.backend_coprocessor_map_parallel(::CryptoBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number
            cg = CryptoAccel.CryptoArray(Float64.(coll))
            results[i] = Array(f.(cg))
        else
            results[i] = map(f, coll)
        end
    end
    results
end

"""
Parallel reduce on crypto coprocessor. Uses hardware-accelerated
homomorphic addition for privacy-preserving aggregation.
"""
function PolyglotFormalisms.backend_coprocessor_reduce_parallel(::CryptoBackend, f, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            cg = CryptoAccel.CryptoArray(Float64.(coll))
            results[i] = sum(cg)
        else
            results[i] = reduce(f, coll)
        end
    end
    results
end

"""
Parallel fold on crypto coprocessor. Accumulates encrypted values
using hardware FHE circuits with bootstrapping for noise management.
"""
function PolyglotFormalisms.backend_coprocessor_fold_parallel(::CryptoBackend, f, init, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if eltype(coll) <: Number && f === +
            cg = CryptoAccel.CryptoArray(Float64.(coll))
            results[i] = sum(cg) + Float64(init)
        else
            results[i] = foldl(f, coll; init=init)
        end
    end
    results
end

"""
Crypto-accelerated symbolic evaluation. Generates zero-knowledge proofs
for expression evaluation, enabling verifiable computation where the
prover demonstrates correctness without revealing the witness.
"""
function PolyglotFormalisms.backend_coprocessor_symbolic_eval(::CryptoBackend, expr, env::Dict)
    CryptoAccel.symbolic_eval(expr, env)
end

end # module PolyglotFormalismsCryptoExt
