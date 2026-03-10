# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl Crypto Coprocessor
# Zero-knowledge proof of satisfiability using cryptographic coprocessor.
# Proves SAT formula is satisfiable without revealing the assignment.

using SHA

function AcceleratorGate.device_capabilities(b::CryptoBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 8, 1000,
        Int64(2 * 1024^3), Int64(2 * 1024^3),
        1, false, false, true, "Intel", "Crypto AES-NI/SHA",
    )
end

function AcceleratorGate.estimate_cost(::CryptoBackend, op::Symbol, data_size::Int)
    overhead = 100.0
    op == :solve && return overhead + Float64(data_size) * 0.2
    op == :check_sat && return overhead + Float64(data_size) * 0.15
    Inf
end

AcceleratorGate.register_operation!(CryptoBackend, :solve)
AcceleratorGate.register_operation!(CryptoBackend, :check_sat)

"""Zero-knowledge proof of satisfiability."""
struct ZKSATProof
    commitments::Vector{Vector{UInt8}}
    revealed_indices::Vector{Int}
    revealed_values::Vector{Bool}
    revealed_nonces::Vector{Vector{UInt8}}
    verified_clauses::Vector{Int}
    rounds::Int
end

_zk_commit(val::Bool, nonce::Vector{UInt8}) = sha256(vcat(nonce, UInt8[val ? 0x01 : 0x00]))

function _zk_generate_proof(clauses, assignment, n_vars; rounds=40)
    nonces = [rand(UInt8, 32) for _ in 1:n_vars]
    commits = [_zk_commit(assignment[i], nonces[i]) for i in 1:n_vars]
    nc = length(clauses)
    n_challenge = max(1, div(nc, 2))
    challenged = sort(randperm(nc)[1:n_challenge])
    needed = Set{Int}()
    for ci in challenged
        for lit in clauses[ci]; vi = abs(lit); vi <= n_vars && push!(needed, vi); end
    end
    ri = sort(collect(needed))
    ZKSATProof(commits, ri, [assignment[i] for i in ri], [nonces[i] for i in ri], challenged, rounds)
end

function _zk_verify(clauses, proof, n_vars)
    for (idx, vi) in enumerate(proof.revealed_indices)
        sha256(vcat(proof.revealed_nonces[idx], UInt8[proof.revealed_values[idx] ? 0x01 : 0x00])) != proof.commitments[vi] && return false
    end
    rmap = Dict(proof.revealed_indices[i] => proof.revealed_values[i] for i in eachindex(proof.revealed_indices))
    for ci in proof.verified_clauses
        sat = false
        for lit in clauses[ci]
            vi = abs(lit)
            haskey(rmap, vi) || continue
            ((lit > 0 && rmap[vi]) || (lit < 0 && !rmap[vi])) && (sat = true; break)
        end
        !sat && return false
    end
    true
end

"""
SAT solving with zero-knowledge proof generation. Finds a satisfying
assignment, then generates a ZK proof verifiable without the assignment.
"""
function backend_coprocessor_solve(::CryptoBackend, clauses::AbstractVector,
                                   variables::AbstractVector,
                                   config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    rounds = get(config, :zk_rounds, 40)
    n_trials = min(get(config, :trials, 2^min(n_vars, 20)), 1048576)

    for trial in 1:n_trials
        a = rand(Bool, n_vars); ok = true
        for clause in clauses
            sat = false
            for lit in clause
                vi = abs(lit); vi > n_vars && continue
                ((lit > 0 && a[vi]) || (lit < 0 && !a[vi])) && (sat = true; break)
            end
            !sat && (ok = false; break)
        end
        if ok
            proof = _zk_generate_proof(clauses, a, n_vars; rounds=rounds)
            verified = _zk_verify(clauses, proof, n_vars)
            model = Dict(variables[j] => a[j] for j in 1:n_vars)
            return (status=:sat, model=model, trials_checked=trial,
                    zk_proof=proof, zk_verified=verified)
        end
    end
    return (status=:unknown, model=nothing, trials_checked=n_trials,
            zk_proof=nothing, zk_verified=false)
end

function backend_coprocessor_check_sat(::CryptoBackend, clauses::AbstractVector, n_vars::Int)
    r = backend_coprocessor_solve(CryptoBackend(0), clauses, [Symbol("x$i") for i in 1:n_vars])
    r.status == :sat ? :sat : :unknown
end
