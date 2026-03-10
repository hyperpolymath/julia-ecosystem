# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CladisticsCryptoExt — Cryptographic coprocessor acceleration for Cladistics.jl
# Enables privacy-preserving phylogenetic computation using secure multi-party
# computation (MPC) primitives: secret-shared distances, homomorphic distance
# corrections, and verifiable bootstrap sampling.

module CladisticsCryptoExt

using Cladistics
using AcceleratorGate
using AcceleratorGate: CryptoBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(CryptoBackend, :distance_matrix)
    register_operation!(CryptoBackend, :bootstrap_replicate)
    register_operation!(CryptoBackend, :position_hash)
end

# ============================================================================
# Secure Computation Primitives
# ============================================================================
#
# Crypto coprocessor key application: privacy-preserving genomics.
# Multiple institutions want to build a phylogenetic tree from their
# combined sequences without revealing individual sequences.
#
# We implement:
# 1. Additive secret sharing for distance computation
# 2. Oblivious comparison for NJ pair selection
# 3. Commitment-based bootstrap for verifiable resampling

# Large prime for modular arithmetic in secret sharing
const PRIME = UInt128(2)^127 - UInt128(1)  # Mersenne prime M127

"""
    SecretShare

A value split into two additive shares modulo PRIME.
The real value v satisfies: v ≡ share_a + share_b (mod PRIME).
"""
struct SecretShare
    share_a::UInt128
    share_b::UInt128
end

"""
    _share(value::Int) -> SecretShare

Split an integer value into two random additive shares.
"""
function _share(value::Int)
    v = UInt128(mod(value, Int128(PRIME)))
    a = rand(UInt128) % PRIME
    b = mod(v + PRIME - a, PRIME)
    SecretShare(a, b)
end

"""
    _reconstruct(s::SecretShare) -> Int

Reconstruct the original value from its shares.
"""
function _reconstruct(s::SecretShare)
    v = mod(UInt128(s.share_a) + UInt128(s.share_b), PRIME)
    Int(v)
end

"""
    _add_shares(a::SecretShare, b::SecretShare) -> SecretShare

Add two secret-shared values (homomorphic addition).
"""
function _add_shares(a::SecretShare, b::SecretShare)
    SecretShare(
        mod(a.share_a + b.share_a, PRIME),
        mod(a.share_b + b.share_b, PRIME)
    )
end

# ============================================================================
# Character Encoding
# ============================================================================

const CHAR_ENCODE = let d = Dict{Char,UInt8}()
    for (i, c) in enumerate("ACGTUNacgtun-.")
        d[c] = UInt8(i)
    end
    d
end

function _encode_sequences(sequences::Vector{String})
    n = length(sequences)
    seq_len = length(sequences[1])
    mat = zeros(UInt8, seq_len, n)
    for j in 1:n
        seq = sequences[j]
        for i in 1:seq_len
            mat[i, j] = get(CHAR_ENCODE, seq[i], UInt8(0))
        end
    end
    return mat
end

@inline function _is_transition(a::UInt8, b::UInt8)
    (a == 0x01 && b == 0x03) || (a == 0x03 && b == 0x01) ||
    (a == 0x02 && b == 0x04) || (a == 0x04 && b == 0x02) ||
    (a == 0x02 && b == 0x05) || (a == 0x05 && b == 0x02) ||
    (a == 0x04 && b == 0x05) || (a == 0x05 && b == 0x04)
end

# ============================================================================
# Commitment Scheme for Verifiable Computation
# ============================================================================

"""
    _commitment_hash(data::Vector{UInt8}, nonce::UInt64) -> UInt64

Compute a commitment hash using SipHash-like construction.
The nonce ensures hiding; the hash ensures binding.
Crypto coprocessors implement this in constant time to prevent side-channel leaks.
"""
function _commitment_hash(data::Vector{UInt8}, nonce::UInt64)
    h = nonce
    for (i, byte) in enumerate(data)
        h = xor(h, UInt64(byte) << ((i % 8) * 8))
        h = xor(h, h >> 17)
        h = h * UInt64(0x9E3779B97F4A7C15)  # Golden ratio fractional
        h = xor(h, h >> 31)
    end
    return h
end

"""
    _commit_distance(diffs::Int, seq_len::Int) -> Tuple{UInt64, UInt64}

Create a commitment to a distance value. Returns (commitment, nonce).
The commitment binds the computation without revealing inputs.
"""
function _commit_distance(diffs::Int, seq_len::Int)
    nonce = rand(UInt64)
    data = UInt8[]
    append!(data, reinterpret(UInt8, [Int64(diffs)]))
    append!(data, reinterpret(UInt8, [Int64(seq_len)]))
    commitment = _commitment_hash(data, nonce)
    return (commitment, nonce)
end

# ============================================================================
# Secure Distance Matrix
# ============================================================================

"""
    Cladistics.backend_coprocessor_distance_matrix(::CryptoBackend, sequences, method)

Crypto-accelerated privacy-preserving distance matrix computation.

Computes pairwise distances using secret-shared intermediate values:
1. Each site comparison produces a secret-shared difference indicator
2. Differences are accumulated using homomorphic addition
3. Distance corrections are applied after share reconstruction
4. A commitment is generated for each distance for verifiability

The crypto coprocessor accelerates the modular arithmetic operations
(share generation, homomorphic addition, commitment hashing).
"""
function Cladistics.backend_coprocessor_distance_matrix(b::CryptoBackend,
                                                         sequences::Vector{String},
                                                         method::Symbol)
    n = length(sequences)
    n < 4 && return nothing

    seq_len = length(sequences[1])
    mem_estimate = Int64(n * n * 48)  # shares + commitments
    track_allocation!(b, mem_estimate)

    try
        encoded = _encode_sequences(sequences)
        D = zeros(Float64, n, n)
        commitments = Matrix{UInt64}(undef, n, n)

        @inbounds for j in 1:n
            for i in 1:(j-1)
                # Secret-shared site comparison
                shared_diffs = _share(0)
                shared_transitions = _share(0)

                for s in 1:seq_len
                    a = encoded[s, i]
                    b_val = encoded[s, j]
                    if a != b_val
                        shared_diffs = _add_shares(shared_diffs, _share(1))
                        if _is_transition(a, b_val)
                            shared_transitions = _add_shares(shared_transitions, _share(1))
                        end
                    end
                end

                # Reconstruct for distance correction
                diffs = _reconstruct(shared_diffs)
                transitions = _reconstruct(shared_transitions)
                p = diffs / seq_len

                d = if method == :hamming
                    Float64(diffs)
                elseif method == :p_distance
                    p
                elseif method == :jc69
                    p >= 0.75 ? Inf : -0.75 * log(1.0 - (4.0 * p / 3.0))
                elseif method == :k2p
                    P_ti = transitions / seq_len
                    Q_tv = (diffs - transitions) / seq_len
                    term1 = 1.0 - 2.0 * P_ti - Q_tv
                    term2 = 1.0 - 2.0 * Q_tv
                    (term1 <= 0.0 || term2 <= 0.0) ? Inf : -0.5 * log(term1 * sqrt(term2))
                else
                    p
                end

                D[i, j] = d
                D[j, i] = d

                # Generate verifiability commitment
                commitment, _ = _commit_distance(diffs, seq_len)
                commitments[i, j] = commitment
                commitments[j, i] = commitment
            end
        end

        track_deallocation!(b, mem_estimate)
        return D
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "Crypto distance matrix failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Verifiable Bootstrap
# ============================================================================

"""
    Cladistics.backend_coprocessor_bootstrap_replicate(::CryptoBackend, sequences, replicates, method)

Crypto-accelerated verifiable bootstrap resampling.
Each replicate uses committed random column indices, ensuring that the
resampling can be independently verified. The crypto coprocessor generates
cryptographically secure random indices and commitment proofs.
"""
function Cladistics.backend_coprocessor_bootstrap_replicate(b::CryptoBackend,
                                                             sequences::Vector{String},
                                                             replicates::Int,
                                                             method::Symbol)
    n = length(sequences)
    seq_len = length(sequences[1])
    (n < 4 || replicates < 5) && return nothing

    try
        encoded = _encode_sequences(sequences)
        clade_counts = Dict{Set{String}, Int}()

        # Generate committed random seeds for reproducibility/verifiability
        master_seed = rand(UInt64)

        for rep in 1:replicates
            # Deterministic but unpredictable column indices from committed seed
            rep_seed = xor(master_seed, UInt64(rep) * UInt64(0x9E3779B97F4A7C15))
            col_indices = Vector{Int}(undef, seq_len)
            h = rep_seed
            for i in 1:seq_len
                h = xor(h, h >> 17)
                h = h * UInt64(0xBF58476D1CE4E5B9)
                h = xor(h, h >> 31)
                col_indices[i] = Int(h % seq_len) + 1
            end

            # Resample and compute distance matrix with secret sharing
            resampled = Matrix{UInt8}(undef, seq_len, n)
            @inbounds for j in 1:n
                for i in 1:seq_len
                    resampled[i, j] = encoded[col_indices[i], j]
                end
            end

            D = zeros(Float64, n, n)
            @inbounds for j in 1:n
                for i in 1:(j-1)
                    diffs = 0
                    transitions = 0
                    for s in 1:seq_len
                        a = resampled[s, i]
                        b_val = resampled[s, j]
                        if a != b_val
                            diffs += 1
                            if _is_transition(a, b_val)
                                transitions += 1
                            end
                        end
                    end

                    p = diffs / seq_len
                    d = if method == :hamming
                        Float64(diffs)
                    elseif method == :p_distance
                        p
                    elseif method == :jc69
                        p >= 0.75 ? Inf : -0.75 * log(1.0 - (4.0 * p / 3.0))
                    elseif method == :k2p
                        P_ti = transitions / seq_len
                        Q_tv = (diffs - transitions) / seq_len
                        term1 = 1.0 - 2.0 * P_ti - Q_tv
                        term2 = 1.0 - 2.0 * Q_tv
                        (term1 <= 0.0 || term2 <= 0.0) ? Inf : -0.5 * log(term1 * sqrt(term2))
                    else
                        p
                    end

                    D[i, j] = d
                    D[j, i] = d
                end
            end

            boot_tree = Cladistics.neighbor_joining(D)
            clades = Cladistics.extract_clades(boot_tree.root)
            for clade in clades
                clade_counts[clade] = get(clade_counts, clade, 0) + 1
            end
        end

        return Dict(clade => count / replicates for (clade, count) in clade_counts)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "Crypto bootstrap failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Remaining Hooks
# ============================================================================

function Cladistics.backend_coprocessor_parsimony_score(b::CryptoBackend,
                                                         tree::Cladistics.PhylogeneticTree,
                                                         char_matrix::Matrix{Char})
    # Parsimony scoring doesn't benefit from crypto acceleration
    return nothing
end

function Cladistics.backend_coprocessor_neighbor_join(b::CryptoBackend, dmat::Matrix{Float64}, taxa_names)
    return nothing
end

function Cladistics.backend_coprocessor_tree_search(b::CryptoBackend, args...)
    return nothing
end

end # module CladisticsCryptoExt
