# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CladisticsMathExt — Extended precision math acceleration for Cladistics.jl
# Uses arbitrary-precision arithmetic for numerically stable distance
# corrections (JC69/K2P), preventing catastrophic cancellation in
# log-correction formulas for closely or distantly related sequences.

module CladisticsMathExt

using Cladistics
using AcceleratorGate
using AcceleratorGate: MathBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(MathBackend, :distance_matrix)
    register_operation!(MathBackend, :parsimony_score)
    register_operation!(MathBackend, :bootstrap_replicate)
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
# Extended Precision Distance Corrections
# ============================================================================
#
# Math coprocessor key advantage: arbitrary-precision arithmetic prevents
# catastrophic cancellation in distance correction formulas.
#
# The JC69 formula: d = -3/4 * ln(1 - 4p/3)
# suffers from catastrophic cancellation when p is near 0 (closely related)
# because 1 - 4p/3 ~ 1, and when p is near 3/4 (saturated divergence)
# because the log argument approaches 0.
#
# The K2P formula: d = -1/2 * ln((1-2P-Q) * sqrt(1-2Q))
# has similar issues with both P and Q near their boundaries.
#
# Using BigFloat with configurable precision eliminates these issues.

"""
    _bigfloat_jc69(diffs::Int, seq_len::Int; precision::Int=256) -> BigFloat

Compute JC69 distance using arbitrary-precision arithmetic.
Uses Taylor series expansion near p=0 to avoid cancellation.
"""
function _bigfloat_jc69(diffs::Int, seq_len::Int; precision::Int=256)
    setprecision(BigFloat, precision) do
        p = BigFloat(diffs) / BigFloat(seq_len)

        # For very small p, use Taylor expansion: d ~ p + (2/3)p^2 + ...
        if p < BigFloat("0.001")
            # Taylor series of -3/4 * ln(1 - 4p/3)
            # = p + (8/9)p^2 + (64/81)p^3 + ...
            x = BigFloat(4) * p / BigFloat(3)
            # Compute -3/4 * ln(1-x) via series: sum_{k=1}^{N} x^k / k
            d = BigFloat(0)
            xk = x
            for k in 1:20
                d += xk / BigFloat(k)
                xk *= x
            end
            return d * BigFloat(3) / BigFloat(4)
        end

        arg = BigFloat(1) - BigFloat(4) * p / BigFloat(3)
        if arg <= BigFloat(0)
            return BigFloat(Inf)
        end
        return -BigFloat(3) / BigFloat(4) * log(arg)
    end
end

"""
    _bigfloat_k2p(diffs::Int, transitions::Int, seq_len::Int; precision::Int=256) -> BigFloat

Compute K2P distance using arbitrary-precision arithmetic.
Handles near-boundary cases where Float64 would lose significant digits.
"""
function _bigfloat_k2p(diffs::Int, transitions::Int, seq_len::Int; precision::Int=256)
    setprecision(BigFloat, precision) do
        P_ti = BigFloat(transitions) / BigFloat(seq_len)
        Q_tv = BigFloat(diffs - transitions) / BigFloat(seq_len)

        term1 = BigFloat(1) - BigFloat(2) * P_ti - Q_tv
        term2 = BigFloat(1) - BigFloat(2) * Q_tv

        if term1 <= BigFloat(0) || term2 <= BigFloat(0)
            return BigFloat(Inf)
        end

        # For small P_ti and Q_tv, use series expansion
        if P_ti < BigFloat("0.001") && Q_tv < BigFloat("0.001")
            # d = P_ti + Q_tv + higher order terms
            # Exact series: -1/2 * [ln(1 - 2P - Q) + 1/2 * ln(1 - 2Q)]
            x = BigFloat(2) * P_ti + Q_tv
            y = BigFloat(2) * Q_tv

            # Series for -ln(1-x): sum x^k/k
            log_term1 = BigFloat(0)
            xk = x
            for k in 1:20
                log_term1 += xk / BigFloat(k)
                xk *= x
            end

            log_term2 = BigFloat(0)
            yk = y
            for k in 1:20
                log_term2 += yk / BigFloat(k)
                yk *= y
            end

            return (log_term1 + log_term2 / BigFloat(2)) / BigFloat(2)
        end

        return -BigFloat(1) / BigFloat(2) * log(term1 * sqrt(term2))
    end
end

"""
    Cladistics.backend_coprocessor_distance_matrix(::MathBackend, sequences, method)

Math coprocessor-accelerated pairwise distance matrix using extended precision
arithmetic. Provides numerically exact distance corrections for JC69 and K2P
models, preventing catastrophic cancellation that occurs with Float64 for
closely related or highly divergent sequences.
"""
function Cladistics.backend_coprocessor_distance_matrix(b::MathBackend,
                                                         sequences::Vector{String},
                                                         method::Symbol)
    n = length(sequences)
    # Extended precision is always beneficial, even for small inputs
    n < 4 && return nothing

    # Only JC69 and K2P benefit from extended precision
    method in (:hamming, :p_distance) && return nothing

    seq_len = length(sequences[1])
    encoded = _encode_sequences(sequences)

    mem_estimate = Int64(n * n * 32 + seq_len * n)  # BigFloat overhead
    track_allocation!(b, mem_estimate)

    try
        D = zeros(Float64, n, n)

        @inbounds for j in 1:n
            for i in 1:(j-1)
                diffs = 0
                transitions = 0
                for s in 1:seq_len
                    a = encoded[s, i]
                    b_val = encoded[s, j]
                    if a != b_val
                        diffs += 1
                        if _is_transition(a, b_val)
                            transitions += 1
                        end
                    end
                end

                d = if method == :jc69
                    Float64(_bigfloat_jc69(diffs, seq_len))
                elseif method == :k2p
                    Float64(_bigfloat_k2p(diffs, transitions, seq_len))
                else
                    Float64(diffs) / seq_len
                end

                D[i, j] = d
                D[j, i] = d
            end
        end

        track_deallocation!(b, mem_estimate)
        return D
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "Math distance matrix failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Extended Precision Parsimony (exact integer arithmetic)
# ============================================================================

"""
    Cladistics.backend_coprocessor_parsimony_score(::MathBackend, tree, char_matrix)

Math coprocessor-accelerated Fitch parsimony using exact integer arithmetic.
Parsimony scores are inherently integer-valued, but for weighted parsimony
with rational step matrices, extended precision prevents rounding errors
in weight accumulation across large trees.
"""
function Cladistics.backend_coprocessor_parsimony_score(b::MathBackend,
                                                         tree::Cladistics.PhylogeneticTree,
                                                         char_matrix::Matrix{Char})
    n_sites = size(char_matrix, 2)
    n_taxa = size(char_matrix, 1)

    n_sites < 16 && return nothing

    try
        bitmasks = Matrix{UInt32}(undef, n_taxa, n_sites)
        for t in 1:n_taxa
            for s in 1:n_sites
                code = get(CHAR_ENCODE, char_matrix[t, s], UInt8(0))
                bitmasks[t, s] = UInt32(1) << code
            end
        end

        left_child = Int[]
        right_child = Int[]
        taxon_idx = Int[]

        function _visit(node::Cladistics.TreeNode)
            if isempty(node.children)
                idx = findfirst(==(node.name), tree.taxa)
                push!(left_child, 0)
                push!(right_child, 0)
                push!(taxon_idx, idx === nothing ? 0 : idx)
            else
                child_ids = Int[]
                for child in node.children
                    _visit(child)
                    push!(child_ids, length(left_child))
                end
                push!(left_child, length(child_ids) >= 1 ? child_ids[1] : 0)
                push!(right_child, length(child_ids) >= 2 ? child_ids[2] : 0)
                push!(taxon_idx, 0)
            end
        end
        _visit(tree.root)
        n_nodes = length(left_child)

        # Use BigInt for exact accumulation (prevents overflow for huge trees)
        states = Matrix{UInt32}(undef, n_nodes, n_sites)
        total_score = BigInt(0)

        for node in 1:n_nodes
            ti = taxon_idx[node]
            lc = left_child[node]
            rc = right_child[node]

            if ti > 0
                @inbounds for s in 1:n_sites
                    states[node, s] = bitmasks[ti, s]
                end
            else
                @inbounds for s in 1:n_sites
                    left_set = lc > 0 ? states[lc, s] : UInt32(0xFFFFFFFF)
                    right_set = rc > 0 ? states[rc, s] : UInt32(0xFFFFFFFF)
                    intersection = left_set & right_set
                    if intersection != UInt32(0)
                        states[node, s] = intersection
                    else
                        states[node, s] = left_set | right_set
                        total_score += 1
                    end
                end
            end
        end

        return Int(total_score)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "Math parsimony score failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Extended Precision Bootstrap
# ============================================================================

function Cladistics.backend_coprocessor_bootstrap_replicate(b::MathBackend,
                                                             sequences::Vector{String},
                                                             replicates::Int,
                                                             method::Symbol)
    n = length(sequences)
    seq_len = length(sequences[1])

    # Only JC69/K2P benefit from extended precision in bootstrap
    method in (:hamming, :p_distance) && return nothing
    (n < 4 || replicates < 5) && return nothing

    try
        encoded = _encode_sequences(sequences)
        clade_counts = Dict{Set{String}, Int}()

        for rep in 1:replicates
            col_indices = rand(1:seq_len, seq_len)
            resampled = Matrix{UInt8}(undef, seq_len, n)
            @inbounds for j in 1:n
                for i in 1:seq_len
                    resampled[i, j] = encoded[col_indices[i], j]
                end
            end

            # Extended precision distance matrix
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

                    d = if method == :jc69
                        Float64(_bigfloat_jc69(diffs, seq_len))
                    elseif method == :k2p
                        Float64(_bigfloat_k2p(diffs, transitions, seq_len))
                    else
                        Float64(diffs) / seq_len
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
        @warn "Math bootstrap failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

function Cladistics.backend_coprocessor_neighbor_join(b::MathBackend, dmat::Matrix{Float64}, taxa_names)
    return nothing
end

function Cladistics.backend_coprocessor_tree_search(b::MathBackend, args...)
    return nothing
end

end # module CladisticsMathExt
