# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CladisticsVPUExt — SIMD vector processing acceleration for Cladistics.jl
# Exploits vector processing units for SIMD-parallel pairwise distance
# computation, vectorized Fitch parsimony, and bootstrap resampling.

module CladisticsVPUExt

using Cladistics
using AcceleratorGate
using AcceleratorGate: VPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(VPUBackend, :distance_matrix)
    register_operation!(VPUBackend, :parsimony_score)
    register_operation!(VPUBackend, :bootstrap_replicate)
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
# SIMD-Vectorized Pairwise Distance
# ============================================================================
#
# VPU key advantage: process 16-64 bytes simultaneously with SIMD comparison
# instructions. We pack sequences as UInt8 vectors and use vectorized
# inequality comparison to count differences.

"""
    _simd_hamming_distance(seq_a::AbstractVector{UInt8}, seq_b::AbstractVector{UInt8}) -> Int

Compute Hamming distance between two encoded sequences using SIMD-friendly
vectorized comparison. Julia's LLVM backend auto-vectorizes the byte
comparison loop when operating on contiguous UInt8 arrays.
"""
function _simd_hamming_distance(seq_a::AbstractVector{UInt8}, seq_b::AbstractVector{UInt8})
    len = length(seq_a)
    diffs = 0

    # Process in chunks of 32 bytes for SIMD alignment
    # The @simd macro hints the compiler to use vector instructions
    @inbounds @simd for i in 1:len
        diffs += (seq_a[i] != seq_b[i])
    end
    return diffs
end

"""
    _simd_transition_count(seq_a::AbstractVector{UInt8}, seq_b::AbstractVector{UInt8}) -> Tuple{Int,Int}

Count differences and transitions simultaneously using vectorized scan.
Returns (total_diffs, transition_count).
"""
function _simd_transition_count(seq_a::AbstractVector{UInt8}, seq_b::AbstractVector{UInt8})
    len = length(seq_a)
    diffs = 0
    transitions = 0

    @inbounds for i in 1:len
        a = seq_a[i]
        b = seq_b[i]
        if a != b
            diffs += 1
            if _is_transition(a, b)
                transitions += 1
            end
        end
    end
    return (diffs, transitions)
end

"""
    _simd_pairwise_distances(encoded::Matrix{UInt8}, method::Symbol) -> Matrix{Float64}

Compute full pairwise distance matrix using SIMD-vectorized sequence comparison.
Column-major layout ensures contiguous memory access per taxon sequence.
"""
function _simd_pairwise_distances(encoded::Matrix{UInt8}, method::Symbol)
    seq_len, n = size(encoded)
    D = zeros(Float64, n, n)

    @inbounds for j in 1:n
        col_j = @view encoded[:, j]
        for i in 1:(j-1)
            col_i = @view encoded[:, i]

            if method == :k2p
                diffs, transitions = _simd_transition_count(col_i, col_j)
                P_ti = transitions / seq_len
                Q_tv = (diffs - transitions) / seq_len
                term1 = 1.0 - 2.0 * P_ti - Q_tv
                term2 = 1.0 - 2.0 * Q_tv
                d = (term1 <= 0.0 || term2 <= 0.0) ? Inf : -0.5 * log(term1 * sqrt(term2))
            else
                diffs = _simd_hamming_distance(col_i, col_j)
                p = diffs / seq_len

                d = if method == :hamming
                    Float64(diffs)
                elseif method == :p_distance
                    p
                elseif method == :jc69
                    p >= 0.75 ? Inf : -0.75 * log(1.0 - (4.0 * p / 3.0))
                else
                    Float64(diffs)
                end
            end

            D[i, j] = d
            D[j, i] = d
        end
    end
    return D
end

"""
    Cladistics.backend_coprocessor_distance_matrix(::VPUBackend, sequences, method)

VPU-accelerated pairwise distance matrix using SIMD-vectorized byte comparison.
Encodes sequences as packed UInt8 vectors and uses @simd-annotated loops for
hardware vector instruction utilization.
"""
function Cladistics.backend_coprocessor_distance_matrix(b::VPUBackend,
                                                         sequences::Vector{String},
                                                         method::Symbol)
    n = length(sequences)
    # VPU is efficient even for moderate sizes due to low overhead
    n < 8 && return nothing

    seq_len = length(sequences[1])
    mem_estimate = Int64(seq_len * n + n * n * 8)
    track_allocation!(b, mem_estimate)

    try
        encoded = _encode_sequences(sequences)
        result = _simd_pairwise_distances(encoded, method)
        track_deallocation!(b, mem_estimate)
        return result
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "VPU distance matrix failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# SIMD-Vectorized Fitch Parsimony
# ============================================================================

"""
    Cladistics.backend_coprocessor_parsimony_score(::VPUBackend, tree, char_matrix)

VPU-accelerated Fitch parsimony scoring using SIMD-vectorized bitwise
operations. The bitmask intersection/union step at each internal node
processes all sites simultaneously using vector bitwise AND/OR.
"""
function Cladistics.backend_coprocessor_parsimony_score(b::VPUBackend,
                                                         tree::Cladistics.PhylogeneticTree,
                                                         char_matrix::Matrix{Char})
    n_sites = size(char_matrix, 2)
    n_taxa = size(char_matrix, 1)

    n_sites < 32 && return nothing

    try
        # Encode as bitmasks -- UInt32 per site per taxon
        bitmasks = Matrix{UInt32}(undef, n_taxa, n_sites)
        for t in 1:n_taxa
            for s in 1:n_sites
                code = get(CHAR_ENCODE, char_matrix[t, s], UInt8(0))
                bitmasks[t, s] = UInt32(1) << code
            end
        end

        # Flatten tree to postorder
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

        # SIMD-vectorized Fitch traversal: process all sites per node
        states = Matrix{UInt32}(undef, n_nodes, n_sites)
        scores = zeros(Int, n_sites)

        for node in 1:n_nodes
            ti = taxon_idx[node]
            lc = left_child[node]
            rc = right_child[node]

            if ti > 0
                # Leaf: vectorized copy
                @inbounds @simd for s in 1:n_sites
                    states[node, s] = bitmasks[ti, s]
                end
            else
                # Internal: SIMD-vectorized bitwise intersection/union
                @inbounds for s in 1:n_sites
                    left_set = lc > 0 ? states[lc, s] : UInt32(0xFFFFFFFF)
                    right_set = rc > 0 ? states[rc, s] : UInt32(0xFFFFFFFF)
                    intersection = left_set & right_set
                    if intersection != UInt32(0)
                        states[node, s] = intersection
                    else
                        states[node, s] = left_set | right_set
                        scores[s] += 1
                    end
                end
            end
        end

        return sum(scores)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "VPU parsimony score failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# SIMD-Vectorized Bootstrap
# ============================================================================

"""
    Cladistics.backend_coprocessor_bootstrap_replicate(::VPUBackend, sequences, replicates, method)

VPU-accelerated bootstrap resampling using SIMD-vectorized distance computation.
Each replicate's distance matrix is computed using the vectorized pairwise
comparison routines.
"""
function Cladistics.backend_coprocessor_bootstrap_replicate(b::VPUBackend,
                                                             sequences::Vector{String},
                                                             replicates::Int,
                                                             method::Symbol)
    n = length(sequences)
    seq_len = length(sequences[1])

    (n < 8 || replicates < 5) && return nothing

    try
        encoded = _encode_sequences(sequences)
        clade_counts = Dict{Set{String}, Int}()

        for rep in 1:replicates
            # Resample columns using vectorized gather
            col_indices = rand(1:seq_len, seq_len)
            resampled = Matrix{UInt8}(undef, seq_len, n)

            # Vectorized column resampling
            @inbounds for j in 1:n
                @simd for i in 1:seq_len
                    resampled[i, j] = encoded[col_indices[i], j]
                end
            end

            # SIMD-vectorized distance computation
            dmat = _simd_pairwise_distances(resampled, method)

            # Tree construction on CPU
            boot_tree = Cladistics.neighbor_joining(dmat)
            clades = Cladistics.extract_clades(boot_tree.root)
            for clade in clades
                clade_counts[clade] = get(clade_counts, clade, 0) + 1
            end
        end

        return Dict(clade => count / replicates for (clade, count) in clade_counts)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "VPU bootstrap failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# VPU Neighbor Join and Tree Search
# ============================================================================

function Cladistics.backend_coprocessor_neighbor_join(b::VPUBackend, dmat::Matrix{Float64}, taxa_names)
    # NJ is inherently sequential; VPU helps more in the distance computation stage
    return nothing
end

function Cladistics.backend_coprocessor_tree_search(b::VPUBackend, args...)
    return nothing
end

end # module CladisticsVPUExt
