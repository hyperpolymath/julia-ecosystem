# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CladisticsROCmExt — ROCm GPU kernels for Cladistics.jl
# Accelerates distance matrix, parsimony scoring, and bootstrap resampling
# on AMD GPUs via KernelAbstractions.jl + AMDGPU.jl.

module CladisticsROCmExt

using Cladistics
using AMDGPU
using AMDGPU: ROCArray, ROCMatrix, ROCVector
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using AcceleratorGate: ROCmBackend, JuliaBackend, _record_diagnostic!

# ============================================================================
# Character Encoding (shared logic with CUDA/Metal ext)
# ============================================================================

const CHAR_ENCODE = let d = Dict{Char,UInt8}()
    for (i, c) in enumerate("ACGTUNacgtun-.")
        d[c] = UInt8(i)
    end
    d
end

"""
    _encode_sequences(sequences::Vector{String}) -> Matrix{UInt8}

Encode aligned sequences into a (seq_len x n_taxa) UInt8 matrix.
"""
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

# ============================================================================
# Transition/Transversion Classification
# ============================================================================

@inline function _is_transition(a::UInt8, b::UInt8)
    (a == 0x01 && b == 0x03) || (a == 0x03 && b == 0x01) ||
    (a == 0x02 && b == 0x04) || (a == 0x04 && b == 0x02) ||
    (a == 0x02 && b == 0x05) || (a == 0x05 && b == 0x02) ||
    (a == 0x04 && b == 0x05) || (a == 0x05 && b == 0x04)
end

# ============================================================================
# GPU Kernel: Pairwise Distance Matrix
# ============================================================================

@kernel function distance_kernel!(dmat, @Const(seqs), seq_len::Int32, method_code::Int32)
    i, j = @index(Global, NTuple)

    if i < j
        diffs = Int32(0)
        transitions = Int32(0)

        for k in Int32(1):seq_len
            a = seqs[k, i]
            b = seqs[k, j]
            if a != b
                diffs += Int32(1)
                if method_code == Int32(4)
                    if _is_transition(a, b)
                        transitions += Int32(1)
                    end
                end
            end
        end

        p = Float64(diffs) / Float64(seq_len)

        dist = if method_code == Int32(1)
            Float64(diffs)
        elseif method_code == Int32(2)
            p
        elseif method_code == Int32(3)
            if p >= 0.75
                Inf64
            else
                -0.75 * log(1.0 - (4.0 * p / 3.0))
            end
        else
            P_ti = Float64(transitions) / Float64(seq_len)
            Q_tv = Float64(diffs - transitions) / Float64(seq_len)
            term1 = 1.0 - 2.0 * P_ti - Q_tv
            term2 = 1.0 - 2.0 * Q_tv
            if term1 <= 0.0 || term2 <= 0.0
                Inf64
            else
                -0.5 * log(term1 * sqrt(term2))
            end
        end

        dmat[i, j] = dist
        dmat[j, i] = dist
    end
end

function _method_code(method::Symbol)
    method == :hamming    && return Int32(1)
    method == :p_distance && return Int32(2)
    method == :jc69       && return Int32(3)
    method == :k2p        && return Int32(4)
    error("Unknown distance method for GPU kernel: $method")
end

# ============================================================================
# GPU Kernel: Fitch Parsimony Score per Site
# ============================================================================

function _flatten_tree_postorder(root::Cladistics.TreeNode, taxa::Vector{String})
    left_child = Int32[]
    right_child = Int32[]
    taxon_idx = Int32[]

    function _visit(node::Cladistics.TreeNode)
        if isempty(node.children)
            idx = findfirst(==(node.name), taxa)
            push!(left_child, Int32(0))
            push!(right_child, Int32(0))
            push!(taxon_idx, idx === nothing ? Int32(0) : Int32(idx))
        else
            child_indices = Int32[]
            for child in node.children
                _visit(child)
                push!(child_indices, Int32(length(left_child)))
            end
            push!(left_child, length(child_indices) >= 1 ? child_indices[1] : Int32(0))
            push!(right_child, length(child_indices) >= 2 ? child_indices[2] : Int32(0))
            push!(taxon_idx, Int32(0))
        end
    end

    _visit(root)
    return left_child, right_child, taxon_idx
end

@inline _char_to_bitmask(c::UInt8) = UInt32(1) << c

@kernel function parsimony_kernel_with_scratch!(scores, scratch, @Const(char_matrix),
                                                @Const(left_child), @Const(right_child),
                                                @Const(taxon_idx), n_nodes::Int32)
    site = @index(Global)
    score = Int32(0)

    for node in Int32(1):n_nodes
        lc = left_child[node]
        rc = right_child[node]
        ti = taxon_idx[node]

        if ti > Int32(0)
            scratch[node, site] = _char_to_bitmask(char_matrix[site, ti])
        else
            left_set = lc > Int32(0) ? scratch[lc, site] : UInt32(0xFFFFFFFF)
            right_set = rc > Int32(0) ? scratch[rc, site] : UInt32(0xFFFFFFFF)

            intersection = left_set & right_set
            if intersection != UInt32(0)
                scratch[node, site] = intersection
            else
                scratch[node, site] = left_set | right_set
                score += Int32(1)
            end
        end
    end

    scores[site] = score
end

# ============================================================================
# GPU Kernel: Bootstrap Column Resampling
# ============================================================================

@kernel function bootstrap_resample_kernel!(out, @Const(seqs), @Const(col_indices),
                                            seq_len::Int32, n_taxa::Int32)
    rep, col = @index(Global, NTuple)
    src_col = col_indices[col, rep]
    for t in Int32(1):n_taxa
        out[col, t, rep] = seqs[src_col, t]
    end
end

# ============================================================================
# Host-Side Backend Implementations
# ============================================================================

function Cladistics.backend_distance_matrix(::ROCmBackend, sequences::Vector{String}, method::Symbol)
    n = length(sequences)
    n < 16 && return nothing

    seq_len = length(sequences[1])
    encoded = _encode_sequences(sequences)

    d_seqs = ROCArray(encoded)
    d_dmat = AMDGPU.zeros(Float64, n, n)

    kernel = distance_kernel!(KernelAbstractions.ROCDevice(), (16, 16))
    kernel(d_dmat, d_seqs, Int32(seq_len), _method_code(method); ndrange=(n, n))
    KernelAbstractions.synchronize(KernelAbstractions.ROCDevice())

    return Array(d_dmat)
end

function Cladistics.backend_parsimony_score(::ROCmBackend, tree::Cladistics.PhylogeneticTree,
                                            char_matrix::Matrix{Char})
    n_sites = size(char_matrix, 2)
    n_sites < 64 && return nothing

    left_child, right_child, taxon_idx = _flatten_tree_postorder(tree.root, tree.taxa)
    n_nodes = Int32(length(left_child))

    n_taxa = size(char_matrix, 1)
    encoded_chars = zeros(UInt8, n_sites, n_taxa)
    for t in 1:n_taxa
        for s in 1:n_sites
            encoded_chars[s, t] = get(CHAR_ENCODE, char_matrix[t, s], UInt8(0))
        end
    end

    d_left = ROCArray(left_child)
    d_right = ROCArray(right_child)
    d_taxon = ROCArray(taxon_idx)
    d_chars = ROCArray(encoded_chars)
    d_scores = AMDGPU.zeros(Int32, n_sites)
    d_scratch = AMDGPU.zeros(UInt32, Int(n_nodes), n_sites)

    kernel = parsimony_kernel_with_scratch!(KernelAbstractions.ROCDevice(), 256)
    kernel(d_scores, d_scratch, d_chars, d_left, d_right, d_taxon, n_nodes;
           ndrange=n_sites)
    KernelAbstractions.synchronize(KernelAbstractions.ROCDevice())

    return Int(sum(Array(d_scores)))
end

function Cladistics.backend_bootstrap_replicate(::ROCmBackend, sequences::Vector{String},
                                                replicates::Int, method::Symbol)
    n = length(sequences)
    seq_len = length(sequences[1])
    (n < 16 || replicates < 10) && return nothing

    encoded = _encode_sequences(sequences)
    d_seqs = ROCArray(encoded)

    col_indices = zeros(Int32, seq_len, replicates)
    for rep in 1:replicates
        for s in 1:seq_len
            col_indices[s, rep] = Int32(rand(1:seq_len))
        end
    end
    d_col_indices = ROCArray(col_indices)

    d_resampled = AMDGPU.zeros(UInt8, seq_len, n, replicates)

    kernel = bootstrap_resample_kernel!(KernelAbstractions.ROCDevice(), (16, 16))
    kernel(d_resampled, d_seqs, d_col_indices, Int32(seq_len), Int32(n);
           ndrange=(replicates, seq_len))
    KernelAbstractions.synchronize(KernelAbstractions.ROCDevice())

    d_dmat = AMDGPU.zeros(Float64, n, n)
    mcode = _method_code(method == :hamming ? :p_distance : method)

    clade_counts = Dict{Set{String}, Int}()

    for rep in 1:replicates
        AMDGPU.fill!(d_dmat, 0.0)
        d_rep_seqs = ROCArray(Array(d_resampled[:, :, rep]))

        dist_kernel = distance_kernel!(KernelAbstractions.ROCDevice(), (16, 16))
        dist_kernel(d_dmat, d_rep_seqs, Int32(seq_len), mcode; ndrange=(n, n))
        KernelAbstractions.synchronize(KernelAbstractions.ROCDevice())

        cpu_dmat = Array(d_dmat)

        boot_tree = if method == :upgma
            Cladistics.upgma(cpu_dmat)
        else
            Cladistics.neighbor_joining(cpu_dmat)
        end

        clades = Cladistics.extract_clades(boot_tree.root)
        for clade in clades
            clade_counts[clade] = get(clade_counts, clade, 0) + 1
        end
    end

    return Dict(clade => count / replicates for (clade, count) in clade_counts)
end

function Cladistics.backend_neighbor_join(::ROCmBackend, dmat::Matrix{Float64}, taxa_names)
    # NJ is inherently sequential; fall back to CPU.
    return nothing
end

function Cladistics.backend_tree_search(::ROCmBackend, args...)
    return nothing
end

end # module CladisticsROCmExt
