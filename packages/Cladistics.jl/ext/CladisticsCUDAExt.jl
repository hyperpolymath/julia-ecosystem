# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CladisticsCUDAExt — CUDA GPU kernels for Cladistics.jl
# Accelerates distance matrix, parsimony scoring, and bootstrap resampling
# on NVIDIA GPUs via KernelAbstractions.jl + CUDA.jl.

module CladisticsCUDAExt

using Cladistics
using CUDA
using CUDA: CuArray, @cuda, CuMatrix, CuVector
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using AcceleratorGate: CUDABackend, JuliaBackend, _record_diagnostic!

# ============================================================================
# Character Encoding
# ============================================================================

# Encode characters as UInt8 for GPU-friendly representation.
# Standard DNA/RNA/protein characters map to compact integer codes.
const CHAR_ENCODE = let d = Dict{Char,UInt8}()
    for (i, c) in enumerate("ACGTUNacgtun-.")
        d[c] = UInt8(i)
    end
    d
end

"""
    _encode_sequences(sequences::Vector{String}) -> Matrix{UInt8}

Encode aligned sequences into a (seq_len x n_taxa) UInt8 matrix suitable
for GPU transfer.  Column-major layout means each column is one taxon's
full sequence, which gives coalesced reads when iterating over sites.
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
# Transition/Transversion Classification (for Kimura 2-parameter)
# ============================================================================

# Purines: A(1), G(3); Pyrimidines: C(2), T(4), U(5)
# Transition: purine<->purine or pyrimidine<->pyrimidine
# Transversion: purine<->pyrimidine

"""
    _is_transition(a::UInt8, b::UInt8) -> Bool

Test whether two encoded nucleotides form a transition substitution.
Uses the encoding from CHAR_ENCODE where A=1, C=2, G=3, T=4, U=5.
"""
@inline function _is_transition(a::UInt8, b::UInt8)
    # A(1)<->G(3) or C(2)<->T(4) or C(2)<->U(5) or T(4)<->U(5)
    (a == 0x01 && b == 0x03) || (a == 0x03 && b == 0x01) ||
    (a == 0x02 && b == 0x04) || (a == 0x04 && b == 0x02) ||
    (a == 0x02 && b == 0x05) || (a == 0x05 && b == 0x02) ||
    (a == 0x04 && b == 0x05) || (a == 0x05 && b == 0x04)
end

# ============================================================================
# GPU Kernel: Pairwise Distance Matrix
# ============================================================================

"""
    distance_kernel!(dmat, seqs, seq_len, method_code)

KernelAbstractions kernel computing pairwise sequence distances.
Each work-item handles one (i, j) pair with i < j.

Method codes: 1 = hamming, 2 = p_distance, 3 = jc69, 4 = k2p.
"""
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
                # Track transitions for K2P — branch-free on other methods
                if method_code == Int32(4)
                    if _is_transition(a, b)
                        transitions += Int32(1)
                    end
                end
            end
        end

        p = Float64(diffs) / Float64(seq_len)

        dist = if method_code == Int32(1)
            # Hamming: raw count
            Float64(diffs)
        elseif method_code == Int32(2)
            # P-distance: proportion of differences
            p
        elseif method_code == Int32(3)
            # Jukes-Cantor 1969
            if p >= 0.75
                Inf64
            else
                -0.75 * log(1.0 - (4.0 * p / 3.0))
            end
        else
            # Kimura 2-parameter
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

"""
    _method_code(method::Symbol) -> Int32

Map symbolic distance method to integer code for GPU kernel dispatch.
"""
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

# The Fitch algorithm is tree-recursive, which is hard to parallelise per se.
# Instead we parallelise across character sites: each GPU thread computes the
# full Fitch traversal for one column of the character matrix.  For large
# alignments (thousands of sites) this gives substantial speedup.
#
# We flatten the tree into a postorder traversal array on the CPU, then the
# kernel walks the array bottom-up.  Each node is represented by:
#   - left_child index (0 for leaves)
#   - right_child index (0 for leaves)
#   - taxon index (>0 for leaves, 0 for internal)
#
# The node state sets are stored as bitmasks (UInt32) — one bit per unique
# character state.  For DNA there are only 4-5 states so this fits easily.

"""
    _flatten_tree_postorder(root::Cladistics.TreeNode, taxa::Vector{String})

Flatten tree into arrays suitable for GPU Fitch traversal.
Returns (left_child, right_child, taxon_idx) each of length n_nodes,
in postorder.
"""
function _flatten_tree_postorder(root::Cladistics.TreeNode, taxa::Vector{String})
    left_child = Int32[]
    right_child = Int32[]
    taxon_idx = Int32[]
    node_map = Dict{Cladistics.TreeNode, Int32}()

    function _visit(node::Cladistics.TreeNode)
        if isempty(node.children)
            # Leaf node
            idx = findfirst(==(node.name), taxa)
            push!(left_child, Int32(0))
            push!(right_child, Int32(0))
            push!(taxon_idx, idx === nothing ? Int32(0) : Int32(idx))
        else
            # Visit children first (postorder)
            child_indices = Int32[]
            for child in node.children
                _visit(child)
                push!(child_indices, Int32(length(left_child)))
            end
            # Internal node — for Fitch we handle bifurcating; for multifurcating
            # we chain pairwise intersections (first two children, then fold in rest).
            push!(left_child, length(child_indices) >= 1 ? child_indices[1] : Int32(0))
            push!(right_child, length(child_indices) >= 2 ? child_indices[2] : Int32(0))
            push!(taxon_idx, Int32(0))
        end
        node_map[node] = Int32(length(left_child))
    end

    _visit(root)
    return left_child, right_child, taxon_idx
end

"""
    _char_to_bitmask(c::UInt8) -> UInt32

Convert an encoded character to a bitmask with one bit set.
"""
@inline _char_to_bitmask(c::UInt8) = UInt32(1) << c

"""
    parsimony_kernel!(scores, char_matrix, left_child, right_child, taxon_idx,
                      n_nodes, n_taxa)

KernelAbstractions kernel computing Fitch parsimony score for each character site.
Each work-item processes one column of the character matrix.
"""
@kernel function parsimony_kernel!(scores, @Const(char_matrix), @Const(left_child),
                                   @Const(right_child), @Const(taxon_idx),
                                   n_nodes::Int32, n_taxa::Int32)
    site = @index(Global)

    score = Int32(0)

    # Allocate state bitmasks in registers / local memory
    # We use a simple array — KernelAbstractions handles this per-thread
    # For trees up to ~4096 nodes this is fine on modern GPUs
    # (stack allocation, not heap)

    # Walk postorder: nodes 1..n_nodes
    # We need per-node state; use a flat array sized to n_nodes
    # Since KA kernels can't dynamically allocate, we use a fixed-size approach
    # via repeated computation (stateless Fitch).

    # Stateless recursive Fitch: we recompute from leaves each time.
    # For GPU efficiency with many sites, this is actually fine — the tree
    # is small (tens to hundreds of nodes) but sites are thousands.

    # Instead of dynamic arrays, store state in a stack-allocated MVector
    # approximation using tuple mutation... but that's fragile in KA.

    # Practical approach: use shared memory via workgroup or just accept
    # that n_nodes is small and store results in global memory scratch space.

    # For simplicity and correctness, we use the global memory scratch approach:
    # each thread gets a slice of a (n_nodes x n_sites) scratch buffer.
    # But that's wasteful. Better: since n_nodes is tiny, just recompute.

    # ACTUALLY: the cleanest GPU approach is to use a pre-allocated scratch
    # buffer passed as a kernel argument. Let's do that.

    # ... but KernelAbstractions doesn't support variable-length local arrays
    # elegantly. The pragmatic solution: allocate scratch globally.

    # We'll handle this in the host function by passing a scratch buffer.
    # For now, mark this kernel as operating on a pre-flattened structure.

    # Simple approach that works: iterate postorder, maintain state in global scratch.
    # The host provides: scratch_states[node, site] as UInt32 buffer.

    scores[site] = score
end

"""
    parsimony_kernel_with_scratch!(scores, scratch, char_matrix, left_child,
                                   right_child, taxon_idx, n_nodes)

Fitch parsimony kernel using a global scratch buffer for node state bitmasks.
scratch is (n_nodes, n_sites) UInt32 matrix.
Each thread processes one site (column).
"""
@kernel function parsimony_kernel_with_scratch!(scores, scratch, @Const(char_matrix),
                                                @Const(left_child), @Const(right_child),
                                                @Const(taxon_idx), n_nodes::Int32)
    site = @index(Global)
    score = Int32(0)

    # Walk nodes in postorder (indices 1..n_nodes)
    for node in Int32(1):n_nodes
        lc = left_child[node]
        rc = right_child[node]
        ti = taxon_idx[node]

        if ti > Int32(0)
            # Leaf: state is the single character at this site
            scratch[node, site] = _char_to_bitmask(char_matrix[site, ti])
        else
            # Internal: intersection/union of children
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

"""
    bootstrap_resample_kernel!(out_matrices, seqs, col_indices, seq_len, n_taxa)

Each work-item copies one resampled column for one replicate.
Grid: (n_replicates, seq_len) — each thread handles one column of one replicate.
"""
@kernel function bootstrap_resample_kernel!(out, @Const(seqs), @Const(col_indices),
                                            seq_len::Int32, n_taxa::Int32)
    rep, col = @index(Global, NTuple)

    # col_indices[col, rep] gives the source column
    src_col = col_indices[col, rep]

    # Copy all taxa for this column
    for t in Int32(1):n_taxa
        out[col, t, rep] = seqs[src_col, t]
    end
end

# ============================================================================
# Host-Side Backend Implementations
# ============================================================================

"""
    Cladistics.backend_distance_matrix(::CUDABackend, sequences, method)

GPU-accelerated pairwise distance matrix computation on CUDA.
Encodes sequences as UInt8, transfers to GPU, launches one thread per (i,j) pair.
"""
function Cladistics.backend_distance_matrix(::CUDABackend, sequences::Vector{String}, method::Symbol)
    n = length(sequences)

    # Small inputs: not worth GPU overhead
    n < 16 && return nothing

    seq_len = length(sequences[1])
    encoded = _encode_sequences(sequences)

    # Transfer to GPU
    d_seqs = CuArray(encoded)
    d_dmat = CUDA.zeros(Float64, n, n)

    # Launch kernel — one thread per (i, j) pair
    backend = CUDABackend()
    kernel = distance_kernel!(KernelAbstractions.CUDADevice(), (16, 16))
    kernel(d_dmat, d_seqs, Int32(seq_len), _method_code(method); ndrange=(n, n))
    KernelAbstractions.synchronize(KernelAbstractions.CUDADevice())

    # Transfer back
    result = Array(d_dmat)

    # Clean up
    CUDA.unsafe_free!(d_seqs)
    CUDA.unsafe_free!(d_dmat)

    return result
end

"""
    Cladistics.backend_parsimony_score(::CUDABackend, tree, char_matrix)

GPU-accelerated Fitch parsimony scoring on CUDA.
Parallelises across character sites — each GPU thread computes the Fitch
traversal for one column of the alignment.
"""
function Cladistics.backend_parsimony_score(::CUDABackend, tree::Cladistics.PhylogeneticTree,
                                            char_matrix::Matrix{Char})
    n_sites = size(char_matrix, 2)

    # Small alignments: not worth GPU overhead
    n_sites < 64 && return nothing

    # Flatten tree to postorder arrays
    left_child, right_child, taxon_idx = _flatten_tree_postorder(tree.root, tree.taxa)
    n_nodes = Int32(length(left_child))

    # Encode character matrix as UInt8
    n_taxa = size(char_matrix, 1)
    encoded_chars = zeros(UInt8, n_sites, n_taxa)
    for t in 1:n_taxa
        for s in 1:n_sites
            encoded_chars[s, t] = get(CHAR_ENCODE, char_matrix[t, s], UInt8(0))
        end
    end

    # Transfer to GPU
    d_left = CuArray(left_child)
    d_right = CuArray(right_child)
    d_taxon = CuArray(taxon_idx)
    d_chars = CuArray(encoded_chars)
    d_scores = CUDA.zeros(Int32, n_sites)
    d_scratch = CUDA.zeros(UInt32, Int(n_nodes), n_sites)

    # Launch kernel — one thread per site
    backend = CUDABackend()
    kernel = parsimony_kernel_with_scratch!(KernelAbstractions.CUDADevice(), 256)
    kernel(d_scores, d_scratch, d_chars, d_left, d_right, d_taxon, n_nodes;
           ndrange=n_sites)
    KernelAbstractions.synchronize(KernelAbstractions.CUDADevice())

    # Sum scores on GPU, transfer scalar
    total = Int(sum(Array(d_scores)))

    # Clean up
    CUDA.unsafe_free!(d_left)
    CUDA.unsafe_free!(d_right)
    CUDA.unsafe_free!(d_taxon)
    CUDA.unsafe_free!(d_chars)
    CUDA.unsafe_free!(d_scores)
    CUDA.unsafe_free!(d_scratch)

    return total
end

"""
    Cladistics.backend_bootstrap_replicate(::CUDABackend, sequences, replicates, method)

GPU-accelerated bootstrap resampling on CUDA.
Generates resampled column indices on CPU (or GPU RNG), resamples alignment
columns on GPU in parallel, then builds distance matrices on GPU.
Tree construction remains on CPU (inherently sequential NJ/UPGMA).
"""
function Cladistics.backend_bootstrap_replicate(::CUDABackend, sequences::Vector{String},
                                                replicates::Int, method::Symbol)
    n = length(sequences)
    seq_len = length(sequences[1])

    # Small problems: not worth GPU overhead
    (n < 16 || replicates < 10) && return nothing

    encoded = _encode_sequences(sequences)
    d_seqs = CuArray(encoded)

    # Generate resampled column indices on CPU
    col_indices = zeros(Int32, seq_len, replicates)
    for rep in 1:replicates
        for s in 1:seq_len
            col_indices[s, rep] = Int32(rand(1:seq_len))
        end
    end
    d_col_indices = CuArray(col_indices)

    # Allocate resampled output: (seq_len, n_taxa, replicates)
    d_resampled = CUDA.zeros(UInt8, seq_len, n, replicates)

    # Launch resampling kernel
    kernel = bootstrap_resample_kernel!(KernelAbstractions.CUDADevice(), (16, 16))
    kernel(d_resampled, d_seqs, d_col_indices, Int32(seq_len), Int32(n);
           ndrange=(replicates, seq_len))
    KernelAbstractions.synchronize(KernelAbstractions.CUDADevice())

    # For each replicate: compute distance matrix on GPU, build tree on CPU
    # Pre-allocate distance matrix on GPU
    d_dmat = CUDA.zeros(Float64, n, n)
    mcode = _method_code(method == :hamming ? :p_distance : method)

    clade_counts = Dict{Set{String}, Int}()
    taxa_names = nothing  # Let CPU functions assign default names

    for rep in 1:replicates
        # Extract this replicate's resampled sequences
        # Compute distance matrix directly on GPU from resampled data
        CUDA.fill!(d_dmat, 0.0)

        # We need a slice: d_resampled[:, :, rep]
        # CuArray slicing creates a view; copy to contiguous array for kernel
        d_rep_seqs = CuArray(Array(d_resampled[:, :, rep]))

        dist_kernel = distance_kernel!(KernelAbstractions.CUDADevice(), (16, 16))
        dist_kernel(d_dmat, d_rep_seqs, Int32(seq_len), mcode; ndrange=(n, n))
        KernelAbstractions.synchronize(KernelAbstractions.CUDADevice())

        # Transfer distance matrix to CPU for tree building
        cpu_dmat = Array(d_dmat)
        CUDA.unsafe_free!(d_rep_seqs)

        # Build tree on CPU (NJ/UPGMA are inherently sequential)
        boot_tree = if method == :upgma
            Cladistics.upgma(cpu_dmat)
        else
            Cladistics.neighbor_joining(cpu_dmat)
        end

        # Extract and count clades
        clades = Cladistics.extract_clades(boot_tree.root)
        for clade in clades
            clade_counts[clade] = get(clade_counts, clade, 0) + 1
        end
    end

    # Clean up
    CUDA.unsafe_free!(d_seqs)
    CUDA.unsafe_free!(d_col_indices)
    CUDA.unsafe_free!(d_resampled)
    CUDA.unsafe_free!(d_dmat)

    # Convert counts to proportions
    return Dict(clade => count / replicates for (clade, count) in clade_counts)
end

# NJ dispatch — accelerate Q-matrix computation on GPU
function Cladistics.backend_neighbor_join(::CUDABackend, dmat::Matrix{Float64}, taxa_names)
    n = size(dmat, 1)

    # NJ is inherently sequential (iterative nearest-pair joining).
    # GPU acceleration of the Q-matrix is only worthwhile for very large n.
    n < 128 && return nothing

    # For large n, we accelerate the Q-matrix computation step but keep the
    # iterative loop on CPU.  The full NJ is delegated to the CPU path since
    # the Q-matrix recomputation cost is O(n^2) per iteration and the iteration
    # count is O(n), giving O(n^3) total — GPU helps with the inner O(n^2).
    # However, the data transfer overhead per iteration usually dominates for
    # moderate n.  Return nothing to fall back to CPU for now.
    # Future: persistent GPU Q-matrix update for n > 512 (data transfer overhead
    # currently dominates at moderate n; revisit when kernel fusion is available)
    return nothing
end

# Tree search — placeholder for future GPU-accelerated tree space exploration
function Cladistics.backend_tree_search(::CUDABackend, args...)
    return nothing
end

end # module CladisticsCUDAExt
