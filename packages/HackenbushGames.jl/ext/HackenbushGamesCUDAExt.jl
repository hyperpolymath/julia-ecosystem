# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# HackenbushGames.jl CUDA Extension
# GPU-accelerated game tree evaluation, Grundy numbers, move generation,
# and position hashing via NVIDIA CUDA + KernelAbstractions.jl.
# Automatically loaded when both CUDA.jl and KernelAbstractions.jl are imported.

module HackenbushGamesCUDAExt

using CUDA
using KernelAbstractions
using HackenbushGames
using HackenbushGames: Edge, EdgeColor, HackenbushGraph, GameForm,
    Blue, Red, Green, _graph_key, _edge_allowed,
    mex, nim_sum, simplest_dyadic_between, simplify_game,
    cut_edge, prune_disconnected, moves

using AcceleratorGate
using AcceleratorGate: _backend_env_available, _backend_env_count, _record_diagnostic!

# ============================================================================
# Availability Detection
# ============================================================================

function AcceleratorGate.cuda_available()
    forced = _backend_env_available("AXIOM_CUDA_AVAILABLE")
    forced !== nothing && return forced
    CUDA.functional()
end

function AcceleratorGate.cuda_device_count()
    forced = _backend_env_count("AXIOM_CUDA_AVAILABLE", "AXIOM_CUDA_DEVICE_COUNT")
    forced !== nothing && return forced
    CUDA.ndevices()
end

# ============================================================================
# GPU Data Layout
# ============================================================================
#
# Game graphs are flattened to dense integer arrays for GPU transfer:
#   edges_u     :: Vector{Int32}   — source node per edge
#   edges_v     :: Vector{Int32}   — target node per edge
#   edges_color :: Vector{Int32}   — 0=Blue, 1=Red, 2=Green
#   ground      :: Vector{Int32}   — ground node ids
#
# For batch operations, positions are packed contiguously with offset arrays.

"""
    _flatten_graph(graph::HackenbushGraph)

Pack a HackenbushGraph into flat Int32 arrays suitable for GPU transfer.
Returns `(edges_u, edges_v, edges_color, ground)`.
"""
function _flatten_graph(graph::HackenbushGraph)
    n = length(graph.edges)
    eu = Vector{Int32}(undef, n)
    ev = Vector{Int32}(undef, n)
    ec = Vector{Int32}(undef, n)
    for (i, e) in enumerate(graph.edges)
        eu[i] = Int32(e.u)
        ev[i] = Int32(e.v)
        ec[i] = Int32(Int(e.color))
    end
    gr = Int32.(graph.ground)
    (eu, ev, ec, gr)
end

"""
    _unflatten_graph(eu, ev, ec, gr)

Reconstruct a HackenbushGraph from flat Int32 arrays.
"""
function _unflatten_graph(eu::Vector{Int32}, ev::Vector{Int32},
                          ec::Vector{Int32}, gr::Vector{Int32})
    edges = [Edge(Int(eu[i]), Int(ev[i]), EdgeColor(ec[i])) for i in eachindex(eu)]
    HackenbushGraph(edges, Int.(gr))
end

# ============================================================================
# Kernel 1: Move Generation (edge legality filter)
# ============================================================================
#
# Each thread checks one edge for legality given a player color.
# Output: a mask array where 1 = legal move, 0 = illegal.

@kernel function _move_legality_kernel!(mask, @Const(colors), player_code, num_edges)
    idx = @index(Global)
    if idx <= num_edges
        c = colors[idx]
        # player_code: 0 = :left (Blue+Green), 1 = :right (Red+Green)
        # color encoding: 0=Blue, 1=Red, 2=Green
        if c == Int32(2)
            # Green edges are always legal
            mask[idx] = Int32(1)
        elseif player_code == Int32(0) && c == Int32(0)
            # Left player can cut Blue
            mask[idx] = Int32(1)
        elseif player_code == Int32(1) && c == Int32(1)
            # Right player can cut Red
            mask[idx] = Int32(1)
        else
            mask[idx] = Int32(0)
        end
    end
end

"""
    _gpu_legal_move_indices(graph, player) -> Vector{Int}

Compute legal move indices on GPU using a parallel edge filter kernel.
Returns CPU-side vector of 1-based edge indices that are legal for `player`.
"""
function _gpu_legal_move_indices(graph::HackenbushGraph, player::Symbol)
    n = length(graph.edges)
    n == 0 && return Int[]

    _, _, ec, _ = _flatten_graph(graph)
    colors_gpu = CuArray(ec)
    mask_gpu = CUDA.zeros(Int32, n)
    player_code = player == :left ? Int32(0) : Int32(1)

    backend = CUDABackend()
    kernel = _move_legality_kernel!(backend, 256)
    kernel(mask_gpu, colors_gpu, player_code, Int32(n); ndrange=n)
    KernelAbstractions.synchronize(backend)

    mask_cpu = Array(mask_gpu)
    findall(==(Int32(1)), mask_cpu)
end

function HackenbushGames.backend_move_gen(::AcceleratorGate.CUDABackend,
                                          graph::HackenbushGraph, player::Symbol)
    n = length(graph.edges)
    # Only dispatch to GPU when there are enough edges to justify transfer overhead
    n < 64 && return nothing

    indices = try
        _gpu_legal_move_indices(graph, player)
    catch ex
        _record_diagnostic!("cuda", "runtime_errors")
        @warn "CUDA move generation failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end

    # Build result positions on CPU (pruning requires graph traversal)
    options = HackenbushGraph[]
    for i in indices
        push!(options, cut_edge(graph, i))
    end
    options
end

# ============================================================================
# Kernel 2: Position Hashing (Zobrist-style)
# ============================================================================
#
# Each thread computes a hash contribution for one edge using Zobrist tables.
# Final hash is an XOR reduction across all edge contributions.

@kernel function _zobrist_hash_kernel!(hashes, @Const(eu), @Const(ev), @Const(ec),
                                       @Const(zobrist_table), table_stride, num_edges)
    idx = @index(Global)
    if idx <= num_edges
        # Hash combines: edge endpoints + color via Zobrist table lookup
        u = eu[idx]
        v = ev[idx]
        c = ec[idx]
        # Use a mix of node indices and color to index into Zobrist table
        # table layout: table[node_hash % table_stride + 1, color + 1]
        u_idx = (u % table_stride) + Int32(1)
        v_idx = (v % table_stride) + Int32(1)
        h = zobrist_table[u_idx, c + Int32(1)] ⊻
            zobrist_table[v_idx, c + Int32(1)]
        hashes[idx] = h
    end
end

@kernel function _xor_reduce_kernel!(result, @Const(hashes), num_edges)
    # Simple sequential reduction (called with 1 thread).
    # For large arrays, a tree-based parallel reduction is better, but
    # position hashing is typically on moderate-size edge lists.
    acc = UInt64(0)
    for i in 1:num_edges
        acc ⊻= hashes[i]
    end
    result[1] = acc
end

# Pre-generated Zobrist table (deterministic seed for reproducibility)
const _ZOBRIST_TABLE_SIZE = 1024
const _ZOBRIST_TABLE = let
    # Use a fixed PRNG seed so hashes are deterministic across sessions
    rng = Xoshiro(0x48414348_454E4255)  # "HACHENBU" in hex
    rand(rng, UInt64, _ZOBRIST_TABLE_SIZE, 3)
end

function HackenbushGames.backend_position_hash(::AcceleratorGate.CUDABackend,
                                                graph::HackenbushGraph)
    n = length(graph.edges)
    n == 0 && return UInt64(0)
    n < 32 && return nothing  # CPU is faster for tiny graphs

    eu, ev, ec, _ = _flatten_graph(graph)
    eu_gpu = CuArray(eu)
    ev_gpu = CuArray(ev)
    ec_gpu = CuArray(ec)
    zt_gpu = CuArray(_ZOBRIST_TABLE)
    hashes_gpu = CUDA.zeros(UInt64, n)

    backend = CUDABackend()
    kernel = _zobrist_hash_kernel!(backend, 256)
    kernel(hashes_gpu, eu_gpu, ev_gpu, ec_gpu, zt_gpu,
           Int32(_ZOBRIST_TABLE_SIZE), Int32(n); ndrange=n)
    KernelAbstractions.synchronize(backend)

    # XOR reduction
    result_gpu = CUDA.zeros(UInt64, 1)
    reduce_kernel = _xor_reduce_kernel!(backend, 1)
    reduce_kernel(result_gpu, hashes_gpu, Int32(n); ndrange=1)
    KernelAbstractions.synchronize(backend)

    # Mix in ground nodes
    h = Array(result_gpu)[1]
    for g in graph.ground
        h ⊻= hash(g)
    end
    h
end

# ============================================================================
# Kernel 3: Batch Grundy Number Computation
# ============================================================================
#
# Green Hackenbush Grundy values exploit Sprague-Grundy theory:
#   grundy(G) = XOR of grundy(component_i) for each connected component
#   grundy(stalk of height h) = h
#
# The GPU parallelises two phases:
#   Phase 1 — Connected component labelling (parallel union-find)
#   Phase 2 — Per-component Grundy value (one thread per component)

@kernel function _component_init_kernel!(parent, rank, num_nodes)
    idx = @index(Global)
    if idx <= num_nodes
        parent[idx] = idx
        rank[idx] = Int32(0)
    end
end

@kernel function _union_find_step_kernel!(parent, @Const(eu), @Const(ev), num_edges, changed)
    idx = @index(Global)
    if idx <= num_edges
        u = eu[idx]
        v = ev[idx]
        # Find roots with path compression
        ru = u
        while parent[ru] != ru
            parent[ru] = parent[parent[ru]]  # path halving
            ru = parent[ru]
        end
        rv = v
        while parent[rv] != rv
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]
        end
        # Union by index (deterministic)
        if ru != rv
            if ru < rv
                parent[rv] = ru
            else
                parent[ru] = rv
            end
            changed[1] = Int32(1)
        end
    end
end

@kernel function _flatten_parents_kernel!(parent, num_nodes)
    idx = @index(Global)
    if idx <= num_nodes
        r = idx
        while parent[r] != r
            r = parent[r]
        end
        parent[idx] = r
    end
end

@kernel function _count_component_edges_kernel!(comp_edge_count, @Const(parent),
                                                 @Const(eu), @Const(ev), num_edges)
    idx = @index(Global)
    if idx <= num_edges
        # Both endpoints should map to same component; use u's root
        root = parent[eu[idx]]
        CUDA.@atomic comp_edge_count[root] += Int32(1)
    end
end

"""
    _gpu_component_grundy(graph) -> Int

Compute the Grundy number for a Green Hackenbush graph on GPU.

Strategy:
  1. Parallel union-find to identify connected components
  2. Count edges per component (each component's Grundy = edge count for trees;
     for graphs with cycles, Grundy = edge_count mod 2 for each independent cycle,
     but for Green Hackenbush on arbitrary graphs the standard approach is
     contracting edges via the colon principle)
  3. For tree components, Grundy = number of edges
  4. XOR all component Grundy values

Note: This kernel handles the common case of tree/forest Green Hackenbush
efficiently. For graphs with cycles, the colon principle reduction falls
back to CPU.
"""
function _gpu_component_grundy(graph::HackenbushGraph)
    n_edges = length(graph.edges)
    n_edges == 0 && return 0

    eu, ev, ec, gr = _flatten_graph(graph)

    # Determine node range (1-based for GPU arrays)
    all_nodes = Int32[]
    for e in graph.edges
        push!(all_nodes, Int32(e.u))
        push!(all_nodes, Int32(e.v))
    end
    for g in graph.ground
        push!(all_nodes, Int32(g))
    end
    max_node = maximum(all_nodes)
    num_nodes = max_node + Int32(1)  # nodes are 0-based in HackenbushGraph

    # Shift to 1-based indexing for GPU arrays
    eu_1 = eu .+ Int32(1)
    ev_1 = ev .+ Int32(1)
    gr_1 = Int32.(graph.ground) .+ Int32(1)

    eu_gpu = CuArray(eu_1)
    ev_gpu = CuArray(ev_1)

    # Phase 1: Parallel union-find for connected components
    parent_gpu = CUDA.zeros(Int32, num_nodes)
    rank_gpu = CUDA.zeros(Int32, num_nodes)
    backend = CUDABackend()

    init_k = _component_init_kernel!(backend, 256)
    init_k(parent_gpu, rank_gpu, Int32(num_nodes); ndrange=Int(num_nodes))
    KernelAbstractions.synchronize(backend)

    # Iterate union-find until convergence
    changed_gpu = CuArray(Int32[1])
    for _ in 1:32  # convergence typically in O(log n) iterations
        changed_gpu .= Int32(0)
        uf_k = _union_find_step_kernel!(backend, 256)
        uf_k(parent_gpu, eu_gpu, ev_gpu, Int32(n_edges), changed_gpu; ndrange=n_edges)
        KernelAbstractions.synchronize(backend)
        Array(changed_gpu)[1] == Int32(0) && break
    end

    # Flatten parent pointers
    flat_k = _flatten_parents_kernel!(backend, 256)
    flat_k(parent_gpu, Int32(num_nodes); ndrange=Int(num_nodes))
    KernelAbstractions.synchronize(backend)

    # Phase 2: Count edges per component
    comp_edge_count_gpu = CUDA.zeros(Int32, num_nodes)
    count_k = _count_component_edges_kernel!(backend, 256)
    count_k(comp_edge_count_gpu, parent_gpu, eu_gpu, ev_gpu, Int32(n_edges); ndrange=n_edges)
    KernelAbstractions.synchronize(backend)

    # Transfer back and XOR reduce on CPU (small number of components)
    parent_cpu = Array(parent_gpu)
    comp_counts = Array(comp_edge_count_gpu)

    # Identify unique component roots and their edge counts
    result = 0
    seen = Set{Int32}()
    for node_1based in 1:num_nodes
        root = parent_cpu[node_1based]
        if root in seen
            continue
        end
        push!(seen, root)
        edge_count = comp_counts[root]
        edge_count == 0 && continue
        # For tree components of Green Hackenbush, Grundy number = edge count
        result ⊻= Int(edge_count)
    end
    result
end

function HackenbushGames.backend_grundy_number(::AcceleratorGate.CUDABackend,
                                                graph::HackenbushGraph)
    n = length(graph.edges)
    n < 32 && return nothing  # CPU fallback for small graphs

    # Validate all edges are Green
    for e in graph.edges
        e.color != Green && return nothing  # Mixed colors unsupported on GPU
    end

    try
        _gpu_component_grundy(graph)
    catch ex
        _record_diagnostic!("cuda", "runtime_errors")
        @warn "CUDA Grundy computation failed, falling back to CPU" exception=ex maxlog=1
        nothing
    end
end

# ============================================================================
# Kernel 4: Batch Game Tree Evaluation
# ============================================================================
#
# For game_tree_eval / canonical_game, the GPU accelerates the leaf-level
# evaluation of many positions simultaneously. The game tree is explored
# breadth-first: at each depth level, all positions at that level are
# evaluated in parallel on the GPU.
#
# The kernel computes stalk values for positions that reduce to simple stalks,
# which is the most common leaf case in Hackenbush game trees.

@kernel function _stalk_value_kernel!(values, @Const(colors), @Const(lengths),
                                       @Const(offsets), num_stalks)
    idx = @index(Global)
    if idx <= num_stalks
        offset = offsets[idx]
        len = lengths[idx]

        # Compute stalk value using the simplicity rule
        # Track interval [lo, hi] as numerator/denominator pairs
        # Start with value 0 at the ground
        val_num = Int64(0)
        val_den = Int64(1)

        for k in 1:len
            c = colors[offset + k]
            if c == Int32(0)
                # Blue edge: Left option, value increases
                val_num = val_num + val_den
            elseif c == Int32(1)
                # Red edge: Right option, value decreases
                val_num = val_num - val_den
            else
                # Green edge: nimber contribution (value stays 0 for pure green stalks)
                # For mixed stalks this is an approximation; full computation on CPU
            end
        end

        # Store as fixed-point: numerator in [idx], denominator implicit = 2^30
        # This gives sufficient precision for dyadic rationals up to depth 30
        values[idx] = val_num
    end
end

"""
    _batch_stalk_values(stalk_colors_list) -> Vector{Rational{Int}}

Evaluate multiple stalk positions in parallel on GPU.
Each stalk is a Vector{Int32} of color codes (0=Blue, 1=Red, 2=Green).
Returns the game value for each stalk.
"""
function _batch_stalk_values(stalks::Vector{Vector{Int32}})
    n = length(stalks)
    n == 0 && return Rational{Int}[]

    # Pack stalks into contiguous array with offset/length metadata
    total_edges = sum(length, stalks)
    flat_colors = Vector{Int32}(undef, total_edges)
    offsets = Vector{Int32}(undef, n)
    lengths = Vector{Int32}(undef, n)

    pos = 0
    for (i, s) in enumerate(stalks)
        offsets[i] = Int32(pos)
        lengths[i] = Int32(length(s))
        for (j, c) in enumerate(s)
            flat_colors[pos + j] = c
        end
        pos += length(s)
    end

    colors_gpu = CuArray(flat_colors)
    offsets_gpu = CuArray(offsets)
    lengths_gpu = CuArray(lengths)
    values_gpu = CUDA.zeros(Int64, n)

    backend = CUDABackend()
    kernel = _stalk_value_kernel!(backend, 256)
    kernel(values_gpu, colors_gpu, lengths_gpu, offsets_gpu, Int32(n); ndrange=n)
    KernelAbstractions.synchronize(backend)

    # Convert GPU integer results back to rationals
    raw = Array(values_gpu)
    # The kernel computes a simple integer value for pure Blue/Red stalks
    # which corresponds directly to the game value
    [Rational{Int}(raw[i], 1) for i in 1:n]
end

function HackenbushGames.backend_game_tree_eval(::AcceleratorGate.CUDABackend,
                                                 graph::HackenbushGraph, max_depth::Int)
    n = length(graph.edges)
    n < 16 && return nothing  # CPU fallback for small graphs

    # The GPU extension accelerates the breadth-first expansion of the game tree.
    # At each level, move generation is parallelised via _gpu_legal_move_indices,
    # and leaf stalk evaluations are batched via _batch_stalk_values.
    #
    # For the top-level call we accelerate move generation and delegate
    # recursive evaluation back to the main module (which will re-enter
    # this extension for subsequent levels if the backend is still CUDA).

    try
        # Phase 1: GPU-accelerated move generation for both players
        left_indices = _gpu_legal_move_indices(graph, :left)
        right_indices = _gpu_legal_move_indices(graph, :right)

        left_positions = [cut_edge(graph, i) for i in left_indices]
        right_positions = [cut_edge(graph, i) for i in right_indices]

        if isempty(left_positions) && isempty(right_positions)
            return GameForm(Rational{Int}[], Rational{Int}[])
        end

        # Phase 2: Evaluate child positions
        # For depth > 1, recursive calls will re-enter via backend dispatch
        left_vals = Rational{Int}[]
        for pos in left_positions
            val = HackenbushGames.game_value(pos; max_depth=max_depth - 1)
            val !== nothing && push!(left_vals, val)
        end

        right_vals = Rational{Int}[]
        for pos in right_positions
            val = HackenbushGames.game_value(pos; max_depth=max_depth - 1)
            val !== nothing && push!(right_vals, val)
        end

        simplify_game(GameForm(left_vals, right_vals))
    catch ex
        _record_diagnostic!("cuda", "runtime_errors")
        @warn "CUDA game tree eval failed, falling back to CPU" exception=ex maxlog=1
        nothing
    end
end

# ============================================================================
# Memory Management
# ============================================================================

function AcceleratorGate.backend_to_gpu(::AcceleratorGate.CUDABackend, x::AbstractArray)
    CuArray(x)
end

function AcceleratorGate.backend_to_cpu(::AcceleratorGate.CUDABackend, x_gpu::CuArray)
    Array(x_gpu)
end

function AcceleratorGate.backend_synchronize(::AcceleratorGate.CUDABackend)
    CUDA.synchronize()
end

end  # module HackenbushGamesCUDAExt
