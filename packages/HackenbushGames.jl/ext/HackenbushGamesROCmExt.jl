# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# HackenbushGames.jl ROCm Extension
# GPU-accelerated game tree evaluation, Grundy numbers, move generation,
# and position hashing via AMD ROCm + KernelAbstractions.jl.
# Automatically loaded when both AMDGPU.jl and KernelAbstractions.jl are imported.

module HackenbushGamesROCmExt

using AMDGPU
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

function AcceleratorGate.rocm_available()
    forced = _backend_env_available("AXIOM_ROCM_AVAILABLE")
    forced !== nothing && return forced
    AMDGPU.functional()
end

function AcceleratorGate.rocm_device_count()
    forced = _backend_env_count("AXIOM_ROCM_AVAILABLE", "AXIOM_ROCM_DEVICE_COUNT")
    forced !== nothing && return forced
    AMDGPU.functional() ? length(AMDGPU.devices()) : 0
end

# ============================================================================
# GPU Data Layout
# ============================================================================

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

# ============================================================================
# Kernel 1: Move Generation (edge legality filter)
# ============================================================================

@kernel function _move_legality_kernel!(mask, @Const(colors), player_code, num_edges)
    idx = @index(Global)
    if idx <= num_edges
        c = colors[idx]
        if c == Int32(2)
            mask[idx] = Int32(1)
        elseif player_code == Int32(0) && c == Int32(0)
            mask[idx] = Int32(1)
        elseif player_code == Int32(1) && c == Int32(1)
            mask[idx] = Int32(1)
        else
            mask[idx] = Int32(0)
        end
    end
end

"""
    _gpu_legal_move_indices(graph, player) -> Vector{Int}

Compute legal move indices on ROCm GPU using a parallel edge filter kernel.
"""
function _gpu_legal_move_indices(graph::HackenbushGraph, player::Symbol)
    n = length(graph.edges)
    n == 0 && return Int[]

    _, _, ec, _ = _flatten_graph(graph)
    colors_gpu = ROCArray(ec)
    mask_gpu = AMDGPU.zeros(Int32, n)
    player_code = player == :left ? Int32(0) : Int32(1)

    backend = ROCBackend()
    kernel = _move_legality_kernel!(backend, 256)
    kernel(mask_gpu, colors_gpu, player_code, Int32(n); ndrange=n)
    KernelAbstractions.synchronize(backend)

    mask_cpu = Array(mask_gpu)
    findall(==(Int32(1)), mask_cpu)
end

function HackenbushGames.backend_move_gen(::AcceleratorGate.ROCmBackend,
                                          graph::HackenbushGraph, player::Symbol)
    n = length(graph.edges)
    n < 64 && return nothing

    indices = try
        _gpu_legal_move_indices(graph, player)
    catch ex
        _record_diagnostic!("rocm", "runtime_errors")
        @warn "ROCm move generation failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end

    options = HackenbushGraph[]
    for i in indices
        push!(options, cut_edge(graph, i))
    end
    options
end

# ============================================================================
# Kernel 2: Position Hashing (Zobrist-style)
# ============================================================================

@kernel function _zobrist_hash_kernel!(hashes, @Const(eu), @Const(ev), @Const(ec),
                                       @Const(zobrist_table), table_stride, num_edges)
    idx = @index(Global)
    if idx <= num_edges
        u = eu[idx]
        v = ev[idx]
        c = ec[idx]
        u_idx = (u % table_stride) + Int32(1)
        v_idx = (v % table_stride) + Int32(1)
        h = zobrist_table[u_idx, c + Int32(1)] ⊻
            zobrist_table[v_idx, c + Int32(1)]
        hashes[idx] = h
    end
end

const _ZOBRIST_TABLE_SIZE = 1024
const _ZOBRIST_TABLE = let
    rng = Xoshiro(0x48414348_454E4255)
    rand(rng, UInt64, _ZOBRIST_TABLE_SIZE, 3)
end

function HackenbushGames.backend_position_hash(::AcceleratorGate.ROCmBackend,
                                                graph::HackenbushGraph)
    n = length(graph.edges)
    n == 0 && return UInt64(0)
    n < 32 && return nothing

    eu, ev, ec, _ = _flatten_graph(graph)
    eu_gpu = ROCArray(eu)
    ev_gpu = ROCArray(ev)
    ec_gpu = ROCArray(ec)
    zt_gpu = ROCArray(_ZOBRIST_TABLE)
    hashes_gpu = AMDGPU.zeros(UInt64, n)

    backend = ROCBackend()
    kernel = _zobrist_hash_kernel!(backend, 256)
    kernel(hashes_gpu, eu_gpu, ev_gpu, ec_gpu, zt_gpu,
           Int32(_ZOBRIST_TABLE_SIZE), Int32(n); ndrange=n)
    KernelAbstractions.synchronize(backend)

    hashes_cpu = Array(hashes_gpu)
    h = UInt64(0)
    for x in hashes_cpu
        h ⊻= x
    end
    for g in graph.ground
        h ⊻= hash(g)
    end
    h
end

# ============================================================================
# Kernel 3: Batch Grundy Number Computation
# ============================================================================

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
        ru = u
        while parent[ru] != ru
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        end
        rv = v
        while parent[rv] != rv
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]
        end
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
                                                 @Const(eu), num_edges)
    idx = @index(Global)
    if idx <= num_edges
        root = parent[eu[idx]]
        # ROCm supports atomics; use @atomic for correctness with cycles
        AMDGPU.@atomic comp_edge_count[root] += Int32(1)
    end
end

function _gpu_component_grundy(graph::HackenbushGraph)
    n_edges = length(graph.edges)
    n_edges == 0 && return 0

    eu, ev, ec, gr = _flatten_graph(graph)

    all_nodes = Int32[]
    for e in graph.edges
        push!(all_nodes, Int32(e.u))
        push!(all_nodes, Int32(e.v))
    end
    for g in graph.ground
        push!(all_nodes, Int32(g))
    end
    max_node = maximum(all_nodes)
    num_nodes = max_node + Int32(1)

    eu_1 = eu .+ Int32(1)
    ev_1 = ev .+ Int32(1)

    eu_gpu = ROCArray(eu_1)
    ev_gpu = ROCArray(ev_1)

    parent_gpu = AMDGPU.zeros(Int32, Int(num_nodes))
    rank_gpu = AMDGPU.zeros(Int32, Int(num_nodes))
    backend = ROCBackend()

    init_k = _component_init_kernel!(backend, 256)
    init_k(parent_gpu, rank_gpu, Int32(num_nodes); ndrange=Int(num_nodes))
    KernelAbstractions.synchronize(backend)

    changed_gpu = ROCArray(Int32[1])
    for _ in 1:32
        changed_gpu .= Int32(0)
        uf_k = _union_find_step_kernel!(backend, 256)
        uf_k(parent_gpu, eu_gpu, ev_gpu, Int32(n_edges), changed_gpu; ndrange=n_edges)
        KernelAbstractions.synchronize(backend)
        Array(changed_gpu)[1] == Int32(0) && break
    end

    flat_k = _flatten_parents_kernel!(backend, 256)
    flat_k(parent_gpu, Int32(num_nodes); ndrange=Int(num_nodes))
    KernelAbstractions.synchronize(backend)

    comp_edge_count_gpu = AMDGPU.zeros(Int32, Int(num_nodes))
    count_k = _count_component_edges_kernel!(backend, 256)
    count_k(comp_edge_count_gpu, parent_gpu, eu_gpu, Int32(n_edges); ndrange=n_edges)
    KernelAbstractions.synchronize(backend)

    parent_cpu = Array(parent_gpu)
    comp_counts = Array(comp_edge_count_gpu)

    result = 0
    seen = Set{Int32}()
    for node_1based in 1:Int(num_nodes)
        root = parent_cpu[node_1based]
        root in seen && continue
        push!(seen, root)
        edge_count = comp_counts[root]
        edge_count == 0 && continue
        result ⊻= Int(edge_count)
    end
    result
end

function HackenbushGames.backend_grundy_number(::AcceleratorGate.ROCmBackend,
                                                graph::HackenbushGraph)
    n = length(graph.edges)
    n < 32 && return nothing

    for e in graph.edges
        e.color != Green && return nothing
    end

    try
        _gpu_component_grundy(graph)
    catch ex
        _record_diagnostic!("rocm", "runtime_errors")
        @warn "ROCm Grundy computation failed, falling back to CPU" exception=ex maxlog=1
        nothing
    end
end

# ============================================================================
# Kernel 4: Batch Game Tree Evaluation
# ============================================================================

@kernel function _stalk_value_kernel!(values, @Const(colors), @Const(lengths),
                                       @Const(offsets), num_stalks)
    idx = @index(Global)
    if idx <= num_stalks
        offset = offsets[idx]
        len = lengths[idx]
        val_num = Int64(0)
        val_den = Int64(1)
        for k in 1:len
            c = colors[offset + k]
            if c == Int32(0)
                val_num = val_num + val_den
            elseif c == Int32(1)
                val_num = val_num - val_den
            end
        end
        values[idx] = val_num
    end
end

function HackenbushGames.backend_game_tree_eval(::AcceleratorGate.ROCmBackend,
                                                 graph::HackenbushGraph, max_depth::Int)
    n = length(graph.edges)
    n < 16 && return nothing

    try
        left_indices = _gpu_legal_move_indices(graph, :left)
        right_indices = _gpu_legal_move_indices(graph, :right)

        left_positions = [cut_edge(graph, i) for i in left_indices]
        right_positions = [cut_edge(graph, i) for i in right_indices]

        if isempty(left_positions) && isempty(right_positions)
            return GameForm(Rational{Int}[], Rational{Int}[])
        end

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
        _record_diagnostic!("rocm", "runtime_errors")
        @warn "ROCm game tree eval failed, falling back to CPU" exception=ex maxlog=1
        nothing
    end
end

# ============================================================================
# Memory Management
# ============================================================================

function AcceleratorGate.backend_to_gpu(::AcceleratorGate.ROCmBackend, x::AbstractArray)
    ROCArray(x)
end

function AcceleratorGate.backend_to_cpu(::AcceleratorGate.ROCmBackend, x_gpu::ROCArray)
    Array(x_gpu)
end

function AcceleratorGate.backend_synchronize(::AcceleratorGate.ROCmBackend)
    AMDGPU.synchronize()
end

end  # module HackenbushGamesROCmExt
