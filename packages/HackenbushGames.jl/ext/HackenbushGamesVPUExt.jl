# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# HackenbushGamesVPUExt — SIMD vector processing acceleration for HackenbushGames.jl
# Exploits VPU SIMD instructions for vectorized move generation, batch
# position evaluation, and SIMD-parallel Grundy number computation.

module HackenbushGamesVPUExt

using HackenbushGames
using HackenbushGames: Edge, EdgeColor, HackenbushGraph, GameForm,
    Blue, Red, Green, _graph_key, _edge_allowed,
    mex, nim_sum, simplest_dyadic_between, simplify_game,
    cut_edge, prune_disconnected, moves
using AcceleratorGate
using AcceleratorGate: VPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(VPUBackend, :move_gen)
    register_operation!(VPUBackend, :position_hash)
    register_operation!(VPUBackend, :grundy_number)
end

# ============================================================================
# SIMD-Vectorized Move Generation
# ============================================================================
#
# VPU key insight: edge legality checking is a data-parallel filter operation.
# Pack edge colors into a contiguous Int32 array and use SIMD comparison
# to produce a bitmask of legal moves in bulk.

"""
    _simd_legal_mask(colors::Vector{Int32}, player::Symbol) -> Vector{Bool}

Compute legal move mask using SIMD-vectorized comparison.
Each element tests whether the edge color is legal for the given player.
"""
function _simd_legal_mask(colors::Vector{Int32}, player::Symbol)
    n = length(colors)
    mask = Vector{Bool}(undef, n)

    if player == :left
        # Left can cut Blue (0) and Green (2)
        @inbounds @simd for i in 1:n
            c = colors[i]
            mask[i] = (c == Int32(0)) | (c == Int32(2))
        end
    else
        # Right can cut Red (1) and Green (2)
        @inbounds @simd for i in 1:n
            c = colors[i]
            mask[i] = (c == Int32(1)) | (c == Int32(2))
        end
    end

    return mask
end

"""
    HackenbushGames.backend_coprocessor_move_gen(::VPUBackend, graph, player)

VPU-accelerated move generation using SIMD-vectorized edge filtering.
Packs edge colors into a contiguous array and uses @simd comparison
to determine legality of all edges simultaneously.
"""
function HackenbushGames.backend_coprocessor_move_gen(b::VPUBackend,
                                                       graph::HackenbushGraph,
                                                       player::Symbol)
    n = length(graph.edges)
    n < 32 && return nothing

    try
        # Pack colors for SIMD processing
        colors = Vector{Int32}(undef, n)
        @inbounds for i in 1:n
            colors[i] = Int32(Int(graph.edges[i].color))
        end

        mask = _simd_legal_mask(colors, player)
        indices = findall(mask)

        result = HackenbushGraph[]
        for i in indices
            push!(result, cut_edge(graph, i))
        end
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "VPU move gen failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# SIMD-Vectorized Position Hashing
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_position_hash(::VPUBackend, graph)

VPU-accelerated Zobrist-style position hashing using SIMD-vectorized
XOR operations. Pre-computed hash contributions for each edge are
XOR-reduced using vector instructions.
"""
function HackenbushGames.backend_coprocessor_position_hash(b::VPUBackend,
                                                             graph::HackenbushGraph)
    n = length(graph.edges)
    n == 0 && return UInt64(0)
    n < 16 && return nothing

    try
        # Pre-compute per-edge hash contributions
        contributions = Vector{UInt64}(undef, n)
        @inbounds for i in 1:n
            e = graph.edges[i]
            # Zobrist-style: combine node indices and color
            h = UInt64(e.u) * UInt64(0x9E3779B97F4A7C15)
            h = xor(h, UInt64(e.v) * UInt64(0xBF58476D1CE4E5B9))
            h = xor(h, UInt64(Int(e.color)) * UInt64(0x94D049BB133111EB))
            contributions[i] = h
        end

        # SIMD-vectorized XOR reduction
        result = UInt64(0)
        @inbounds @simd for i in 1:n
            result = xor(result, contributions[i])
        end

        # Mix in ground nodes
        for g in graph.ground
            result = xor(result, hash(g))
        end

        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "VPU position hash failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# SIMD-Vectorized Grundy Number
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_grundy_number(::VPUBackend, graph)

VPU-accelerated Grundy number computation for Green Hackenbush.
Uses SIMD-parallel union-find edge processing and vectorized XOR reduction.
"""
function HackenbushGames.backend_coprocessor_grundy_number(b::VPUBackend,
                                                             graph::HackenbushGraph)
    n = length(graph.edges)
    n < 16 && return nothing

    for e in graph.edges
        e.color != Green && return nothing
    end

    try
        # Build node set
        all_nodes = Set{Int}()
        for e in graph.edges
            push!(all_nodes, e.u)
            push!(all_nodes, e.v)
        end
        for g in graph.ground
            push!(all_nodes, g)
        end

        # Union-find with SIMD-friendly flat arrays
        node_list = sort(collect(all_nodes))
        node_idx = Dict(v => i for (i, v) in enumerate(node_list))
        nn = length(node_list)

        parent = collect(1:nn)
        rank_arr = zeros(Int, nn)

        function _find(v)
            while parent[v] != v
                parent[v] = parent[parent[v]]
                v = parent[v]
            end
            v
        end

        # Process edges (packed for SIMD)
        eu = Vector{Int}(undef, n)
        ev = Vector{Int}(undef, n)
        @inbounds for i in 1:n
            eu[i] = node_idx[graph.edges[i].u]
            ev[i] = node_idx[graph.edges[i].v]
        end

        for i in 1:n
            ru = _find(eu[i])
            rv = _find(ev[i])
            if ru != rv
                if rank_arr[ru] < rank_arr[rv]
                    parent[ru] = rv
                elseif rank_arr[ru] > rank_arr[rv]
                    parent[rv] = ru
                else
                    parent[rv] = ru
                    rank_arr[ru] += 1
                end
            end
        end

        # Count edges per component (SIMD-friendly accumulation)
        comp_edges = zeros(Int, nn)
        @inbounds for i in 1:n
            root = _find(eu[i])
            comp_edges[root] += 1
        end

        # SIMD-vectorized XOR reduction
        result = 0
        @inbounds @simd for i in 1:nn
            if comp_edges[i] > 0
                result = xor(result, comp_edges[i])
            end
        end

        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "VPU Grundy computation failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Remaining Hooks
# ============================================================================

function HackenbushGames.backend_coprocessor_game_tree_eval(b::VPUBackend,
                                                              graph::HackenbushGraph,
                                                              max_depth::Int)
    return nothing
end

function HackenbushGames.backend_coprocessor_minimax_search(b::VPUBackend, args...)
    return nothing
end

end # module HackenbushGamesVPUExt
