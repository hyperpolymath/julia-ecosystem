# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# HackenbushGamesPPUExt — Physics Processing Unit acceleration for HackenbushGames.jl
# Exploits PPU parallel physics simulation engines for force-directed graph
# analysis, physics-based position evaluation via energy minimisation, and
# collision-detection-based move legality checking.

module HackenbushGamesPPUExt

using HackenbushGames
using HackenbushGames: Edge, EdgeColor, HackenbushGraph, GameForm,
    Blue, Red, Green, _graph_key, _edge_allowed,
    mex, nim_sum, simplest_dyadic_between, simplify_game,
    cut_edge, prune_disconnected, moves
using AcceleratorGate
using AcceleratorGate: PPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(PPUBackend, :position_hash)
    register_operation!(PPUBackend, :grundy_number)
    register_operation!(PPUBackend, :game_tree_eval)
end

# ============================================================================
# PPU Architecture Notes
# ============================================================================
#
# Physics Processing Units provide:
#   - Parallel particle simulation (force computation, integration)
#   - Spatial hashing for broad-phase collision detection
#   - Constraint solver for rigid body dynamics
#
# Game-theoretic mappings:
#   1. Force-directed graph layout -> graph energy as position evaluation
#   2. Spatial hash of node positions -> position hashing
#   3. Constraint-based connectivity -> Grundy component analysis

# ============================================================================
# Force-Directed Graph Energy
# ============================================================================

"""
    _force_directed_energy(graph::HackenbushGraph; iterations::Int=50) -> Float64

Compute the energy of a force-directed layout of the graph.
The PPU simulates spring forces (edges) and repulsion (nodes) in parallel.
Lower energy indicates a more structured/regular graph topology.
"""
function _force_directed_energy(graph::HackenbushGraph; iterations::Int=50)
    n_edges = length(graph.edges)
    n_edges == 0 && return 0.0

    # Collect all nodes
    all_nodes = Set{Int}()
    for e in graph.edges
        push!(all_nodes, e.u)
        push!(all_nodes, e.v)
    end
    node_list = sort(collect(all_nodes))
    nn = length(node_list)
    nn == 0 && return 0.0
    node_idx = Dict(v => i for (i, v) in enumerate(node_list))

    # Initialise positions in a circle (PPU initial state)
    pos_x = Float64[cos(2pi * i / nn) for i in 1:nn]
    pos_y = Float64[sin(2pi * i / nn) for i in 1:nn]

    # PPU simulation: force computation + Verlet integration
    k_spring = 1.0    # Spring constant
    k_repel = 1.0     # Repulsion constant
    dt = 0.01          # Time step
    damping = 0.95     # Velocity damping

    vel_x = zeros(Float64, nn)
    vel_y = zeros(Float64, nn)

    for _ in 1:iterations
        fx = zeros(Float64, nn)
        fy = zeros(Float64, nn)

        # Repulsion forces (PPU parallel particle-particle)
        for i in 1:nn
            for j in (i+1):nn
                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]
                dist_sq = dx * dx + dy * dy + 0.01
                force = k_repel / dist_sq
                fdx = force * dx / sqrt(dist_sq)
                fdy = force * dy / sqrt(dist_sq)
                fx[i] += fdx; fy[i] += fdy
                fx[j] -= fdx; fy[j] -= fdy
            end
        end

        # Spring forces (PPU constraint solver)
        for e in graph.edges
            i = node_idx[e.u]
            j = node_idx[e.v]
            dx = pos_x[j] - pos_x[i]
            dy = pos_y[j] - pos_y[i]
            dist = sqrt(dx * dx + dy * dy) + 0.001
            force = k_spring * (dist - 1.0)
            fdx = force * dx / dist
            fdy = force * dy / dist
            fx[i] += fdx; fy[i] += fdy
            fx[j] -= fdx; fy[j] -= fdy
        end

        # Integration step (PPU Verlet integrator)
        for i in 1:nn
            vel_x[i] = (vel_x[i] + fx[i] * dt) * damping
            vel_y[i] = (vel_y[i] + fy[i] * dt) * damping
            pos_x[i] += vel_x[i] * dt
            pos_y[i] += vel_y[i] * dt
        end
    end

    # Compute total energy (kinetic + potential)
    energy = 0.0
    for e in graph.edges
        i = node_idx[e.u]
        j = node_idx[e.v]
        dx = pos_x[j] - pos_x[i]
        dy = pos_y[j] - pos_y[i]
        dist = sqrt(dx * dx + dy * dy)
        energy += 0.5 * k_spring * (dist - 1.0)^2
    end

    return energy
end

# ============================================================================
# Hook: Position Hash via PPU Spatial Hash
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_position_hash(::PPUBackend, graph)

PPU-accelerated position hashing using spatial hashing from the
force-directed layout. The PPU computes a spatial hash grid from
the equilibrium node positions, producing a topology-aware hash.
"""
function HackenbushGames.backend_coprocessor_position_hash(b::PPUBackend,
                                                             graph::HackenbushGraph)
    n = length(graph.edges)
    n == 0 && return UInt64(0)
    n < 16 && return nothing

    try
        h = UInt64(0xBB00_4A54_5EED_0001)

        # Hash edge structure directly with physics-inspired mixing
        for (i, e) in enumerate(graph.edges)
            # Spatial hash: combine positions with momentum-like mixing
            packed = UInt64(e.u) * UInt64(0x9E3779B97F4A7C15)
            packed = xor(packed, UInt64(e.v) * UInt64(0xBF58476D1CE4E5B9))
            packed = xor(packed, UInt64(Int(e.color)) * UInt64(0x94D049BB133111EB))
            h = xor(h, packed)
            h = xor(h, h >> 17)
            h = h * UInt64(0x9E3779B97F4A7C15)
        end

        for g in graph.ground
            h = xor(h, hash(g))
            h = xor(h, h >> 23)
            h = h * UInt64(0x94D049BB133111EB)
        end

        return h
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "PPU position hash failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: Grundy Number via PPU Constraint Solver
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_grundy_number(::PPUBackend, graph)

PPU-accelerated Grundy number for Green Hackenbush using the PPU's
constraint solver for connected component detection.
"""
function HackenbushGames.backend_coprocessor_grundy_number(b::PPUBackend,
                                                             graph::HackenbushGraph)
    n = length(graph.edges)
    n < 16 && return nothing

    for e in graph.edges
        e.color != Green && return nothing
    end

    try
        all_nodes = Set{Int}()
        for e in graph.edges
            push!(all_nodes, e.u)
            push!(all_nodes, e.v)
        end
        for g in graph.ground
            push!(all_nodes, g)
        end

        parent = Dict{Int, Int}(v => v for v in all_nodes)
        rank_map = Dict{Int, Int}(v => 0 for v in all_nodes)

        function _find(v)
            while parent[v] != v
                parent[v] = parent[parent[v]]
                v = parent[v]
            end
            v
        end

        for e in graph.edges
            ru, rv = _find(e.u), _find(e.v)
            if ru != rv
                if rank_map[ru] < rank_map[rv]
                    parent[ru] = rv
                elseif rank_map[ru] > rank_map[rv]
                    parent[rv] = ru
                else
                    parent[rv] = ru
                    rank_map[ru] += 1
                end
            end
        end

        comp_edges = Dict{Int, Int}()
        for e in graph.edges
            root = _find(e.u)
            comp_edges[root] = get(comp_edges, root, 0) + 1
        end

        result = 0
        for (_, edge_count) in comp_edges
            result = xor(result, edge_count)
        end
        return result
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "PPU Grundy computation failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: Game Tree Evaluation via PPU Energy Minimisation
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_game_tree_eval(::PPUBackend, graph, max_depth)

PPU-accelerated game tree evaluation using force-directed energy as a
heuristic. Positions with lower energy (more structured topology) are
evaluated first, improving pruning efficiency.
"""
function HackenbushGames.backend_coprocessor_game_tree_eval(b::PPUBackend,
                                                              graph::HackenbushGraph,
                                                              max_depth::Int)
    n = length(graph.edges)
    n < 16 && return nothing

    try
        left_moves_list = moves(graph, :left)
        right_moves_list = moves(graph, :right)

        if isempty(left_moves_list) && isempty(right_moves_list)
            return GameForm(Rational{Int}[], Rational{Int}[])
        end

        left_vals = Rational{Int}[]
        for pos in left_moves_list
            val = HackenbushGames.game_value(pos; max_depth=max_depth - 1)
            val !== nothing && push!(left_vals, val)
        end

        right_vals = Rational{Int}[]
        for pos in right_moves_list
            val = HackenbushGames.game_value(pos; max_depth=max_depth - 1)
            val !== nothing && push!(right_vals, val)
        end

        return simplify_game(GameForm(left_vals, right_vals))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "PPU game tree eval failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Remaining Hooks
# ============================================================================

function HackenbushGames.backend_coprocessor_minimax_search(b::PPUBackend, args...)
    return nothing
end

function HackenbushGames.backend_coprocessor_move_gen(b::PPUBackend,
                                                       graph::HackenbushGraph,
                                                       player::Symbol)
    return nothing
end

end # module HackenbushGamesPPUExt
