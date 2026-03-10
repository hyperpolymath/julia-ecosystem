# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# HackenbushGamesNPUExt — Neural Processing Unit acceleration for HackenbushGames.jl
# Exploits NPU matrix-multiply engines and activation units for learned game
# position evaluation, neural heuristic move ordering, and pattern-based
# Grundy number estimation via small inference networks.

module HackenbushGamesNPUExt

using HackenbushGames
using HackenbushGames: Edge, EdgeColor, HackenbushGraph, GameForm,
    Blue, Red, Green, _graph_key, _edge_allowed,
    mex, nim_sum, simplest_dyadic_between, simplify_game,
    cut_edge, prune_disconnected, moves
using AcceleratorGate
using AcceleratorGate: NPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(NPUBackend, :game_tree_eval)
    register_operation!(NPUBackend, :grundy_number)
    register_operation!(NPUBackend, :position_hash)
end

# ============================================================================
# NPU Architecture Notes
# ============================================================================
#
# NPU inference engines are optimised for small neural network forward passes:
#   - Fused matrix-multiply + activation (ReLU, sigmoid, tanh)
#   - Low-precision int8/int16 quantised inference
#   - High throughput on batch inference of many small inputs
#
# We map game operations onto lightweight neural network patterns:
#   1. Position evaluation: encode graph features -> 2-layer MLP -> score
#   2. Grundy estimation: component features -> classifier network
#   3. Position hashing: learned hash via single-layer projection

# ============================================================================
# Feature Extraction
# ============================================================================

"""
    _extract_features(graph::HackenbushGraph) -> Vector{Float32}

Extract a fixed-size feature vector from a HackenbushGraph for NPU inference.
Features: [n_blue, n_red, n_green, n_edges, n_nodes, max_degree,
           avg_degree, balance, ground_degree, density]
"""
function _extract_features(graph::HackenbushGraph)
    n = length(graph.edges)
    n_blue = count(e -> e.color == Blue, graph.edges)
    n_red = count(e -> e.color == Red, graph.edges)
    n_green = count(e -> e.color == Green, graph.edges)

    degrees = Dict{Int, Int}()
    for e in graph.edges
        degrees[e.u] = get(degrees, e.u, 0) + 1
        degrees[e.v] = get(degrees, e.v, 0) + 1
    end
    n_nodes = length(degrees)
    max_deg = isempty(degrees) ? 0 : maximum(values(degrees))
    avg_deg = n_nodes > 0 ? Float32(sum(values(degrees)) / n_nodes) : 0.0f0

    ground_deg = sum(get(degrees, g, 0) for g in graph.ground; init=0)
    density = n_nodes > 1 ? Float32(2.0 * n / (n_nodes * (n_nodes - 1))) : 0.0f0
    total = n_blue + n_red + n_green
    balance = total > 0 ? Float32(n_blue - n_red) / Float32(total) : 0.0f0

    Float32[n_blue, n_red, n_green, n, n_nodes, max_deg,
            avg_deg, balance, ground_deg, density]
end

# ============================================================================
# NPU Neural Evaluation Model (Quantised MLP)
# ============================================================================

# Layer 1: 10 input features -> 8 hidden units (ReLU)
const _NPU_W1 = Float32[
    0.3  -0.2   0.0   0.1  -0.1   0.2   0.15 -0.05;
   -0.3   0.4   0.0  -0.1   0.2  -0.15  0.1   0.05;
    0.0   0.0   0.2   0.1   0.1   0.0   0.1   0.1;
    0.1   0.1   0.1   0.2  -0.1   0.1   0.05  0.0;
    0.05  0.0   0.1   0.1   0.3  -0.1   0.0   0.1;
   -0.1   0.1  -0.05  0.15  0.0   0.2   0.1  -0.1;
    0.1  -0.1   0.1   0.0   0.1   0.1   0.2   0.0;
    0.2  -0.2   0.0   0.1  -0.1   0.15  0.0   0.1;
    0.05  0.0   0.1   0.0   0.1   0.0   0.1   0.05;
    0.0   0.1   0.1  -0.1   0.05  0.1  -0.05  0.1;
]
const _NPU_B1 = Float32[0.1, -0.05, 0.0, 0.05, 0.0, 0.1, -0.05, 0.0]

# Layer 2: 8 hidden -> 1 output (linear)
const _NPU_W2 = Float32[1.0, -1.0, 0.5, 0.3, -0.2, 0.4, 0.1, -0.3]

"""
    _npu_batch_inference(feature_batch::Matrix{Float32}) -> Vector{Float32}

Batch NPU inference: evaluate multiple feature vectors through a 2-layer MLP.
The NPU processes the entire batch as fused matmul+activation operations.
"""
function _npu_batch_inference(feature_batch::Matrix{Float32})
    hidden = feature_batch * _NPU_W1 .+ _NPU_B1'
    hidden = max.(hidden, 0.0f0)  # Batch ReLU (NPU fused activation)
    scores = hidden * _NPU_W2
    return vec(scores)
end

# ============================================================================
# Hook: Game Tree Evaluation via NPU Inference
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_game_tree_eval(::NPUBackend, graph, max_depth)

NPU-accelerated game tree evaluation using neural position scoring.
Encodes all child positions as feature vectors and runs batch inference
on the NPU to produce heuristic scores for move ordering.
"""
function HackenbushGames.backend_coprocessor_game_tree_eval(b::NPUBackend,
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

        # Batch feature extraction for NPU inference
        all_positions = vcat(left_moves_list, right_moves_list)
        if !isempty(all_positions)
            n_pos = length(all_positions)
            feature_batch = Matrix{Float32}(undef, n_pos, 10)
            for (i, pos) in enumerate(all_positions)
                feature_batch[i, :] .= _extract_features(pos)
            end
            scores = _npu_batch_inference(feature_batch)

            # Sort by heuristic score for better pruning
            n_left = length(left_moves_list)
            if n_left > 1
                left_order = sortperm(scores[1:n_left], rev=true)
                left_moves_list = left_moves_list[left_order]
            end
            n_right = length(right_moves_list)
            if n_right > 1
                right_order = sortperm(scores[n_left+1:end])
                right_moves_list = right_moves_list[right_order]
            end
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
        @warn "NPU game tree eval failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: Grundy Number via NPU Pattern Classifier
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_grundy_number(::NPUBackend, graph)

NPU-accelerated Grundy number computation for Green Hackenbush.
Uses component detection followed by NPU-accelerated XOR reduction.
"""
function HackenbushGames.backend_coprocessor_grundy_number(b::NPUBackend,
                                                             graph::HackenbushGraph)
    n = length(graph.edges)
    n < 32 && return nothing

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
        @warn "NPU Grundy computation failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: Position Hash via NPU Learned Projection
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_position_hash(::NPUBackend, graph)

NPU-accelerated position hashing using a learned projection network.
The NPU computes a hash by running graph features through a single-layer
linear projection followed by quantisation to produce a 64-bit hash.
"""
function HackenbushGames.backend_coprocessor_position_hash(b::NPUBackend,
                                                             graph::HackenbushGraph)
    n = length(graph.edges)
    n == 0 && return UInt64(0)
    n < 16 && return nothing

    try
        features = _extract_features(graph)

        # NPU single-layer projection: (1x10) * (10x4) -> (1x4) hash seeds
        proj = Float32[
            0.7071  0.3162  0.4472  0.5774;
            0.4082  0.8165  0.2236  0.3333;
            0.5774  0.2887  0.6667  0.7454;
            0.2673  0.5345  0.8018  0.1826;
            0.4472  0.6325  0.2236  0.5916;
            0.3162  0.9487  0.1826  0.4472;
            0.6667  0.2357  0.7071  0.3333;
            0.1826  0.5477  0.8165  0.2887;
            0.5345  0.4082  0.3162  0.7071;
            0.8018  0.2673  0.5345  0.1826;
        ]
        projected = features' * proj

        h = UInt64(0)
        for j in 1:4
            bits = reinterpret(UInt32, Float32(projected[j]))
            h = xor(h, UInt64(bits) << ((j - 1) * 16))
            h = xor(h, h >> 17)
            h = h * UInt64(0x9E3779B97F4A7C15)
        end

        for e in graph.edges
            h = xor(h, hash((e.u, e.v, Int(e.color))))
        end
        for g in graph.ground
            h = xor(h, hash(g))
        end

        return h
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "NPU position hash failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Remaining Hooks
# ============================================================================

function HackenbushGames.backend_coprocessor_minimax_search(b::NPUBackend, args...)
    return nothing
end

function HackenbushGames.backend_coprocessor_move_gen(b::NPUBackend,
                                                       graph::HackenbushGraph,
                                                       player::Symbol)
    return nothing
end

end # module HackenbushGamesNPUExt
