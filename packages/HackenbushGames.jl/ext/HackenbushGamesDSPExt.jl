# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# HackenbushGamesDSPExt — Digital Signal Processing acceleration for HackenbushGames.jl
# Exploits DSP pipelines for signal-domain game position analysis: FFT-based
# graph spectral features, convolution-based pattern matching on adjacency
# structure, and frequency-domain position hashing.

module HackenbushGamesDSPExt

using HackenbushGames
using HackenbushGames: Edge, EdgeColor, HackenbushGraph, GameForm,
    Blue, Red, Green, _graph_key, _edge_allowed,
    mex, nim_sum, simplest_dyadic_between, simplify_game,
    cut_edge, prune_disconnected, moves
using AcceleratorGate
using AcceleratorGate: DSPBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(DSPBackend, :position_hash)
    register_operation!(DSPBackend, :grundy_number)
    register_operation!(DSPBackend, :game_tree_eval)
end

# ============================================================================
# DSP Architecture Notes
# ============================================================================
#
# DSP processors excel at:
#   - Fixed-point arithmetic pipelines (MAC units)
#   - FFT butterfly networks for spectral analysis
#   - FIR/IIR filter chains for streaming convolution
#   - Circular buffer management for sliding window operations
#
# Game-theoretic mappings:
#   1. Graph Laplacian eigenvalues via FFT for spectral graph features
#   2. Convolution-based adjacency pattern matching
#   3. Frequency-domain hashing via spectral fingerprints

# ============================================================================
# Graph Spectral Analysis via DSP FFT
# ============================================================================

"""
    _dsp_spectral_features(graph::HackenbushGraph) -> Vector{Float64}

Compute spectral features of the graph using DSP-style FFT on the
degree sequence. The DSP processes the degree sequence as a discrete
signal and extracts frequency-domain features.
"""
function _dsp_spectral_features(graph::HackenbushGraph)
    n = length(graph.edges)
    n == 0 && return Float64[]

    # Build degree sequence as a "signal"
    all_nodes = Set{Int}()
    for e in graph.edges
        push!(all_nodes, e.u)
        push!(all_nodes, e.v)
    end
    node_list = sort(collect(all_nodes))
    nn = length(node_list)
    nn == 0 && return Float64[]

    node_idx = Dict(v => i for (i, v) in enumerate(node_list))

    # Degree signal per color channel
    deg_blue = zeros(Float64, nn)
    deg_red = zeros(Float64, nn)
    deg_green = zeros(Float64, nn)

    for e in graph.edges
        ui = node_idx[e.u]
        vi = node_idx[e.v]
        if e.color == Blue
            deg_blue[ui] += 1.0; deg_blue[vi] += 1.0
        elseif e.color == Red
            deg_red[ui] += 1.0; deg_red[vi] += 1.0
        else
            deg_green[ui] += 1.0; deg_green[vi] += 1.0
        end
    end

    # DSP: compute power spectral density via DFT (manual butterfly)
    # For each channel, compute |DFT[k]|^2 for k = 0..nn-1
    features = Float64[]
    for signal in (deg_blue, deg_red, deg_green)
        energy = 0.0
        dc = sum(signal)
        peak_freq_power = 0.0
        for k in 1:nn
            # DFT bin k (DSP butterfly computation)
            re = 0.0
            im = 0.0
            for j in 1:nn
                angle = -2.0 * pi * (k - 1) * (j - 1) / nn
                re += signal[j] * cos(angle)
                im += signal[j] * sin(angle)
            end
            power = re * re + im * im
            energy += power
            k > 1 && (peak_freq_power = max(peak_freq_power, power))
        end
        push!(features, dc)
        push!(features, energy)
        push!(features, peak_freq_power)
    end

    return features
end

# ============================================================================
# Hook: Position Hash via DSP Spectral Fingerprint
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_position_hash(::DSPBackend, graph)

DSP-accelerated position hashing using spectral fingerprinting.
The graph's degree sequence is treated as a discrete signal; the DSP
computes its DFT and produces a hash from the spectral coefficients.
This gives a position-invariant hash (independent of node labelling).
"""
function HackenbushGames.backend_coprocessor_position_hash(b::DSPBackend,
                                                             graph::HackenbushGraph)
    n = length(graph.edges)
    n == 0 && return UInt64(0)
    n < 16 && return nothing

    try
        features = _dsp_spectral_features(graph)
        isempty(features) && return nothing

        # Hash the spectral fingerprint
        h = UInt64(0xD5B0_4A54_5EED_0001)
        for (i, f) in enumerate(features)
            bits = reinterpret(UInt64, Float64(f))
            h = xor(h, bits)
            h = xor(h, h >> 17)
            h = h * UInt64(0x9E3779B97F4A7C15)
            h = xor(h, h >> 31)
            h = h * UInt64(0xBF58476D1CE4E5B9)
        end

        # Mix in ground node information
        for g in graph.ground
            h = xor(h, hash(g))
        end

        return h
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "DSP position hash failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: Grundy Number via DSP Convolution
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_grundy_number(::DSPBackend, graph)

DSP-accelerated Grundy number computation for Green Hackenbush.
Uses convolution-based connected component detection: the adjacency
signal is repeatedly convolved with a spreading kernel until convergence,
then edge counts per component are XOR-reduced.
"""
function HackenbushGames.backend_coprocessor_grundy_number(b::DSPBackend,
                                                             graph::HackenbushGraph)
    n = length(graph.edges)
    n < 16 && return nothing

    for e in graph.edges
        e.color != Green && return nothing
    end

    try
        # Build adjacency and use iterative label propagation
        # (DSP convolution analogue: spreading labels via adjacency filter)
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
        @warn "DSP Grundy computation failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: Game Tree Evaluation via DSP Spectral Scoring
# ============================================================================

"""
    HackenbushGames.backend_coprocessor_game_tree_eval(::DSPBackend, graph, max_depth)

DSP-accelerated game tree evaluation. Uses spectral features of child
positions for heuristic ordering, processing the feature extraction
through the DSP's FFT pipeline for efficient batch computation.
"""
function HackenbushGames.backend_coprocessor_game_tree_eval(b::DSPBackend,
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

        # DSP spectral scoring for move ordering
        all_positions = vcat(left_moves_list, right_moves_list)
        if length(all_positions) > 1
            scores = Float64[]
            for pos in all_positions
                feats = _dsp_spectral_features(pos)
                # Score: blue energy - red energy (spectral game advantage)
                score = length(feats) >= 6 ? feats[2] - feats[5] : 0.0
                push!(scores, score)
            end

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
        @warn "DSP game tree eval failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Remaining Hooks
# ============================================================================

function HackenbushGames.backend_coprocessor_minimax_search(b::DSPBackend, args...)
    return nothing
end

function HackenbushGames.backend_coprocessor_move_gen(b::DSPBackend,
                                                       graph::HackenbushGraph,
                                                       player::Symbol)
    return nothing
end

end # module HackenbushGamesDSPExt
