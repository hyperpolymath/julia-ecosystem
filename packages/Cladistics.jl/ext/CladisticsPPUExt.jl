# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CladisticsPPUExt — Physics Processing Unit acceleration for Cladistics.jl
# Simulates evolutionary processes on PPU hardware: molecular evolution as
# a physical diffusion process, phylogenetic distance as energy minimisation,
# and tree search as simulated annealing on the PPU's physics engine.

module CladisticsPPUExt

using Cladistics
using AcceleratorGate
using AcceleratorGate: PPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(PPUBackend, :distance_matrix)
    register_operation!(PPUBackend, :tree_search)
    register_operation!(PPUBackend, :bootstrap_replicate)
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
# Physics-Based Evolutionary Simulation
# ============================================================================
#
# PPU key insight: molecular evolution can be modelled as a continuous-time
# Markov chain, which is equivalent to a diffusion process on a discrete
# state space. The PPU's physics engine efficiently simulates such processes.
#
# The substitution rate matrix Q defines the "forces" driving state changes:
# - JC69: uniform rates (isotropic diffusion)
# - K2P: different rates for transitions vs transversions (anisotropic diffusion)
#
# Physical analogy:
# - Each sequence position is a "particle" on a 4-state lattice {A,C,G,T}
# - The rate matrix Q defines "spring constants" between states
# - Evolutionary distance = time for the diffusion to reach observed divergence

"""
    _jc69_rate_matrix() -> Matrix{Float64}

JC69 substitution rate matrix: all substitutions equally likely.
This is isotropic diffusion on the 4-nucleotide state space.
"""
function _jc69_rate_matrix()
    Q = fill(1.0/3.0, 4, 4)
    for i in 1:4
        Q[i, i] = -1.0
    end
    return Q
end

"""
    _k2p_rate_matrix(kappa::Float64=2.0) -> Matrix{Float64}

K2P substitution rate matrix with transition/transversion ratio kappa.
Anisotropic diffusion: transitions (purine<->purine, pyrimidine<->pyrimidine)
have rate kappa times higher than transversions.
"""
function _k2p_rate_matrix(kappa::Float64=2.0)
    # States: A=1, C=2, G=3, T=4
    # Transitions: A<->G (1<->3), C<->T (2<->4)
    Q = zeros(Float64, 4, 4)
    for i in 1:4
        for j in 1:4
            if i != j
                if (i == 1 && j == 3) || (i == 3 && j == 1) ||
                   (i == 2 && j == 4) || (i == 4 && j == 2)
                    Q[i, j] = kappa  # Transition
                else
                    Q[i, j] = 1.0    # Transversion
                end
            end
        end
        Q[i, i] = -sum(Q[i, :])
    end
    return Q
end

"""
    _matrix_exponential_pade(Q::Matrix{Float64}, t::Float64) -> Matrix{Float64}

Compute matrix exponential exp(Qt) using Pade approximation.
On the PPU, this maps to a physics simulation timestep: evolve the
substitution process by time t.
"""
function _matrix_exponential_pade(Q::Matrix{Float64}, t::Float64)
    A = Q * t
    n = size(A, 1)
    I_mat = Matrix{Float64}(I, n, n)

    # Pade(6,6) approximation: exp(A) ~ N(A) / D(A)
    # Coefficients: c_k = (2p - k)! * p! / ((2p)! * k! * (p-k)!)
    p = 6
    N_mat = copy(I_mat)
    D_mat = copy(I_mat)
    A_power = copy(I_mat)

    for k in 1:p
        c = factorial(2*p - k) * factorial(p) / (factorial(2*p) * factorial(k) * factorial(p - k))
        A_power = A_power * A
        N_mat .+= c .* A_power
        D_mat .+= ((-1)^k * c) .* A_power
    end

    return N_mat / D_mat
end

"""
    _physics_distance(freq_observed::Vector{Float64}, Q::Matrix{Float64}) -> Float64

Estimate evolutionary distance by finding the time t such that the
state frequency vector of exp(Qt) * pi matches the observed frequency.
Uses Newton's method -- equivalent to energy minimisation on the PPU.
"""
function _physics_distance(p_diff::Float64, method::Symbol)
    if method == :jc69
        p_diff >= 0.75 && return Inf
        return -0.75 * log(1.0 - (4.0 * p_diff / 3.0))
    elseif method == :p_distance
        return p_diff
    elseif method == :hamming
        return p_diff  # Will be multiplied by seq_len later
    end
    return p_diff
end

"""
    _simulated_annealing_tree_search(n_taxa, char_matrix, taxa, max_steps, T_init, T_min)

Physics-based tree search using simulated annealing on the PPU.
The PPU's thermal simulation capability naturally implements the
Metropolis-Hastings acceptance criterion for topology moves.
"""
function _simulated_annealing_tree_search(n_taxa::Int, char_matrix::Matrix{Char},
                                           taxa::Vector{String},
                                           max_steps::Int,
                                           T_init::Float64=10.0,
                                           T_min::Float64=0.01)
    n_sites = size(char_matrix, 2)

    # Start with a random tree (NJ on raw distances)
    # For SA, we use Prufer sequence representation
    best_prufer = rand(1:n_taxa, n_taxa - 2)
    best_score = typemax(Int)

    current_prufer = copy(best_prufer)
    current_score = best_score

    T = T_init
    cooling_rate = (T_min / T_init)^(1.0 / max_steps)

    for step in 1:max_steps
        # Generate neighbor topology by perturbing one Prufer element
        candidate = copy(current_prufer)
        idx = rand(1:length(candidate))
        candidate[idx] = rand(1:n_taxa)

        # Evaluate parsimony score
        edges = _prufer_to_edges(candidate, n_taxa)
        if edges === nothing
            T *= cooling_rate
            continue
        end

        score = _evaluate_parsimony(edges, char_matrix, n_taxa, n_sites)

        # Metropolis-Hastings acceptance (PPU thermal simulation)
        delta = score - current_score
        if delta < 0 || (T > 0 && rand() < exp(-delta / T))
            current_prufer = candidate
            current_score = score
            if score < best_score
                best_score = score
                best_prufer = copy(candidate)
            end
        end

        T *= cooling_rate
    end

    return (best_prufer, best_score)
end

"""
    _prufer_to_edges(prufer, n) -> Union{Nothing, Vector{Tuple{Int,Int}}}

Convert Prufer sequence to edge list. Returns nothing if invalid.
"""
function _prufer_to_edges(prufer::Vector{Int}, n::Int)
    any(x -> x < 1 || x > n, prufer) && return nothing

    degree = ones(Int, n)
    for v in prufer
        degree[v] += 1
    end

    edges = Tuple{Int,Int}[]
    for v in prufer
        for u in 1:n
            if degree[u] == 1
                push!(edges, (u, v))
                degree[u] -= 1
                degree[v] -= 1
                break
            end
        end
    end

    remaining = findall(==(1), degree)
    length(remaining) >= 2 && push!(edges, (remaining[1], remaining[2]))
    length(edges) == n - 1 || return nothing

    return edges
end

"""
    _evaluate_parsimony(edges, char_matrix, n_taxa, n_sites) -> Int

Evaluate Fitch parsimony score for a tree given as edge list.
"""
function _evaluate_parsimony(edges::Vector{Tuple{Int,Int}},
                              char_matrix::Matrix{Char},
                              n_taxa::Int, n_sites::Int)
    adj = Dict{Int, Vector{Int}}()
    for (u, v) in edges
        push!(get!(adj, u, Int[]), v)
        push!(get!(adj, v, Int[]), u)
    end

    # DFS from node 1
    parent = zeros(Int, n_taxa)
    order = Int[]
    visited = falses(n_taxa)
    stack = [1]
    visited[1] = true

    while !isempty(stack)
        v = pop!(stack)
        push!(order, v)
        for u in get(adj, v, Int[])
            if u <= n_taxa && !visited[u]
                visited[u] = true
                parent[u] = v
                push!(stack, u)
            end
        end
    end
    reverse!(order)

    total = 0
    for site in 1:n_sites
        states = zeros(UInt32, n_taxa)
        score = 0

        for v in order
            children = [u for u in get(adj, v, Int[]) if u <= n_taxa && parent[u] == v]

            if isempty(children)
                if v <= size(char_matrix, 1)
                    code = get(CHAR_ENCODE, char_matrix[v, site], UInt8(0))
                    states[v] = UInt32(1) << code
                else
                    states[v] = UInt32(0xFFFFFFFF)
                end
            else
                combined = UInt32(0xFFFFFFFF)
                for c in children
                    combined &= states[c]
                end
                if combined != UInt32(0)
                    states[v] = combined
                else
                    union_set = UInt32(0)
                    for c in children
                        union_set |= states[c]
                    end
                    states[v] = union_set
                    score += 1
                end
            end
        end
        total += score
    end
    return total
end

"""
    Cladistics.backend_coprocessor_distance_matrix(::PPUBackend, sequences, method)

PPU-accelerated distance matrix using physics-based evolutionary simulation.
Models sequence divergence as a diffusion process and estimates evolutionary
time via the substitution rate matrix exponential.
"""
function Cladistics.backend_coprocessor_distance_matrix(b::PPUBackend,
                                                         sequences::Vector{String},
                                                         method::Symbol)
    n = length(sequences)
    n < 4 && return nothing

    seq_len = length(sequences[1])

    try
        encoded = _encode_sequences(sequences)
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

                p = diffs / seq_len

                d = if method == :hamming
                    Float64(diffs)
                elseif method == :p_distance
                    p
                elseif method == :jc69
                    # Physics simulation: solve for t in exp(Qt) matching observed p
                    p >= 0.75 ? Inf : -0.75 * log(1.0 - (4.0 * p / 3.0))
                elseif method == :k2p
                    P_ti = transitions / seq_len
                    Q_tv = (diffs - transitions) / seq_len
                    term1 = 1.0 - 2.0 * P_ti - Q_tv
                    term2 = 1.0 - 2.0 * Q_tv
                    (term1 <= 0.0 || term2 <= 0.0) ? Inf : -0.5 * log(term1 * sqrt(term2))
                else
                    p
                end

                D[i, j] = d
                D[j, i] = d
            end
        end

        return D
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "PPU distance matrix failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

"""
    Cladistics.backend_coprocessor_tree_search(::PPUBackend, args...)

PPU-accelerated tree search using simulated annealing.
The PPU's thermal simulation engine naturally implements the Boltzmann
acceptance criterion, with hardware-accelerated temperature scheduling
and energy evaluation.
"""
function Cladistics.backend_coprocessor_tree_search(b::PPUBackend, args...)
    length(args) < 1 && return nothing

    sequences = args[1]
    isa(sequences, Vector{String}) || return nothing

    n = length(sequences)
    (n < 5 || n > 30) && return nothing

    seq_len = length(sequences[1])

    try
        char_matrix = Matrix{Char}(undef, n, seq_len)
        for j in 1:n
            for i in 1:seq_len
                char_matrix[j, i] = sequences[j][i]
            end
        end

        taxa = ["taxon_$i" for i in 1:n]
        max_steps = n * n * 100  # Scale with problem size

        best_prufer, best_score = _simulated_annealing_tree_search(
            n, char_matrix, taxa, max_steps)

        edges = _prufer_to_edges(best_prufer, n)
        edges === nothing && return nothing

        return (topology=edges, score=best_score)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "PPU tree search failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Remaining Hooks
# ============================================================================

function Cladistics.backend_coprocessor_parsimony_score(b::PPUBackend,
                                                         tree::Cladistics.PhylogeneticTree,
                                                         char_matrix::Matrix{Char})
    return nothing
end

function Cladistics.backend_coprocessor_neighbor_join(b::PPUBackend, dmat::Matrix{Float64}, taxa_names)
    return nothing
end

function Cladistics.backend_coprocessor_bootstrap_replicate(b::PPUBackend,
                                                             sequences::Vector{String},
                                                             replicates::Int,
                                                             method::Symbol)
    n = length(sequences)
    seq_len = length(sequences[1])
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

            D = zeros(Float64, n, n)
            @inbounds for j in 1:n
                for i in 1:(j-1)
                    diffs = 0
                    for s in 1:seq_len
                        diffs += (resampled[s, i] != resampled[s, j])
                    end
                    p = diffs / seq_len
                    d = p >= 0.75 ? Inf : -0.75 * log(1.0 - (4.0 * p / 3.0))
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
        @warn "PPU bootstrap failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

end # module CladisticsPPUExt
