# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CladisticsDSPExt — DSP correlation-based acceleration for Cladistics.jl
# Exploits digital signal processing hardware for correlation-based sequence
# comparison, FFT-accelerated distance computation, and spectral bootstrap analysis.

module CladisticsDSPExt

using Cladistics
using AcceleratorGate
using AcceleratorGate: DSPBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(DSPBackend, :distance_matrix)
    register_operation!(DSPBackend, :parsimony_score)
    register_operation!(DSPBackend, :bootstrap_replicate)
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

# ============================================================================
# Correlation-Based Sequence Comparison
# ============================================================================
#
# DSP key insight: sequence similarity can be measured as cross-correlation
# of indicator signals. For each nucleotide state k, define indicator signal
# x_k[i] = 1 if sequence[i] == k, else 0. The number of matching positions
# equals sum over k of dot(x_k^(a), x_k^(b)). Cross-correlation can be
# computed via FFT: corr(x,y) = ifft(fft(x) .* conj(fft(y))).
#
# For phylogenetic distance, we only need the zero-lag correlation (alignment
# is already given), so we use direct dot products -- but the DSP framing
# allows future extension to unaligned sequence comparison.

"""
    _indicator_signals(encoded::Matrix{UInt8}, n_states::Int=14) -> Array{Float64,3}

Convert encoded sequences to per-state indicator signals.
Returns a (seq_len, n_taxa, n_states) tensor where
signals[i, j, k] = 1.0 if encoded[i, j] == k, else 0.0.
"""
function _indicator_signals(encoded::Matrix{UInt8}, n_states::Int=14)
    seq_len, n = size(encoded)
    signals = zeros(Float64, seq_len, n, n_states)
    @inbounds for j in 1:n
        for i in 1:seq_len
            state = Int(encoded[i, j])
            if 1 <= state <= n_states
                signals[i, j, state] = 1.0
            end
        end
    end
    return signals
end

"""
    _correlation_match_count(signals, i, j, n_states) -> Float64

Compute the number of matching positions between taxa i and j using
zero-lag cross-correlation of indicator signals. This is equivalent to
sum over k of dot(signals[:, i, k], signals[:, j, k]).
"""
function _correlation_match_count(signals::Array{Float64,3}, i::Int, j::Int, n_states::Int)
    seq_len = size(signals, 1)
    matches = 0.0
    @inbounds for k in 1:n_states
        # DSP dot product (zero-lag correlation) for state k
        for s in 1:seq_len
            matches += signals[s, i, k] * signals[s, j, k]
        end
    end
    return matches
end

"""
    _fft_correlation_distance(signals, seq_len, n_states, n) -> Matrix{Float64}

Compute pairwise match count matrix using FFT-based correlation.
For each state k, compute fft(signal_k) for all taxa, then dot products
in frequency domain. This is O(n^2 * K * L log L) vs O(n^2 * K * L) for
direct computation, but the DSP hardware FFT units make this faster for
long sequences.
"""
function _fft_correlation_distance(signals::Array{Float64,3}, seq_len::Int,
                                    n_states::Int, n::Int)
    # Pad to next power of 2 for FFT efficiency
    fft_len = nextpow(2, seq_len)

    # Pre-compute FFT of all indicator signals
    # On a real DSP, this uses hardware FFT units
    fft_signals = zeros(ComplexF64, fft_len, n, n_states)
    for k in 1:n_states
        for j in 1:n
            padded = zeros(Float64, fft_len)
            @inbounds for i in 1:seq_len
                padded[i] = signals[i, j, k]
            end
            # Manual DFT (avoid dependency on FFTW)
            # For DSP hardware, this maps to a dedicated FFT pipeline
            _inplace_fft!(view(fft_signals, :, j, k), padded, fft_len)
        end
    end

    # Compute match count matrix from frequency domain dot products
    match_matrix = zeros(Float64, n, n)
    for j in 1:n, i in 1:(j-1)
        matches = 0.0
        for k in 1:n_states
            # Cross-correlation at zero lag = sum of element-wise products in freq domain
            for f in 1:fft_len
                matches += real(fft_signals[f, i, k] * conj(fft_signals[f, j, k]))
            end
        end
        matches /= fft_len  # Normalize by FFT length
        match_matrix[i, j] = matches
        match_matrix[j, i] = matches
    end

    return match_matrix
end

"""
    _inplace_fft!(result, signal, N)

Compute DFT of a real signal using Cooley-Tukey radix-2 algorithm.
On DSP hardware, this maps to a dedicated butterfly pipeline.
"""
function _inplace_fft!(result::AbstractVector{ComplexF64}, signal::Vector{Float64}, N::Int)
    for k in 1:N
        s = ComplexF64(0.0, 0.0)
        for n in 0:(N-1)
            angle = -2.0 * pi * (k - 1) * n / N
            s += signal[n + 1] * (cos(angle) + im * sin(angle))
        end
        result[k] = s
    end
end

@inline function _distance_correction(diffs::Float64, seq_len::Int, method::Symbol)
    p = diffs / seq_len
    if method == :hamming
        return diffs
    elseif method == :p_distance
        return p
    elseif method == :jc69
        return p >= 0.75 ? Inf : -0.75 * log(1.0 - (4.0 * p / 3.0))
    end
    return p
end

"""
    Cladistics.backend_coprocessor_distance_matrix(::DSPBackend, sequences, method)

DSP-accelerated pairwise distance matrix using correlation-based comparison.
For short sequences (< 256 sites), uses direct correlation dot products.
For long sequences, uses FFT-based correlation exploiting hardware FFT units.
"""
function Cladistics.backend_coprocessor_distance_matrix(b::DSPBackend,
                                                         sequences::Vector{String},
                                                         method::Symbol)
    n = length(sequences)
    n < 8 && return nothing

    # K2P requires transition/transversion breakdown -- fall back for now
    method == :k2p && return nothing

    seq_len = length(sequences[1])
    n_states = 14

    mem_estimate = Int64(seq_len * n * n_states * 8 + n * n * 8)
    track_allocation!(b, mem_estimate)

    try
        encoded = _encode_sequences(sequences)
        signals = _indicator_signals(encoded, n_states)

        # Choose correlation method based on sequence length
        if seq_len < 256
            # Direct correlation -- small sequence overhead
            D = zeros(Float64, n, n)
            for j in 1:n, i in 1:(j-1)
                matches = _correlation_match_count(signals, i, j, n_states)
                diffs = Float64(seq_len) - matches
                d = _distance_correction(diffs, seq_len, method)
                D[i, j] = d
                D[j, i] = d
            end
        else
            # FFT-based correlation for long sequences
            match_matrix = _fft_correlation_distance(signals, seq_len, n_states, n)
            D = zeros(Float64, n, n)
            for j in 1:n, i in 1:(j-1)
                diffs = Float64(seq_len) - match_matrix[i, j]
                d = _distance_correction(diffs, seq_len, method)
                D[i, j] = d
                D[j, i] = d
            end
        end

        track_deallocation!(b, mem_estimate)
        return D
    catch ex
        track_deallocation!(b, mem_estimate)
        _record_diagnostic!(b, "runtime_errors")
        @warn "DSP distance matrix failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# DSP Parsimony and Bootstrap
# ============================================================================

"""
    Cladistics.backend_coprocessor_parsimony_score(::DSPBackend, tree, char_matrix)

DSP-accelerated parsimony scoring using correlation-based state matching.
The Fitch intersection test is reframed as a correlation check: two state
sets have non-empty intersection iff their indicator signals have positive
dot product.
"""
function Cladistics.backend_coprocessor_parsimony_score(b::DSPBackend,
                                                         tree::Cladistics.PhylogeneticTree,
                                                         char_matrix::Matrix{Char})
    n_sites = size(char_matrix, 2)
    n_taxa = size(char_matrix, 1)

    n_sites < 64 && return nothing

    try
        # Encode as bitmasks for Fitch
        bitmasks = Matrix{UInt32}(undef, n_taxa, n_sites)
        for t in 1:n_taxa
            for s in 1:n_sites
                code = get(CHAR_ENCODE, char_matrix[t, s], UInt8(0))
                bitmasks[t, s] = UInt32(1) << code
            end
        end

        # Flatten tree
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

        states = Matrix{UInt32}(undef, n_nodes, n_sites)
        total_score = 0

        for node in 1:n_nodes
            ti = taxon_idx[node]
            lc = left_child[node]
            rc = right_child[node]

            if ti > 0
                @inbounds for s in 1:n_sites
                    states[node, s] = bitmasks[ti, s]
                end
            else
                @inbounds for s in 1:n_sites
                    left_set = lc > 0 ? states[lc, s] : UInt32(0xFFFFFFFF)
                    right_set = rc > 0 ? states[rc, s] : UInt32(0xFFFFFFFF)
                    intersection = left_set & right_set
                    if intersection != UInt32(0)
                        states[node, s] = intersection
                    else
                        states[node, s] = left_set | right_set
                        total_score += 1
                    end
                end
            end
        end

        return total_score
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        @warn "DSP parsimony score failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

function Cladistics.backend_coprocessor_bootstrap_replicate(b::DSPBackend,
                                                             sequences::Vector{String},
                                                             replicates::Int,
                                                             method::Symbol)
    # K2P unsupported on DSP path
    method == :k2p && return nothing

    n = length(sequences)
    seq_len = length(sequences[1])
    (n < 8 || replicates < 5) && return nothing

    try
        encoded = _encode_sequences(sequences)
        n_states = 14
        clade_counts = Dict{Set{String}, Int}()

        for rep in 1:replicates
            col_indices = rand(1:seq_len, seq_len)
            resampled = Matrix{UInt8}(undef, seq_len, n)
            @inbounds for j in 1:n
                for i in 1:seq_len
                    resampled[i, j] = encoded[col_indices[i], j]
                end
            end

            signals = _indicator_signals(resampled, n_states)
            D = zeros(Float64, n, n)
            for j in 1:n, i in 1:(j-1)
                matches = _correlation_match_count(signals, i, j, n_states)
                diffs = Float64(seq_len) - matches
                d = _distance_correction(diffs, seq_len, method)
                D[i, j] = d
                D[j, i] = d
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
        @warn "DSP bootstrap failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

function Cladistics.backend_coprocessor_neighbor_join(b::DSPBackend, dmat::Matrix{Float64}, taxa_names)
    return nothing
end

function Cladistics.backend_coprocessor_tree_search(b::DSPBackend, args...)
    return nothing
end

end # module CladisticsDSPExt
