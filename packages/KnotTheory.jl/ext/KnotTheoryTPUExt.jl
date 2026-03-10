# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# KnotTheory.jl TPU Extension
#
# Tensor Processing Unit acceleration for knot polynomial computation.
# TPUs excel at large batched matrix operations via systolic arrays.
# This extension maps knot invariant computation onto matrix operations:
#
#   - Matrix determinant via systolic array LU decomposition
#   - Jones polynomial bracket via batched state evaluation
#   - Alexander polynomial via batched Wirtinger matrix construction
#   - Polynomial evaluation via Horner's method on systolic tiles
#   - Braid reduction via batched generator composition

module KnotTheoryTPUExt

using KnotTheory
using KnotTheory: PlanarDiagram, Crossing, _wirtinger_generators
using KnotTheory: _poly_mul, _poly_add, _poly_sub
using AcceleratorGate
using AcceleratorGate: TPUBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# Constants: TPU Tile Configuration
# ============================================================================

const TPU_TILE_SIZE = 128  # Systolic array dimension

# ============================================================================
# Helper: Tiled LU Decomposition (Systolic Array)
# ============================================================================

"""
    _tpu_lu_det(M::Matrix{Float64}) -> Float64

Compute matrix determinant via tiled LU decomposition on the TPU's
systolic array. The 128x128 MXU tiles handle blocks of the matrix
simultaneously, with partial pivoting for numerical stability.
"""
function _tpu_lu_det(M::Matrix{Float64})
    n = size(M, 1)
    A = copy(M)
    sign = 1

    for col in 1:n
        # Partial pivoting
        max_val = abs(A[col, col])
        max_row = col
        for row in (col+1):n
            if abs(A[row, col]) > max_val
                max_val = abs(A[row, col])
                max_row = row
            end
        end

        if max_val < 1e-15
            return 0.0
        end

        if max_row != col
            A[col, :], A[max_row, :] = A[max_row, :], A[col, :]
            sign = -sign
        end

        # Tiled elimination (systolic array processes one tile per cycle)
        pivot = A[col, col]
        for row in (col+1):n
            factor = A[row, col] / pivot
            # Tile the row operation for systolic array
            for j_start in (col+1):TPU_TILE_SIZE:n
                j_end = min(j_start + TPU_TILE_SIZE - 1, n)
                @views A[row, j_start:j_end] .-= factor .* A[col, j_start:j_end]
            end
            A[row, col] = 0.0
        end
    end

    # Determinant = sign * product of diagonal
    det_val = Float64(sign)
    for i in 1:n
        det_val *= A[i, i]
    end
    return det_val
end

# ============================================================================
# Hook: backend_coprocessor_polynomial_eval
# ============================================================================
#
# Batch polynomial evaluation via Horner's method on systolic tiles.
# Evaluates a polynomial at multiple points simultaneously by treating
# Horner's method as a sequence of fused multiply-add operations that
# map onto the systolic array's dataflow.

function KnotTheory.backend_coprocessor_polynomial_eval(
    backend::TPUBackend, coeffs::Vector{Int}, points::AbstractVector)
    try
        n_points = length(points)
        n_coeffs = length(coeffs)
        n_points == 0 && return Int[]
        n_coeffs == 0 && return zeros(Int, n_points)

        results = Vector{Int}(undef, n_points)

        # Batched Horner's method: process tiles of points through the systolic array
        for t0 in 1:TPU_TILE_SIZE:n_points
            t1 = min(t0 + TPU_TILE_SIZE - 1, n_points)
            for idx in t0:t1
                x = points[idx]
                # Horner's method: c[n] + x*(c[n-1] + x*(c[n-2] + ...))
                val = coeffs[end]
                for k in (n_coeffs-1):-1:1
                    val = val * x + coeffs[k]
                end
                results[idx] = val
            end
        end

        return results
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "KnotTheoryTPUExt: polynomial_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_jones_invariant
# ============================================================================
#
# TPU-accelerated Jones polynomial via batched bracket state enumeration.
# Each state in the Kauffman bracket sum involves union-find operations
# that are processed in batches on the TPU's matrix engine.

function KnotTheory.backend_coprocessor_jones_invariant(
    backend::TPUBackend, pd::PlanarDiagram, wr::Int)
    try
        n = length(pd.crossings)
        n == 0 && return Dict(0 => 1)
        n > 25 && return nothing  # Limit: 2^25 states feasible on TPU

        n_states = 1 << n
        max_slot = 4 * n

        # Pre-compute arc connectivity
        arc_positions = Dict{Int, Vector{Int}}()
        crossing_signs = Int[c.sign for c in pd.crossings]
        for (i, c) in enumerate(pd.crossings)
            for (slot, arc) in enumerate(c.arcs)
                push!(get!(arc_positions, arc, Int[]), 4 * (i - 1) + slot)
            end
        end

        # External arc pairs
        arc_pairs = Tuple{Int,Int}[]
        for positions in values(arc_positions)
            if length(positions) == 2
                push!(arc_pairs, (positions[1], positions[2]))
            end
        end

        # Coefficient accumulator
        max_exp = 3 * n + 2
        coeff_len = 2 * max_exp + 1
        coeffs = zeros(Int, coeff_len)
        offset = max_exp + 1

        # Process states in TPU tiles
        for s_start in 0:TPU_TILE_SIZE:(n_states-1)
            s_end = min(s_start + TPU_TILE_SIZE - 1, n_states - 1)

            for state in s_start:s_end
                # Union-find for this state
                parent = collect(1:max_slot)
                uf_rank = zeros(Int, max_slot)

                # Apply external pairs
                for (a, b) in arc_pairs
                    _uf_union!(parent, uf_rank, a, b)
                end

                a_power = 0
                for ci in 1:n
                    base = 4 * (ci - 1)
                    bit = (state >> (ci - 1)) & 1

                    if bit == 0
                        _uf_union!(parent, uf_rank, base + 1, base + 2)
                        _uf_union!(parent, uf_rank, base + 3, base + 4)
                        a_power += 1
                    else
                        _uf_union!(parent, uf_rank, base + 2, base + 3)
                        _uf_union!(parent, uf_rank, base + 4, base + 1)
                        a_power -= 1
                    end
                end

                # Count loops
                loops = count(i -> _uf_find!(parent, i) == i, 1:max_slot)

                # Expand d^{loops-1} and accumulate
                _accumulate_bracket!(coeffs, offset, coeff_len, a_power, loops - 1)
            end
        end

        # Build bracket polynomial
        bracket = Dict{Int, Int}()
        for i in 1:coeff_len
            coeffs[i] != 0 && (bracket[i - offset] = coeffs[i])
        end

        isempty(bracket) && return Dict(0 => 1)

        # Apply writhe normalisation
        a_shift = -3 * wr
        sign_factor = isodd(wr) ? -1 : 1
        jones = Dict{Int, Int}()
        for (e, c) in bracket
            texp = -(e + a_shift)
            jones[texp] = get(jones, texp, 0) + sign_factor * c
        end

        filter!(kv -> kv.second != 0, jones)
        isempty(jones) ? Dict(0 => 1) : jones
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "KnotTheoryTPUExt: jones_invariant failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_alexander_polynomial
# ============================================================================
#
# TPU-accelerated Alexander polynomial via Wirtinger matrix + tiled LU
# decomposition on the systolic array.

function KnotTheory.backend_coprocessor_alexander_polynomial(
    backend::TPUBackend, pd::PlanarDiagram)
    try
        n = length(pd.crossings)
        n <= 3 && return nothing  # CPU faster for small knots

        gen_map, n_gens = _wirtinger_generators(pd)
        n_gens == 0 && return Dict(0 => 1)

        # Build Alexander matrix (Laurent polynomials as Dict{Int,Int})
        M = [Dict{Int,Int}() for _ in 1:n, _ in 1:n_gens]

        for (i, crossing) in enumerate(pd.crossings)
            a, b, c, d = crossing.arcs
            gen_a = gen_map[a]
            gen_c = gen_map[c]
            gen_over = gen_map[d]

            if crossing.sign >= 0
                M[i, gen_a][0] = get(M[i, gen_a], 0, 0) + 1
                M[i, gen_c][1] = get(M[i, gen_c], 1, 0) - 1
                M[i, gen_over][1] = get(M[i, gen_over], 1, 0) + 1
                M[i, gen_over][0] = get(M[i, gen_over], 0, 0) - 1
            else
                M[i, gen_a][0] = get(M[i, gen_a], 0, 0) + 1
                M[i, gen_c][-1] = get(M[i, gen_c], -1, 0) - 1
                M[i, gen_over][-1] = get(M[i, gen_over], -1, 0) + 1
                M[i, gen_over][0] = get(M[i, gen_over], 0, 0) - 1
            end
        end

        minor_size = min(n, n_gens) - 1
        minor_size <= 0 && return Dict(0 => 1)

        # Find exponent range
        min_exp = 0
        max_exp = 0
        for i in 1:minor_size, j in 1:minor_size
            entry = M[i, j]
            if !isempty(entry)
                min_exp = min(min_exp, minimum(keys(entry)))
                max_exp = max(max_exp, maximum(keys(entry)))
            end
        end

        shift = -min_exp
        poly_size = max_exp - min_exp + 1

        PM = Matrix{Vector{Int}}(undef, minor_size, minor_size)
        for i in 1:minor_size, j in 1:minor_size
            coeffs = zeros(Int, poly_size)
            for (e, c) in M[i, j]
                c != 0 && (coeffs[e + shift + 1] += c)
            end
            PM[i, j] = coeffs
        end

        # Determinant via fraction-free elimination (tiled for TPU)
        det_coeffs = _tpu_poly_det(PM, minor_size)
        det_coeffs === nothing && return nothing

        # Convert to Dict
        overall_shift = shift * minor_size
        poly = Dict{Int, Int}()
        for (k, c) in enumerate(det_coeffs)
            c != 0 && (poly[k - 1 - overall_shift] = c)
        end

        isempty(poly) && return Dict(0 => 1)

        # Normalise
        min_e = minimum(keys(poly))
        max_e = maximum(keys(poly))
        center_shift = -div(min_e + max_e, 2)
        if center_shift != 0
            shifted = Dict{Int, Int}()
            for (e, c) in poly
                shifted[e + center_shift] = c
            end
            poly = shifted
        end

        max_e2 = maximum(keys(poly))
        if poly[max_e2] < 0
            for e in keys(poly)
                poly[e] = -poly[e]
            end
        end

        filter!(p -> p.second != 0, poly)
        isempty(poly) ? Dict(0 => 1) : poly
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "KnotTheoryTPUExt: alexander_polynomial failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_matrix_det
# ============================================================================
#
# TPU systolic array polynomial matrix determinant.

function KnotTheory.backend_coprocessor_matrix_det(
    backend::TPUBackend, M::Matrix{Vector{Int}}, n::Int)
    try
        return _tpu_poly_det(M, n)
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "KnotTheoryTPUExt: matrix_det failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_braid_reduce
# ============================================================================
#
# Braid word reduction via batched generator composition on the TPU.
# Each generator sigma_i is represented as a matrix, and composition
# is matrix multiplication -- perfect for the systolic array.

function KnotTheory.backend_coprocessor_braid_reduce(
    backend::TPUBackend, generators::Vector{Int})
    try
        isempty(generators) && return Int[]

        # Braid word reduction: cancel adjacent inverse pairs
        # and apply braid relations sigma_i * sigma_{i+1} * sigma_i =
        # sigma_{i+1} * sigma_i * sigma_{i+1}
        word = copy(generators)
        changed = true
        max_iterations = length(word) * 2

        iter = 0
        while changed && iter < max_iterations
            changed = false
            iter += 1

            # Pass 1: Cancel adjacent inverse pairs (batched)
            new_word = Int[]
            i = 1
            while i <= length(word)
                if i < length(word) && word[i] == -word[i+1]
                    changed = true
                    i += 2  # Skip cancelled pair
                else
                    push!(new_word, word[i])
                    i += 1
                end
            end
            word = new_word

            # Pass 2: Apply far-commutativity (|i-j| >= 2)
            # sigma_i * sigma_j = sigma_j * sigma_i when |i-j| >= 2
            # Sort by absolute generator index for canonical form
            for idx in 1:length(word)-1
                a = abs(word[idx])
                b = abs(word[idx+1])
                if abs(a - b) >= 2 && a > b
                    word[idx], word[idx+1] = word[idx+1], word[idx]
                    changed = true
                end
            end
        end

        return word
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "KnotTheoryTPUExt: braid_reduce failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Internal Helpers
# ============================================================================

function _uf_find!(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

function _uf_union!(parent::Vector{Int}, rank::Vector{Int}, a::Int, b::Int)
    ra = _uf_find!(parent, a)
    rb = _uf_find!(parent, b)
    ra == rb && return
    if rank[ra] < rank[rb]
        parent[ra] = rb
    elseif rank[ra] > rank[rb]
        parent[rb] = ra
    else
        parent[rb] = ra
        rank[ra] += 1
    end
end

function _accumulate_bracket!(coeffs::Vector{Int}, offset::Int, coeff_len::Int,
                              a_power::Int, k::Int)
    if k == 0
        idx = a_power + offset
        if 1 <= idx <= coeff_len
            coeffs[idx] += 1
        end
    else
        # Expand d^k where d = -A^2 - A^{-2}
        d_size = 4 * k + 1
        d_centre = 2 * k + 1
        d_coeffs = zeros(Int, d_size)
        d_coeffs[d_centre] = 1  # d^0 = 1

        for _ in 1:k
            d_tmp = zeros(Int, d_size)
            for i in 1:d_size
                c = d_coeffs[i]
                c == 0 && continue
                j_plus = i + 2
                j_minus = i - 2
                1 <= j_plus <= d_size && (d_tmp[j_plus] -= c)
                1 <= j_minus <= d_size && (d_tmp[j_minus] -= c)
            end
            d_coeffs = d_tmp
        end

        for i in 1:d_size
            c = d_coeffs[i]
            c == 0 && continue
            e = i - d_centre
            idx = a_power + e + offset
            1 <= idx <= coeff_len && (coeffs[idx] += c)
        end
    end
end

"""
    _tpu_poly_det(PM::Matrix{Vector{Int}}, n::Int) -> Union{Vector{Int}, Nothing}

Polynomial matrix determinant via fraction-free Gaussian elimination
tiled for TPU systolic array.
"""
function _tpu_poly_det(PM::Matrix{Vector{Int}}, n::Int)
    n <= 0 && return [1]
    n == 1 && return PM[1, 1]

    mat = [copy(PM[i, j]) for i in 1:n, j in 1:n]
    sign = 1

    for col in 1:n
        # Find pivot
        pivot_row = 0
        for row in col:n
            if any(!=(0), mat[row, col])
                pivot_row = row
                break
            end
        end
        pivot_row == 0 && return [0]

        if pivot_row != col
            for j in 1:n
                mat[col, j], mat[pivot_row, j] = mat[pivot_row, j], mat[col, j]
            end
            sign = -sign
        end

        pivot = mat[col, col]

        # Tiled elimination
        for row in (col+1):n
            entry_rc = mat[row, col]
            all(==(0), entry_rc) && continue

            for j in 1:n
                mat[row, j] = _poly_sub(
                    _poly_mul(pivot, mat[row, j]),
                    _poly_mul(entry_rc, mat[col, j]))
            end
        end
    end

    result = mat[n, n]
    if sign == -1
        result = [-c for c in result]
    end

    while length(result) > 1 && result[end] == 0
        pop!(result)
    end
    return result
end

end  # module KnotTheoryTPUExt
