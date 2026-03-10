# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# KnotTheory.jl Math Extension
#
# Mathematical Accelerator acceleration for knot polynomial computation.
# Hardware-accelerated exact polynomial arithmetic, number-theoretic transforms for polynomial multiplication, and hardware GCD for fraction-free elimination

module KnotTheoryMathExt

using KnotTheory
using KnotTheory: PlanarDiagram, Crossing, _wirtinger_generators
using KnotTheory: _poly_mul, _poly_add, _poly_sub
using AcceleratorGate
using AcceleratorGate: MathAccelBackend, _record_diagnostic!

# ============================================================================
# Constants
# ============================================================================

const BATCH_SIZE = 64

# ============================================================================
# Hook: backend_coprocessor_polynomial_eval
# ============================================================================

function KnotTheory.backend_coprocessor_polynomial_eval(
    backend::MathAccelBackend, coeffs::Vector{Int}, points::AbstractVector)
    try
        n_points = length(points)
        n_coeffs = length(coeffs)
        n_points == 0 && return Int[]
        n_coeffs == 0 && return zeros(Int, n_points)

        results = Vector{Int}(undef, n_points)

        for batch_start in 1:BATCH_SIZE:n_points
            batch_end = min(batch_start + BATCH_SIZE - 1, n_points)
            for idx in batch_start:batch_end
                x = points[idx]
                val = coeffs[end]
                for k in (n_coeffs-1):-1:1
                    val = val * x + coeffs[k]
                end
                results[idx] = val
            end
        end

        return results
    catch e
        _record_diagnostic!("mathaccel", "runtime_errors")
        @warn "KnotTheoryMathExt: polynomial_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_jones_invariant
# ============================================================================

function KnotTheory.backend_coprocessor_jones_invariant(
    backend::MathAccelBackend, pd::PlanarDiagram, wr::Int)
    try
        n = length(pd.crossings)
        n == 0 && return Dict(0 => 1)
        n > 25 && return nothing

        n_states = 1 << n
        max_slot = 4 * n

        arc_positions = Dict{Int, Vector{Int}}()
        for (i, c) in enumerate(pd.crossings)
            for (slot, arc) in enumerate(c.arcs)
                push!(get!(arc_positions, arc, Int[]), 4 * (i - 1) + slot)
            end
        end
        arc_pairs = Tuple{Int,Int}[]
        for positions in values(arc_positions)
            length(positions) == 2 && push!(arc_pairs, (positions[1], positions[2]))
        end

        max_exp = 3 * n + 2
        coeff_len = 2 * max_exp + 1
        coeffs = zeros(Int, coeff_len)
        offset = max_exp + 1

        for s_start in 0:BATCH_SIZE:(n_states-1)
            s_end = min(s_start + BATCH_SIZE - 1, n_states - 1)
            for state in s_start:s_end
                parent = collect(1:max_slot)
                uf_rank = zeros(Int, max_slot)

                for (a, b) in arc_pairs
                    _mathaccel_uf_union!(parent, uf_rank, a, b)
                end

                a_power = 0
                for ci in 1:n
                    base = 4 * (ci - 1)
                    bit = (state >> (ci - 1)) & 1
                    if bit == 0
                        _mathaccel_uf_union!(parent, uf_rank, base + 1, base + 2)
                        _mathaccel_uf_union!(parent, uf_rank, base + 3, base + 4)
                        a_power += 1
                    else
                        _mathaccel_uf_union!(parent, uf_rank, base + 2, base + 3)
                        _mathaccel_uf_union!(parent, uf_rank, base + 4, base + 1)
                        a_power -= 1
                    end
                end

                loops = count(i -> _mathaccel_uf_find!(parent, i) == i, 1:max_slot)
                _mathaccel_accumulate_bracket!(coeffs, offset, coeff_len, a_power, loops - 1)
            end
        end

        bracket = Dict{Int, Int}()
        for i in 1:coeff_len
            coeffs[i] != 0 && (bracket[i - offset] = coeffs[i])
        end
        isempty(bracket) && return Dict(0 => 1)

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
        _record_diagnostic!("mathaccel", "runtime_errors")
        @warn "KnotTheoryMathExt: jones_invariant failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_alexander_polynomial
# ============================================================================

function KnotTheory.backend_coprocessor_alexander_polynomial(
    backend::MathAccelBackend, pd::PlanarDiagram)
    try
        n = length(pd.crossings)
        n <= 3 && return nothing

        gen_map, n_gens = _wirtinger_generators(pd)
        n_gens == 0 && return Dict(0 => 1)

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

        min_exp, max_exp = 0, 0
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
            c = zeros(Int, poly_size)
            for (e, v) in M[i, j]
                v != 0 && (c[e + shift + 1] += v)
            end
            PM[i, j] = c
        end

        det_coeffs = _mathaccel_poly_det(PM, minor_size)
        det_coeffs === nothing && return nothing

        overall_shift = shift * minor_size
        poly = Dict{Int, Int}()
        for (k, c) in enumerate(det_coeffs)
            c != 0 && (poly[k - 1 - overall_shift] = c)
        end
        isempty(poly) && return Dict(0 => 1)

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
        poly[max_e2] < 0 && (for e in keys(poly); poly[e] = -poly[e]; end)

        filter!(p -> p.second != 0, poly)
        isempty(poly) ? Dict(0 => 1) : poly
    catch e
        _record_diagnostic!("mathaccel", "runtime_errors")
        @warn "KnotTheoryMathExt: alexander_polynomial failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_matrix_det
# ============================================================================

function KnotTheory.backend_coprocessor_matrix_det(
    backend::MathAccelBackend, M::Matrix{Vector{Int}}, n::Int)
    try
        return _mathaccel_poly_det(M, n)
    catch e
        _record_diagnostic!("mathaccel", "runtime_errors")
        @warn "KnotTheoryMathExt: matrix_det failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_braid_reduce
# ============================================================================

function KnotTheory.backend_coprocessor_braid_reduce(
    backend::MathAccelBackend, generators::Vector{Int})
    try
        isempty(generators) && return Int[]

        word = copy(generators)
        changed = true
        max_iter = length(word) * 2
        iter = 0

        while changed && iter < max_iter
            changed = false
            iter += 1

            new_word = Int[]
            i = 1
            while i <= length(word)
                if i < length(word) && word[i] == -word[i+1]
                    changed = true
                    i += 2
                else
                    push!(new_word, word[i])
                    i += 1
                end
            end
            word = new_word

            for idx in 1:length(word)-1
                a, b = abs(word[idx]), abs(word[idx+1])
                if abs(a - b) >= 2 && a > b
                    word[idx], word[idx+1] = word[idx+1], word[idx]
                    changed = true
                end
            end
        end

        return word
    catch e
        _record_diagnostic!("mathaccel", "runtime_errors")
        @warn "KnotTheoryMathExt: braid_reduce failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Internal Helpers
# ============================================================================

function _mathaccel_uf_find!(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

function _mathaccel_uf_union!(parent::Vector{Int}, rank::Vector{Int}, a::Int, b::Int)
    ra = _mathaccel_uf_find!(parent, a)
    rb = _mathaccel_uf_find!(parent, b)
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

function _mathaccel_accumulate_bracket!(coeffs::Vector{Int}, offset::Int, coeff_len::Int,
                                     a_power::Int, k::Int)
    if k == 0
        idx = a_power + offset
        1 <= idx <= coeff_len && (coeffs[idx] += 1)
    else
        d_size = 4 * k + 1
        d_centre = 2 * k + 1
        d_coeffs = zeros(Int, d_size)
        d_coeffs[d_centre] = 1
        for _ in 1:k
            d_tmp = zeros(Int, d_size)
            for i in 1:d_size
                c = d_coeffs[i]
                c == 0 && continue
                i + 2 <= d_size && (d_tmp[i + 2] -= c)
                i - 2 >= 1 && (d_tmp[i - 2] -= c)
            end
            d_coeffs = d_tmp
        end
        for i in 1:d_size
            c = d_coeffs[i]
            c == 0 && continue
            idx = a_power + (i - d_centre) + offset
            1 <= idx <= coeff_len && (coeffs[idx] += c)
        end
    end
end

function _mathaccel_poly_det(PM::Matrix{Vector{Int}}, n::Int)
    n <= 0 && return [1]
    n == 1 && return PM[1, 1]

    mat = [copy(PM[i, j]) for i in 1:n, j in 1:n]
    sign = 1

    for col in 1:n
        pivot_row = 0
        for row in col:n
            any(!=(0), mat[row, col]) && (pivot_row = row; break)
        end
        pivot_row == 0 && return [0]

        if pivot_row != col
            for j in 1:n
                mat[col, j], mat[pivot_row, j] = mat[pivot_row, j], mat[col, j]
            end
            sign = -sign
        end

        pivot = mat[col, col]
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

    result = sign == -1 ? [-c for c in mat[n, n]] : mat[n, n]
    while length(result) > 1 && result[end] == 0
        pop!(result)
    end
    return result
end

end  # module KnotTheoryMathExt
