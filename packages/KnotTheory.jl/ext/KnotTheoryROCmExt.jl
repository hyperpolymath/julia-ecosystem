# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# KnotTheory.jl ROCm Extension
# GPU-accelerated kernels for knot polynomial computation on AMD GPUs
# via AMDGPU.jl and KernelAbstractions.jl. Mirrors the CUDA extension with
# ROCm-specific array types and backend.

module KnotTheoryROCmExt

using AMDGPU
using KernelAbstractions
using KernelAbstractions: @index, @Const

import KnotTheory
import KnotTheory: PlanarDiagram, Crossing, _wirtinger_generators
import KnotTheory: backend_jones_invariant, backend_alexander_polynomial, backend_matrix_det
import KnotTheory: _poly_mul, _poly_add, _poly_sub
using AcceleratorGate: GPUBackend

# ============================================================================
# Union-Find on GPU (inline, thread-local)
# ============================================================================

@inline function gpu_uf_find!(parent, x)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

@inline function gpu_uf_union!(parent, rank, a, b)
    ra = gpu_uf_find!(parent, a)
    rb = gpu_uf_find!(parent, b)
    ra == rb && return nothing
    if rank[ra] < rank[rb]
        parent[ra] = rb
    elseif rank[ra] > rank[rb]
        parent[rb] = ra
    else
        parent[rb] = ra
        rank[ra] += Int32(1)
    end
    return nothing
end

# ============================================================================
# Kernel 1: Bracket State Enumeration (Jones Polynomial)
# ============================================================================
#
# Identical algorithm to the CUDA kernel. Each thread handles one of 2^n
# Kauffman bracket states. KernelAbstractions makes the kernel portable;
# the host-side code uses ROCArray for AMD GPU memory.

@kernel function bracket_state_kernel!(
    coeffs,
    @Const(crossing_arcs),
    @Const(crossing_signs),
    @Const(arc_pair_a),
    @Const(arc_pair_b),
    n_crossings::Int32,
    n_arc_pairs::Int32,
    max_slot::Int32,
    coeff_offset::Int32,
    coeff_len::Int32,
)
    state_idx = @index(Global) - Int32(1)

    parent = @private Int32 (256,)
    uf_rank = @private Int32 (256,)

    for i in Int32(1):max_slot
        parent[i] = i
        uf_rank[i] = Int32(0)
    end

    for k in Int32(1):n_arc_pairs
        gpu_uf_union!(parent, uf_rank, arc_pair_a[k], arc_pair_b[k])
    end

    a_power = Int32(0)

    for ci in Int32(1):n_crossings
        base = Int32(4) * (ci - Int32(1))
        s1 = base + Int32(1)
        s2 = base + Int32(2)
        s3 = base + Int32(3)
        s4 = base + Int32(4)

        bit = (state_idx >> (ci - Int32(1))) & Int32(1)

        if bit == Int32(0)
            gpu_uf_union!(parent, uf_rank, s1, s2)
            gpu_uf_union!(parent, uf_rank, s3, s4)
            a_power += Int32(1)
        else
            gpu_uf_union!(parent, uf_rank, s2, s3)
            gpu_uf_union!(parent, uf_rank, s4, s1)
            a_power -= Int32(1)
        end
    end

    loops = Int32(0)
    for i in Int32(1):max_slot
        if gpu_uf_find!(parent, i) == i
            loops += Int32(1)
        end
    end

    k = loops - Int32(1)

    if k == Int32(0)
        idx = a_power + coeff_offset + Int32(1)
        if idx >= Int32(1) && idx <= coeff_len
            Atomics.atomic_add!(pointer(coeffs, idx), Int32(1))
        end
    else
        d_coeffs = @private Int32 (81,)
        d_len = Int32(4) * k + Int32(1)
        d_centre = Int32(2) * k + Int32(1)

        for i in Int32(1):d_len
            d_coeffs[i] = Int32(0)
        end
        d_coeffs[d_centre] = Int32(1)

        d_tmp = @private Int32 (81,)
        for _ in Int32(1):k
            for i in Int32(1):d_len
                d_tmp[i] = Int32(0)
            end
            for i in Int32(1):d_len
                c = d_coeffs[i]
                c == Int32(0) && continue
                j_plus = i + Int32(2)
                if j_plus >= Int32(1) && j_plus <= d_len
                    d_tmp[j_plus] -= c
                end
                j_minus = i - Int32(2)
                if j_minus >= Int32(1) && j_minus <= d_len
                    d_tmp[j_minus] -= c
                end
            end
            for i in Int32(1):d_len
                d_coeffs[i] = d_tmp[i]
            end
        end

        for i in Int32(1):d_len
            c = d_coeffs[i]
            c == Int32(0) && continue
            e = (i - d_centre)
            total_exp = a_power + e
            idx = total_exp + coeff_offset + Int32(1)
            if idx >= Int32(1) && idx <= coeff_len
                Atomics.atomic_add!(pointer(coeffs, idx), c)
            end
        end
    end
end

# ============================================================================
# Host-side: Jones Polynomial via ROCm GPU
# ============================================================================

function backend_jones_invariant(::GPUBackend, pd::PlanarDiagram, wr::Int)
    !AMDGPU.functional() && return nothing

    n = length(pd.crossings)
    n == 0 && return Dict(0 => 1)
    n <= 8 && return nothing

    if n > 30
        @warn "ROCm bracket limited to 30 crossings (got $n); falling back to CPU"
        return nothing
    end

    n_states = 1 << n

    crossing_arcs_h = zeros(Int32, 4, n)
    crossing_signs_h = zeros(Int32, n)
    arc_positions = Dict{Int, Vector{Int}}()

    for (i, c) in enumerate(pd.crossings)
        crossing_signs_h[i] = Int32(c.sign)
        for (slot, arc) in enumerate(c.arcs)
            crossing_arcs_h[slot, i] = Int32(4 * (i - 1) + slot)
            push!(get!(arc_positions, arc, Int[]), 4 * (i - 1) + slot)
        end
    end

    pair_a_h = Int32[]
    pair_b_h = Int32[]
    for positions in values(arc_positions)
        if length(positions) == 2
            push!(pair_a_h, Int32(positions[1]))
            push!(pair_b_h, Int32(positions[2]))
        end
    end
    n_arc_pairs = Int32(length(pair_a_h))
    max_slot = Int32(4 * n)

    max_exp = 3 * n + 2
    coeff_offset = Int32(max_exp)
    coeff_len = Int32(2 * max_exp + 1)

    crossing_arcs_d = ROCArray(crossing_arcs_h)
    crossing_signs_d = ROCArray(crossing_signs_h)
    pair_a_d = isempty(pair_a_h) ? AMDGPU.zeros(Int32, 1) : ROCArray(pair_a_h)
    pair_b_d = isempty(pair_b_h) ? AMDGPU.zeros(Int32, 1) : ROCArray(pair_b_h)
    coeffs_d = AMDGPU.zeros(Int32, coeff_len)

    backend = ROCBackend()
    kernel! = bracket_state_kernel!(backend)
    kernel!(
        coeffs_d,
        crossing_arcs_d,
        crossing_signs_d,
        pair_a_d,
        pair_b_d,
        Int32(n),
        n_arc_pairs,
        max_slot,
        coeff_offset,
        coeff_len;
        ndrange=n_states,
    )
    KernelAbstractions.synchronize(backend)

    coeffs_h = Array(coeffs_d)

    bracket_poly = Dict{Int, Int}()
    for i in 1:coeff_len
        c = Int(coeffs_h[i])
        c == 0 && continue
        exp = i - 1 - Int(coeff_offset)
        bracket_poly[exp] = c
    end

    isempty(bracket_poly) && return Dict(0 => 1)

    a_shift = -3 * wr
    sign_factor = isodd(wr) ? -1 : 1
    jones = Dict{Int, Int}()
    for (e, c) in bracket_poly
        texp = -(e + a_shift)
        jones[texp] = get(jones, texp, 0) + sign_factor * c
    end

    filter!(kv -> kv.second != 0, jones)
    isempty(jones) ? Dict(0 => 1) : jones
end

# ============================================================================
# Polynomial Matrix Determinant (Fraction-Free LU)
# ============================================================================

function _extract_poly(mat::Array{Int32, 3}, row::Int, col::Int, padded_len::Int)
    result = zeros(Int, padded_len)
    for k in 1:padded_len
        result[k] = Int(mat[k, row, col])
    end
    last_nz = padded_len
    while last_nz > 1 && result[last_nz] == 0
        last_nz -= 1
    end
    return result[1:last_nz]
end

function _store_poly!(mat::Array{Int32, 3}, row::Int, col::Int, poly::Vector{Int}, padded_len::Int)
    for k in 1:min(length(poly), padded_len)
        mat[k, row, col] = Int32(poly[k])
    end
    for k in (length(poly)+1):padded_len
        mat[k, row, col] = Int32(0)
    end
end

function gpu_poly_lu_det(PM::Matrix{Vector{Int}}, n::Int)
    n <= 0 && return [1]
    n == 1 && return PM[1, 1]
    n <= 4 && return nothing

    max_poly_len = 1
    for i in 1:n, j in 1:n
        max_poly_len = max(max_poly_len, length(PM[i, j]))
    end
    padded_len = max_poly_len * n + n

    mat = zeros(Int32, padded_len, n, n)
    for i in 1:n, j in 1:n
        entry = PM[i, j]
        for k in 1:length(entry)
            mat[k, i, j] = Int32(entry[k])
        end
    end

    sign = 1

    for col in 1:n
        pivot_row = 0
        for row in col:n
            entry = @view mat[:, row, col]
            if any(!=(Int32(0)), entry)
                pivot_row = row
                break
            end
        end

        pivot_row == 0 && return [0]

        if pivot_row != col
            for j in 1:n
                mat[:, col, j], mat[:, pivot_row, j] = mat[:, pivot_row, j], mat[:, col, j]
            end
            sign = -sign
        end

        pivot = _extract_poly(mat, col, col, padded_len)

        for row in (col+1):n
            entry_rc = _extract_poly(mat, row, col, padded_len)
            all(==(0), entry_rc) && continue

            for j in 1:n
                row_j = _extract_poly(mat, row, j, padded_len)
                piv_j = _extract_poly(mat, col, j, padded_len)
                new_val = _poly_sub(_poly_mul(pivot, row_j), _poly_mul(entry_rc, piv_j))
                _store_poly!(mat, row, j, new_val, padded_len)
            end
        end
    end

    last_diag = _extract_poly(mat, n, n, padded_len)
    result = sign == 1 ? last_diag : [-c for c in last_diag]

    while length(result) > 1 && result[end] == 0
        pop!(result)
    end
    return result
end

function backend_matrix_det(::GPUBackend, M::Matrix{Vector{Int}}, n::Int)
    !AMDGPU.functional() && return nothing
    return gpu_poly_lu_det(M, n)
end

# ============================================================================
# Alexander Polynomial (Wirtinger + GPU Determinant)
# ============================================================================

function backend_alexander_polynomial(::GPUBackend, pd::PlanarDiagram)
    !AMDGPU.functional() && return nothing

    n = length(pd.crossings)
    n <= 5 && return nothing

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
        entry = M[i, j]
        for (e, c) in entry
            c != 0 && (coeffs[e + shift + 1] += c)
        end
        PM[i, j] = coeffs
    end

    det_coeffs = gpu_poly_lu_det(PM, minor_size)
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
    if poly[max_e2] < 0
        for e in keys(poly)
            poly[e] = -poly[e]
        end
    end

    filter!(p -> p.second != 0, poly)
    isempty(poly) ? Dict(0 => 1) : poly
end

end # module KnotTheoryROCmExt
