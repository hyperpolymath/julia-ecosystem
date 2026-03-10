# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl VPU Coprocessor
# SIMD-vectorized clause evaluation using Vector Processing Units.
# Exploits wide SIMD lanes to evaluate multiple clauses per instruction.

function AcceleratorGate.device_capabilities(b::VPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 8, 2000,
        Int64(2 * 1024^3), Int64(2 * 1024^3),
        512, true, true, true, "Intel", "VPU AVX-512",
    )
end

function AcceleratorGate.estimate_cost(::VPUBackend, op::Symbol, data_size::Int)
    overhead = 10.0
    op == :solve && return overhead + Float64(data_size) * 0.05
    op == :check_sat && return overhead + Float64(data_size) * 0.03
    op == :model_eval && return overhead + Float64(data_size) * 0.04
    op == :simplify && return overhead + Float64(data_size) * 0.02
    Inf
end

AcceleratorGate.register_operation!(VPUBackend, :solve)
AcceleratorGate.register_operation!(VPUBackend, :check_sat)
AcceleratorGate.register_operation!(VPUBackend, :model_eval)
AcceleratorGate.register_operation!(VPUBackend, :simplify)

const _VPU_SIMD_LANES = 8

"""SIMD-batched clause evaluation: processes _VPU_SIMD_LANES clauses per iteration."""
function _vpu_simd_evaluate(pos_masks::Vector{UInt64}, neg_masks::Vector{UInt64},
                             assignment::UInt64, n_padded::Int)
    neg = ~assignment
    i = 1
    while i + _VPU_SIMD_LANES - 1 <= n_padded
        ok = true
        @simd for lane in 0:(_VPU_SIMD_LANES - 1)
            c = i + lane
            ((pos_masks[c] & assignment) | (neg_masks[c] & neg)) == UInt64(0) && (ok = false)
        end
        !ok && return false
        i += _VPU_SIMD_LANES
    end
    while i <= n_padded
        ((pos_masks[i] & assignment) | (neg_masks[i] & neg)) == UInt64(0) && return false
        i += 1
    end
    return true
end

"""Encode clauses as bitmasks aligned to SIMD lane width."""
function _vpu_encode(clauses::AbstractVector, n_vars::Int)
    nc = length(clauses)
    padded = cld(nc, _VPU_SIMD_LANES) * _VPU_SIMD_LANES
    pos = zeros(UInt64, padded); neg = zeros(UInt64, padded)
    for i in (nc+1):padded; pos[i] = typemax(UInt64); end
    for (i, clause) in enumerate(clauses)
        for lit in clause
            vi = abs(lit); (vi > n_vars || vi > 64) && continue
            bit = UInt64(1) << (vi - 1)
            lit > 0 ? (pos[i] |= bit) : (neg[i] |= bit)
        end
    end
    return pos, neg, padded
end

"""SIMD-vectorized SAT solving for problems with <= 64 variables."""
function backend_coprocessor_solve(::VPUBackend, clauses::AbstractVector,
                                   variables::AbstractVector,
                                   config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    n_vars > 64 && return backend_solve(JuliaBackend(), clauses, variables, config)
    pos, neg, padded = _vpu_encode(clauses, n_vars)
    mask = (UInt64(1) << n_vars) - UInt64(1)
    n_trials = min(get(config, :trials, 2^min(n_vars, 22)), 4194304)
    for trial in 1:n_trials
        a = rand(UInt64) & mask
        if _vpu_simd_evaluate(pos, neg, a, padded)
            model = Dict(variables[j] => (a >> (j-1)) & UInt64(1) == UInt64(1) for j in 1:n_vars)
            return (status=:sat, model=model, trials_checked=trial)
        end
    end
    return (status=:unknown, model=nothing, trials_checked=n_trials)
end

function backend_coprocessor_check_sat(::VPUBackend, clauses::AbstractVector, n_vars::Int)
    n_vars > 64 && return backend_check_sat(JuliaBackend(), clauses, n_vars)
    pos, neg, padded = _vpu_encode(clauses, n_vars)
    mask = (UInt64(1) << n_vars) - UInt64(1)
    for _ in 1:min(2^min(n_vars, 22), 4194304)
        _vpu_simd_evaluate(pos, neg, rand(UInt64) & mask, padded) && return :sat
    end
    :unknown
end

function backend_coprocessor_model_eval(::VPUBackend, formula_matrix::AbstractMatrix,
                                        model_batch::AbstractMatrix)
    Float64.(model_batch) * Float64.(formula_matrix)
end

function backend_coprocessor_simplify(::VPUBackend, expression_batch::AbstractMatrix)
    result = Float64.(expression_batch)
    @simd for i in eachindex(result)
        abs(result[i]) < 1.0e-10 && (result[i] = 0.0)
    end
    for row in 1:size(result, 1)
        rm = 0.0
        @simd for j in 1:size(result, 2); rm = max(rm, abs(result[row, j])); end
        if rm > 1.0e-30
            inv_rm = 1.0 / rm
            @simd for j in 1:size(result, 2); result[row, j] *= inv_rm; end
        end
    end
    result
end
