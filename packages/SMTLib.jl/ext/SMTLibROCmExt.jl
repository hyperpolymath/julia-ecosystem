# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl ROCm Extension
# Provides GPU-accelerated SAT clause evaluation via AMD ROCm/HIP.
# Automatically loaded when AMDGPU.jl is imported alongside SMTLib.

module SMTLibROCmExt

using AMDGPU
using SMTLib
using AcceleratorGate

# ============================================================================
# Availability Detection
# ============================================================================

function AcceleratorGate.rocm_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_ROCM_AVAILABLE")
    forced !== nothing && return forced
    AMDGPU.functional()
end

function AcceleratorGate.rocm_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_ROCM_AVAILABLE", "AXIOM_ROCM_DEVICE_COUNT")
    forced !== nothing && return forced
    length(AMDGPU.devices())
end

# ============================================================================
# Device Capabilities
# ============================================================================

function AcceleratorGate.device_capabilities(b::ROCmBackend)
    !AMDGPU.functional() && return nothing
    dev = AMDGPU.devices()[b.device + 1]
    AcceleratorGate.DeviceCapabilities(
        b,
        64,   # CUs (typical)
        1500, # MHz (typical)
        Int64(16 * 1024^3), # 16GB typical
        Int64(14 * 1024^3), # estimated available
        256,   # max workgroup
        true,  # f64
        true,  # f16
        true,  # int8
        "AMD",
        "ROCm",
    )
end

function AcceleratorGate.estimate_cost(::ROCmBackend, op::Symbol, data_size::Int)
    overhead = 600.0
    op == :solve && return overhead + Float64(data_size) * 0.012
    op == :check_sat && return overhead + Float64(data_size) * 0.006
    op == :model_eval && return overhead + Float64(data_size) * 0.025
    op == :simplify && return overhead + Float64(data_size) * 0.035
    Inf
end

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    AcceleratorGate.register_operation!(ROCmBackend, :solve)
    AcceleratorGate.register_operation!(ROCmBackend, :check_sat)
    AcceleratorGate.register_operation!(ROCmBackend, :model_eval)
    AcceleratorGate.register_operation!(ROCmBackend, :simplify)
end

# ============================================================================
# ROCm-Accelerated SAT Operations
# ============================================================================

"""
    backend_solve(::ROCmBackend, clauses, variables, config)

Parallel SAT solving on AMD GPUs via ROCm. Each wavefront evaluates a
different candidate assignment against the clause set.
"""
function SMTLib.backend_solve(::ROCmBackend, clauses::AbstractVector,
                              variables::AbstractVector, config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    n_clauses = length(clauses)
    max_clause_len = maximum(length(c) for c in clauses)

    clause_matrix = zeros(Int32, n_clauses, max_clause_len)
    for (i, clause) in enumerate(clauses)
        for (j, lit) in enumerate(clause)
            clause_matrix[i, j] = Int32(lit)
        end
    end
    clause_gpu = ROCArray(clause_matrix)

    n_trials = min(get(config, :trials, 2^min(n_vars, 20)), 2^20)
    assignments_gpu = AMDGPU.rand(Float32, n_trials, n_vars) .> 0.5f0

    # Evaluate each trial: count satisfied clauses
    sat_counts = AMDGPU.zeros(Int32, n_trials)

    for c_idx in 1:n_clauses
        clause_satisfied = AMDGPU.zeros(Bool, n_trials)
        for k in 1:max_clause_len
            lit = clause_matrix[c_idx, k]
            lit == 0 && break
            var_idx = abs(lit)
            var_idx > n_vars && continue
            if lit > 0
                clause_satisfied .|= assignments_gpu[:, var_idx]
            else
                clause_satisfied .|= .!assignments_gpu[:, var_idx]
            end
        end
        sat_counts .+= Int32.(clause_satisfied)
    end

    host_counts = Array(sat_counts)
    for i in 1:n_trials
        if host_counts[i] == n_clauses
            assignment = Array(assignments_gpu[i, :])
            model = Dict(variables[j] => Bool(assignment[j]) for j in 1:n_vars)
            return (status=:sat, model=model, trials_checked=i)
        end
    end

    return (status=:unknown, model=nothing, trials_checked=n_trials)
end

function SMTLib.backend_check_sat(::ROCmBackend, clauses::AbstractVector, n_vars::Int)
    n_clauses = length(clauses)
    max_clause_len = maximum(length(c) for c in clauses)
    n_batch = min(2^min(n_vars, 18), 262144)

    assignments_gpu = AMDGPU.rand(Float32, n_batch, n_vars) .> 0.5f0
    sat_counts = AMDGPU.zeros(Int32, n_batch)

    for c_idx in 1:n_clauses
        clause_satisfied = AMDGPU.zeros(Bool, n_batch)
        for k in 1:max_clause_len
            lit = clauses[c_idx][k > length(clauses[c_idx]) ? 1 : k]
            k > length(clauses[c_idx]) && break
            var_idx = abs(lit)
            var_idx > n_vars && continue
            if lit > 0
                clause_satisfied .|= assignments_gpu[:, var_idx]
            else
                clause_satisfied .|= .!assignments_gpu[:, var_idx]
            end
        end
        sat_counts .+= Int32.(clause_satisfied)
    end

    host_counts = Array(sat_counts)
    any(c -> c >= n_clauses, host_counts) && return :sat
    return :unknown
end

function SMTLib.backend_model_eval(::ROCmBackend, formula_matrix::AbstractMatrix,
                                   model_batch::AbstractMatrix)
    f_gpu = ROCArray(Float32.(formula_matrix))
    m_gpu = ROCArray(Float32.(model_batch))
    results_gpu = m_gpu * f_gpu
    return Array(results_gpu)
end

function SMTLib.backend_simplify(::ROCmBackend, expression_batch::AbstractMatrix)
    expr_gpu = ROCArray(Float32.(expression_batch))
    threshold = 1.0f-10
    simplified = expr_gpu .* (abs.(expr_gpu) .> threshold)
    row_maxes = maximum(abs.(simplified), dims=2)
    row_maxes = max.(row_maxes, 1.0f-30)
    normalized = simplified ./ row_maxes
    return Array(normalized)
end

end # module SMTLibROCmExt
