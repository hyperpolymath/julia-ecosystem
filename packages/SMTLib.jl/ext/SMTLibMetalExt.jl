# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl Metal Extension
# Provides GPU-accelerated SAT clause evaluation via Apple Metal.
# Automatically loaded when Metal.jl is imported alongside SMTLib.

module SMTLibMetalExt

using Metal
using SMTLib
using AcceleratorGate

# ============================================================================
# Availability Detection
# ============================================================================

function AcceleratorGate.metal_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_METAL_AVAILABLE")
    forced !== nothing && return forced
    Metal.functional()
end

function AcceleratorGate.metal_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_METAL_AVAILABLE", "AXIOM_METAL_DEVICE_COUNT")
    forced !== nothing && return forced
    Metal.functional() ? 1 : 0
end

# ============================================================================
# Device Capabilities
# ============================================================================

function AcceleratorGate.device_capabilities(b::MetalBackend)
    !Metal.functional() && return nothing
    AcceleratorGate.DeviceCapabilities(
        b,
        32,   # GPU cores (Apple Silicon typical)
        1398, # MHz
        Int64(16 * 1024^3),
        Int64(12 * 1024^3),
        1024,
        false, # Metal does not support f64 natively
        true,  # f16
        true,  # int8
        "Apple",
        "Metal",
    )
end

function AcceleratorGate.estimate_cost(::MetalBackend, op::Symbol, data_size::Int)
    overhead = 400.0
    op == :solve && return overhead + Float64(data_size) * 0.015
    op == :check_sat && return overhead + Float64(data_size) * 0.008
    op == :model_eval && return overhead + Float64(data_size) * 0.03
    op == :simplify && return overhead + Float64(data_size) * 0.04
    Inf
end

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    AcceleratorGate.register_operation!(MetalBackend, :solve)
    AcceleratorGate.register_operation!(MetalBackend, :check_sat)
    AcceleratorGate.register_operation!(MetalBackend, :model_eval)
    AcceleratorGate.register_operation!(MetalBackend, :simplify)
end

# ============================================================================
# Metal-Accelerated SAT Operations
# ============================================================================

"""
    backend_solve(::MetalBackend, clauses, variables, config)

Parallel SAT solving on Apple Silicon GPUs. Uses Metal compute shaders
for clause evaluation across threadgroups.
"""
function SMTLib.backend_solve(::MetalBackend, clauses::AbstractVector,
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
    clause_gpu = MtlArray(clause_matrix)

    n_trials = min(get(config, :trials, 2^min(n_vars, 18)), 2^18)
    # Metal uses Float32 - generate random assignments
    assignments_gpu = Metal.rand(Float32, n_trials, n_vars) .> 0.5f0

    sat_counts = Metal.zeros(Int32, n_trials)

    for c_idx in 1:n_clauses
        clause_satisfied = Metal.zeros(Int32, n_trials)
        for k in 1:max_clause_len
            lit = clause_matrix[c_idx, k]
            lit == 0 && break
            var_idx = abs(lit)
            var_idx > n_vars && continue
            if lit > 0
                clause_satisfied .+= Int32.(assignments_gpu[:, var_idx])
            else
                clause_satisfied .+= Int32.(.!assignments_gpu[:, var_idx])
            end
        end
        # Clause is satisfied if any literal was true
        sat_counts .+= Int32.(clause_satisfied .> Int32(0))
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

function SMTLib.backend_check_sat(::MetalBackend, clauses::AbstractVector, n_vars::Int)
    n_clauses = length(clauses)
    n_batch = min(2^min(n_vars, 16), 65536)
    assignments_gpu = Metal.rand(Float32, n_batch, n_vars) .> 0.5f0

    sat_counts = Metal.zeros(Int32, n_batch)
    for (c_idx, clause) in enumerate(clauses)
        clause_satisfied = Metal.zeros(Int32, n_batch)
        for lit in clause
            var_idx = abs(lit)
            var_idx > n_vars && continue
            if lit > 0
                clause_satisfied .+= Int32.(assignments_gpu[:, var_idx])
            else
                clause_satisfied .+= Int32.(.!assignments_gpu[:, var_idx])
            end
        end
        sat_counts .+= Int32.(clause_satisfied .> Int32(0))
    end

    host_counts = Array(sat_counts)
    any(c -> c >= n_clauses, host_counts) && return :sat
    return :unknown
end

function SMTLib.backend_model_eval(::MetalBackend, formula_matrix::AbstractMatrix,
                                   model_batch::AbstractMatrix)
    f_gpu = MtlArray(Float32.(formula_matrix))
    m_gpu = MtlArray(Float32.(model_batch))
    return Array(m_gpu * f_gpu)
end

function SMTLib.backend_simplify(::MetalBackend, expression_batch::AbstractMatrix)
    expr_gpu = MtlArray(Float32.(expression_batch))
    threshold = 1.0f-10
    simplified = expr_gpu .* (abs.(expr_gpu) .> threshold)
    row_maxes = maximum(abs.(simplified), dims=2)
    row_maxes = max.(row_maxes, 1.0f-30)
    return Array(simplified ./ row_maxes)
end

end # module SMTLibMetalExt
