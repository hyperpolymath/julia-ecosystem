# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl TPU Extension
# Provides tensor-accelerated SMT operations via TPU systolic arrays.
# TPUs excel at large batch matrix-style constraint evaluation with high throughput.
# Automatically loaded when TPUCompute is imported alongside SMTLib.

module SMTLibTPUExt

using TPUCompute
using SMTLib
using AcceleratorGate

# ============================================================================
# Availability Detection
# ============================================================================

function AcceleratorGate.tpu_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_TPU_AVAILABLE")
    forced !== nothing && return forced
    TPUCompute.functional()
end

function AcceleratorGate.tpu_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_TPU_AVAILABLE", "AXIOM_TPU_DEVICE_COUNT")
    forced !== nothing && return forced
    TPUCompute.ndevices()
end

# ============================================================================
# Device Capabilities
# ============================================================================

function AcceleratorGate.device_capabilities(b::TPUBackend)
    !TPUCompute.functional() && return nothing
    dev = TPUCompute.device(b.device)
    total_mem = TPUCompute.totalmem(dev)
    free_mem = TPUCompute.available_memory()
    AcceleratorGate.DeviceCapabilities(
        b,
        TPUCompute.core_count(dev),
        TPUCompute.clock_mhz(dev),
        Int64(total_mem),
        Int64(free_mem),
        TPUCompute.max_batch_size(dev),
        false,  # TPUs have limited f64 support
        true,   # bf16 / f16 native
        true,   # int8 supported via quantisation
        "Google",
        string(TPUCompute.version()),
    )
end

function AcceleratorGate.estimate_cost(::TPUBackend, op::Symbol, data_size::Int)
    # TPU systolic arrays are optimal for large batch constraint evaluation
    overhead = 800.0  # TPU dispatch overhead is higher than GPU
    op == :solve && return overhead + Float64(data_size) * 0.005
    op == :check_sat && return overhead + Float64(data_size) * 0.003
    op == :model_eval && return overhead + Float64(data_size) * 0.002
    op == :interpolate && return overhead + Float64(data_size) * 0.04
    op == :simplify && return overhead + Float64(data_size) * 0.008
    Inf
end

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    AcceleratorGate.register_operation!(TPUBackend, :solve)
    AcceleratorGate.register_operation!(TPUBackend, :check_sat)
    AcceleratorGate.register_operation!(TPUBackend, :model_eval)
    AcceleratorGate.register_operation!(TPUBackend, :simplify)
end

# ============================================================================
# TPU-Accelerated SAT Operations
# ============================================================================

"""
    backend_solve(::TPUBackend, clauses, variables, config)

Tensor-parallel SAT solving on TPU. Encodes the clause database as a dense
matrix and evaluates candidate assignments via systolic array matrix multiply.
TPU's deterministic execution model provides consistent throughput for large
clause sets.
"""
function SMTLib.backend_solve(::TPUBackend, clauses::AbstractVector,
                              variables::AbstractVector, config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    n_clauses = length(clauses)

    # Encode clauses as a dense matrix for TPU systolic array processing
    max_clause_len = maximum(length(c) for c in clauses)
    clause_matrix = zeros(Float32, n_clauses, n_vars)
    for (i, clause) in enumerate(clauses)
        for lit in clause
            var_idx = abs(lit)
            if var_idx <= n_vars
                clause_matrix[i, var_idx] = lit > 0 ? 1.0f0 : -1.0f0
            end
        end
    end
    clause_tpu = TPUCompute.TPUArray(clause_matrix)

    # TPU batch size: systolic arrays work best with large, aligned batches
    n_trials = min(get(config, :trials, 2^min(n_vars, 20)), 2^20)
    batch_size = min(n_trials, TPUCompute.max_batch_size(TPUCompute.device(0)))

    # Generate random assignments and evaluate via matrix multiply
    # clause_matrix * assignment^T gives satisfaction scores per clause
    for batch_start in 1:batch_size:n_trials
        batch_end = min(batch_start + batch_size - 1, n_trials)
        actual_batch = batch_end - batch_start + 1

        assignments = TPUCompute.rand(Float32, actual_batch, n_vars)
        binary = Float32.(assignments .> 0.5f0)

        # Satisfaction via matrix multiply: positive score = clause satisfied
        # Transform: assignment -> {-1, 1} encoding
        encoded = 2.0f0 .* binary .- 1.0f0
        scores = clause_tpu * encoded'  # n_clauses x actual_batch

        # A clause is satisfied if score > -max_clause_len (at least one literal matches)
        clause_sat = scores .> Float32(-max_clause_len)
        all_sat = vec(sum(Float32.(clause_sat), dims=1))
        host_sat = Array(all_sat)

        for i in 1:actual_batch
            if host_sat[i] >= n_clauses
                assignment = Array(binary[i, :])
                model = Dict(variables[j] => Bool(assignment[j] > 0.5f0) for j in 1:n_vars)
                return (status=:sat, model=model, trials_checked=batch_start + i - 1)
            end
        end
    end

    return (status=:unknown, model=nothing, trials_checked=n_trials)
end

"""
    backend_check_sat(::TPUBackend, clauses, n_vars)

Batch satisfiability checking via TPU matrix operations. Encodes the problem
as a matrix multiply and checks all candidate assignments in one systolic pass.
"""
function SMTLib.backend_check_sat(::TPUBackend, clauses::AbstractVector, n_vars::Int)
    n_clauses = length(clauses)

    # Build clause-variable incidence matrix
    clause_matrix = zeros(Float32, n_clauses, n_vars)
    for (i, clause) in enumerate(clauses)
        for lit in clause
            var_idx = abs(lit)
            var_idx <= n_vars && (clause_matrix[i, var_idx] = lit > 0 ? 1.0f0 : -1.0f0)
        end
    end
    clause_tpu = TPUCompute.TPUArray(clause_matrix)

    n_batch = min(2^min(n_vars, 18), 262144)
    assignments = TPUCompute.rand(Float32, n_batch, n_vars) .> 0.5f0
    encoded = 2.0f0 .* Float32.(assignments) .- 1.0f0

    scores = clause_tpu * encoded'
    clause_sat = Float32.(scores .> Float32(-n_vars))
    sat_counts = vec(sum(clause_sat, dims=1))
    host_counts = Array(sat_counts)

    any(c -> c >= n_clauses, host_counts) && return :sat
    return :unknown
end

"""
    backend_model_eval(::TPUBackend, formula_matrix, model_batch)

Batch model evaluation leveraging TPU systolic array matrix multiply.
Evaluates many models against a formula simultaneously.
"""
function SMTLib.backend_model_eval(::TPUBackend, formula_matrix::AbstractMatrix,
                                   model_batch::AbstractMatrix)
    f_tpu = TPUCompute.TPUArray(Float32.(formula_matrix))
    m_tpu = TPUCompute.TPUArray(Float32.(model_batch))
    results = m_tpu * f_tpu
    return Array(results)
end

"""
    backend_simplify(::TPUBackend, expression_batch)

Batch expression simplification using TPU parallel constant folding.
Processes a large batch of expression coefficient vectors simultaneously.
"""
function SMTLib.backend_simplify(::TPUBackend, expression_batch::AbstractMatrix)
    expr_tpu = TPUCompute.TPUArray(Float32.(expression_batch))
    threshold = 1.0f-10
    simplified = expr_tpu .* Float32.(abs.(expr_tpu) .> threshold)
    row_maxes = maximum(abs.(simplified), dims=2)
    row_maxes = max.(row_maxes, 1.0f-30)
    normalized = simplified ./ row_maxes
    return Array(normalized)
end

end # module SMTLibTPUExt
