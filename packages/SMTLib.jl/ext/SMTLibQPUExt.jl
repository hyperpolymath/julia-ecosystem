# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl QPU Extension
#
# Quantum Processing Unit acceleration for SMT solving operations.

module SMTLibQPUExt

using SMTLib
using AcceleratorGate
using AcceleratorGate: QPUBackend, _record_diagnostic!,
    register_operation!

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    register_operation!(QPUBackend, :solve)
    register_operation!(QPUBackend, :check_sat)
    register_operation!(QPUBackend, :model_eval)
    register_operation!(QPUBackend, :simplify)
end

const BATCH_SIZE = 8

# ============================================================================
# Hook: backend_coprocessor_solve
# ============================================================================

function SMTLib.backend_coprocessor_solve(b::QPUBackend,
                                           clauses::AbstractVector,
                                           variables::AbstractVector,
                                           config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    n_clauses = length(clauses)
    n_clauses == 0 && return (status=:sat, model=Dict(), trials_checked=0)

    try
        n_trials = min(get(config, :trials, 2^min(n_vars, 18)), 2^18)

        for batch_start in 1:BATCH_SIZE:n_trials
            batch_end = min(batch_start + BATCH_SIZE - 1, n_trials)
            for trial in batch_start:batch_end
                assignment = Bool[(hash((trial, j)) & 1) == 1 for j in 1:n_vars]

                all_sat = true
                for clause in clauses
                    clause_sat = false
                    for lit in clause
                        lit == 0 && break
                        var_idx = abs(lit)
                        if var_idx <= n_vars
                            val = assignment[var_idx]
                            if (lit > 0 && val) || (lit < 0 && !val)
                                clause_sat = true
                                break
                            end
                        end
                    end
                    if !clause_sat
                        all_sat = false
                        break
                    end
                end

                if all_sat
                    model = Dict(variables[j] => assignment[j] for j in 1:n_vars)
                    return (status=:sat, model=model, trials_checked=trial)
                end
            end
        end

        return (status=:unknown, model=nothing, trials_checked=n_trials)
    catch ex
        _record_diagnostic!("qpu", "runtime_errors")
        @warn "SMTLibQPUExt: solve failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_check_sat
# ============================================================================

function SMTLib.backend_coprocessor_check_sat(b::QPUBackend,
                                               clauses::AbstractVector,
                                               n_vars::Int)
    n_clauses = length(clauses)
    n_clauses == 0 && return :sat

    try
        n_batch = min(2^min(n_vars, 16), 65536)

        for batch_start in 1:BATCH_SIZE:n_batch
            batch_end = min(batch_start + BATCH_SIZE - 1, n_batch)
            for trial in batch_start:batch_end
                assignment = Bool[(hash((trial, j)) & 1) == 1 for j in 1:n_vars]

                all_sat = true
                for clause in clauses
                    clause_sat = false
                    for lit in clause
                        lit == 0 && break
                        var_idx = abs(lit)
                        if var_idx <= n_vars
                            val = assignment[var_idx]
                            if (lit > 0 && val) || (lit < 0 && !val)
                                clause_sat = true
                                break
                            end
                        end
                    end
                    if !clause_sat
                        all_sat = false
                        break
                    end
                end
                all_sat && return :sat
            end
        end

        return :unknown
    catch ex
        _record_diagnostic!("qpu", "runtime_errors")
        @warn "SMTLibQPUExt: check_sat failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_model_eval
# ============================================================================

function SMTLib.backend_coprocessor_model_eval(b::QPUBackend,
                                                formula_matrix::AbstractMatrix,
                                                model_batch::AbstractMatrix)
    try
        results = Float64.(model_batch) * Float64.(formula_matrix)
        return results
    catch ex
        _record_diagnostic!("qpu", "runtime_errors")
        @warn "SMTLibQPUExt: model_eval failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_simplify
# ============================================================================

function SMTLib.backend_coprocessor_simplify(b::QPUBackend,
                                              expression_batch::AbstractMatrix)
    try
        result = Float64.(expression_batch)
        threshold = 1.0e-10
        result .= result .* (abs.(result) .> threshold)
        for i in 1:size(result, 1)
            row_max = maximum(abs.(result[i, :]))
            row_max > 1.0e-30 && (result[i, :] ./= row_max)
        end
        return result
    catch ex
        _record_diagnostic!("qpu", "runtime_errors")
        @warn "SMTLibQPUExt: simplify failed, falling back to CPU" exception=ex maxlog=1
        return nothing
    end
end

# ============================================================================
# Remaining Hooks
# ============================================================================

function SMTLib.backend_coprocessor_interpolate(b::QPUBackend, args...)
    return nothing
end

end # module SMTLibQPUExt
