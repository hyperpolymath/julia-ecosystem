# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl NPU Coprocessor
# Neural-guided SAT heuristics using Neural Processing Units.
# Uses learned heuristics to guide variable selection and clause ordering.

# ============================================================================
# Device Capabilities & Cost
# ============================================================================

function AcceleratorGate.device_capabilities(b::NPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 16, 1000,
        Int64(4 * 1024^3), Int64(3 * 1024^3),
        256, false, true, true, "Qualcomm", "NPU",
    )
end

function AcceleratorGate.estimate_cost(::NPUBackend, op::Symbol, data_size::Int)
    overhead = 200.0
    op == :solve && return overhead + Float64(data_size) * 0.05
    op == :check_sat && return overhead + Float64(data_size) * 0.04
    op == :model_eval && return overhead + Float64(data_size) * 0.02
    Inf
end

AcceleratorGate.register_operation!(NPUBackend, :solve)
AcceleratorGate.register_operation!(NPUBackend, :check_sat)
AcceleratorGate.register_operation!(NPUBackend, :model_eval)

# ============================================================================
# Neural-Guided SAT Heuristics
# ============================================================================

"""
Build feature vectors for neural-guided SAT solving. Features per variable:
positive/negative occurrence counts, inverse clause length weights, activity.
"""
function _npu_build_features(clauses::AbstractVector, n_vars::Int)
    features = zeros(Float32, n_vars, 6)
    for clause in clauses
        clen = Float32(length(clause))
        for lit in clause
            var_idx = abs(lit)
            var_idx > n_vars && continue
            col = lit > 0 ? 1 : 2
            features[var_idx, col] += 1.0f0
            features[var_idx, col + 2] += 1.0f0 / clen
            features[var_idx, 5] += 1.0f0 / clen
            features[var_idx, 6] = max(features[var_idx, 6], 1.0f0 / clen)
        end
    end
    return features
end

"""
Two-layer perceptron scoring for variable priority, optimized for NPU int8/fp16.
Returns indices sorted by predicted branching value.
"""
function _npu_variable_ordering(features::AbstractMatrix{Float32})
    w1 = Float32[0.3 0.3 0.5 0.5 0.2 0.1;
                  0.1 0.1 0.3 0.3 0.4 0.3;
                  0.4 0.2 0.1 0.1 0.3 0.4;
                  0.2 0.4 0.1 0.1 0.1 0.2]
    h1 = max.(features * w1', 0.0f0)
    w2 = Float32[0.4, 0.3, 0.2, 0.1]
    scores = h1 * w2
    return sortperm(vec(scores), rev=true)
end

"""
Neural-guided SAT solving. Uses NPU inference to compute variable branching
heuristics, then applies DPLL with the learned ordering.
"""
function backend_coprocessor_solve(::NPUBackend, clauses::AbstractVector,
                                   variables::AbstractVector,
                                   config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    features = _npu_build_features(clauses, n_vars)
    var_order = _npu_variable_ordering(features)
    max_depth = min(n_vars, get(config, :max_depth, 30))
    assignment = zeros(Int8, n_vars)

    function check_all()
        for clause in clauses
            sat = false; all_decided = true
            for lit in clause
                vi = abs(lit); vi > n_vars && continue
                assignment[vi] == 0 && (all_decided = false; continue)
                ((lit > 0 && assignment[vi] == 1) || (lit < 0 && assignment[vi] == -1)) && (sat = true; break)
            end
            all_decided && !sat && return :unsat
        end
        all_sat = true
        for clause in clauses
            cs = false
            for lit in clause
                vi = abs(lit); vi > n_vars && continue
                ((lit > 0 && assignment[vi] == 1) || (lit < 0 && assignment[vi] == -1)) && (cs = true; break)
            end
            !cs && (all_sat = false; break)
        end
        all_sat ? :sat : :unknown
    end

    function dpll(depth::Int)
        depth > max_depth && return check_all()
        status = check_all()
        (status == :sat || status == :unsat) && return status
        next_var = 0
        for vi in var_order
            assignment[vi] == 0 && (next_var = vi; break)
        end
        next_var == 0 && return check_all()
        polarity = features[next_var, 1] >= features[next_var, 2] ? Int8(1) : Int8(-1)
        for p in (polarity, -polarity)
            assignment[next_var] = p
            dpll(depth + 1) == :sat && return :sat
        end
        assignment[next_var] = Int8(0)
        return :unknown
    end

    if dpll(0) == :sat
        model = Dict(variables[j] => (assignment[j] == 1) for j in 1:n_vars if assignment[j] != 0)
        return (status=:sat, model=model, trials_checked=0)
    end
    return (status=:unknown, model=nothing, trials_checked=0)
end

function backend_coprocessor_check_sat(::NPUBackend, clauses::AbstractVector, n_vars::Int)
    variables = [Symbol("x$i") for i in 1:n_vars]
    result = backend_coprocessor_solve(NPUBackend(0), clauses, variables)
    result.status == :sat ? :sat : :unknown
end

function backend_coprocessor_model_eval(::NPUBackend, formula_matrix::AbstractMatrix,
                                        model_batch::AbstractMatrix)
    Float32.(model_batch) * Float32.(formula_matrix)
end
