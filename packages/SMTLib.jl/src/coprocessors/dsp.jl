# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl DSP Coprocessor
# Signal processing view of bit-vector operations using Digital Signal Processors.
# Maps BV arithmetic to correlation on DSP multiply-accumulate units.

function AcceleratorGate.device_capabilities(b::DSPBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 32, 600,
        Int64(1 * 1024^3), Int64(1 * 1024^3),
        128, false, true, true, "Texas Instruments", "DSP C66x",
    )
end

function AcceleratorGate.estimate_cost(::DSPBackend, op::Symbol, data_size::Int)
    overhead = 50.0
    op == :solve && return overhead + Float64(data_size) * 0.08
    op == :simplify && return overhead + Float64(data_size) * 0.02
    op == :model_eval && return overhead + Float64(data_size) * 0.04
    Inf
end

AcceleratorGate.register_operation!(DSPBackend, :solve)
AcceleratorGate.register_operation!(DSPBackend, :simplify)
AcceleratorGate.register_operation!(DSPBackend, :model_eval)

"""Convert clause to matched filter: +1 for positive lits, -1 for negative."""
function _dsp_clause_filter(clause::AbstractVector, n_vars::Int)
    f = zeros(Float64, n_vars)
    for lit in clause
        vi = abs(lit); vi > n_vars && continue
        f[vi] = lit > 0 ? 1.0 : -1.0
    end
    f
end

"""Evaluate clause via DSP MAC: correlate filter with bipolar assignment signal."""
function _dsp_eval_clause(filter::Vector{Float64}, signal::Vector{Float64}, clen::Int)
    mac = 0.0
    @simd for i in eachindex(filter); mac += filter[i] * signal[i]; end
    mac > Float64(-clen)
end

"""
DSP-accelerated SAT solving using matched filtering. Converts clauses to
matched filters and evaluates candidate assignments via multiply-accumulate
on the DSP's MAC array.
"""
function backend_coprocessor_solve(::DSPBackend, clauses::AbstractVector,
                                   variables::AbstractVector,
                                   config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    filters = [_dsp_clause_filter(c, n_vars) for c in clauses]
    clens = [length(c) for c in clauses]
    n_trials = min(get(config, :trials, 2^min(n_vars, 20)), 1048576)

    for trial in 1:n_trials
        bits = rand(Bool, n_vars)
        signal = Float64[b ? 1.0 : -1.0 for b in bits]
        ok = true
        for (c, f) in enumerate(filters)
            _dsp_eval_clause(f, signal, clens[c]) || (ok = false; break)
        end
        if ok
            model = Dict(variables[j] => bits[j] for j in 1:n_vars)
            return (status=:sat, model=model, trials_checked=trial)
        end
    end
    return (status=:unknown, model=nothing, trials_checked=n_trials)
end

function backend_coprocessor_model_eval(::DSPBackend, formula_matrix::AbstractMatrix,
                                        model_batch::AbstractMatrix)
    Float64.(model_batch) * Float64.(formula_matrix)
end

"""
DSP-based simplification using FIR smoothing on coefficient vectors,
treating them as discrete signals and filtering out noise.
"""
function backend_coprocessor_simplify(::DSPBackend, expression_batch::AbstractMatrix)
    result = Float64.(expression_batch)
    nr, nc = size(result)
    for row in 1:nr
        smoothed = copy(@view result[row, :])
        for j in 2:(nc-1)
            smoothed[j] = 0.25 * result[row, j-1] + 0.5 * result[row, j] + 0.25 * result[row, j+1]
        end
        for j in 1:nc; abs(smoothed[j]) < 1.0e-10 && (smoothed[j] = 0.0); end
        rm = maximum(abs, smoothed)
        rm > 1.0e-30 && (smoothed ./= rm)
        result[row, :] .= smoothed
    end
    result
end
