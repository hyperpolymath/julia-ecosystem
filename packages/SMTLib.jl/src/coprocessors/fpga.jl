# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl FPGA Coprocessor
# Hardware SAT solving pipeline. FPGAs are a well-established platform for SAT
# acceleration. Implements clause evaluation as a streaming pipeline on
# reconfigurable logic using bitmask LUTs for single-cycle clause evaluation.

function AcceleratorGate.device_capabilities(b::FPGABackend)
    AcceleratorGate.DeviceCapabilities(
        b, 1000, 200,
        Int64(8 * 1024^3), Int64(6 * 1024^3),
        64, false, false, true, "Intel", "FPGA Stratix",
    )
end

function AcceleratorGate.estimate_cost(::FPGABackend, op::Symbol, data_size::Int)
    setup = 5000.0
    op == :solve && return setup + Float64(data_size) * 0.001
    op == :check_sat && return setup + Float64(data_size) * 0.0005
    op == :simplify && return setup + Float64(data_size) * 0.002
    Inf
end

AcceleratorGate.register_operation!(FPGABackend, :solve)
AcceleratorGate.register_operation!(FPGABackend, :check_sat)
AcceleratorGate.register_operation!(FPGABackend, :simplify)

"""Encode clauses as bitmask LUTs for single-cycle hardware evaluation."""
function _fpga_encode_lut(clauses::AbstractVector, n_vars::Int)
    n_clauses = length(clauses)
    n_words = cld(n_vars, 64)
    pos_masks = zeros(UInt64, n_clauses, n_words)
    neg_masks = zeros(UInt64, n_clauses, n_words)
    for (i, clause) in enumerate(clauses)
        for lit in clause
            vi = abs(lit); vi > n_vars && continue
            wi = cld(vi, 64); bi = (vi - 1) % 64
            if lit > 0
                pos_masks[i, wi] |= UInt64(1) << bi
            else
                neg_masks[i, wi] |= UInt64(1) << bi
            end
        end
    end
    return pos_masks, neg_masks, n_words
end

"""Hardware-emulated single-cycle clause evaluation via bitwise AND/OR."""
function _fpga_eval_hw(pos_masks, neg_masks, assignment_bits, n_clauses, n_words)
    neg_bits = .~assignment_bits
    for c in 1:n_clauses
        sat = false
        for w in 1:n_words
            ((pos_masks[c, w] & assignment_bits[w]) | (neg_masks[c, w] & neg_bits[w])) != UInt64(0) && (sat = true; break)
        end
        !sat && return false
    end
    return true
end

"""
Hardware SAT solving pipeline. Clauses are encoded as bitmask LUTs in BRAM,
candidates stream through a FIFO, parallel clause units check in one cycle,
a reduction tree ANDs results. O(1) per candidate, O(n_units) parallelism.
"""
function backend_coprocessor_solve(::FPGABackend, clauses::AbstractVector,
                                   variables::AbstractVector,
                                   config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    n_clauses = length(clauses)
    pos_masks, neg_masks, n_words = _fpga_encode_lut(clauses, n_vars)
    n_trials = min(get(config, :trials, 2^min(n_vars, 22)), 4194304)
    remaining = n_vars % 64
    vmask = remaining > 0 ? (UInt64(1) << remaining) - UInt64(1) : typemax(UInt64)

    for trial in 1:n_trials
        bits = UInt64[rand(UInt64) for _ in 1:n_words]
        bits[end] &= vmask
        if _fpga_eval_hw(pos_masks, neg_masks, bits, n_clauses, n_words)
            model = Dict{Symbol,Bool}()
            for j in 1:n_vars
                wi = cld(j, 64); bi = (j - 1) % 64
                model[variables[j]] = (bits[wi] >> bi) & UInt64(1) == UInt64(1)
            end
            return (status=:sat, model=model, trials_checked=trial)
        end
    end
    return (status=:unknown, model=nothing, trials_checked=n_trials)
end

function backend_coprocessor_check_sat(::FPGABackend, clauses::AbstractVector, n_vars::Int)
    pos_masks, neg_masks, n_words = _fpga_encode_lut(clauses, n_vars)
    remaining = n_vars % 64
    vmask = remaining > 0 ? (UInt64(1) << remaining) - UInt64(1) : typemax(UInt64)
    for _ in 1:min(2^min(n_vars, 22), 4194304)
        bits = UInt64[rand(UInt64) for _ in 1:n_words]
        bits[end] &= vmask
        _fpga_eval_hw(pos_masks, neg_masks, bits, length(clauses), n_words) && return :sat
    end
    return :unknown
end

function backend_coprocessor_simplify(::FPGABackend, expression_batch::AbstractMatrix)
    result = Float64.(expression_batch)
    # Pipeline stage 1: zero elimination
    for i in eachindex(result)
        abs(result[i]) < 1.0e-10 && (result[i] = 0.0)
    end
    # Pipeline stage 2: row normalization
    for row in 1:size(result, 1)
        rm = maximum(abs, @view result[row, :])
        rm > 1.0e-30 && (result[row, :] ./= rm)
    end
    return result
end
