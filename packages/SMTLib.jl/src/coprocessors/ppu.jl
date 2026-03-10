# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl PPU Coprocessor
# Physics-based SAT solving via simulated annealing.
# Maps clauses to energy constraints; satisfying assignments are ground states.

function AcceleratorGate.device_capabilities(b::PPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 64, 500,
        Int64(4 * 1024^3), Int64(3 * 1024^3),
        256, true, true, false, "NVIDIA", "PPU PhysX",
    )
end

function AcceleratorGate.estimate_cost(::PPUBackend, op::Symbol, data_size::Int)
    overhead = 300.0
    op == :solve && return overhead + Float64(data_size) * 0.06
    op == :check_sat && return overhead + Float64(data_size) * 0.05
    Inf
end

AcceleratorGate.register_operation!(PPUBackend, :solve)
AcceleratorGate.register_operation!(PPUBackend, :check_sat)

"""Compute energy (number of unsatisfied clauses) for an assignment."""
function _ppu_energy(clauses, assignment, n_vars)
    e = 0
    for clause in clauses
        sat = false
        for lit in clause
            vi = abs(lit); vi > n_vars && continue
            ((lit > 0 && assignment[vi]) || (lit < 0 && !assignment[vi])) && (sat = true; break)
        end
        !sat && (e += 1)
    end
    e
end

"""Compute energy delta from flipping a single variable."""
function _ppu_flip_delta(clauses, assignment, var_idx, n_vars)
    delta = 0
    for clause in clauses
        has_var = false
        for lit in clause; abs(lit) == var_idx && (has_var = true; break); end
        !has_var && continue
        cur_sat = false
        for lit in clause
            vi = abs(lit); vi > n_vars && continue
            ((lit > 0 && assignment[vi]) || (lit < 0 && !assignment[vi])) && (cur_sat = true; break)
        end
        old = assignment[var_idx]; assignment[var_idx] = !old
        new_sat = false
        for lit in clause
            vi = abs(lit); vi > n_vars && continue
            ((lit > 0 && assignment[vi]) || (lit < 0 && !assignment[vi])) && (new_sat = true; break)
        end
        assignment[var_idx] = old
        cur_sat && !new_sat && (delta += 1)
        !cur_sat && new_sat && (delta -= 1)
    end
    delta
end

"""
Physics-based SAT via simulated annealing. Variables are particles,
clauses define energy constraints, satisfying assignments are ground states.
"""
function backend_coprocessor_solve(::PPUBackend, clauses::AbstractVector,
                                   variables::AbstractVector,
                                   config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    T_init = get(config, :T_initial, 2.0)
    T_final = get(config, :T_final, 0.01)
    cool = get(config, :cooling_rate, 0.995)
    restarts = get(config, :restarts, 5)
    steps = max(10, n_vars)

    best_a = nothing; best_e = length(clauses) + 1
    for _ in 1:restarts
        a = rand(Bool, n_vars)
        e = _ppu_energy(clauses, a, n_vars)
        e == 0 && return (status=:sat, model=Dict(variables[j] => a[j] for j in 1:n_vars), trials_checked=1)
        T = T_init
        while T > T_final
            for _ in 1:steps
                vi = rand(1:n_vars)
                d = _ppu_flip_delta(clauses, a, vi, n_vars)
                if d <= 0 || rand() < exp(-d / T)
                    a[vi] = !a[vi]; e += d
                end
                e == 0 && return (status=:sat, model=Dict(variables[j] => a[j] for j in 1:n_vars), trials_checked=1)
                e < best_e && (best_e = e; best_a = copy(a))
            end
            T *= cool
        end
    end
    best_e == 0 && best_a !== nothing && return (status=:sat, model=Dict(variables[j] => best_a[j] for j in 1:n_vars), trials_checked=restarts)
    return (status=:unknown, model=nothing, trials_checked=restarts)
end

function backend_coprocessor_check_sat(::PPUBackend, clauses::AbstractVector, n_vars::Int)
    r = backend_coprocessor_solve(PPUBackend(0), clauses, [Symbol("x$i") for i in 1:n_vars])
    r.status == :sat ? :sat : :unknown
end
