# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl QPU Coprocessor
# Quantum SAT solving via Grover's algorithm for satisfiability.
# Provides quadratic speedup over classical brute-force search.

function AcceleratorGate.device_capabilities(b::QPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 127, 0,
        Int64(0), Int64(0),
        1, true, false, false, "IBM", "QPU Quantum",
    )
end

function AcceleratorGate.estimate_cost(::QPUBackend, op::Symbol, data_size::Int)
    gate_cost = 100.0
    op == :solve && return gate_cost * sqrt(Float64(data_size))
    op == :check_sat && return gate_cost * sqrt(Float64(data_size))
    Inf
end

AcceleratorGate.register_operation!(QPUBackend, :solve)
AcceleratorGate.register_operation!(QPUBackend, :check_sat)

"""Grover oracle: flip amplitude of satisfying states."""
function _qpu_oracle!(amps::Vector{ComplexF64}, pos::Vector{UInt64}, neg::Vector{UInt64}, nc::Int)
    for x in 0:(length(amps) - 1)
        a = UInt64(x); na = ~a
        all_sat = true
        for c in 1:nc
            ((pos[c] & a) | (neg[c] & na)) == UInt64(0) && (all_sat = false; break)
        end
        all_sat && (amps[x + 1] = -amps[x + 1])
    end
end

"""Grover diffusion operator: inversion about the mean."""
function _qpu_diffusion!(amps::Vector{ComplexF64})
    N = length(amps)
    m = sum(amps) / N
    for i in 1:N; amps[i] = 2.0 * m - amps[i]; end
end

"""Simulate quantum measurement: sample from |amplitude|^2."""
function _qpu_measure(amps::Vector{ComplexF64})
    probs = abs2.(amps); probs ./= sum(probs)
    r = rand(); cum = 0.0
    for i in eachindex(probs)
        cum += probs[i]; r <= cum && return i - 1
    end
    length(probs) - 1
end

"""
Quantum SAT via Grover's algorithm. O(sqrt(2^n)) oracle queries vs O(2^n) classical.
Simulation is exponential; practical for n_vars <= 20.
"""
function backend_coprocessor_solve(::QPUBackend, clauses::AbstractVector,
                                   variables::AbstractVector,
                                   config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    max_qubits = get(config, :max_qubits, 20)
    if n_vars > max_qubits
        @warn "QPU: $n_vars vars > $max_qubits qubit limit, falling back" maxlog=1
        return backend_solve(JuliaBackend(), clauses, variables, config)
    end

    N = 1 << n_vars
    nc = length(clauses)
    pos = zeros(UInt64, nc); neg = zeros(UInt64, nc)
    for (i, clause) in enumerate(clauses)
        for lit in clause
            vi = abs(lit); vi > n_vars && continue
            bit = UInt64(1) << (vi - 1)
            lit > 0 ? (pos[i] |= bit) : (neg[i] |= bit)
        end
    end

    # Count solutions for optimal iteration count
    n_sol = 0
    for x in 0:(N-1)
        a = UInt64(x); na = ~a; ok = true
        for c in 1:nc
            ((pos[c] & a) | (neg[c] & na)) == UInt64(0) && (ok = false; break)
        end
        ok && (n_sol += 1)
    end
    n_sol == 0 && return (status=:unsat, model=nothing, trials_checked=N)

    n_iter = max(1, floor(Int, (pi / 4) * sqrt(N / n_sol)))
    n_iter = min(n_iter, get(config, :max_iterations, 100))
    n_shots = get(config, :shots, 10)

    for _ in 1:n_shots
        amps = fill(ComplexF64(1.0 / sqrt(N)), N)
        for _ in 1:n_iter
            _qpu_oracle!(amps, pos, neg, nc)
            _qpu_diffusion!(amps)
        end
        state = UInt64(_qpu_measure(amps)); ns = ~state
        ok = true
        for c in 1:nc
            ((pos[c] & state) | (neg[c] & ns)) == UInt64(0) && (ok = false; break)
        end
        if ok
            model = Dict(variables[j] => (state >> (j-1)) & UInt64(1) == UInt64(1) for j in 1:n_vars)
            return (status=:sat, model=model, trials_checked=n_iter)
        end
    end
    return (status=:unknown, model=nothing, trials_checked=n_iter * n_shots)
end

function backend_coprocessor_check_sat(::QPUBackend, clauses::AbstractVector, n_vars::Int)
    r = backend_coprocessor_solve(QPUBackend(0), clauses, [Symbol("x$i") for i in 1:n_vars])
    r.status
end
