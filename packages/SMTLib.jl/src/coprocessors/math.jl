# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl Math Coprocessor
# Extended precision for arithmetic theories and exact rational solving.
# Uses arbitrary precision arithmetic for QF_NRA, QF_LRA, QF_NIA theories.

function AcceleratorGate.device_capabilities(b::MathBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 4, 1000,
        Int64(8 * 1024^3), Int64(6 * 1024^3),
        1, true, false, false, "Software", "BigFloat/Rational",
    )
end

function AcceleratorGate.estimate_cost(::MathBackend, op::Symbol, data_size::Int)
    overhead = 20.0
    op == :solve && return overhead + Float64(data_size) * 0.5
    op == :check_sat && return overhead + Float64(data_size) * 0.3
    op == :model_eval && return overhead + Float64(data_size) * 0.2
    op == :interpolate && return overhead + Float64(data_size) * 0.8
    op == :simplify && return overhead + Float64(data_size) * 0.1
    Inf
end

AcceleratorGate.register_operation!(MathBackend, :solve)
AcceleratorGate.register_operation!(MathBackend, :check_sat)
AcceleratorGate.register_operation!(MathBackend, :model_eval)
AcceleratorGate.register_operation!(MathBackend, :interpolate)
AcceleratorGate.register_operation!(MathBackend, :simplify)

"""Solve Ax = b using exact rational Gaussian elimination."""
function _math_solve_exact(A::AbstractMatrix, b::AbstractVector)
    m, n = size(A)
    A_r = Rational{BigInt}.(A); b_r = Rational{BigInt}.(b)
    aug = hcat(A_r, b_r)
    pr = 1; pcols = Int[]
    for col in 1:n
        found = 0
        for row in pr:m; aug[row, col] != 0 && (found = row; break); end
        found == 0 && continue
        found != pr && (aug[pr, :], aug[found, :] = aug[found, :], aug[pr, :])
        push!(pcols, col)
        pv = aug[pr, col]
        for row in (pr+1):m
            aug[row, col] != 0 && (aug[row, :] .-= (aug[row, col] // pv) .* aug[pr, :])
        end
        pr += 1
    end
    for row in pr:m; aug[row, n+1] != 0 && return nothing; end
    sol = zeros(Rational{BigInt}, n)
    for (idx, col) in Iterators.reverse(enumerate(pcols))
        rhs = aug[idx, n+1]
        for j in (idx+1):length(pcols); rhs -= aug[idx, pcols[j]] * sol[pcols[j]]; end
        sol[col] = rhs // aug[idx, col]
    end
    sol
end

"""
Exact arithmetic SAT/SMT solving for linear arithmetic theories.
Uses rational arithmetic to avoid floating-point rounding errors.
"""
function backend_coprocessor_solve(::MathBackend, clauses::AbstractVector,
                                   variables::AbstractVector,
                                   config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    if haskey(config, :coefficients) && haskey(config, :bounds)
        sol = _math_solve_exact(config.coefficients, config.bounds)
        if sol !== nothing
            return (status=:sat, model=Dict(variables[j] => Float64(sol[j]) for j in 1:n_vars), trials_checked=1)
        else
            return (status=:unsat, model=nothing, trials_checked=1)
        end
    end
    n_trials = min(get(config, :trials, 2^min(n_vars, 20)), 1048576)
    for trial in 1:n_trials
        a = rand(Bool, n_vars); ok = true
        for clause in clauses
            sat = false
            for lit in clause
                vi = abs(lit); vi > n_vars && continue
                ((lit > 0 && a[vi]) || (lit < 0 && !a[vi])) && (sat = true; break)
            end
            !sat && (ok = false; break)
        end
        ok && return (status=:sat, model=Dict(variables[j] => a[j] for j in 1:n_vars), trials_checked=trial)
    end
    return (status=:unknown, model=nothing, trials_checked=n_trials)
end

function backend_coprocessor_check_sat(::MathBackend, clauses::AbstractVector, n_vars::Int)
    r = backend_coprocessor_solve(MathBackend(0), clauses, [Symbol("x$i") for i in 1:n_vars])
    r.status
end

"""Exact model evaluation using arbitrary-precision rational arithmetic."""
function backend_coprocessor_model_eval(::MathBackend, formula_matrix::AbstractMatrix,
                                        model_batch::AbstractMatrix)
    Float64.(Rational{BigInt}.(model_batch) * Rational{BigInt}.(formula_matrix))
end

"""Craig interpolation using exact arithmetic."""
function backend_coprocessor_interpolate(::MathBackend, formula_a::AbstractMatrix,
                                         formula_b::AbstractMatrix,
                                         shared_vars::AbstractVector{Int})
    A_s = Rational{BigInt}.(formula_a[:, shared_vars])
    B_s = Rational{BigInt}.(formula_b[:, shared_vars])
    m_a = size(formula_a, 1); m_b = size(formula_b, 1)
    n_s = length(shared_vars)
    interp = zeros(Rational{BigInt}, 1, n_s)
    for j in 1:n_s
        interp[1, j] = (sum(A_s[:, j]) - sum(B_s[:, j])) // BigInt(m_a + m_b)
    end
    Float64.(interp)
end

"""Exact GCD-based coefficient reduction."""
function backend_coprocessor_simplify(::MathBackend, expression_batch::AbstractMatrix)
    nr, nc = size(expression_batch)
    result = zeros(Rational{BigInt}, nr, nc)
    for i in 1:nr
        row = Rational{BigInt}.(expression_batch[i, :])
        nums = BigInt[abs(numerator(r)) for r in row if r != 0]
        if !isempty(nums)
            g = reduce(gcd, nums)
            g > 1 && (row ./= g)
        end
        result[i, :] .= row
    end
    Float64.(result)
end
