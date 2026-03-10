# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# KnotTheory.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- KnotTheory
# ============================================================================

# Julia fallback implementations
backend_polynomial_eval(::JuliaBackend, args...) = nothing
backend_jones_invariant(::JuliaBackend, args...) = nothing
backend_alexander_polynomial(::JuliaBackend, args...) = nothing
backend_matrix_det(::JuliaBackend, args...) = nothing
backend_braid_reduce(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_polynomial_eval end
function backend_coprocessor_jones_invariant end
function backend_coprocessor_alexander_polynomial end
function backend_coprocessor_matrix_det end
function backend_coprocessor_braid_reduce end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_polynomial_eval, :backend_polynomial_eval),
    (:backend_coprocessor_jones_invariant, :backend_jones_invariant),
    (:backend_coprocessor_alexander_polynomial, :backend_alexander_polynomial),
    (:backend_coprocessor_matrix_det, :backend_matrix_det),
    (:backend_coprocessor_braid_reduce, :backend_braid_reduce),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_polynomial_eval(b::CoprocessorBackend, args...) = backend_coprocessor_polynomial_eval(b, args...)
backend_jones_invariant(b::CoprocessorBackend, args...) = backend_coprocessor_jones_invariant(b, args...)
backend_alexander_polynomial(b::CoprocessorBackend, args...) = backend_coprocessor_alexander_polynomial(b, args...)
backend_matrix_det(b::CoprocessorBackend, args...) = backend_coprocessor_matrix_det(b, args...)
backend_braid_reduce(b::CoprocessorBackend, args...) = backend_coprocessor_braid_reduce(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_polynomial_eval(b::GPUBackend, args...) = backend_polynomial_eval(JuliaBackend(), args...)
backend_jones_invariant(b::GPUBackend, args...) = backend_jones_invariant(JuliaBackend(), args...)
backend_alexander_polynomial(b::GPUBackend, args...) = backend_alexander_polynomial(JuliaBackend(), args...)
backend_matrix_det(b::GPUBackend, args...) = backend_matrix_det(JuliaBackend(), args...)
backend_braid_reduce(b::GPUBackend, args...) = backend_braid_reduce(JuliaBackend(), args...)
