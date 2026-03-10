# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- SMTLib
# ============================================================================

# Julia fallback implementations
backend_solve(::JuliaBackend, args...) = nothing
backend_check_sat(::JuliaBackend, args...) = nothing
backend_model_eval(::JuliaBackend, args...) = nothing
backend_interpolate(::JuliaBackend, args...) = nothing
backend_simplify(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_solve end
function backend_coprocessor_check_sat end
function backend_coprocessor_model_eval end
function backend_coprocessor_interpolate end
function backend_coprocessor_simplify end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_solve, :backend_solve),
    (:backend_coprocessor_check_sat, :backend_check_sat),
    (:backend_coprocessor_model_eval, :backend_model_eval),
    (:backend_coprocessor_interpolate, :backend_interpolate),
    (:backend_coprocessor_simplify, :backend_simplify),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_solve(b::CoprocessorBackend, args...) = backend_coprocessor_solve(b, args...)
backend_check_sat(b::CoprocessorBackend, args...) = backend_coprocessor_check_sat(b, args...)
backend_model_eval(b::CoprocessorBackend, args...) = backend_coprocessor_model_eval(b, args...)
backend_interpolate(b::CoprocessorBackend, args...) = backend_coprocessor_interpolate(b, args...)
backend_simplify(b::CoprocessorBackend, args...) = backend_coprocessor_simplify(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_solve(b::GPUBackend, args...) = backend_solve(JuliaBackend(), args...)
backend_check_sat(b::GPUBackend, args...) = backend_check_sat(JuliaBackend(), args...)
backend_model_eval(b::GPUBackend, args...) = backend_model_eval(JuliaBackend(), args...)
backend_interpolate(b::GPUBackend, args...) = backend_interpolate(JuliaBackend(), args...)
backend_simplify(b::GPUBackend, args...) = backend_simplify(JuliaBackend(), args...)
