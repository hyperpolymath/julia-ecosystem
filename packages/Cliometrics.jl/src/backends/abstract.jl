# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cliometrics.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- Cliometrics
# ============================================================================

# Julia fallback implementations
backend_regression(::JuliaBackend, args...) = nothing
backend_decomposition(::JuliaBackend, args...) = nothing
backend_convergence_test(::JuliaBackend, args...) = nothing
backend_time_series_filter(::JuliaBackend, args...) = nothing
backend_growth_accounting(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_regression end
function backend_coprocessor_decomposition end
function backend_coprocessor_convergence_test end
function backend_coprocessor_time_series_filter end
function backend_coprocessor_growth_accounting end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_regression, :backend_regression),
    (:backend_coprocessor_decomposition, :backend_decomposition),
    (:backend_coprocessor_convergence_test, :backend_convergence_test),
    (:backend_coprocessor_time_series_filter, :backend_time_series_filter),
    (:backend_coprocessor_growth_accounting, :backend_growth_accounting),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_regression(b::CoprocessorBackend, args...) = backend_coprocessor_regression(b, args...)
backend_decomposition(b::CoprocessorBackend, args...) = backend_coprocessor_decomposition(b, args...)
backend_convergence_test(b::CoprocessorBackend, args...) = backend_coprocessor_convergence_test(b, args...)
backend_time_series_filter(b::CoprocessorBackend, args...) = backend_coprocessor_time_series_filter(b, args...)
backend_growth_accounting(b::CoprocessorBackend, args...) = backend_coprocessor_growth_accounting(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_regression(b::GPUBackend, args...) = backend_regression(JuliaBackend(), args...)
backend_decomposition(b::GPUBackend, args...) = backend_decomposition(JuliaBackend(), args...)
backend_convergence_test(b::GPUBackend, args...) = backend_convergence_test(JuliaBackend(), args...)
backend_time_series_filter(b::GPUBackend, args...) = backend_time_series_filter(JuliaBackend(), args...)
backend_growth_accounting(b::GPUBackend, args...) = backend_growth_accounting(JuliaBackend(), args...)
