# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Causals.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- Causals
# ============================================================================

# Julia fallback implementations
backend_bayesian_update(::JuliaBackend, args...) = nothing
backend_causal_inference(::JuliaBackend, args...) = nothing
backend_uncertainty_propagate(::JuliaBackend, args...) = nothing
backend_network_eval(::JuliaBackend, args...) = nothing
backend_monte_carlo(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_bayesian_update end
function backend_coprocessor_causal_inference end
function backend_coprocessor_uncertainty_propagate end
function backend_coprocessor_network_eval end
function backend_coprocessor_monte_carlo end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_bayesian_update, :backend_bayesian_update),
    (:backend_coprocessor_causal_inference, :backend_causal_inference),
    (:backend_coprocessor_uncertainty_propagate, :backend_uncertainty_propagate),
    (:backend_coprocessor_network_eval, :backend_network_eval),
    (:backend_coprocessor_monte_carlo, :backend_monte_carlo),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_bayesian_update(b::CoprocessorBackend, args...) = backend_coprocessor_bayesian_update(b, args...)
backend_causal_inference(b::CoprocessorBackend, args...) = backend_coprocessor_causal_inference(b, args...)
backend_uncertainty_propagate(b::CoprocessorBackend, args...) = backend_coprocessor_uncertainty_propagate(b, args...)
backend_network_eval(b::CoprocessorBackend, args...) = backend_coprocessor_network_eval(b, args...)
backend_monte_carlo(b::CoprocessorBackend, args...) = backend_coprocessor_monte_carlo(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_bayesian_update(b::GPUBackend, args...) = backend_bayesian_update(JuliaBackend(), args...)
backend_causal_inference(b::GPUBackend, args...) = backend_causal_inference(JuliaBackend(), args...)
backend_uncertainty_propagate(b::GPUBackend, args...) = backend_uncertainty_propagate(JuliaBackend(), args...)
backend_network_eval(b::GPUBackend, args...) = backend_network_eval(JuliaBackend(), args...)
backend_monte_carlo(b::GPUBackend, args...) = backend_monte_carlo(JuliaBackend(), args...)
