# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- ZeroProb
# ============================================================================

# Julia fallback implementations
backend_probability_eval(::JuliaBackend, args...) = nothing
backend_bayesian_update(::JuliaBackend, args...) = nothing
backend_log_likelihood(::JuliaBackend, args...) = nothing
backend_sampling(::JuliaBackend, args...) = nothing
backend_marginalize(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_probability_eval end
function backend_coprocessor_bayesian_update end
function backend_coprocessor_log_likelihood end
function backend_coprocessor_sampling end
function backend_coprocessor_marginalize end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_probability_eval, :backend_probability_eval),
    (:backend_coprocessor_bayesian_update, :backend_bayesian_update),
    (:backend_coprocessor_log_likelihood, :backend_log_likelihood),
    (:backend_coprocessor_sampling, :backend_sampling),
    (:backend_coprocessor_marginalize, :backend_marginalize),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_probability_eval(b::CoprocessorBackend, args...) = backend_coprocessor_probability_eval(b, args...)
backend_bayesian_update(b::CoprocessorBackend, args...) = backend_coprocessor_bayesian_update(b, args...)
backend_log_likelihood(b::CoprocessorBackend, args...) = backend_coprocessor_log_likelihood(b, args...)
backend_sampling(b::CoprocessorBackend, args...) = backend_coprocessor_sampling(b, args...)
backend_marginalize(b::CoprocessorBackend, args...) = backend_coprocessor_marginalize(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_probability_eval(b::GPUBackend, args...) = backend_probability_eval(JuliaBackend(), args...)
backend_bayesian_update(b::GPUBackend, args...) = backend_bayesian_update(JuliaBackend(), args...)
backend_log_likelihood(b::GPUBackend, args...) = backend_log_likelihood(JuliaBackend(), args...)
backend_sampling(b::GPUBackend, args...) = backend_sampling(JuliaBackend(), args...)
backend_marginalize(b::GPUBackend, args...) = backend_marginalize(JuliaBackend(), args...)
