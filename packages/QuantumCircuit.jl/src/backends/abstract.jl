# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# QuantumCircuit.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- QuantumCircuit
# ============================================================================

# Julia fallback implementations
backend_gate_apply(::JuliaBackend, args...) = nothing
backend_tensor_contract(::JuliaBackend, args...) = nothing
backend_state_evolve(::JuliaBackend, args...) = nothing
backend_measurement(::JuliaBackend, args...) = nothing
backend_entangle(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_gate_apply end
function backend_coprocessor_tensor_contract end
function backend_coprocessor_state_evolve end
function backend_coprocessor_measurement end
function backend_coprocessor_entangle end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_gate_apply, :backend_gate_apply),
    (:backend_coprocessor_tensor_contract, :backend_tensor_contract),
    (:backend_coprocessor_state_evolve, :backend_state_evolve),
    (:backend_coprocessor_measurement, :backend_measurement),
    (:backend_coprocessor_entangle, :backend_entangle),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_gate_apply(b::CoprocessorBackend, args...) = backend_coprocessor_gate_apply(b, args...)
backend_tensor_contract(b::CoprocessorBackend, args...) = backend_coprocessor_tensor_contract(b, args...)
backend_state_evolve(b::CoprocessorBackend, args...) = backend_coprocessor_state_evolve(b, args...)
backend_measurement(b::CoprocessorBackend, args...) = backend_coprocessor_measurement(b, args...)
backend_entangle(b::CoprocessorBackend, args...) = backend_coprocessor_entangle(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_gate_apply(b::GPUBackend, args...) = backend_gate_apply(JuliaBackend(), args...)
backend_tensor_contract(b::GPUBackend, args...) = backend_tensor_contract(JuliaBackend(), args...)
backend_state_evolve(b::GPUBackend, args...) = backend_state_evolve(JuliaBackend(), args...)
backend_measurement(b::GPUBackend, args...) = backend_measurement(JuliaBackend(), args...)
backend_entangle(b::GPUBackend, args...) = backend_entangle(JuliaBackend(), args...)
