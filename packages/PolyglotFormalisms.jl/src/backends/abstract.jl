# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- PolyglotFormalisms
# ============================================================================

# Julia fallback implementations
backend_fold_parallel(::JuliaBackend, args...) = nothing
backend_map_parallel(::JuliaBackend, args...) = nothing
backend_reduce_parallel(::JuliaBackend, args...) = nothing
backend_tensor_contract(::JuliaBackend, args...) = nothing
backend_symbolic_eval(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_fold_parallel end
function backend_coprocessor_map_parallel end
function backend_coprocessor_reduce_parallel end
function backend_coprocessor_tensor_contract end
function backend_coprocessor_symbolic_eval end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_fold_parallel, :backend_fold_parallel),
    (:backend_coprocessor_map_parallel, :backend_map_parallel),
    (:backend_coprocessor_reduce_parallel, :backend_reduce_parallel),
    (:backend_coprocessor_tensor_contract, :backend_tensor_contract),
    (:backend_coprocessor_symbolic_eval, :backend_symbolic_eval),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_fold_parallel(b::CoprocessorBackend, args...) = backend_coprocessor_fold_parallel(b, args...)
backend_map_parallel(b::CoprocessorBackend, args...) = backend_coprocessor_map_parallel(b, args...)
backend_reduce_parallel(b::CoprocessorBackend, args...) = backend_coprocessor_reduce_parallel(b, args...)
backend_tensor_contract(b::CoprocessorBackend, args...) = backend_coprocessor_tensor_contract(b, args...)
backend_symbolic_eval(b::CoprocessorBackend, args...) = backend_coprocessor_symbolic_eval(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_fold_parallel(b::GPUBackend, args...) = backend_fold_parallel(JuliaBackend(), args...)
backend_map_parallel(b::GPUBackend, args...) = backend_map_parallel(JuliaBackend(), args...)
backend_reduce_parallel(b::GPUBackend, args...) = backend_reduce_parallel(JuliaBackend(), args...)
backend_tensor_contract(b::GPUBackend, args...) = backend_tensor_contract(JuliaBackend(), args...)
backend_symbolic_eval(b::GPUBackend, args...) = backend_symbolic_eval(JuliaBackend(), args...)
