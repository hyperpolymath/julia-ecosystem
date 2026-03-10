# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SiliconCore.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- SiliconCore
# ============================================================================

# Julia fallback implementations
backend_vector_op(::JuliaBackend, args...) = nothing
backend_matrix_op(::JuliaBackend, args...) = nothing
backend_simd_dispatch(::JuliaBackend, args...) = nothing
backend_cache_prefetch(::JuliaBackend, args...) = nothing
backend_intrinsic_call(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_vector_op end
function backend_coprocessor_matrix_op end
function backend_coprocessor_simd_dispatch end
function backend_coprocessor_cache_prefetch end
function backend_coprocessor_intrinsic_call end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_vector_op, :backend_vector_op),
    (:backend_coprocessor_matrix_op, :backend_matrix_op),
    (:backend_coprocessor_simd_dispatch, :backend_simd_dispatch),
    (:backend_coprocessor_cache_prefetch, :backend_cache_prefetch),
    (:backend_coprocessor_intrinsic_call, :backend_intrinsic_call),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_vector_op(b::CoprocessorBackend, args...) = backend_coprocessor_vector_op(b, args...)
backend_matrix_op(b::CoprocessorBackend, args...) = backend_coprocessor_matrix_op(b, args...)
backend_simd_dispatch(b::CoprocessorBackend, args...) = backend_coprocessor_simd_dispatch(b, args...)
backend_cache_prefetch(b::CoprocessorBackend, args...) = backend_coprocessor_cache_prefetch(b, args...)
backend_intrinsic_call(b::CoprocessorBackend, args...) = backend_coprocessor_intrinsic_call(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_vector_op(b::GPUBackend, args...) = backend_vector_op(JuliaBackend(), args...)
backend_matrix_op(b::GPUBackend, args...) = backend_matrix_op(JuliaBackend(), args...)
backend_simd_dispatch(b::GPUBackend, args...) = backend_simd_dispatch(JuliaBackend(), args...)
backend_cache_prefetch(b::GPUBackend, args...) = backend_cache_prefetch(JuliaBackend(), args...)
backend_intrinsic_call(b::GPUBackend, args...) = backend_intrinsic_call(JuliaBackend(), args...)
