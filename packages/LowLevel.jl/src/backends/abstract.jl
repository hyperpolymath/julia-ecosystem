# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# LowLevel.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- LowLevel
# ============================================================================

# Julia fallback implementations
backend_kernel_dispatch(::JuliaBackend, args...) = nothing
backend_memory_op(::JuliaBackend, args...) = nothing
backend_dma_transfer(::JuliaBackend, args...) = nothing
backend_interrupt_handle(::JuliaBackend, args...) = nothing
backend_register_op(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_kernel_dispatch end
function backend_coprocessor_memory_op end
function backend_coprocessor_dma_transfer end
function backend_coprocessor_interrupt_handle end
function backend_coprocessor_register_op end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_kernel_dispatch, :backend_kernel_dispatch),
    (:backend_coprocessor_memory_op, :backend_memory_op),
    (:backend_coprocessor_dma_transfer, :backend_dma_transfer),
    (:backend_coprocessor_interrupt_handle, :backend_interrupt_handle),
    (:backend_coprocessor_register_op, :backend_register_op),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_kernel_dispatch(b::CoprocessorBackend, args...) = backend_coprocessor_kernel_dispatch(b, args...)
backend_memory_op(b::CoprocessorBackend, args...) = backend_coprocessor_memory_op(b, args...)
backend_dma_transfer(b::CoprocessorBackend, args...) = backend_coprocessor_dma_transfer(b, args...)
backend_interrupt_handle(b::CoprocessorBackend, args...) = backend_coprocessor_interrupt_handle(b, args...)
backend_register_op(b::CoprocessorBackend, args...) = backend_coprocessor_register_op(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_kernel_dispatch(b::GPUBackend, args...) = backend_kernel_dispatch(JuliaBackend(), args...)
backend_memory_op(b::GPUBackend, args...) = backend_memory_op(JuliaBackend(), args...)
backend_dma_transfer(b::GPUBackend, args...) = backend_dma_transfer(JuliaBackend(), args...)
backend_interrupt_handle(b::GPUBackend, args...) = backend_interrupt_handle(JuliaBackend(), args...)
backend_register_op(b::GPUBackend, args...) = backend_register_op(JuliaBackend(), args...)
