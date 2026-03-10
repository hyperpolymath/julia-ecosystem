# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ProvenCrypto.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- ProvenCrypto
# ============================================================================

# Julia fallback implementations
backend_encrypt(::JuliaBackend, args...) = nothing
backend_decrypt(::JuliaBackend, args...) = nothing
backend_hash(::JuliaBackend, args...) = nothing
backend_sign(::JuliaBackend, args...) = nothing
backend_verify(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_encrypt end
function backend_coprocessor_decrypt end
function backend_coprocessor_hash end
function backend_coprocessor_sign end
function backend_coprocessor_verify end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_encrypt, :backend_encrypt),
    (:backend_coprocessor_decrypt, :backend_decrypt),
    (:backend_coprocessor_hash, :backend_hash),
    (:backend_coprocessor_sign, :backend_sign),
    (:backend_coprocessor_verify, :backend_verify),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_encrypt(b::CoprocessorBackend, args...) = backend_coprocessor_encrypt(b, args...)
backend_decrypt(b::CoprocessorBackend, args...) = backend_coprocessor_decrypt(b, args...)
backend_hash(b::CoprocessorBackend, args...) = backend_coprocessor_hash(b, args...)
backend_sign(b::CoprocessorBackend, args...) = backend_coprocessor_sign(b, args...)
backend_verify(b::CoprocessorBackend, args...) = backend_coprocessor_verify(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_encrypt(b::GPUBackend, args...) = backend_encrypt(JuliaBackend(), args...)
backend_decrypt(b::GPUBackend, args...) = backend_decrypt(JuliaBackend(), args...)
backend_hash(b::GPUBackend, args...) = backend_hash(JuliaBackend(), args...)
backend_sign(b::GPUBackend, args...) = backend_sign(JuliaBackend(), args...)
backend_verify(b::GPUBackend, args...) = backend_verify(JuliaBackend(), args...)
