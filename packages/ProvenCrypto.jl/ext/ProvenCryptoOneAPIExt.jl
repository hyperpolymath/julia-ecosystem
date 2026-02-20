# SPDX-License-Identifier: PMPL-1.0-or-later
module ProvenCryptoOneAPIExt
using ..ProvenCrypto, oneAPI

# Override the default availability check
ProvenCrypto.oneapi_available() = true

"""
    create_oneapi_backend() -> ProvenCrypto.oneAPIBackend

Create a oneAPI backend for Intel GPUs/NPUs.
"""
function ProvenCrypto.create_oneapi_backend()
    device_id = 0  # Default device
    # TODO: Add more specific Intel hardware detection logic
    return ProvenCrypto.oneAPIBackend(device_id, false, 0)
end

"""
    backend_lattice_multiply(backend::ProvenCrypto.oneAPIBackend, A::AbstractMatrix, x::AbstractVector)

oneAPI-specific implementation for lattice multiplication.
"""
function ProvenCrypto.backend_lattice_multiply(backend::ProvenCrypto.oneAPIBackend, A::AbstractMatrix, x::AbstractVector)
    # Placeholder: Implement oneAPI-accelerated lattice multiplication
    @warn "oneAPI lattice multiplication not yet implemented; falling back to CPU"
    return ProvenCrypto.backend_lattice_multiply(ProvenCrypto.CPUBackend(:intel, Threads.nthreads()), A, x)
end

"""
    backend_ntt_transform(backend::ProvenCrypto.oneAPIBackend, poly::AbstractVector, modulus::Integer)

oneAPI-specific implementation for Number Theoretic Transform (NTT).
"""
function ProvenCrypto.backend_ntt_transform(backend::ProvenCrypto.oneAPIBackend, poly::AbstractVector, modulus::Integer)
    # Placeholder: Implement oneAPI-accelerated NTT
    @warn "oneAPI NTT not yet implemented; falling back to CPU"
    return ProvenCrypto.backend_ntt_transform(ProvenCrypto.CPUBackend(:intel, Threads.nthreads()), poly, modulus)
end

"""
    backend_polynomial_multiply(backend::ProvenCrypto.oneAPIBackend, a::AbstractVector, b::AbstractVector, modulus::Integer)

oneAPI-specific implementation for polynomial multiplication.
"""
function ProvenCrypto.backend_polynomial_multiply(backend::ProvenCrypto.oneAPIBackend, a::AbstractVector, b::AbstractVector, modulus::Integer)
    # Placeholder: Implement oneAPI-accelerated polynomial multiplication
    @warn "oneAPI polynomial multiplication not yet implemented; falling back to CPU"
    return ProvenCrypto.backend_polynomial_multiply(ProvenCrypto.CPUBackend(:intel, Threads.nthreads()), a, b, modulus)
end

"""
    backend_sampling(backend::ProvenCrypto.oneAPIBackend, distribution::Symbol, params...)

oneAPI-specific implementation for cryptographic sampling.
"""
function ProvenCrypto.backend_sampling(backend::ProvenCrypto.oneAPIBackend, distribution::Symbol, params...)
    # Placeholder: Implement oneAPI-accelerated sampling
    @warn "oneAPI sampling not yet implemented; falling back to CPU"
    return ProvenCrypto.backend_sampling(ProvenCrypto.CPUBackend(:intel, Threads.nthreads()), distribution, params...)
end
end
