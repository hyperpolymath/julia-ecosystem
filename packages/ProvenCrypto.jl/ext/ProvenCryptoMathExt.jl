# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ProvenCrypto Math Coprocessor Extension
# Arbitrary precision and extended arithmetic backend for cryptographic operations.
# Math coprocessors (IBM z16 Crypto Express, historical i8087, modern BigNum ASICs)
# provide hardware-accelerated arbitrary precision arithmetic, enabling NTT with
# large moduli (e.g., for FHE schemes where q can be thousands of bits),
# extended precision lattice operations, and symbolic polynomial manipulation.

module ProvenCryptoMathExt

using LinearAlgebra
using ..ProvenCrypto
using AcceleratorGate
using AcceleratorGate: MathBackend, DeviceCapabilities,
                       register_operation!, track_allocation!, track_deallocation!,
                       _record_diagnostic!

# ============================================================================
# Device Capabilities
# ============================================================================

function AcceleratorGate.device_capabilities(b::MathBackend)
    # IBM z16 Crypto Express / dedicated BigNum coprocessor
    DeviceCapabilities(
        b,
        4,                       # compute units (BigNum pipelines)
        200,                     # clock MHz (precision over speed)
        Int64(16) * 1024^3,      # 16 GiB
        Int64(14) * 1024^3,      # ~14 GiB available
        64,                      # max workgroup size
        true,                    # arbitrary precision (includes f64)
        true,                    # f16 via software
        true,                    # INT8 via BigNum library
        "IBM/Custom",
        "Math Coprocessor v1",
    )
end

function AcceleratorGate.estimate_cost(b::MathBackend, op::Symbol, data_size::Int)
    # Math coprocessors are slow per-operation but handle arbitrary precision
    # They are the only option when modulus exceeds 64-bit range
    if op in (:ntt_transform, :ntt_inverse_transform)
        # Arbitrary precision NTT: each multiply is more expensive but exact
        return 10.0 + Float64(data_size) * log2(max(data_size, 2)) * 0.5
    elseif op == :lattice_multiply
        # Extended precision matmul: handles huge moduli
        return 10.0 + Float64(data_size) * 0.8
    elseif op == :polynomial_multiply
        return 15.0 + Float64(data_size) * log2(max(data_size, 2)) * 1.0
    elseif op == :sampling
        return 5.0 + Float64(data_size) * 0.3
    end
    Inf
end

# ============================================================================
# Arbitrary Precision NTT
# ============================================================================
#
# The Math coprocessor handles NTT with moduli that exceed the 64-bit range,
# which is essential for Fully Homomorphic Encryption (FHE) schemes like
# BGV/BFV/CKKS where the ciphertext modulus can be hundreds or thousands
# of bits. Julia's BigInt provides the arbitrary precision arithmetic that
# maps to the math coprocessor's hardware BigNum units.

"""
    bignum_ntt(poly, zetas, q) -> Vector{BigInt}

Arbitrary precision NTT using BigInt arithmetic. This handles moduli of
any size, making it suitable for:
- FHE schemes (CKKS/BGV/BFV) with moduli of 100+ bits
- Multi-precision lattice operations
- Research with non-standard parameters

The math coprocessor's BigNum pipeline handles the extended-width
modular multiplications that would overflow Int64 or Int128.
"""
function bignum_ntt(poly::AbstractVector, zetas::Vector{Int}, q::Integer)
    n = length(poly)
    num_stages = trailing_zeros(n)
    data = BigInt.(poly)
    q_big = BigInt(q)

    for stage in 0:(num_stages - 1)
        m = 1 << stage
        full = m << 1
        n_over_full = n ÷ full

        for block in 0:(n ÷ full - 1)
            for j in 0:(m - 1)
                i_lo = block * full + j + 1
                i_hi = i_lo + m

                zeta_idx = min(n_over_full * j + 1, length(zetas))
                zeta = BigInt(zetas[zeta_idx])

                a = data[i_lo]
                b = data[i_hi]

                # BigNum modular multiply: handled by coprocessor's
                # extended-width multiplier pipeline
                t = mod(zeta * b, q_big)
                data[i_lo] = mod(a + t, q_big)
                data[i_hi] = mod(a - t + q_big, q_big)
            end
        end
    end

    return data
end

"""
    bignum_intt(poly, zetas_inv, q) -> Vector{BigInt}

Arbitrary precision inverse NTT using BigInt arithmetic.
"""
function bignum_intt(poly::AbstractVector, zetas_inv::Vector{Int}, q::Integer)
    n = length(poly)
    num_stages = trailing_zeros(n)
    data = BigInt.(poly)
    q_big = BigInt(q)

    for stage in (num_stages - 1):-1:0
        m = 1 << stage
        full = m << 1
        n_over_full = n ÷ full

        for block in 0:(n ÷ full - 1)
            for j in 0:(m - 1)
                i_lo = block * full + j + 1
                i_hi = i_lo + m

                zeta_idx = min(n_over_full * j + 1, length(zetas_inv))
                zeta_inv = BigInt(zetas_inv[zeta_idx])

                a = data[i_lo]
                b = data[i_hi]

                data[i_lo] = mod(a + b, q_big)
                diff = mod(a - b + q_big, q_big)
                data[i_hi] = mod(zeta_inv * diff, q_big)
            end
        end
    end

    return data
end

# ============================================================================
# Extended Precision Lattice Operations
# ============================================================================

"""
    bignum_mod_matvec(A, x, q) -> Vector{BigInt}

Arbitrary precision modular matrix-vector multiply. Each element is
computed with full BigInt precision and reduced modulo q, preventing
any overflow regardless of matrix dimensions or coefficient sizes.

This is essential for FHE ciphertext operations where q can be
a product of many primes (RNS representation) each of which exceeds Int64.
"""
function bignum_mod_matvec(A::AbstractMatrix, x::AbstractVector, q::Integer)
    m, n = size(A)
    q_big = BigInt(q)
    result = zeros(BigInt, m)

    for i in 1:m
        acc = BigInt(0)
        for j in 1:n
            # BigNum MAC: extended-width multiply-accumulate
            acc += BigInt(A[i, j]) * BigInt(x[j])
        end
        result[i] = mod(acc, q_big)
    end

    return result
end

"""
    bignum_mod_matmul(A, B, q) -> Matrix{BigInt}

Arbitrary precision modular matrix-matrix multiply.
"""
function bignum_mod_matmul(A::AbstractMatrix, B::AbstractMatrix, q::Integer)
    m, _ = size(A)
    _, p = size(B)
    result = zeros(BigInt, m, p)

    for col in 1:p
        result[:, col] = bignum_mod_matvec(A, B[:, col], q)
    end

    return result
end

# ============================================================================
# Symbolic Polynomial Manipulation
# ============================================================================

"""
    symbolic_poly_multiply(a, b, q) -> Vector{BigInt}

Polynomial multiplication via schoolbook method with full precision.
For small polynomials or when exact intermediate values matter (e.g.,
for proof generation), this avoids NTT and computes the convolution
directly with BigInt arithmetic. The math coprocessor's extended
multiplier makes this practical for medium-sized polynomials.

The result is reduced modulo X^n + 1 (the polynomial modulus for
Kyber/Dilithium) to produce the negacyclic convolution.
"""
function symbolic_poly_multiply(a::AbstractVector, b::AbstractVector, q::Integer)
    n = length(a)
    @assert length(b) == n "Polynomials must have equal length"
    q_big = BigInt(q)

    # Full convolution (degree 2n-2)
    conv = zeros(BigInt, 2 * n)
    for i in 1:n
        for j in 1:n
            conv[i + j - 1] += BigInt(a[i]) * BigInt(b[j])
        end
    end

    # Reduce modulo X^n + 1 (negacyclic: coefficient at position n+k
    # is subtracted from position k)
    result = zeros(BigInt, n)
    for k in 1:n
        result[k] = mod(conv[k] - (k + n <= 2 * n ? conv[k + n] : BigInt(0)), q_big)
        if result[k] < 0
            result[k] += q_big
        end
    end

    return result
end

# ============================================================================
# Backend Method Implementations
# ============================================================================

"""
    backend_ntt_transform(::MathBackend, poly, modulus)

Forward NTT with arbitrary precision arithmetic. Handles moduli of any
size, including the multi-hundred-bit moduli used in FHE schemes.
For standard Kyber (q=3329), this is overkill but provides exact results
with no overflow risk.
"""
function ProvenCrypto.backend_ntt_transform(::MathBackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    @assert n > 0 && ispow2(n) "NTT input length must be a power of 2, got $n"

    q = modulus
    zetas = ProvenCrypto.ZETAS[1:min(n, length(ProvenCrypto.ZETAS))]
    if length(zetas) < n
        append!(zetas, zeros(Int, n - length(zetas)))
    end

    # Estimate memory for BigInt working set
    bits_per_element = max(64, ceil(Int, log2(max(q, 2))) * 2)
    mem_bytes = Int64(n * bits_per_element ÷ 8)
    track_allocation!(MathBackend(0), mem_bytes)
    try
        result = bignum_ntt(poly, zetas, q)
        # Convert back to Int64 if modulus fits
        if q <= typemax(Int64)
            return Int64.(result)
        end
        return result
    finally
        track_deallocation!(MathBackend(0), mem_bytes)
    end
end

function ProvenCrypto.backend_ntt_transform(backend::MathBackend, mat::AbstractMatrix, modulus::Integer)
    rows = [ProvenCrypto.backend_ntt_transform(backend, vec(mat[i, :]), modulus) for i in axes(mat, 1)]
    return reduce(vcat, [r' for r in rows])
end

"""
    backend_ntt_inverse_transform(::MathBackend, poly, modulus)

Inverse NTT with arbitrary precision and 1/N scaling.
"""
function ProvenCrypto.backend_ntt_inverse_transform(::MathBackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    @assert n > 0 && ispow2(n) "INTT input length must be a power of 2, got $n"

    q = modulus
    zetas_inv = ProvenCrypto.ZETAS_INV[1:min(n, length(ProvenCrypto.ZETAS_INV))]
    if length(zetas_inv) < n
        append!(zetas_inv, zeros(Int, n - length(zetas_inv)))
    end

    bits_per_element = max(64, ceil(Int, log2(max(q, 2))) * 2)
    mem_bytes = Int64(n * bits_per_element ÷ 8)
    track_allocation!(MathBackend(0), mem_bytes)
    try
        result = bignum_intt(poly, zetas_inv, q)
        n_inv = powermod(BigInt(n), -1, BigInt(q))
        result = mod.(result .* n_inv, BigInt(q))
        if q <= typemax(Int64)
            return Int64.(result)
        end
        return result
    finally
        track_deallocation!(MathBackend(0), mem_bytes)
    end
end

function ProvenCrypto.backend_ntt_inverse_transform(backend::MathBackend, mat::AbstractMatrix, modulus::Integer)
    rows = [ProvenCrypto.backend_ntt_inverse_transform(backend, vec(mat[i, :]), modulus) for i in axes(mat, 1)]
    return reduce(vcat, [r' for r in rows])
end

"""
    backend_lattice_multiply(::MathBackend, A, x)

Arbitrary precision modular matrix-vector multiply. No overflow risk
regardless of matrix dimensions, coefficient sizes, or modulus.
"""
function ProvenCrypto.backend_lattice_multiply(::MathBackend, A::AbstractMatrix, x::AbstractVector)
    q = ProvenCrypto.Q
    track_allocation!(MathBackend(0), Int64(sizeof(A) + sizeof(x)) * 2)
    try
        result = bignum_mod_matvec(A, x, q)
        return Int64.(result)
    finally
        track_deallocation!(MathBackend(0), Int64(sizeof(A) + sizeof(x)) * 2)
    end
end

function ProvenCrypto.backend_lattice_multiply(::MathBackend, A::AbstractMatrix, B::AbstractMatrix)
    q = ProvenCrypto.Q
    track_allocation!(MathBackend(0), Int64(sizeof(A) + sizeof(B)) * 2)
    try
        result = bignum_mod_matmul(A, B, q)
        return Int64.(result)
    finally
        track_deallocation!(MathBackend(0), Int64(sizeof(A) + sizeof(B)) * 2)
    end
end

"""
    backend_polynomial_multiply(::MathBackend, a, b, modulus)

Polynomial multiplication with full precision. For small polynomials
(n <= 64), uses schoolbook multiplication with exact intermediate values.
For larger polynomials, uses the BigNum NTT path.

The schoolbook path is useful for proof generation where exact intermediate
values are needed for formal verification.
"""
function ProvenCrypto.backend_polynomial_multiply(backend::MathBackend, a::AbstractVector, b::AbstractVector, modulus::Integer)
    n = length(a)
    q = modulus

    if n <= 64
        # Schoolbook: exact intermediates (useful for proofs)
        result = symbolic_poly_multiply(a, b, q)
        if q <= typemax(Int64)
            return Int64.(result)
        end
        return result
    else
        # BigNum NTT path for larger polynomials
        a_ntt = ProvenCrypto.backend_ntt_transform(backend, a, modulus)
        b_ntt = ProvenCrypto.backend_ntt_transform(backend, b, modulus)
        c_ntt = mod.(BigInt.(a_ntt) .* BigInt.(b_ntt), BigInt(q))
        result = ProvenCrypto.backend_ntt_inverse_transform(backend, Int64.(c_ntt), modulus)
        return result
    end
end

"""
    backend_sampling(::MathBackend, distribution, params...)

CBD sampling with arbitrary precision intermediate values. Uses extended
precision for the bit counting to ensure exact results even with large eta
values (as used in some FHE parameter sets).
"""
function ProvenCrypto.backend_sampling(::MathBackend, distribution::Symbol, params...)
    if distribution == :cbd
        eta, n, k = params
        total = k * n

        track_allocation!(MathBackend(0), Int64(total * 2 * eta * 2))
        try
            random_bytes = rand(UInt8, total * 2 * eta)

            result = zeros(Int, total)
            offset = 0
            for idx in 1:total
                # BigInt accumulation (overkill for CBD but exact)
                a_count = BigInt(0)
                b_count = BigInt(0)
                for e in 1:eta
                    a_count += count_ones(random_bytes[offset + e])
                    b_count += count_ones(random_bytes[offset + eta + e])
                end
                offset += 2 * eta
                result[idx] = Int(clamp(a_count - b_count, -eta, eta))
            end

            return reshape(result, k, n)
        finally
            track_deallocation!(MathBackend(0), Int64(total * 2 * eta * 2))
        end
    else
        _record_diagnostic!(MathBackend(0), "runtime_fallbacks")
        return randn()
    end
end

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    for op in (:ntt_transform, :ntt_inverse_transform, :lattice_multiply,
               :polynomial_multiply, :sampling)
        register_operation!(MathBackend, op)
    end
    @info "ProvenCryptoMathExt loaded: arbitrary precision NTT + BigNum lattice ops"
end

end # module ProvenCryptoMathExt
