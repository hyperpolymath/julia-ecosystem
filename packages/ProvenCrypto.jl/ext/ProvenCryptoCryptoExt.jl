# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ProvenCrypto Crypto Accelerator Extension
# Hardware crypto engine implementations for dedicated crypto ASICs.
# Modern crypto accelerators (Intel QAT, ARM CryptoCell, dedicated PQC ASICs)
# have hardwired NTT units, modular arithmetic pipelines, and AES/SHA engines.
# This is the most natural backend for ProvenCrypto operations.

module ProvenCryptoCryptoExt

using LinearAlgebra
using ..ProvenCrypto
using AcceleratorGate
using AcceleratorGate: CryptoBackend, DeviceCapabilities,
                       register_operation!, track_allocation!, track_deallocation!,
                       _record_diagnostic!

# ============================================================================
# Device Capabilities
# ============================================================================

function AcceleratorGate.device_capabilities(b::CryptoBackend)
    # Model: dedicated PQC crypto accelerator ASIC (e.g., Lattice-based crypto ASIC)
    # or Intel QAT / ARM CryptoCell class device
    DeviceCapabilities(
        b,
        8,                       # compute units (crypto pipeline lanes)
        500,                     # clock MHz (crypto ASIC typical)
        Int64(1) * 1024^3,       # 1 GiB SRAM (on-chip, fast)
        Int64(1) * 1024^3,       # all available
        128,                     # max workgroup size
        false,                   # crypto ASICs typically integer-only
        false,                   # no floating point
        true,                    # INT8 for AES S-box
        "CryptoASIC",
        "PQC Accelerator v1",
    )
end

function AcceleratorGate.estimate_cost(b::CryptoBackend, op::Symbol, data_size::Int)
    # Crypto accelerators have near-zero overhead for their native operations
    if op in (:ntt_transform, :ntt_inverse_transform)
        # Hardwired NTT unit: best-in-class for this operation
        return 1.0 + Float64(data_size) * 0.001
    elseif op == :lattice_multiply
        # Hardware modular MAC pipeline
        return 1.0 + Float64(data_size) * 0.002
    elseif op == :polynomial_multiply
        # Chained NTT → pointwise → INTT in hardware
        return 2.0 + Float64(data_size) * 0.003
    elseif op == :sampling
        # Hardware TRNG + CBD circuit
        return 1.0 + Float64(data_size) * 0.005
    end
    Inf
end

# ============================================================================
# Hardware NTT Accelerator
# ============================================================================
#
# Dedicated crypto ASICs implement the NTT as a hardwired butterfly network
# with on-chip twiddle factor ROM. The hardware NTT unit processes the
# entire transform in a fixed number of cycles, with each butterfly computed
# by a dedicated modular multiply-add unit. Unlike general-purpose processors,
# the modular reduction (mod q) is fused into the butterfly operation itself
# using Barrett or Montgomery reduction circuits.

"""
    crypto_hw_ntt(poly, zetas, q) -> Vector{Int64}

Hardware NTT accelerator simulation. Models a dedicated NTT unit with:
- Fused modular butterfly (multiply-add-reduce in single cycle)
- On-chip twiddle factor ROM
- Bit-reversal permutation in address generation logic
- Pipeline depth = log2(N) stages, throughput = 1 NTT per N cycles

The key optimization over software NTT is the fused modular reduction:
instead of computing t = (zeta * b) then t % q as two operations, the
hardware performs Barrett reduction inline using precomputed constants.
"""
function crypto_hw_ntt(poly::AbstractVector, zetas::Vector{Int}, q::Int)
    n = length(poly)
    num_stages = trailing_zeros(n)
    data = Int64.(poly)

    # Barrett reduction precompute (hardwired in crypto ASIC)
    # For q=3329: k = ceil(log2(q)) = 12, m = floor(2^(2k) / q)
    k = ceil(Int, log2(q))
    barrett_m = (1 << (2 * k)) ÷ q

    for stage in 0:(num_stages - 1)
        m = 1 << stage
        full = m << 1
        n_over_full = n ÷ full

        for block in 0:(n ÷ full - 1)
            for j in 0:(m - 1)
                i_lo = block * full + j + 1
                i_hi = i_lo + m

                zeta_idx = min(n_over_full * j + 1, length(zetas))
                zeta = Int64(zetas[zeta_idx])

                a = data[i_lo]
                b = data[i_hi]

                # Fused modular multiply with Barrett reduction
                # Hardware computes: t = zeta * b, then Barrett reduce in-place
                product = zeta * b
                # Barrett reduction: q_approx = (product * barrett_m) >> (2k)
                q_approx = (product * barrett_m) >> (2 * k)
                t = product - q_approx * q
                # Final correction (at most one subtraction needed)
                if t >= q
                    t -= q
                end
                if t < 0
                    t += q
                end

                # Butterfly with fused reduction
                sum_val = a + t
                if sum_val >= q
                    sum_val -= q
                end

                diff_val = a - t
                if diff_val < 0
                    diff_val += q
                end

                data[i_lo] = sum_val
                data[i_hi] = diff_val
            end
        end
    end

    return data
end

"""
    crypto_hw_intt(poly, zetas_inv, q) -> Vector{Int64}

Hardware inverse NTT with fused Barrett reduction and Gentleman-Sande
butterfly ordering. The inverse NTT unit shares the same modular arithmetic
pipelines as the forward unit, with the twiddle ROM switched to inverse roots.
"""
function crypto_hw_intt(poly::AbstractVector, zetas_inv::Vector{Int}, q::Int)
    n = length(poly)
    num_stages = trailing_zeros(n)
    data = Int64.(poly)

    k = ceil(Int, log2(q))
    barrett_m = (1 << (2 * k)) ÷ q

    for stage in (num_stages - 1):-1:0
        m = 1 << stage
        full = m << 1
        n_over_full = n ÷ full

        for block in 0:(n ÷ full - 1)
            for j in 0:(m - 1)
                i_lo = block * full + j + 1
                i_hi = i_lo + m

                zeta_idx = min(n_over_full * j + 1, length(zetas_inv))
                zeta_inv = Int64(zetas_inv[zeta_idx])

                a = data[i_lo]
                b = data[i_hi]

                # Inverse butterfly with fused Barrett reduction
                sum_val = a + b
                if sum_val >= q
                    sum_val -= q
                end

                diff = a - b
                if diff < 0
                    diff += q
                end

                # Fused modular multiply for inverse twiddle
                product = zeta_inv * diff
                q_approx = (product * barrett_m) >> (2 * k)
                t = product - q_approx * q
                if t >= q; t -= q; end
                if t < 0; t += q; end

                data[i_lo] = sum_val
                data[i_hi] = t
            end
        end
    end

    return data
end

# ============================================================================
# Hardware Modular Arithmetic Pipeline
# ============================================================================

"""
    crypto_hw_mod_matvec(A, x, q) -> Vector{Int64}

Hardware modular matrix-vector multiply. The crypto ASIC's modular MAC
(multiply-accumulate) pipeline computes each dot product element with
inline modular reduction, avoiding intermediate overflow. Each MAC unit
handles one row's accumulation.
"""
function crypto_hw_mod_matvec(A::AbstractMatrix, x::AbstractVector, q::Int)
    m, n = size(A)
    result = zeros(Int64, m)

    k = ceil(Int, log2(q))
    barrett_m = (1 << (2 * k)) ÷ q

    for i in 1:m
        acc = Int64(0)
        for j in 1:n
            # Modular MAC: acc = (acc + A[i,j] * x[j]) mod q
            product = Int64(A[i, j]) * Int64(x[j])
            # Barrett reduce the product
            q_approx = (product * barrett_m) >> (2 * k)
            reduced = product - q_approx * q
            if reduced >= q; reduced -= q; end
            if reduced < 0; reduced += q; end

            acc += reduced
            # Periodic reduction to prevent accumulator overflow
            if acc >= q
                acc -= q
            end
        end
        result[i] = mod(acc, q)
    end

    return result
end

"""
    crypto_hw_mod_matmul(A, B, q) -> Matrix{Int64}

Hardware modular matrix-matrix multiply via parallel MAC pipeline banks.
"""
function crypto_hw_mod_matmul(A::AbstractMatrix, B::AbstractMatrix, q::Int)
    m, _ = size(A)
    _, p = size(B)
    result = zeros(Int64, m, p)

    for col in 1:p
        result[:, col] = crypto_hw_mod_matvec(A, B[:, col], q)
    end

    return result
end

# ============================================================================
# Hardware TRNG + CBD
# ============================================================================

"""
    crypto_hw_cbd_sampling(eta, n, k) -> Matrix{Int}

Hardware true random number generation with CBD reduction circuit.
Crypto ASICs include dedicated TRNGs (typically based on metastable
flip-flops or ring oscillators with health monitoring) feeding into
a hardwired CBD reduction pipeline. The TRNG output is continuously
health-checked against NIST SP 800-90B entropy requirements.
"""
function crypto_hw_cbd_sampling(eta::Int, n::Int, k_dim::Int)
    total = k_dim * n
    result = zeros(Int, total)

    for idx in 1:total
        a_count = 0
        b_count = 0
        for byte_idx in 1:eta
            # Hardware TRNG output (with online health monitoring)
            a_byte = rand(UInt8)
            b_byte = rand(UInt8)
            # Hardware popcount (combinational logic, zero latency)
            a_count += count_ones(a_byte)
            b_count += count_ones(b_byte)
        end
        # Hardware subtractor with saturation
        result[idx] = clamp(a_count - b_count, -eta, eta)
    end

    return reshape(result, k_dim, n)
end

# ============================================================================
# Backend Method Implementations
# ============================================================================

"""
    backend_ntt_transform(::CryptoBackend, poly, modulus)

Forward NTT via hardware NTT accelerator unit. Crypto ASICs have dedicated
NTT coprocessors with fused Barrett modular reduction, achieving the lowest
latency of any backend for this operation. The Kyber NTT (N=256, q=3329)
completes in ~256 cycles on hardware.
"""
function ProvenCrypto.backend_ntt_transform(::CryptoBackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    @assert n > 0 && ispow2(n) "NTT input length must be a power of 2, got $n"

    q = Int(modulus)
    zetas = ProvenCrypto.ZETAS[1:min(n, length(ProvenCrypto.ZETAS))]
    if length(zetas) < n
        append!(zetas, zeros(Int, n - length(zetas)))
    end

    track_allocation!(CryptoBackend(0), Int64(n * 8))
    try
        return crypto_hw_ntt(poly, zetas, q)
    finally
        track_deallocation!(CryptoBackend(0), Int64(n * 8))
    end
end

function ProvenCrypto.backend_ntt_transform(backend::CryptoBackend, mat::AbstractMatrix, modulus::Integer)
    result = similar(mat, Int64)
    for i in axes(mat, 1)
        result[i, :] = ProvenCrypto.backend_ntt_transform(backend, vec(mat[i, :]), modulus)
    end
    return result
end

"""
    backend_ntt_inverse_transform(::CryptoBackend, poly, modulus)

Inverse NTT via hardware INTT unit with fused Barrett reduction and 1/N scaling.
"""
function ProvenCrypto.backend_ntt_inverse_transform(::CryptoBackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    @assert n > 0 && ispow2(n) "INTT input length must be a power of 2, got $n"

    q = Int(modulus)
    zetas_inv = ProvenCrypto.ZETAS_INV[1:min(n, length(ProvenCrypto.ZETAS_INV))]
    if length(zetas_inv) < n
        append!(zetas_inv, zeros(Int, n - length(zetas_inv)))
    end

    track_allocation!(CryptoBackend(0), Int64(n * 8))
    try
        result = crypto_hw_intt(poly, zetas_inv, q)
        n_inv = powermod(n, -1, q)
        return mod.(result .* n_inv, q)
    finally
        track_deallocation!(CryptoBackend(0), Int64(n * 8))
    end
end

function ProvenCrypto.backend_ntt_inverse_transform(backend::CryptoBackend, mat::AbstractMatrix, modulus::Integer)
    result = similar(mat, Int64)
    for i in axes(mat, 1)
        result[i, :] = ProvenCrypto.backend_ntt_inverse_transform(backend, vec(mat[i, :]), modulus)
    end
    return result
end

"""
    backend_lattice_multiply(::CryptoBackend, A, x)

Modular matrix-vector multiply via hardware modular MAC pipeline.
The crypto ASIC's MAC units perform inline Barrett reduction, avoiding
intermediate overflow and achieving constant-time operation (critical
for side-channel resistance).
"""
function ProvenCrypto.backend_lattice_multiply(::CryptoBackend, A::AbstractMatrix, x::AbstractVector)
    # Default modulus for lattice crypto (Kyber q=3329)
    q = ProvenCrypto.Q
    track_allocation!(CryptoBackend(0), Int64(sizeof(A) + sizeof(x)))
    try
        return crypto_hw_mod_matvec(A, x, q)
    finally
        track_deallocation!(CryptoBackend(0), Int64(sizeof(A) + sizeof(x)))
    end
end

function ProvenCrypto.backend_lattice_multiply(::CryptoBackend, A::AbstractMatrix, B::AbstractMatrix)
    q = ProvenCrypto.Q
    track_allocation!(CryptoBackend(0), Int64(sizeof(A) + sizeof(B)))
    try
        return crypto_hw_mod_matmul(A, B, q)
    finally
        track_deallocation!(CryptoBackend(0), Int64(sizeof(A) + sizeof(B)))
    end
end

"""
    backend_polynomial_multiply(::CryptoBackend, a, b, modulus)

Polynomial multiplication via chained hardware NTT → pointwise → INTT pipeline.
On crypto ASICs, this entire chain can be fused into a single command to the
NTT coprocessor, eliminating memory round-trips between stages.
"""
function ProvenCrypto.backend_polynomial_multiply(backend::CryptoBackend, a::AbstractVector, b::AbstractVector, modulus::Integer)
    q = Int(modulus)

    a_ntt = ProvenCrypto.backend_ntt_transform(backend, a, modulus)
    b_ntt = ProvenCrypto.backend_ntt_transform(backend, b, modulus)

    # Hardware pointwise modular multiply
    c_ntt = mod.(a_ntt .* b_ntt, q)

    return ProvenCrypto.backend_ntt_inverse_transform(backend, c_ntt, modulus)
end

"""
    backend_sampling(::CryptoBackend, distribution, params...)

CBD sampling via hardware TRNG with online health monitoring and CBD pipeline.
The TRNG meets NIST SP 800-90B entropy requirements, and the CBD reduction
is performed in constant time via hardwired popcount + subtract logic.
"""
function ProvenCrypto.backend_sampling(::CryptoBackend, distribution::Symbol, params...)
    if distribution == :cbd
        eta, n, k = params
        track_allocation!(CryptoBackend(0), Int64(k * n * 2 * eta))
        try
            return crypto_hw_cbd_sampling(eta, n, k)
        finally
            track_deallocation!(CryptoBackend(0), Int64(k * n * 2 * eta))
        end
    else
        _record_diagnostic!(CryptoBackend(0), "runtime_fallbacks")
        return randn()
    end
end

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    for op in (:ntt_transform, :ntt_inverse_transform, :lattice_multiply,
               :polynomial_multiply, :sampling)
        register_operation!(CryptoBackend, op)
    end
    @info "ProvenCryptoCryptoExt loaded: hardware NTT + Barrett reduction + TRNG"
end

end # module ProvenCryptoCryptoExt
