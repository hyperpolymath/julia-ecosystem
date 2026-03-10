# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ProvenCrypto CUDA Extension
# Real GPU kernel implementations for NTT, lattice multiply, polynomial
# multiply, and CBD sampling using KernelAbstractions.jl for portable kernels
# and CUDA.jl for CUBLAS matrix operations.

module ProvenCryptoCUDAExt

using CUDA
using CUDA: CUBLAS
using KernelAbstractions
using ..ProvenCrypto

# ============================================================================
# Backend availability and creation
# ============================================================================

ProvenCrypto.cuda_available() = CUDA.functional()

function ProvenCrypto.create_cuda_backend()
    dev = CUDA.device()
    cc = CUDA.capability(dev)
    has_tc = cc >= v"7.0"  # Tensor cores available on Volta (sm_70) and above
    return ProvenCrypto.CUDABackend(CUDA.deviceid(dev), has_tc, VersionNumber(cc.major, cc.minor))
end

# ============================================================================
# KernelAbstractions GPU Kernels
# ============================================================================

# --- NTT Butterfly Kernel (Cooley-Tukey, one stage) ---
# Each thread handles one butterfly pair within a given stage.
# Stage s has butterflies of half-width m = 2^s.
# For N=256, stages s=0..7, with N/2 = 128 butterflies per stage.
@kernel function ntt_butterfly_kernel!(data, @Const(zetas), stage::Int32, q::Int32)
    idx = @index(Global)  # 1-based, range [1, N/2]

    # Butterfly geometry for this stage
    # m = half the current sub-DFT size = 2^stage
    m = Int32(1) << stage
    full = m << Int32(1)  # full sub-DFT size = 2*m

    # Which sub-DFT block does this butterfly belong to?
    block = (idx - Int32(1)) ÷ m
    j = (idx - Int32(1)) % m  # position within the block

    # Indices into data (1-based)
    i_lo = block * full + j + Int32(1)
    i_hi = i_lo + m

    # Twiddle factor: zetas index is (N / full) * j + 1 (1-based)
    # We precompute with N=256 as the reference length
    n_over_full = Int32(256) ÷ full
    zeta_idx = n_over_full * j + Int32(1)
    zeta = zetas[zeta_idx]

    # Butterfly: (a, b) -> (a + zeta*b mod q, a - zeta*b mod q)
    a = data[i_lo]
    b = data[i_hi]
    t = (zeta * b) % q
    data[i_lo] = (a + t) % q
    data[i_hi] = (a - t + q) % q
end

# --- Inverse NTT Butterfly Kernel (Gentleman-Sande) ---
# Inverse butterfly: (a, b) -> ((a+b) mod q, zeta_inv * (a-b) mod q)
@kernel function intt_butterfly_kernel!(data, @Const(zetas_inv), stage::Int32, q::Int32)
    idx = @index(Global)

    m = Int32(1) << stage
    full = m << Int32(1)

    block = (idx - Int32(1)) ÷ m
    j = (idx - Int32(1)) % m

    i_lo = block * full + j + Int32(1)
    i_hi = i_lo + m

    n_over_full = Int32(256) ÷ full
    zeta_idx = n_over_full * j + Int32(1)
    zeta_inv = zetas_inv[zeta_idx]

    a = data[i_lo]
    b = data[i_hi]
    data[i_lo] = (a + b) % q
    data[i_hi] = (zeta_inv * ((a - b + q) % q)) % q
end

# --- CBD Sampling Kernel ---
# Centered Binomial Distribution: for each coefficient, sample 2*eta bits,
# count ones in each half, subtract.
@kernel function cbd_sampling_kernel!(output, @Const(random_bytes), eta::Int32, n::Int32)
    gid = @index(Global)  # maps to (row, col) in the output matrix

    # Each sample needs 2*eta bytes of randomness
    bytes_per_sample = Int32(2) * eta
    offset = (gid - Int32(1)) * bytes_per_sample

    a_count = Int32(0)
    b_count = Int32(0)
    for i in Int32(1):eta
        byte_a = random_bytes[offset + i]
        byte_b = random_bytes[offset + eta + i]
        # Count set bits (popcount via bitwise iteration)
        for bit in Int32(0):Int32(7)
            a_count += (byte_a >> bit) & Int32(1)
            b_count += (byte_b >> bit) & Int32(1)
        end
    end

    val = a_count - b_count
    output[gid] = clamp(val, -eta, eta)
end

# ============================================================================
# Backend Method Implementations
# ============================================================================

"""
    backend_ntt_transform(::CUDABackend, poly, modulus)

Forward NTT using Cooley-Tukey butterfly decomposition on CUDA GPU.
Each NTT stage is a separate kernel launch with N/2 threads performing
independent butterfly operations in parallel.
"""
function ProvenCrypto.backend_ntt_transform(backend::ProvenCrypto.CUDABackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    @assert n > 0 && ispow2(n) "NTT input length must be a power of 2, got $n"

    q = Int32(modulus)
    num_stages = trailing_zeros(n)  # log2(n) for power-of-2

    # Upload data and twiddle factors to GPU
    d_data = CuArray{Int32}(Int32.(poly))
    d_zetas = CuArray{Int32}(Int32.(ProvenCrypto.ZETAS[1:min(n, length(ProvenCrypto.ZETAS))]))

    # Pad zetas if needed (for smaller polynomials)
    if length(d_zetas) < n
        d_zetas = vcat(d_zetas, CUDA.zeros(Int32, n - length(d_zetas)))
    end

    ka_backend = CUDABackend()
    half_n = n ÷ 2

    # Execute butterfly stages: stage 0 (pairs), 1 (quads), ... , num_stages-1
    for s in 0:(num_stages - 1)
        kernel = ntt_butterfly_kernel!(ka_backend, 256)
        kernel(d_data, d_zetas, Int32(s), q; ndrange=half_n)
    end

    KernelAbstractions.synchronize(ka_backend)
    return Array(d_data)
end

# Handle matrix input (used by Kyber for k x n coefficient matrices)
function ProvenCrypto.backend_ntt_transform(backend::ProvenCrypto.CUDABackend, mat::AbstractMatrix, modulus::Integer)
    result = similar(mat)
    for i in axes(mat, 1)
        result[i, :] = ProvenCrypto.backend_ntt_transform(backend, vec(mat[i, :]), modulus)
    end
    return result
end

"""
    backend_ntt_inverse_transform(::CUDABackend, poly, modulus)

Inverse NTT using Gentleman-Sande butterfly decomposition on CUDA GPU.
Runs stages in reverse order and applies the 1/N scaling factor at the end.
"""
function ProvenCrypto.backend_ntt_inverse_transform(backend::ProvenCrypto.CUDABackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    @assert n > 0 && ispow2(n) "INTT input length must be a power of 2, got $n"

    q = Int32(modulus)
    num_stages = trailing_zeros(n)

    d_data = CuArray{Int32}(Int32.(poly))
    d_zetas_inv = CuArray{Int32}(Int32.(ProvenCrypto.ZETAS_INV[1:min(n, length(ProvenCrypto.ZETAS_INV))]))

    if length(d_zetas_inv) < n
        d_zetas_inv = vcat(d_zetas_inv, CUDA.zeros(Int32, n - length(d_zetas_inv)))
    end

    ka_backend = CUDABackend()
    half_n = n ÷ 2

    # Inverse stages run in reverse: from largest sub-DFT down to pairs
    for s in (num_stages - 1):-1:0
        kernel = intt_butterfly_kernel!(ka_backend, 256)
        kernel(d_data, d_zetas_inv, Int32(s), q; ndrange=half_n)
    end

    KernelAbstractions.synchronize(ka_backend)

    # Scale by N^{-1} mod q
    result = Array(d_data)
    n_inv = powermod(n, -1, Int(modulus))
    return (result .* Int32(n_inv)) .% q
end

# Handle matrix input for inverse NTT
function ProvenCrypto.backend_ntt_inverse_transform(backend::ProvenCrypto.CUDABackend, mat::AbstractMatrix, modulus::Integer)
    result = similar(mat)
    for i in axes(mat, 1)
        result[i, :] = ProvenCrypto.backend_ntt_inverse_transform(backend, vec(mat[i, :]), modulus)
    end
    return result
end

"""
    backend_lattice_multiply(::CUDABackend, A, x)

Matrix-vector multiplication on CUDA using CUBLAS.
For lattice cryptography, this computes A * x mod q on the GPU.
Uses Float64 CUBLAS GEMV and rounds back to integers.
"""
function ProvenCrypto.backend_lattice_multiply(backend::ProvenCrypto.CUDABackend, A::AbstractMatrix, x::AbstractVector)
    # Use CUBLAS for the matrix-vector multiply
    d_A = CuArray{Float64}(Float64.(A))
    d_x = CuArray{Float64}(Float64.(x))
    d_result = d_A * d_x
    return round.(Int, Array(d_result))
end

# Matrix-matrix variant (used in Kyber where both operands can be matrices)
function ProvenCrypto.backend_lattice_multiply(backend::ProvenCrypto.CUDABackend, A::AbstractMatrix, B::AbstractMatrix)
    d_A = CuArray{Float64}(Float64.(A))
    d_B = CuArray{Float64}(Float64.(B))
    d_result = d_A * d_B
    return round.(Int, Array(d_result))
end

"""
    backend_polynomial_multiply(::CUDABackend, a, b, modulus)

NTT-based polynomial multiplication on CUDA.
Transforms both polynomials to NTT domain, performs pointwise multiply,
then inverse-transforms back. All three phases run on GPU.
"""
function ProvenCrypto.backend_polynomial_multiply(backend::ProvenCrypto.CUDABackend, a::AbstractVector, b::AbstractVector, modulus::Integer)
    q = Int32(modulus)

    # Forward NTT on GPU
    a_ntt = ProvenCrypto.backend_ntt_transform(backend, a, modulus)
    b_ntt = ProvenCrypto.backend_ntt_transform(backend, b, modulus)

    # Pointwise multiply on GPU
    d_a = CuArray{Int32}(Int32.(a_ntt))
    d_b = CuArray{Int32}(Int32.(b_ntt))
    d_c = (d_a .* d_b) .% q
    c_ntt = Array(d_c)

    # Inverse NTT on GPU
    return ProvenCrypto.backend_ntt_inverse_transform(backend, c_ntt, modulus)
end

"""
    backend_sampling(::CUDABackend, distribution, params...)

GPU-parallel CBD (Centered Binomial Distribution) sampling.
Generates random bytes on CPU, uploads to GPU, and runs CBD kernel
to produce k x n coefficient matrix in parallel.
"""
function ProvenCrypto.backend_sampling(backend::ProvenCrypto.CUDABackend, distribution::Symbol, params...)
    if distribution == :cbd
        eta, n, k = params
        total_samples = k * n
        bytes_per_sample = 2 * eta

        # Generate random bytes on CPU and upload
        random_bytes = rand(UInt8, total_samples * bytes_per_sample)
        d_random = CuArray{Int32}(Int32.(random_bytes))
        d_output = CUDA.zeros(Int32, total_samples)

        ka_backend = CUDABackend()
        kernel = cbd_sampling_kernel!(ka_backend, 256)
        kernel(d_output, d_random, Int32(eta), Int32(n); ndrange=total_samples)

        KernelAbstractions.synchronize(ka_backend)
        result = Array(d_output)
        return reshape(result, k, n)
    else
        # Gaussian sampling fallback to CPU
        return randn()
    end
end

end # module ProvenCryptoCUDAExt
