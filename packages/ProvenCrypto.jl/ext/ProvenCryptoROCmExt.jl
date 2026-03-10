# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ProvenCrypto ROCm Extension
# Real GPU kernel implementations for AMD GPUs (MI-series, RDNA) using
# KernelAbstractions.jl for portable kernels and AMDGPU.jl for device arrays.

module ProvenCryptoROCmExt

using AMDGPU
using KernelAbstractions
using ..ProvenCrypto

# ============================================================================
# Backend availability and creation
# ============================================================================

ProvenCrypto.rocm_available() = AMDGPU.functional()

function ProvenCrypto.create_rocm_backend()
    dev = AMDGPU.device()
    agent = AMDGPU.get_default_agent()
    gcn_arch = string(AMDGPU.device_id(agent))
    has_matrix = occursin(r"gfx9[0-9]{2}", gcn_arch)  # MI-series GPUs
    return ProvenCrypto.ROCmBackend(0, has_matrix, gcn_arch)
end

# ============================================================================
# KernelAbstractions GPU Kernels
# ============================================================================

@kernel function ntt_butterfly_kernel!(data, @Const(zetas), stage::Int32, q::Int32)
    idx = @index(Global)

    m = Int32(1) << stage
    full = m << Int32(1)

    block = (idx - Int32(1)) ÷ m
    j = (idx - Int32(1)) % m

    i_lo = block * full + j + Int32(1)
    i_hi = i_lo + m

    n_over_full = Int32(256) ÷ full
    zeta_idx = n_over_full * j + Int32(1)
    zeta = zetas[zeta_idx]

    a = data[i_lo]
    b = data[i_hi]
    t = (zeta * b) % q
    data[i_lo] = (a + t) % q
    data[i_hi] = (a - t + q) % q
end

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

@kernel function cbd_sampling_kernel!(output, @Const(random_bytes), eta::Int32, n::Int32)
    gid = @index(Global)

    bytes_per_sample = Int32(2) * eta
    offset = (gid - Int32(1)) * bytes_per_sample

    a_count = Int32(0)
    b_count = Int32(0)
    for i in Int32(1):eta
        byte_a = random_bytes[offset + i]
        byte_b = random_bytes[offset + eta + i]
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
    backend_ntt_transform(::ROCmBackend, poly, modulus)

Forward NTT using Cooley-Tukey butterflies on AMD ROCm GPU.
"""
function ProvenCrypto.backend_ntt_transform(backend::ProvenCrypto.ROCmBackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    @assert n > 0 && ispow2(n) "NTT input length must be a power of 2, got $n"

    q = Int32(modulus)
    num_stages = trailing_zeros(n)

    d_data = ROCArray{Int32}(Int32.(poly))
    zetas_padded = Int32.(ProvenCrypto.ZETAS[1:min(n, length(ProvenCrypto.ZETAS))])
    if length(zetas_padded) < n
        append!(zetas_padded, zeros(Int32, n - length(zetas_padded)))
    end
    d_zetas = ROCArray{Int32}(zetas_padded)

    ka_backend = ROCBackend()
    half_n = n ÷ 2

    for s in 0:(num_stages - 1)
        kernel = ntt_butterfly_kernel!(ka_backend, 256)
        kernel(d_data, d_zetas, Int32(s), q; ndrange=half_n)
    end

    KernelAbstractions.synchronize(ka_backend)
    return Array(d_data)
end

function ProvenCrypto.backend_ntt_transform(backend::ProvenCrypto.ROCmBackend, mat::AbstractMatrix, modulus::Integer)
    result = similar(mat)
    for i in axes(mat, 1)
        result[i, :] = ProvenCrypto.backend_ntt_transform(backend, vec(mat[i, :]), modulus)
    end
    return result
end

"""
    backend_ntt_inverse_transform(::ROCmBackend, poly, modulus)

Inverse NTT using Gentleman-Sande butterflies on AMD ROCm GPU.
"""
function ProvenCrypto.backend_ntt_inverse_transform(backend::ProvenCrypto.ROCmBackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    @assert n > 0 && ispow2(n) "INTT input length must be a power of 2, got $n"

    q = Int32(modulus)
    num_stages = trailing_zeros(n)

    d_data = ROCArray{Int32}(Int32.(poly))
    zetas_inv_padded = Int32.(ProvenCrypto.ZETAS_INV[1:min(n, length(ProvenCrypto.ZETAS_INV))])
    if length(zetas_inv_padded) < n
        append!(zetas_inv_padded, zeros(Int32, n - length(zetas_inv_padded)))
    end
    d_zetas_inv = ROCArray{Int32}(zetas_inv_padded)

    ka_backend = ROCBackend()
    half_n = n ÷ 2

    for s in (num_stages - 1):-1:0
        kernel = intt_butterfly_kernel!(ka_backend, 256)
        kernel(d_data, d_zetas_inv, Int32(s), q; ndrange=half_n)
    end

    KernelAbstractions.synchronize(ka_backend)

    result = Array(d_data)
    n_inv = powermod(n, -1, Int(modulus))
    return (result .* Int32(n_inv)) .% q
end

function ProvenCrypto.backend_ntt_inverse_transform(backend::ProvenCrypto.ROCmBackend, mat::AbstractMatrix, modulus::Integer)
    result = similar(mat)
    for i in axes(mat, 1)
        result[i, :] = ProvenCrypto.backend_ntt_inverse_transform(backend, vec(mat[i, :]), modulus)
    end
    return result
end

"""
    backend_lattice_multiply(::ROCmBackend, A, x)

Matrix multiplication on AMD ROCm GPU using rocBLAS via AMDGPU.jl.
"""
function ProvenCrypto.backend_lattice_multiply(backend::ProvenCrypto.ROCmBackend, A::AbstractMatrix, x::AbstractVector)
    d_A = ROCArray{Float64}(Float64.(A))
    d_x = ROCArray{Float64}(Float64.(x))
    d_result = d_A * d_x
    return round.(Int, Array(d_result))
end

function ProvenCrypto.backend_lattice_multiply(backend::ProvenCrypto.ROCmBackend, A::AbstractMatrix, B::AbstractMatrix)
    d_A = ROCArray{Float64}(Float64.(A))
    d_B = ROCArray{Float64}(Float64.(B))
    d_result = d_A * d_B
    return round.(Int, Array(d_result))
end

"""
    backend_polynomial_multiply(::ROCmBackend, a, b, modulus)

NTT-based polynomial multiplication on AMD ROCm GPU.
"""
function ProvenCrypto.backend_polynomial_multiply(backend::ProvenCrypto.ROCmBackend, a::AbstractVector, b::AbstractVector, modulus::Integer)
    q = Int32(modulus)

    a_ntt = ProvenCrypto.backend_ntt_transform(backend, a, modulus)
    b_ntt = ProvenCrypto.backend_ntt_transform(backend, b, modulus)

    d_a = ROCArray{Int32}(Int32.(a_ntt))
    d_b = ROCArray{Int32}(Int32.(b_ntt))
    d_c = (d_a .* d_b) .% q
    c_ntt = Array(d_c)

    return ProvenCrypto.backend_ntt_inverse_transform(backend, c_ntt, modulus)
end

"""
    backend_sampling(::ROCmBackend, distribution, params...)

GPU-parallel CBD sampling on AMD ROCm.
"""
function ProvenCrypto.backend_sampling(backend::ProvenCrypto.ROCmBackend, distribution::Symbol, params...)
    if distribution == :cbd
        eta, n, k = params
        total_samples = k * n
        bytes_per_sample = 2 * eta

        random_bytes = rand(UInt8, total_samples * bytes_per_sample)
        d_random = ROCArray{Int32}(Int32.(random_bytes))
        d_output = AMDGPU.zeros(Int32, total_samples)

        ka_backend = ROCBackend()
        kernel = cbd_sampling_kernel!(ka_backend, 256)
        kernel(d_output, d_random, Int32(eta), Int32(n); ndrange=total_samples)

        KernelAbstractions.synchronize(ka_backend)
        result = Array(d_output)
        return reshape(result, k, n)
    else
        return randn()
    end
end

end # module ProvenCryptoROCmExt
