# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Hardware acceleration backend detection and abstraction.

Supports:
- NVIDIA CUDA (Tensor cores for lattice crypto)
- AMD ROCm (Matrix cores)
- Apple Metal (Neural Engine for post-quantum)
- Intel oneAPI (NPU/GPU)
- Google TPU (via XLA)
- FPGA (OpenCL)
- CPU SIMD (AVX2, AVX-512, NEON, SVE)
"""

abstract type AbstractCryptoBackend end

struct CPUBackend <: AbstractCryptoBackend
    simd_level::Symbol  # :none, :sse, :avx, :avx2, :avx512, :neon, :sve
    threads::Int
end

struct CUDABackend <: AbstractCryptoBackend
    device::Int
    has_tensor_cores::Bool
    compute_capability::VersionNumber
end

struct ROCmBackend <: AbstractCryptoBackend
    device::Int
    has_matrix_cores::Bool
    gcn_arch::String
end

struct MetalBackend <: AbstractCryptoBackend
    device::Int
    has_neural_engine::Bool
    apple_silicon_generation::Int  # M1=1, M2=2, M3=3, M4=4
end

struct OneAPIBackend <: AbstractCryptoBackend
    device::Int
    has_npu::Bool
    device_type::Symbol  # :gpu, :cpu, :fpga
end

struct TPUBackend <: AbstractCryptoBackend
    version::Int
    topology::String
end

# --- Backend Availability Detection ---
# Default implementations (overridden by package extensions)
"""
    cuda_available() -> Bool

Check if CUDA is available. Override in extensions.
"""
cuda_available() = false

"""
    rocm_available() -> Bool

Check if ROCm is available. Override in extensions.
"""
rocm_available() = false

"""
    metal_available() -> Bool

Check if Metal is available. Override in extensions.
"""
metal_available() = false

"""
    oneapi_available() -> Bool

Check if oneAPI is available. Override in extensions.
"""
oneapi_available() = false

"""
    tpu_available() -> Bool

Check if TPU is available. Override in extensions.
"""
tpu_available() = false

# --- Hardware Detection Utilities ---
"""
    detect_simd_level() -> Symbol

Detect CPU SIMD instruction set support.
Returns highest available: :avx512, :avx2, :avx, :sse, :neon, :sve, :none
"""
function detect_simd_level()
    if Sys.isapple() && Sys.ARCH === :aarch64
        return :neon  # Apple Silicon always has NEON
    elseif Sys.ARCH === :aarch64
        if haskey(ENV, "JULIA_CPU_TARGET") && occursin("sve", ENV["JULIA_CPU_TARGET"])
            return :sve
        else
            return :neon
        end
    elseif Sys.ARCH === :x86_64
        try
            cpu_info = lowercase(Sys.CPU_NAME)
            if occursin("avx512", cpu_info)
                return :avx512
            elseif occursin("avx2", cpu_info)
                return :avx2
            elseif occursin("avx", cpu_info)
                return :avx
            else
                return :sse  # Baseline x86-64 has SSE2
            end
        catch
            return :sse
        end
    else
        return :none
    end
end

"""
    detect_apple_silicon_generation() -> Int

Detect Apple Silicon generation (M1=1, M2=2, M3=3, M4=4, etc.)
"""
function detect_apple_silicon_generation()
    if !Sys.isapple() || Sys.ARCH !== :aarch64
        return 0
    end
    try
        output = read(`system_profiler SPHardwareDataType`, String)
        if occursin("M4", output)
            return 4
        elseif occursin("M3", output)
            return 3
        elseif occursin("M2", output)
            return 2
        elseif occursin("M1", output)
            return 1
        end
    catch
        return 1  # Fallback: assume M1
    end
    return 0
end

"""
    detect_hardware() -> AbstractCryptoBackend

Auto-detect best available hardware backend for cryptography.
Priority order: TPU > CUDA > Metal > oneAPI > ROCm > CPU.
"""
function detect_hardware()
    # TPU (highest priority)
    if tpu_available()
        try
            if haskey(ENV, "TPU_NAME")
                return TPUBackend(4, ENV["TPU_NAME"])
            end
        catch e
            @warn "TPU detection failed" exception=e
        end
    end

    # CUDA
    if cuda_available()
        try
            return CUDABackend(0, true, v"8.0")  # Placeholder; override in extension
        catch e
            @warn "CUDA backend creation failed" exception=e
        end
    end

    # Metal
    if metal_available()
        try
            gen = detect_apple_silicon_generation()
            has_ne = gen >= 2  # Neural Engine improved in M2+
            return MetalBackend(0, has_ne, gen)
        catch e
            @warn "Metal backend creation failed" exception=e
        end
    end

    # oneAPI
    if oneapi_available()
        try
            return OneAPIBackend(0, false, :gpu)  # Placeholder; override in extension
        catch e
            @warn "oneAPI backend creation failed" exception=e
        end
    end

    # ROCm
    if rocm_available()
        try
            return ROCmBackend(0, false, "gfx900")  # Placeholder; override in extension
        catch e
            @warn "ROCm backend creation failed" exception=e
        end
    end

    # Fallback to CPU
    simd = detect_simd_level()
    threads = Threads.nthreads()
    return CPUBackend(simd, threads)
end

# --- Backend-Specific Operations ---
# Dispatch to backend-specific implementations (defined in extensions)
"""
    backend_lattice_multiply(backend::AbstractCryptoBackend, args...)
"""
function backend_lattice_multiply(backend::AbstractCryptoBackend, args...)
    throw(MethodError(backend_lattice_multiply, (backend, args...)))
end

"""
    backend_ntt_transform(backend::AbstractCryptoBackend, args...)
"""
function backend_ntt_transform(backend::AbstractCryptoBackend, args...)
    throw(MethodError(backend_ntt_transform, (backend, args...)))
end

"""
    backend_polynomial_multiply(backend::AbstractCryptoBackend, args...)
"""
function backend_polynomial_multiply(backend::AbstractCryptoBackend, args...)
    throw(MethodError(backend_polynomial_multiply, (backend, args...)))
end

"""
    backend_sampling(backend::AbstractCryptoBackend, args...)
"""
function backend_sampling(backend::AbstractCryptoBackend, args...)
    throw(MethodError(backend_sampling, (backend, args...)))
end

# CPU fallback implementations
function backend_lattice_multiply(::CPUBackend, A::AbstractMatrix, x::AbstractVector)
    return A * x  # Basic fallback
end

# Kyber NTT parameters
const Q = 3329
const ZETAS = [
    228, 222, 216, 210, 204, 198, 192, 186, 180, 174, 168, 162, 156, 150, 144, 138, 132, 126, 120, 114,
    108, 102, 96, 90, 84, 78, 72, 66, 60, 54, 48, 42, 36, 30, 24, 18, 12, 6, 0, 3323, 3317, 3311, 3305,
    3299, 3293, 3287, 3281, 3275, 3269, 3263, 3257, 3251, 3245, 3239, 3233, 3227, 3221, 3215, 3209,
    3203, 3197, 3191, 3185, 3179, 3173, 3167, 3161, 3155, 3149, 3143, 3137, 3131, 3125, 3119, 3113,
    3107, 3101, 3095, 3089, 3083, 3077, 3071, 3065, 3059, 3053, 3047, 3041, 3035, 3029, 3023, 3017,
    3011, 3005, 2999, 2993, 2987, 2981, 2975, 2969, 2963, 2957, 2951, 2945, 2939, 2933, 2927, 2921,
    2915, 2909, 2903, 2897, 2891, 2885, 2879, 2873, 2867, 2861, 2855, 2849, 2843, 2837, 2831, 2825
]

function ntt_cooley_tukey(p::AbstractVector, zetas::Vector{Int}, q::Int)
    n = length(p)
    if n == 1
        return p
    end

    even = ntt_cooley_tukey(p[1:2:end], zetas, q)
    odd = ntt_cooley_tukey(p[2:2:end], zetas, q)

    zetas_len = length(zetas)
    result = similar(p)
    for i in 1:n÷2
        zeta = zetas[zetas_len * (i-1) ÷ (n÷2) + 1]
        t = (zeta * odd[i]) % q
        result[i] = (even[i] + t) % q
        result[i + n÷2] = (even[i] - t + q) % q
    end
    return result
end

const ZETAS_INV = [
    3102, 3113, 3124, 3135, 3146, 3157, 3168, 3179, 3190, 3201, 3212, 3223, 3234, 3245, 3256, 3267,
    3278, 3289, 3300, 3311, 3322, 6, 17, 28, 39, 50, 61, 72, 83, 94, 105, 116, 127, 138, 149, 160,
    171, 182, 193, 204, 215, 226, 237, 248, 259, 270, 281, 292, 303, 314, 325, 336, 347, 358, 369,
    380, 391, 402, 413, 424, 435, 446, 457, 468, 479, 490, 501, 512, 523, 534, 545, 556, 567, 578,
    589, 600, 611, 622, 633, 644, 655, 666, 677, 688, 699, 710, 721, 732, 743, 754, 765, 776, 787,
    798, 809, 820, 831, 842, 853, 864, 875, 886, 897, 908, 919, 930, 941, 952, 963, 974, 985, 996,
    1007, 1018, 1029, 1040, 1051, 1062, 1073, 1084, 1095, 1106, 1117, 1128, 1139, 1150, 1161, 1172
]

function ntt_inverse_cooley_tukey(p::AbstractVector, zetas_inv::Vector{Int}, q::Int)
    n = length(p)
    if n == 1
        return p
    end

    even = ntt_inverse_cooley_tukey(p[1:2:end], zetas_inv, q)
    odd = ntt_inverse_cooley_tukey(p[2:2:end], zetas_inv, q)

    zetas_len = length(zetas_inv)
    result = similar(p)
    for i in 1:n÷2
        zeta_inv = zetas_inv[zetas_len * (i-1) ÷ (n÷2) + 1]
        t = (zeta_inv * odd[i]) % q
        result[i] = (even[i] + t) % q
        result[i + n÷2] = (even[i] - t + q) % q
    end
    return result
end

function backend_ntt_transform(::CPUBackend, poly::AbstractVector, modulus::Integer)
    return ntt_cooley_tukey(poly, ZETAS, modulus)
end

function backend_ntt_inverse_transform(::CPUBackend, poly::AbstractVector, modulus::Integer)
    n = length(poly)
    n_inv = powermod(n, -1, modulus)
    result = ntt_inverse_cooley_tukey(poly, ZETAS_INV, modulus)
    return (result .* n_inv) .% modulus
end

function backend_polynomial_multiply(::CPUBackend, a::AbstractVector, b::AbstractVector, modulus::Integer)
    a_ntt = backend_ntt_transform(CPUBackend(:none, 1), a, modulus)
    b_ntt = backend_ntt_transform(CPUBackend(:none, 1), b, modulus)
    c_ntt = a_ntt .* b_ntt
    return backend_ntt_inverse_transform(CPUBackend(:none, 1), c_ntt, modulus)
end

function backend_sampling(::CPUBackend, distribution::Symbol, params...)
    if distribution == :cbd
        eta, n, k = params
        # Centered binomial distribution sampling
        # This is a simplified, non-constant-time implementation for now
        return [sum(rand(Bool, eta) for _ in 1:eta) - sum(rand(Bool, eta) for _ in 1:eta) for i in 1:k, j in 1:n]
    else
        return randn()
    end
end

# --- Pretty Printing ---
Base.show(io::IO, b::CPUBackend) = print(io, "CPUBackend($(b.simd_level), $(b.threads) threads)")
Base.show(io::IO, b::CUDABackend) = print(io, "CUDABackend(device=$(b.device), CC=$(b.compute_capability))")
Base.show(io::IO, b::ROCmBackend) = print(io, "ROCmBackend(device=$(b.device), arch=$(b.gcn_arch))")
Base.show(io::IO, b::MetalBackend) = print(io, "MetalBackend(M$(b.apple_silicon_generation)$(b.has_neural_engine ? " + Neural Engine" : ""))")
Base.show(io::IO, b::OneAPIBackend) = print(io, "OneAPIBackend($(b.device_type)$(b.has_npu ? " + NPU" : ""))")
Base.show(io::IO, b::TPUBackend) = print(io, "TPUBackend(v$(b.version), $(b.topology))")

# --- Package Extensions ---
# Extensions should define:
# - `cuda_available()`, `create_cuda_backend()`, etc.
# - Backend-specific methods for `backend_lattice_multiply`, etc.
