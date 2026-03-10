# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# QuantumCircuitROCmExt.jl - ROCm GPU kernels for QuantumCircuit.jl
#
# Provides KernelAbstractions-based GPU kernels for quantum circuit simulation
# on AMD GPUs via AMDGPU.jl. Loaded as a package extension when both AMDGPU
# and KernelAbstractions are available.

module QuantumCircuitROCmExt

using QuantumCircuit
using AMDGPU
using KernelAbstractions
using LinearAlgebra

using AcceleratorGate: ROCmBackend

# ============================================================================
# Kernel: Single-Qubit Gate Application
# ============================================================================

@kernel function _rocm_apply_single_gate_kernel!(
    amps::AbstractVector{ComplexF64},
    @Const(g11::ComplexF64), @Const(g12::ComplexF64),
    @Const(g21::ComplexF64), @Const(g22::ComplexF64),
    @Const(target_bit::Int)
)
    idx = @index(Global)
    pair_idx = idx - 1

    step  = 1 << target_bit
    block = 1 << (target_bit + 1)

    block_num = pair_idx ÷ (block ÷ 2)
    local_idx = pair_idx % (block ÷ 2)

    i0 = block_num * block + local_idx + 1
    i1 = i0 + step

    a0 = amps[i0]
    a1 = amps[i1]

    amps[i0] = g11 * a0 + g12 * a1
    amps[i1] = g21 * a0 + g22 * a1
end

# ============================================================================
# Kernel: Two-Qubit (Controlled) Gate Application
# ============================================================================

@kernel function _rocm_apply_two_qubit_gate_kernel!(
    amps::AbstractVector{ComplexF64},
    @Const(gate_flat::AbstractVector{ComplexF64}),
    @Const(bit_high::Int),
    @Const(bit_low::Int),
    @Const(ctrl_bit::Int),
    @Const(tgt_bit::Int)
)
    idx = @index(Global)
    pair_idx = idx - 1

    mask_low = (1 << bit_low) - 1

    lower  = pair_idx & mask_low
    upper  = pair_idx >> bit_low
    temp   = (upper << (bit_low + 1)) | lower

    mask_high = (1 << bit_high) - 1
    lower2 = temp & mask_high
    upper2 = temp >> bit_high
    base   = (upper2 << (bit_high + 1)) | lower2

    i00 = base + 1
    i01 = base + (1 << tgt_bit) + 1
    i10 = base + (1 << ctrl_bit) + 1
    i11 = base + (1 << ctrl_bit) + (1 << tgt_bit) + 1

    a00 = amps[i00]
    a01 = amps[i01]
    a10 = amps[i10]
    a11 = amps[i11]

    amps[i00] = gate_flat[1]*a00 + gate_flat[5]*a01 + gate_flat[9]*a10  + gate_flat[13]*a11
    amps[i01] = gate_flat[2]*a00 + gate_flat[6]*a01 + gate_flat[10]*a10 + gate_flat[14]*a11
    amps[i10] = gate_flat[3]*a00 + gate_flat[7]*a01 + gate_flat[11]*a10 + gate_flat[15]*a11
    amps[i11] = gate_flat[4]*a00 + gate_flat[8]*a01 + gate_flat[12]*a10 + gate_flat[16]*a11
end

# ============================================================================
# Kernel: Measurement Probability Computation
# ============================================================================

@kernel function _rocm_prob_kernel!(
    probs::AbstractVector{Float64},
    @Const(amps::AbstractVector{ComplexF64})
)
    idx = @index(Global)
    a = amps[idx]
    probs[idx] = real(a) * real(a) + imag(a) * imag(a)
end

# ============================================================================
# Kernel: State Evolution (Diagonal Phase Application)
# ============================================================================

@kernel function _rocm_diagonal_phase_kernel!(
    amps::AbstractVector{ComplexF64},
    @Const(diag_real::AbstractVector{Float64}),
    @Const(diag_imag::AbstractVector{Float64}),
    @Const(dt::Float64)
)
    idx = @index(Global)
    phase_angle = -(diag_real[idx] * dt)
    damping     =   diag_imag[idx] * dt
    factor = ComplexF64(exp(damping) * cos(phase_angle), exp(damping) * sin(phase_angle))
    amps[idx] = amps[idx] * factor
end

# ============================================================================
# Kernel: Kronecker (Tensor) Product
# ============================================================================

@kernel function _rocm_kronecker_kernel!(
    C::AbstractVector{ComplexF64},
    @Const(A::AbstractVector{ComplexF64}),
    @Const(B::AbstractVector{ComplexF64}),
    @Const(len_b::Int)
)
    idx = @index(Global)
    i = div(idx - 1, len_b) + 1
    j = mod(idx - 1, len_b) + 1
    C[idx] = A[i] * B[j]
end

# ============================================================================
# Backend Dispatch Overloads
# ============================================================================

const WORKGROUP_SIZE = 256

"""Return the KernelAbstractions backend for ROCm kernel launches."""
_ka_backend() = get_backend(ROCVector{Float32}(undef, 0))

function _launch_single_gate!(d_amps, gate_matrix, target, nq)
    n_pairs = 1 << (nq - 1)
    target_bit = target - 1

    g11 = ComplexF64(gate_matrix[1, 1])
    g12 = ComplexF64(gate_matrix[1, 2])
    g21 = ComplexF64(gate_matrix[2, 1])
    g22 = ComplexF64(gate_matrix[2, 2])

    ka = _ka_backend()
    kernel = _rocm_apply_single_gate_kernel!(ka, WORKGROUP_SIZE)
    kernel(d_amps, g11, g12, g21, g22, target_bit; ndrange=n_pairs)
    KernelAbstractions.synchronize(ka)
end

function _launch_two_qubit_gate!(d_amps, gate_matrix, ctrl, tgt, nq)
    n_groups = 1 << (nq - 2)
    ctrl_bit = ctrl - 1
    tgt_bit  = tgt - 1
    bit_high = max(ctrl_bit, tgt_bit)
    bit_low  = min(ctrl_bit, tgt_bit)

    gate_flat = ROCVector{ComplexF64}(vec(gate_matrix))

    ka = _ka_backend()
    kernel = _rocm_apply_two_qubit_gate_kernel!(ka, WORKGROUP_SIZE)
    kernel(d_amps, gate_flat, bit_high, bit_low, ctrl_bit, tgt_bit; ndrange=n_groups)
    KernelAbstractions.synchronize(ka)
end

# -- Gate Application --------------------------------------------------------

function QuantumCircuit.backend_gate_apply(
    ::AcceleratorGate.ROCmBackend, amps::Vector{ComplexF64},
    gate_matrix::Matrix{ComplexF64}, target::Int, nq::Int
)
    d_amps = ROCVector{ComplexF64}(amps)

    if size(gate_matrix) == (2, 2)
        _launch_single_gate!(d_amps, gate_matrix, target, nq)
    else
        return nothing
    end

    result = Vector{ComplexF64}(d_amps)
    return result
end

# -- Tensor Product ----------------------------------------------------------

function QuantumCircuit.backend_tensor_contract(
    ::AcceleratorGate.ROCmBackend, a::Vector{ComplexF64}, b::Vector{ComplexF64}
)
    len_a = length(a)
    len_b = length(b)
    len_c = len_a * len_b

    d_a = ROCVector{ComplexF64}(a)
    d_b = ROCVector{ComplexF64}(b)
    d_c = ROCVector{ComplexF64}(undef, len_c)

    ka = _ka_backend()
    kernel = _rocm_kronecker_kernel!(ka, WORKGROUP_SIZE)
    kernel(d_c, d_a, d_b, len_b; ndrange=len_c)
    KernelAbstractions.synchronize(ka)

    return Vector{ComplexF64}(d_c)
end

# -- Measurement -------------------------------------------------------------

function QuantumCircuit.backend_measurement(
    ::AcceleratorGate.ROCmBackend, amps::Vector{ComplexF64}, nq::Int
)
    dim = length(amps)
    d_amps = ROCVector{ComplexF64}(amps)
    d_probs = ROCVector{Float64}(undef, dim)

    ka = _ka_backend()
    prob_kernel = _rocm_prob_kernel!(ka, WORKGROUP_SIZE)
    prob_kernel(d_probs, d_amps; ndrange=dim)
    KernelAbstractions.synchronize(ka)

    # Prefix sum for CDF, then sample on host
    d_cdf = accumulate(+, d_probs)
    cdf = Vector{Float64}(d_cdf)
    r = rand()
    outcome = searchsortedfirst(cdf, r) - 1
    outcome = clamp(outcome, 0, dim - 1)

    collapsed = zeros(ComplexF64, dim)
    collapsed[outcome + 1] = 1.0 + 0.0im

    return (outcome, collapsed)
end

# -- State Evolution (Trotter Decomposition) ---------------------------------

function QuantumCircuit.backend_state_evolve(
    ::AcceleratorGate.ROCmBackend, amps::Vector{ComplexF64},
    hamiltonian::Matrix{ComplexF64}, dt::Float64, nq::Int
)
    dim = 2^nq

    if dim <= 65536  # up to 16 qubits
        F = eigen(Hermitian(hamiltonian))
        eigenvalues = F.values
        eigenvectors = F.vectors

        d_coeffs = ROCVector{ComplexF64}(eigenvectors' * amps)
        d_eig_real = ROCVector{Float64}(eigenvalues)
        d_eig_imag = ROCVector{Float64}(zeros(dim))

        ka = _ka_backend()
        phase_kernel = _rocm_diagonal_phase_kernel!(ka, WORKGROUP_SIZE)
        phase_kernel(d_coeffs, d_eig_real, d_eig_imag, dt; ndrange=dim)
        KernelAbstractions.synchronize(ka)

        coeffs = Vector{ComplexF64}(d_coeffs)
        return eigenvectors * coeffs
    end

    return nothing
end

# -- Entanglement (placeholder) ---------------------------------------------

function QuantumCircuit.backend_entangle(::AcceleratorGate.ROCmBackend, args...)
    return nothing
end

end # module QuantumCircuitROCmExt
