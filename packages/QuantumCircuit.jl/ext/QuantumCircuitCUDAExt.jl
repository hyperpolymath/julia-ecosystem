# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# QuantumCircuitCUDAExt.jl - CUDA GPU kernels for QuantumCircuit.jl
#
# Provides KernelAbstractions-based GPU kernels for quantum circuit simulation
# on NVIDIA GPUs via CUDA.jl. Loaded as a package extension when both CUDA and
# KernelAbstractions are available.

module QuantumCircuitCUDAExt

using QuantumCircuit
using CUDA
using KernelAbstractions
using LinearAlgebra

using AcceleratorGate: CUDABackend

# ============================================================================
# Kernel: Single-Qubit Gate Application
# ============================================================================
#
# For n qubits the state vector has 2^n amplitudes. A single-qubit gate on
# qubit `target` (1-based) pairs amplitudes whose indices differ only in bit
# position `target - 1` (0-based bit). Each GPU thread processes one pair.

@kernel function _cuda_apply_single_gate_kernel!(
    amps::AbstractVector{ComplexF64},
    @Const(g11::ComplexF64), @Const(g12::ComplexF64),
    @Const(g21::ComplexF64), @Const(g22::ComplexF64),
    @Const(target_bit::Int)
)
    idx = @index(Global)            # 1-based thread index
    pair_idx = idx - 1              # 0-based pair index

    step  = 1 << target_bit         # distance between paired amplitudes
    block = 1 << (target_bit + 1)   # size of one "block" of pairs

    block_num = pair_idx ÷ (block ÷ 2)
    local_idx = pair_idx % (block ÷ 2)

    i0 = block_num * block + local_idx + 1   # 1-based index of |...0...>
    i1 = i0 + step                            # 1-based index of |...1...>

    a0 = amps[i0]
    a1 = amps[i1]

    amps[i0] = g11 * a0 + g12 * a1
    amps[i1] = g21 * a0 + g22 * a1
end

# ============================================================================
# Kernel: Two-Qubit (Controlled) Gate Application
# ============================================================================
#
# A two-qubit gate acts on 4-amplitude blocks. For control qubit `ctrl` and
# target qubit `tgt`, each thread processes one group of 4 amplitudes
# |00>, |01>, |10>, |11> in the (ctrl, tgt) subspace.

@kernel function _cuda_apply_two_qubit_gate_kernel!(
    amps::AbstractVector{ComplexF64},
    @Const(gate_flat::AbstractVector{ComplexF64}),  # 16-element column-major 4x4
    @Const(bit_high::Int),   # max(ctrl_bit, tgt_bit)
    @Const(bit_low::Int),    # min(ctrl_bit, tgt_bit)
    @Const(ctrl_bit::Int),
    @Const(tgt_bit::Int)
)
    idx = @index(Global)
    pair_idx = idx - 1

    # Remove the two qubit-bit positions from the linear index to find the
    # base index among the remaining (n-2) qubit bits.
    # Strip out bit_high first (higher bit), then bit_low.
    mask_low  = (1 << bit_low) - 1
    mask_mid  = (1 << (bit_high - 1)) - 1  # after removing bit_low conceptually

    # More directly: enumerate the 2^(n-2) "rest" indices, then insert 0-bits
    # at positions bit_low and bit_high.
    # Insert 0-bit at bit_low position
    lower  = pair_idx & mask_low
    upper  = pair_idx >> bit_low
    temp   = (upper << (bit_low + 1)) | lower

    # Insert 0-bit at bit_high position
    mask_high = (1 << bit_high) - 1
    lower2 = temp & mask_high
    upper2 = temp >> bit_high
    base   = (upper2 << (bit_high + 1)) | lower2

    # The four basis-state indices (1-based) for |ctrl,tgt> = |00>, |01>, |10>, |11>
    i00 = base + 1
    i01 = base + (1 << tgt_bit) + 1
    i10 = base + (1 << ctrl_bit) + 1
    i11 = base + (1 << ctrl_bit) + (1 << tgt_bit) + 1

    a00 = amps[i00]
    a01 = amps[i01]
    a10 = amps[i10]
    a11 = amps[i11]

    # Apply the 4x4 gate matrix (column-major flat storage)
    amps[i00] = gate_flat[1]*a00 + gate_flat[5]*a01 + gate_flat[9]*a10  + gate_flat[13]*a11
    amps[i01] = gate_flat[2]*a00 + gate_flat[6]*a01 + gate_flat[10]*a10 + gate_flat[14]*a11
    amps[i10] = gate_flat[3]*a00 + gate_flat[7]*a01 + gate_flat[11]*a10 + gate_flat[15]*a11
    amps[i11] = gate_flat[4]*a00 + gate_flat[8]*a01 + gate_flat[12]*a10 + gate_flat[16]*a11
end

# ============================================================================
# Kernel: Measurement Probability Computation
# ============================================================================
#
# Computes |amplitude|^2 for each basis state in parallel. The resulting
# probability vector is then prefix-summed on the GPU for CDF sampling.

@kernel function _cuda_prob_kernel!(
    probs::AbstractVector{Float64},
    @Const(amps::AbstractVector{ComplexF64})
)
    idx = @index(Global)
    a = amps[idx]
    probs[idx] = real(a) * real(a) + imag(a) * imag(a)
end

# ============================================================================
# Kernel: State Evolution (First-Order Trotter Step)
# ============================================================================
#
# For Hamiltonian simulation via first-order Trotter decomposition, each
# diagonal + off-diagonal term exp(-i * h_jk * dt) is applied as a rotation.
# This kernel applies a diagonal phase: amp[i] *= exp(-i * diag[i] * dt).

@kernel function _cuda_diagonal_phase_kernel!(
    amps::AbstractVector{ComplexF64},
    @Const(diag_real::AbstractVector{Float64}),
    @Const(diag_imag::AbstractVector{Float64}),
    @Const(dt::Float64)
)
    idx = @index(Global)
    # Phase factor: exp(-i * (re + i*im) * dt) = exp(im_part * dt) * exp(-i * re_part * dt)
    phase_angle = -(diag_real[idx] * dt)
    damping     =   diag_imag[idx] * dt
    factor = ComplexF64(exp(damping) * cos(phase_angle), exp(damping) * sin(phase_angle))
    amps[idx] = amps[idx] * factor
end

# ============================================================================
# Kernel: Kronecker (Tensor) Product
# ============================================================================
#
# Reuses the KA kernel pattern from the main module but launched on CUDADevice.

@kernel function _cuda_kronecker_kernel!(
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

"""
    _launch_single_gate!(d_amps, gate_matrix, target, nq)

Launch the single-qubit gate kernel on the CUDA device. `target` is 1-based
qubit index; internally converted to 0-based bit position.
"""
"""Return the KernelAbstractions backend for CUDA kernel launches."""
_ka_backend() = get_backend(CuVector{Float32}(undef, 0))

function _launch_single_gate!(d_amps, gate_matrix, target, nq)
    n_pairs = 1 << (nq - 1)
    target_bit = target - 1  # 0-based bit position

    g11 = ComplexF64(gate_matrix[1, 1])
    g12 = ComplexF64(gate_matrix[1, 2])
    g21 = ComplexF64(gate_matrix[2, 1])
    g22 = ComplexF64(gate_matrix[2, 2])

    ka = _ka_backend()
    kernel = _cuda_apply_single_gate_kernel!(ka, WORKGROUP_SIZE)
    kernel(d_amps, g11, g12, g21, g22, target_bit; ndrange=n_pairs)
    KernelAbstractions.synchronize(ka)
end

"""
    _launch_two_qubit_gate!(d_amps, gate_matrix, ctrl, tgt, nq)

Launch the two-qubit gate kernel. `ctrl` and `tgt` are 1-based qubit indices.
"""
function _launch_two_qubit_gate!(d_amps, gate_matrix, ctrl, tgt, nq)
    n_groups = 1 << (nq - 2)
    ctrl_bit = ctrl - 1
    tgt_bit  = tgt - 1
    bit_high = max(ctrl_bit, tgt_bit)
    bit_low  = min(ctrl_bit, tgt_bit)

    # Flatten the 4x4 gate matrix to a 16-element column-major vector on device
    gate_flat = CuVector{ComplexF64}(vec(gate_matrix))

    ka = _ka_backend()
    kernel = _cuda_apply_two_qubit_gate_kernel!(ka, WORKGROUP_SIZE)
    kernel(d_amps, gate_flat, bit_high, bit_low, ctrl_bit, tgt_bit; ndrange=n_groups)
    KernelAbstractions.synchronize(ka)
end

# -- Gate Application --------------------------------------------------------

function QuantumCircuit.backend_gate_apply(
    ::AcceleratorGate.CUDABackend, amps::Vector{ComplexF64},
    gate_matrix::Matrix{ComplexF64}, target::Int, nq::Int
)
    dim = length(amps)
    d_amps = CuVector{ComplexF64}(amps)

    if size(gate_matrix) == (2, 2)
        _launch_single_gate!(d_amps, gate_matrix, target, nq)
    else
        # For larger gate matrices, fall back to nothing (let Julia handle it)
        return nothing
    end

    result = Vector{ComplexF64}(d_amps)
    CUDA.unsafe_free!(d_amps)
    return result
end

# -- Tensor Product ----------------------------------------------------------

function QuantumCircuit.backend_tensor_contract(
    ::AcceleratorGate.CUDABackend, a::Vector{ComplexF64}, b::Vector{ComplexF64}
)
    len_a = length(a)
    len_b = length(b)
    len_c = len_a * len_b

    d_a = CuVector{ComplexF64}(a)
    d_b = CuVector{ComplexF64}(b)
    d_c = CuVector{ComplexF64}(undef, len_c)

    ka = _ka_backend()
    kernel = _cuda_kronecker_kernel!(ka, WORKGROUP_SIZE)
    kernel(d_c, d_a, d_b, len_b; ndrange=len_c)
    KernelAbstractions.synchronize(ka)

    result = Vector{ComplexF64}(d_c)
    CUDA.unsafe_free!(d_a)
    CUDA.unsafe_free!(d_b)
    CUDA.unsafe_free!(d_c)
    return result
end

# -- Measurement -------------------------------------------------------------

function QuantumCircuit.backend_measurement(
    ::AcceleratorGate.CUDABackend, amps::Vector{ComplexF64}, nq::Int
)
    dim = length(amps)
    d_amps = CuVector{ComplexF64}(amps)
    d_probs = CuVector{Float64}(undef, dim)

    # Compute probabilities in parallel
    ka = _ka_backend()
    prob_kernel = _cuda_prob_kernel!(ka, WORKGROUP_SIZE)
    prob_kernel(d_probs, d_amps; ndrange=dim)
    KernelAbstractions.synchronize(ka)

    # GPU prefix sum for cumulative distribution function
    d_cdf = accumulate(+, d_probs)

    # Sample: draw random number, binary search on CDF (on host for simplicity)
    cdf = Vector{Float64}(d_cdf)
    r = rand()
    outcome = searchsortedfirst(cdf, r) - 1  # 0-based
    outcome = clamp(outcome, 0, dim - 1)

    # Collapse to measured state
    collapsed = zeros(ComplexF64, dim)
    collapsed[outcome + 1] = 1.0 + 0.0im

    CUDA.unsafe_free!(d_amps)
    CUDA.unsafe_free!(d_probs)

    return (outcome, collapsed)
end

# -- State Evolution (Trotter Decomposition) ---------------------------------

function QuantumCircuit.backend_state_evolve(
    ::AcceleratorGate.CUDABackend, amps::Vector{ComplexF64},
    hamiltonian::Matrix{ComplexF64}, dt::Float64, nq::Int
)
    dim = 2^nq

    # Decompose Hamiltonian: apply diagonal phases on GPU, then off-diagonal
    # terms as single-qubit rotations via the gate kernel.
    #
    # Strategy: diagonalise if small enough, otherwise first-order Trotter.
    # For moderate systems (nq <= 16), direct expm on GPU is feasible via
    # host-side eigen-decomposition + GPU-side phase application.

    if dim <= 65536  # up to 16 qubits: eigendecomposition is tractable
        # Eigen-decompose on host (Hermitian Hamiltonian)
        F = eigen(Hermitian(hamiltonian))
        eigenvalues = F.values       # real for Hermitian
        eigenvectors = F.vectors     # unitary columns

        # Transform state to eigenbasis on host, apply phases on GPU
        d_coeffs = CuVector{ComplexF64}(eigenvectors' * amps)
        d_eig_real = CuVector{Float64}(eigenvalues)
        d_eig_imag = CuVector{Float64}(zeros(dim))

        ka = _ka_backend()
        phase_kernel = _cuda_diagonal_phase_kernel!(ka, WORKGROUP_SIZE)
        phase_kernel(d_coeffs, d_eig_real, d_eig_imag, dt; ndrange=dim)
        KernelAbstractions.synchronize(ka)

        # Transform back from eigenbasis
        coeffs = Vector{ComplexF64}(d_coeffs)
        new_amps = eigenvectors * coeffs

        CUDA.unsafe_free!(d_coeffs)
        CUDA.unsafe_free!(d_eig_real)
        CUDA.unsafe_free!(d_eig_imag)

        return new_amps
    end

    # For very large systems, apply first-order Trotter: decompose H into
    # single-qubit terms and apply each as a gate rotation on GPU.
    # Fall back to Julia for Hamiltonians we cannot decompose.
    return nothing
end

# -- Entanglement (placeholder for future Bell-pair generation) --------------

function QuantumCircuit.backend_entangle(
    ::AcceleratorGate.CUDABackend, args...
)
    # Not yet implemented; fall back to Julia.
    return nothing
end

end # module QuantumCircuitCUDAExt
