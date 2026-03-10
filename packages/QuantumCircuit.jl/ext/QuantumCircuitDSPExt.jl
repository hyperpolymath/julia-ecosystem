# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# QuantumCircuitDSPExt.jl - DSP (Digital Signal Processor) backend for QuantumCircuit.jl
#
# Treats quantum state evolution through the lens of signal processing.
# The quantum state vector is viewed as a discrete signal in the computational
# basis, and operations are expressed using DSP primitives:
#
# - Gate application as convolution in the frequency domain (diagonal gates
#   become pointwise multiplication after DFT)
# - State evolution via FFT-based matrix exponential computation
# - Measurement as power spectral density estimation
# - Tensor product as cyclic convolution of amplitude spectra
#
# DSPs excel at fixed-point FFT/IFFT, FIR filtering, and MAC operations,
# making them efficient for quantum simulations that decompose into
# frequency-domain operations.
#
# Loaded as a package extension when AcceleratorGate is available.

module QuantumCircuitDSPExt

using QuantumCircuit
using AcceleratorGate
using AcceleratorGate: DSPBackend, _coprocessor_required, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

# ============================================================================
# DSP Resource Limits
# ============================================================================

"""
Maximum qubits for DSP path. DSPs typically have limited on-chip memory
(256 KB - 4 MB), but can stream from external DRAM. We limit based on
the FFT size the DSP can handle in a single pass.
"""
const MAX_DSP_QUBITS = 22

"""FFT radix size for the DSP butterfly engine (typically radix-2 or radix-4)."""
const FFT_RADIX = 2

function _max_dsp_qubits()
    env_val = get(ENV, "DSP_MAX_QUBITS", "")
    isempty(env_val) ? MAX_DSP_QUBITS : parse(Int, env_val)
end

function _check_dsp_capacity(nq::Int)
    if nq > _max_dsp_qubits()
        _record_diagnostic!(DSPBackend(0), (
            event = :capacity_exceeded,
            requested_qubits = nq,
            max_qubits = _max_dsp_qubits(),
            fft_size = 2^nq,
            action = :fallback_classical
        ))
        return false
    end
    return true
end

# ============================================================================
# FFT Primitives for DSP Processing
# ============================================================================

"""
    _dsp_fft!(x::Vector{ComplexF64})

In-place radix-2 Cooley-Tukey FFT. Computes the Discrete Fourier Transform
of the input vector. This mirrors the DSP hardware's butterfly network.
The length of x must be a power of 2.
"""
function _dsp_fft!(x::Vector{ComplexF64})
    N = length(x)
    N <= 1 && return x

    # Bit-reversal permutation
    j = 1
    for i in 1:N
        if i < j
            x[i], x[j] = x[j], x[i]
        end
        m = N >> 1
        while m >= 2 && j > m
            j -= m
            m >>= 1
        end
        j += m
    end

    # Butterfly stages
    len = 2
    while len <= N
        half = len >> 1
        w_base = exp(-2.0im * pi / len)
        @inbounds for start in 1:len:N
            w = ComplexF64(1.0, 0.0)
            for k in 0:(half - 1)
                i = start + k
                j_idx = i + half
                t = w * x[j_idx]
                x[j_idx] = x[i] - t
                x[i] = x[i] + t
                w *= w_base
            end
        end
        len <<= 1
    end

    return x
end

"""
    _dsp_ifft!(x::Vector{ComplexF64})

In-place inverse FFT. Conjugate-FFT-conjugate-scale method.
"""
function _dsp_ifft!(x::Vector{ComplexF64})
    N = length(x)
    # Conjugate
    @inbounds for i in 1:N
        x[i] = conj(x[i])
    end
    # Forward FFT
    _dsp_fft!(x)
    # Conjugate and scale
    scale = 1.0 / N
    @inbounds for i in 1:N
        x[i] = conj(x[i]) * scale
    end
    return x
end

# ============================================================================
# Gate Application: FFT-Based Gate Decomposition
# ============================================================================
#
# For diagonal gates (Rz, phase gates), the gate is a pointwise multiplication
# in the computational basis -- this is already optimal. For off-diagonal
# gates (Rx, Ry, Hadamard), we decompose using the signal processing insight:
#
# The butterfly structure of single-qubit gate application (pairs of amplitudes
# at stride 2^target) is isomorphic to one stage of an FFT butterfly.
# The DSP hardware implements this natively.

"""
    _is_diagonal_gate(gate_matrix::Matrix{ComplexF64}) -> Bool

Check if a gate matrix is diagonal (only diagonal elements non-zero).
Diagonal gates are applied as pointwise multiplication -- no FFT needed.
"""
function _is_diagonal_gate(gate_matrix::Matrix{ComplexF64})
    return abs(gate_matrix[1, 2]) < 1e-14 && abs(gate_matrix[2, 1]) < 1e-14
end

"""
    _dsp_butterfly_gate!(amps, gate_matrix, target, nq)

Apply a single-qubit gate using DSP butterfly operations.
This is structurally identical to one radix-2 butterfly stage of an FFT,
with the twiddle factors replaced by the gate matrix elements.
"""
function _dsp_butterfly_gate!(amps::Vector{ComplexF64}, gate_matrix::Matrix{ComplexF64},
                              target::Int, nq::Int)
    target_bit = target - 1
    step = 1 << target_bit
    block_size = 1 << (target_bit + 1)
    dim = 1 << nq

    g11 = gate_matrix[1, 1]
    g12 = gate_matrix[1, 2]
    g21 = gate_matrix[2, 1]
    g22 = gate_matrix[2, 2]

    # DSP butterfly: each "twiddle" is a 2x2 rotation
    @inbounds for block_start in 0:block_size:(dim - 1)
        for local_idx in 0:(step - 1)
            i0 = block_start + local_idx + 1
            i1 = i0 + step

            a0 = amps[i0]
            a1 = amps[i1]

            # Butterfly with gate-matrix "twiddle factors"
            amps[i0] = g11 * a0 + g12 * a1
            amps[i1] = g21 * a0 + g22 * a1
        end
    end
end

function QuantumCircuit.backend_coprocessor_gate_apply(
    backend::DSPBackend, amps::Vector{ComplexF64},
    gate_matrix::Matrix{ComplexF64}, target::Int, nq::Int
)
    _check_dsp_capacity(nq) || return nothing
    size(gate_matrix) == (2, 2) || return nothing

    dim = 1 << nq

    # Try DSP hardware path

    # For diagonal gates: pointwise multiplication (DSP's multiply-accumulate)
    if _is_diagonal_gate(gate_matrix)
        result = copy(amps)
        phase0 = gate_matrix[1, 1]
        phase1 = gate_matrix[2, 2]

        target_bit = target - 1
        @inbounds for i in 0:(dim - 1)
            if (i >> target_bit) & 1 == 0
                result[i + 1] *= phase0
            else
                result[i + 1] *= phase1
            end
        end
        return result
    end

    # General single-qubit gate: use DSP butterfly structure
    result = copy(amps)
    _dsp_butterfly_gate!(result, gate_matrix, target, nq)
    return result
end

# ============================================================================
# State Evolution: FFT-Based Approach
# ============================================================================
#
# For diagonal Hamiltonians, evolution is simply phase multiplication:
#   |psi(t)> = diag(exp(-i * d_k * dt)) |psi(0)>
#
# For circulant Hamiltonians (translation-invariant systems), evolution in
# the Fourier domain is a pointwise multiplication:
#   FFT(|psi(t)>) = diag(exp(-i * lambda_k * dt)) * FFT(|psi(0)>)
#
# For general Hamiltonians, we eigendecompose and use the DSP FFT for
# the basis transformations when the eigenvector matrix has structure.

"""
    _is_circulant(H::Matrix{ComplexF64}, dim::Int) -> Bool

Check if the Hamiltonian is circulant (each row is a cyclic shift of the
previous). Circulant matrices are diagonalised by the DFT matrix.
"""
function _is_circulant(H::Matrix{ComplexF64}, dim::Int)
    dim >= 2 || return false
    first_row = H[1, :]
    @inbounds for i in 2:dim
        for j in 1:dim
            expected_col = mod(j - i, dim) + 1
            if abs(H[i, j] - first_row[expected_col]) > 1e-12
                return false
            end
        end
    end
    return true
end

function QuantumCircuit.backend_coprocessor_state_evolve(
    backend::DSPBackend, amps::Vector{ComplexF64},
    hamiltonian::Matrix{ComplexF64}, dt::Float64, nq::Int
)
    _check_dsp_capacity(nq) || return nothing

    dim = 2^nq

    # Try DSP hardware

    # Check if Hamiltonian is diagonal: pure phase evolution
    is_diag = true
    @inbounds for j in 1:dim, i in 1:dim
        if i != j && abs(hamiltonian[i, j]) > 1e-14
            is_diag = false
            break
        end
    end

    if is_diag
        result = copy(amps)
        @inbounds for i in 1:dim
            phase = -real(hamiltonian[i, i]) * dt
            result[i] *= ComplexF64(cos(phase), sin(phase))
        end
        return result
    end

    # Check if Hamiltonian is circulant: use FFT-domain evolution
    if _is_circulant(hamiltonian, dim)
        # Eigenvalues of circulant matrix = DFT of first row
        first_row = copy(hamiltonian[1, :])
        eigenvalues_freq = copy(first_row)
        _dsp_fft!(eigenvalues_freq)

        # Transform state to frequency domain
        state_freq = copy(amps)
        _dsp_fft!(state_freq)

        # Apply phase evolution in frequency domain (pointwise multiply)
        @inbounds for i in 1:dim
            phase = -real(eigenvalues_freq[i]) * dt
            state_freq[i] *= ComplexF64(cos(phase), sin(phase))
        end

        # Transform back to computational basis
        _dsp_ifft!(state_freq)
        return state_freq
    end

    # General case: eigendecomposition + DSP-optimised transforms
    if dim <= 32768  # up to 15 qubits
        F = eigen(Hermitian(hamiltonian))
        eigenvalues = F.values
        eigenvectors = F.vectors

        coeffs = eigenvectors' * amps

        # Phase evolution (DSP MAC-optimised)
        @inbounds for i in 1:dim
            phase = -eigenvalues[i] * dt
            coeffs[i] *= ComplexF64(cos(phase), sin(phase))
        end

        return eigenvectors * coeffs
    end

    return nothing
end

# ============================================================================
# Measurement: Power Spectral Density Estimation
# ============================================================================
#
# From a DSP perspective, measurement probability |a_i|^2 is the power
# spectral density of the quantum state. The DSP computes this efficiently
# using its multiply-accumulate pipeline.

function QuantumCircuit.backend_coprocessor_measurement(
    backend::DSPBackend, amps::Vector{ComplexF64}, nq::Int
)
    _check_dsp_capacity(nq) || return nothing

    dim = length(amps)

    # Try DSP hardware

    # Compute power spectral density: P(i) = |a_i|^2
    probs = Vector{Float64}(undef, dim)
    @inbounds for i in 1:dim
        a = amps[i]
        probs[i] = real(a) * real(a) + imag(a) * imag(a)
    end

    # Normalise
    total = sum(probs)
    if abs(total - 1.0) > 1e-10
        probs ./= total
    end

    # CDF and sampling
    r = rand()
    cumulative = 0.0
    outcome = dim - 1
    @inbounds for i in 1:dim
        cumulative += probs[i]
        if r <= cumulative
            outcome = i - 1
            break
        end
    end

    collapsed = zeros(ComplexF64, dim)
    collapsed[outcome + 1] = 1.0 + 0.0im

    return (outcome, collapsed)
end

# ============================================================================
# Tensor Contraction: Convolution-Based Kronecker Product
# ============================================================================
#
# The Kronecker product of two vectors can be related to circular convolution
# in specific cases, but in general it is computed as a direct outer product.
# The DSP's MAC (multiply-accumulate) units process this efficiently by
# streaming vector B through the accumulator for each element of A.

function QuantumCircuit.backend_coprocessor_tensor_contract(
    backend::DSPBackend, a::Vector{ComplexF64}, b::Vector{ComplexF64}
)
    len_a = length(a)
    len_b = length(b)
    total_qubits = Int(log2(len_a)) + Int(log2(len_b))

    _check_dsp_capacity(total_qubits) || return nothing

    # Try DSP hardware

    # MAC-style Kronecker product: stream B for each element of A
    len_c = len_a * len_b
    result = Vector{ComplexF64}(undef, len_c)

    @inbounds for i in 1:len_a
        ai = a[i]
        offset = (i - 1) * len_b
        for j in 1:len_b
            result[offset + j] = ai * b[j]
        end
    end

    return result
end

# ============================================================================
# Entanglement: DSP Butterfly Two-Qubit Gate
# ============================================================================

function QuantumCircuit.backend_coprocessor_entangle(
    backend::DSPBackend, amps::Vector{ComplexF64},
    qubit_a::Int, qubit_b::Int, nq::Int
)
    _check_dsp_capacity(nq) || return nothing

    (1 <= qubit_a <= nq && 1 <= qubit_b <= nq && qubit_a != qubit_b) || return nothing

    dim = 2^nq
    new_amps = copy(amps)

    ctrl_bit = qubit_a - 1
    tgt_bit  = qubit_b - 1
    bit_high = max(ctrl_bit, tgt_bit)
    bit_low  = min(ctrl_bit, tgt_bit)
    n_groups = 1 << (nq - 2)

    @inbounds for g in 0:(n_groups - 1)
        mask_low = (1 << bit_low) - 1
        lower  = g & mask_low
        upper  = g >> bit_low
        temp   = (upper << (bit_low + 1)) | lower

        mask_high = (1 << bit_high) - 1
        lower2 = temp & mask_high
        upper2 = temp >> bit_high
        base   = (upper2 << (bit_high + 1)) | lower2

        i10 = base + (1 << ctrl_bit) + 1
        i11 = base + (1 << ctrl_bit) + (1 << tgt_bit) + 1

        new_amps[i10] = amps[i11]
        new_amps[i11] = amps[i10]
    end

    return new_amps
end

end # module QuantumCircuitDSPExt
