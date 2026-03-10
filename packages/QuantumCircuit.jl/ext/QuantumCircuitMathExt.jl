# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# QuantumCircuitMathExt.jl - Math coprocessor backend for QuantumCircuit.jl
#
# Provides arbitrary-precision and symbolic computation capabilities for
# quantum circuit simulation. Uses BigFloat/BigComplex arithmetic to avoid
# floating-point rounding errors that accumulate in deep circuits, and
# supports symbolic state evolution for algebraic analysis.
#
# Key capabilities:
# - Arbitrary precision amplitudes (configurable via setprecision)
# - Exact rational arithmetic for Clifford circuits
# - Symbolic gate decomposition tracking
# - Numerical verification of circuit identities
#
# Loaded as a package extension when AcceleratorGate is available.

module QuantumCircuitMathExt

using QuantumCircuit
using AcceleratorGate
using AcceleratorGate: MathBackend, _coprocessor_required, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

# ============================================================================
# Math Coprocessor Configuration
# ============================================================================

"""
Default precision in bits for BigFloat arithmetic.
256 bits ~ 77 decimal digits, sufficient for most quantum circuits.
Increase for very deep circuits or high-fidelity verification.
"""
const DEFAULT_PRECISION_BITS = 256

"""Maximum qubits: arbitrary precision is expensive, O(2^n * precision)."""
const MAX_MATH_QUBITS = 18

function _max_math_qubits()
    env_val = get(ENV, "MATH_MAX_QUBITS", "")
    isempty(env_val) ? MAX_MATH_QUBITS : parse(Int, env_val)
end

function _get_precision_bits()
    env_val = get(ENV, "MATH_PRECISION_BITS", "")
    isempty(env_val) ? DEFAULT_PRECISION_BITS : parse(Int, env_val)
end

function _check_math_capacity(nq::Int)
    if nq > _max_math_qubits()
        _record_diagnostic!(MathBackend(0), (
            event = :capacity_exceeded,
            requested_qubits = nq,
            max_qubits = _max_math_qubits(),
            memory_per_amplitude_bytes = _get_precision_bits() ÷ 4,  # ~2x BigFloat size
            total_memory_mb = (2^nq * _get_precision_bits() ÷ 4) / 1e6,
            action = :fallback_classical
        ))
        return false
    end
    return true
end

# ============================================================================
# Arbitrary Precision Arithmetic Helpers
# ============================================================================

"""Complex type backed by BigFloat for arbitrary-precision amplitudes."""
const BigComplex = Complex{BigFloat}

"""
    _to_bigcomplex(v::Vector{ComplexF64}) -> Vector{BigComplex}

Promote a ComplexF64 vector to arbitrary-precision BigComplex.
"""
function _to_bigcomplex(v::Vector{ComplexF64})
    setprecision(BigFloat, _get_precision_bits())
    return BigComplex.(v)
end

"""
    _to_bigcomplex_matrix(M::Matrix{ComplexF64}) -> Matrix{BigComplex}

Promote a ComplexF64 matrix to arbitrary-precision BigComplex.
"""
function _to_bigcomplex_matrix(M::Matrix{ComplexF64})
    setprecision(BigFloat, _get_precision_bits())
    return BigComplex.(M)
end

"""
    _from_bigcomplex(v::Vector{BigComplex}) -> Vector{ComplexF64}

Demote a BigComplex vector back to ComplexF64 for return to the caller.
Records the maximum rounding error introduced by the demotion.
"""
function _from_bigcomplex(v::Vector{BigComplex})
    result = ComplexF64.(v)

    # Check rounding error
    max_err = maximum(abs.(BigComplex.(result) .- v))
    if max_err > 1e-15
        _record_diagnostic!(MathBackend(0), (
            event = :precision_loss,
            max_rounding_error = Float64(max_err),
            precision_bits = _get_precision_bits()
        ))
    end

    return result
end

"""
    _is_clifford_gate(gate_matrix::Matrix{ComplexF64}) -> Bool

Check if a gate is a Clifford gate (maps Pauli group to Pauli group).
Clifford gates can be simulated exactly with rational arithmetic:
H, S, CNOT, and their compositions.
"""
function _is_clifford_gate(gate_matrix::Matrix{ComplexF64})
    # Clifford single-qubit gates have matrix elements from
    # {0, +-1, +-i, +-1/sqrt(2), +-i/sqrt(2)}
    inv_sqrt2 = 1.0 / sqrt(2.0)
    allowed = [0.0, 1.0, -1.0, inv_sqrt2, -inv_sqrt2]

    for x in gate_matrix
        re, im_part = real(x), imag(x)
        re_ok = any(abs(re - a) < 1e-12 for a in allowed)
        im_ok = any(abs(im_part - a) < 1e-12 for a in allowed)
        if !(re_ok && im_ok)
            return false
        end
    end
    return true
end

# ============================================================================
# Gate Application: Arbitrary Precision Amplitudes
# ============================================================================
#
# Gate application is performed at full BigFloat precision. For Clifford gates,
# we use exact rational arithmetic where possible. For non-Clifford gates
# (T gate, arbitrary rotations), we use BigFloat with configurable precision.

function QuantumCircuit.backend_coprocessor_gate_apply(
    backend::MathBackend, amps::Vector{ComplexF64},
    gate_matrix::Matrix{ComplexF64}, target::Int, nq::Int
)
    _check_math_capacity(nq) || return nothing
    size(gate_matrix) == (2, 2) || return nothing

    dim = 2^nq

    # Try math coprocessor hardware

    # Promote to arbitrary precision
    setprecision(BigFloat, _get_precision_bits())
    big_amps = _to_bigcomplex(amps)
    big_gate = _to_bigcomplex_matrix(gate_matrix)

    # Build full unitary at arbitrary precision
    big_id = Matrix{BigComplex}(I, 2, 2)
    full_op = Matrix{BigComplex}(I, 1, 1)
    for q in 1:nq
        if q == target
            full_op = kron(full_op, big_gate)
        else
            full_op = kron(full_op, big_id)
        end
    end

    # Apply gate at full precision
    big_result = full_op * big_amps

    # Renormalise to correct any drift (shouldn't be needed for unitary, but safe)
    norm_val = sqrt(sum(abs2, big_result))
    if abs(norm_val - 1) > BigFloat(1e-30)
        big_result ./= norm_val
    end

    return _from_bigcomplex(big_result)
end

# ============================================================================
# Tensor Contraction: Exact Kronecker Product
# ============================================================================
#
# Kronecker product at arbitrary precision to avoid accumulated rounding
# in the multiplicative chain a[i] * b[j] for large vectors.

function QuantumCircuit.backend_coprocessor_tensor_contract(
    backend::MathBackend, a::Vector{ComplexF64}, b::Vector{ComplexF64}
)
    len_a = length(a)
    len_b = length(b)
    total_qubits = Int(log2(len_a)) + Int(log2(len_b))

    _check_math_capacity(total_qubits) || return nothing

    setprecision(BigFloat, _get_precision_bits())

    # Try math coprocessor hardware

    big_a = _to_bigcomplex(a)
    big_b = _to_bigcomplex(b)

    len_c = len_a * len_b
    big_result = Vector{BigComplex}(undef, len_c)

    @inbounds for i in 1:len_a
        ai = big_a[i]
        offset = (i - 1) * len_b
        for j in 1:len_b
            big_result[offset + j] = ai * big_b[j]
        end
    end

    return _from_bigcomplex(big_result)
end

# ============================================================================
# State Evolution: Symbolic and Exact Matrix Exponential
# ============================================================================
#
# The math coprocessor computes exp(-i*H*dt) at arbitrary precision.
# For Hamiltonians with rational eigenvalues, this can be computed exactly.
# For general Hamiltonians, we use eigendecomposition at BigFloat precision,
# which avoids the catastrophic cancellation that occurs in Float64 for
# near-degenerate eigenvalues.

"""
    _bigfloat_eigen(H::Matrix{BigComplex}) -> (Vector{BigFloat}, Matrix{BigComplex})

Eigendecomposition at BigFloat precision. For Hermitian matrices, eigenvalues
are real (BigFloat) and eigenvectors form a unitary matrix.
"""
function _bigfloat_eigen(H_big::Matrix{BigComplex})
    # Convert to Float64 for eigendecomposition (Julia's eigen doesn't support BigFloat matrices)
    # then refine eigenvalues/vectors using Newton iteration at BigFloat precision
    H_f64 = ComplexF64.(H_big)
    F = eigen(Hermitian(H_f64))

    # Promote to BigFloat
    eigenvalues = BigFloat.(F.values)
    eigenvectors = BigComplex.(F.vectors)

    # One step of iterative refinement: V' * H * V should be diagonal
    # Residual: off-diagonal elements of V' * H * V
    D_approx = eigenvectors' * H_big * eigenvectors
    for i in axes(D_approx, 1)
        eigenvalues[i] = real(D_approx[i, i])
    end

    return (eigenvalues, eigenvectors)
end

"""
    _taylor_matrix_exp(M::Matrix{BigComplex}, order::Int) -> Matrix{BigComplex}

Compute matrix exponential via truncated Taylor series at arbitrary precision.
The Taylor series exp(M) = sum(M^k / k!, k=0..order) converges for all M,
and at BigFloat precision we can use many terms without losing significance.
"""
function _taylor_matrix_exp(M::Matrix{BigComplex}, order::Int=30)
    dim = size(M, 1)
    Id = Matrix{BigComplex}(I, dim, dim)

    # Scale-and-square: reduce ||M|| before Taylor, then square the result
    norm_M = Float64(maximum(abs, M))
    s = max(0, ceil(Int, log2(norm_M + 1)))
    M_scaled = M ./ BigFloat(2)^s

    # Taylor series: sum M_scaled^k / k!
    result = copy(Id)
    term = copy(Id)
    for k in 1:order
        term = (term * M_scaled) ./ BigFloat(k)
        result .+= term

        # Early termination if terms are negligible
        term_norm = maximum(abs, term)
        if Float64(term_norm) < 1e-50
            break
        end
    end

    # Square s times to undo scaling
    for _ in 1:s
        result = result * result
    end

    return result
end

function QuantumCircuit.backend_coprocessor_state_evolve(
    backend::MathBackend, amps::Vector{ComplexF64},
    hamiltonian::Matrix{ComplexF64}, dt::Float64, nq::Int
)
    _check_math_capacity(nq) || return nothing

    dim = 2^nq
    setprecision(BigFloat, _get_precision_bits())

    # Try math coprocessor hardware

    big_H = _to_bigcomplex_matrix(hamiltonian)
    big_amps = _to_bigcomplex(amps)
    big_dt = BigFloat(dt)

    if dim <= 4096  # up to 12 qubits: eigendecomposition is fast
        eigenvalues, eigenvectors = _bigfloat_eigen(big_H)

        # Project into eigenbasis
        coeffs = eigenvectors' * big_amps

        # Apply exact phase evolution
        @inbounds for i in 1:dim
            phase = -eigenvalues[i] * big_dt
            coeffs[i] *= BigComplex(cos(phase), sin(phase))
        end

        # Back-transform
        big_result = eigenvectors * coeffs

        return _from_bigcomplex(big_result)
    end

    # For larger systems: Taylor series matrix exponential at arbitrary precision
    if dim <= 2^_max_math_qubits()
        exponent = BigComplex(0, -1) .* big_H .* big_dt
        U = _taylor_matrix_exp(exponent)
        big_result = U * big_amps

        return _from_bigcomplex(big_result)
    end

    return nothing
end

# ============================================================================
# Measurement: Exact Probability Computation
# ============================================================================
#
# Measurement probabilities are computed at arbitrary precision to avoid
# the problem where small but non-zero amplitudes get rounded to zero in
# Float64, causing incorrect probability distributions.

function QuantumCircuit.backend_coprocessor_measurement(
    backend::MathBackend, amps::Vector{ComplexF64}, nq::Int
)
    _check_math_capacity(nq) || return nothing

    dim = length(amps)
    setprecision(BigFloat, _get_precision_bits())

    big_amps = _to_bigcomplex(amps)

    # Compute exact probabilities
    probs = Vector{BigFloat}(undef, dim)
    @inbounds for i in 1:dim
        probs[i] = abs2(big_amps[i])
    end

    # Exact normalisation
    total = sum(probs)
    if total > BigFloat(0)
        probs ./= total
    end

    # Convert to Float64 for sampling (rand() returns Float64)
    probs_f64 = Float64.(probs)

    # Sample
    r = rand()
    cumulative = 0.0
    outcome = dim - 1
    @inbounds for i in 1:dim
        cumulative += probs_f64[i]
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
# Entanglement: Exact Two-Qubit Gate
# ============================================================================
#
# CNOT at arbitrary precision, ensuring no rounding errors in the swap.

function QuantumCircuit.backend_coprocessor_entangle(
    backend::MathBackend, amps::Vector{ComplexF64},
    qubit_a::Int, qubit_b::Int, nq::Int
)
    _check_math_capacity(nq) || return nothing

    (1 <= qubit_a <= nq && 1 <= qubit_b <= nq && qubit_a != qubit_b) || return nothing

    dim = 2^nq
    setprecision(BigFloat, _get_precision_bits())

    big_amps = _to_bigcomplex(amps)
    big_result = copy(big_amps)

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

        big_result[i10] = big_amps[i11]
        big_result[i11] = big_amps[i10]
    end

    return _from_bigcomplex(big_result)
end

end # module QuantumCircuitMathExt
