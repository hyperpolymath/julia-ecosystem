# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# QuantumCircuit.jl - Quantum circuit simulation with AcceleratorGate backend dispatch.

module QuantumCircuit

using LinearAlgebra

export Qubit, QuantumGate, QuantumCircuitObj, QuantumState
export apply_gate, measure, tensor_product, state_evolve

# ============================================================================
# Core Types
# ============================================================================

"""
    Qubit(index::Int)

A reference to a qubit by its index within a quantum register.
"""
struct Qubit
    index::Int
end

"""
    QuantumGate(name::String, matrix::Matrix{ComplexF64}, target_qubits::Vector{Qubit})

A quantum gate defined by its unitary matrix and the qubits it acts on.
"""
struct QuantumGate
    name::String
    matrix::Matrix{ComplexF64}
    target_qubits::Vector{Qubit}
end

"""
    QuantumState(amplitudes::Vector{ComplexF64})

A quantum state represented as a vector of complex amplitudes.
The length must be a power of 2.
"""
struct QuantumState
    amplitudes::Vector{ComplexF64}

    function QuantumState(amplitudes::Vector{ComplexF64})
        n = length(amplitudes)
        ispow2(n) || throw(ArgumentError("State vector length must be a power of 2, got $n"))
        return new(amplitudes)
    end
end

"""
    num_qubits(state::QuantumState) -> Int

Return the number of qubits represented by a quantum state.
"""
num_qubits(state::QuantumState) = Int(log2(length(state.amplitudes)))

"""
    QuantumCircuitObj(num_qubits::Int, gates::Vector{QuantumGate})

A quantum circuit composed of a sequence of gates applied to a register of qubits.
Named `QuantumCircuitObj` to avoid conflict with the module name.
"""
struct QuantumCircuitObj
    num_qubits::Int
    gates::Vector{QuantumGate}
end

QuantumCircuitObj(num_qubits::Int) = QuantumCircuitObj(num_qubits, QuantumGate[])

# ============================================================================
# Common Gates
# ============================================================================

const HADAMARD = ComplexF64[1 1; 1 -1] / sqrt(2.0)
const PAULI_X  = ComplexF64[0 1; 1 0]
const PAULI_Y  = ComplexF64[0 -im; im 0]
const PAULI_Z  = ComplexF64[1 0; 0 -1]

# ============================================================================
# Backend Abstraction (AcceleratorGate integration)
# Load before operations so backend hooks are available for dispatch.
# ============================================================================

include("backends/abstract.jl")

# ============================================================================
# Internal Helpers
# ============================================================================

"""Expand a single-qubit gate matrix to the full n-qubit Hilbert space."""
function _expand_single_gate(gate_matrix::Matrix{ComplexF64}, target::Int, n_qubits::Int)
    op = Matrix{ComplexF64}(I, 1, 1)
    for q in 1:n_qubits
        if q == target
            op = kron(op, gate_matrix)
        else
            op = kron(op, Matrix{ComplexF64}(I, 2, 2))
        end
    end
    return op
end

# ============================================================================
# Operations (with backend dispatch)
# ============================================================================

"""
    apply_gate(state::QuantumState, gate::QuantumGate) -> QuantumState

Apply a single-qubit gate to the given quantum state. For single-qubit gates,
the gate matrix is expanded to the full Hilbert space via tensor product with
identity matrices, then multiplied with the state vector.

For non-Julia backends, dispatches through `backend_gate_apply` which may
offload the matrix-vector product to a coprocessor for large state vectors.
"""
function apply_gate(state::QuantumState, gate::QuantumGate)
    nq = num_qubits(state)
    length(gate.target_qubits) == 1 || throw(ArgumentError("Only single-qubit gates supported in apply_gate"))

    target = gate.target_qubits[1].index
    (1 <= target <= nq) || throw(BoundsError("Qubit index $target out of range 1:$nq"))

    backend = current_backend()
    if !(backend isa JuliaBackend)
        # Attempt coprocessor-accelerated gate application.
        # The backend receives the raw amplitudes, gate matrix, target qubit index,
        # and total qubit count so it can build the full operator on-device.
        result = backend_gate_apply(backend, state.amplitudes, gate.matrix, target, nq)
        if result !== nothing
            return QuantumState(result)
        end
        # Backend returned nothing -- fall through to pure-Julia path.
    end

    # Pure-Julia implementation: build full operator I ⊗ ... ⊗ G ⊗ ... ⊗ I
    full_op = _expand_single_gate(gate.matrix, target, nq)
    new_amps = full_op * state.amplitudes
    return QuantumState(new_amps)
end

"""
    measure(state::QuantumState) -> (Int, QuantumState)

Perform a computational basis measurement on the quantum state.
Returns the measured basis state index (0-based) and the collapsed state.

For non-Julia backends, dispatches through `backend_measurement` which may
parallelise the probability computation and sampling on a coprocessor.
"""
function measure(state::QuantumState)
    nq = num_qubits(state)
    backend = current_backend()
    if !(backend isa JuliaBackend)
        # Backend measurement returns (outcome::Int, collapsed_amps::Vector{ComplexF64})
        # or nothing if the backend cannot handle this operation.
        result = backend_measurement(backend, state.amplitudes, nq)
        if result !== nothing
            outcome, collapsed_amps = result
            return (outcome, QuantumState(collapsed_amps))
        end
        # Backend returned nothing -- fall through to pure-Julia path.
    end

    # Pure-Julia implementation
    probs = abs2.(state.amplitudes)
    r = rand()
    cumulative = 0.0
    outcome = length(probs) - 1  # default to last state
    for i in eachindex(probs)
        cumulative += probs[i]
        if r <= cumulative
            outcome = i - 1  # 0-based
            break
        end
    end

    # Collapse to measured state
    collapsed = zeros(ComplexF64, length(state.amplitudes))
    collapsed[outcome + 1] = 1.0 + 0.0im
    return (outcome, QuantumState(collapsed))
end

"""
    tensor_product(a::QuantumState, b::QuantumState) -> QuantumState

Compute the tensor (Kronecker) product of two quantum states.

For non-Julia backends, dispatches through `backend_tensor_contract` which
may use GPU-parallel kernels for the Kronecker product computation.
"""
function tensor_product(a::QuantumState, b::QuantumState)
    backend = current_backend()
    if !(backend isa JuliaBackend)
        result = backend_tensor_contract(backend, a.amplitudes, b.amplitudes)
        if result !== nothing
            return QuantumState(result)
        end
        # Backend returned nothing -- fall through to pure-Julia path.
    end

    return QuantumState(kron(a.amplitudes, b.amplitudes))
end

"""
    state_evolve(state::QuantumState, hamiltonian::Matrix{ComplexF64}, dt::Float64) -> QuantumState

Evolve a quantum state under a Hamiltonian for a time step dt using the
matrix exponential: |psi(t+dt)> = exp(-i * H * dt) |psi(t)>.

For non-Julia backends, dispatches through `backend_state_evolve` which may
offload the matrix exponential and state-vector multiplication to a coprocessor.

# Arguments
- `state::QuantumState`: The initial quantum state.
- `hamiltonian::Matrix{ComplexF64}`: The Hamiltonian operator (must be Hermitian).
- `dt::Float64`: The time step for evolution.

# Returns
A new `QuantumState` representing the evolved state.

# Examples

```julia
# Evolve a single qubit under the Pauli-Z Hamiltonian
state = QuantumState(ComplexF64[1/sqrt(2), 1/sqrt(2)])
H = ComplexF64[1 0; 0 -1]  # Pauli-Z
evolved = state_evolve(state, H, 0.1)
```
"""
function state_evolve(state::QuantumState, hamiltonian::Matrix{ComplexF64}, dt::Float64)
    nq = num_qubits(state)
    dim = 2^nq
    size(hamiltonian) == (dim, dim) || throw(DimensionMismatch(
        "Hamiltonian size $(size(hamiltonian)) does not match state dimension ($dim, $dim)"))

    backend = current_backend()
    if !(backend isa JuliaBackend)
        result = backend_state_evolve(backend, state.amplitudes, hamiltonian, dt, nq)
        if result !== nothing
            return QuantumState(result)
        end
        # Backend returned nothing -- fall through to pure-Julia path.
    end

    # Pure-Julia: compute U = exp(-i * H * dt), then apply U * |psi>
    U = exp(-im * hamiltonian * dt)
    new_amps = U * state.amplitudes
    return QuantumState(new_amps)
end

# ============================================================================
# KernelAbstractions GPU Kernel for Tensor Product (Kronecker)
# ============================================================================
#
# The Kronecker product is element-wise independent:
#   C[i*len_b + j + 1] = A[i+1] * B[j+1]
# for i in 0:len_a-1, j in 0:len_b-1
#
# This is trivially parallel and maps well to GPU threads.
# The kernel is loaded conditionally when KernelAbstractions is available.

"""
    _ka_kronecker!(C, A, B, len_b)

KernelAbstractions-compatible kernel for computing the Kronecker product of
two vectors A and B, writing the result into C.

Each work item computes one element C[idx] = A[div(idx-1, len_b) + 1] * B[mod(idx-1, len_b) + 1].

This kernel is used by GPU backends that support KernelAbstractions.jl to
parallelise the tensor product across thousands of GPU threads.

# Arguments
- `C`: Output vector of length len_a * len_b (pre-allocated).
- `A`: First input vector of length len_a.
- `B`: Second input vector of length len_b.
- `len_b`: Length of B (passed explicitly to avoid GPU-side length() calls).
"""
function _ka_kronecker! end

# Conditional load: only define the kernel if KernelAbstractions is available.
# This avoids a hard dependency -- the kernel is used by GPU backend extensions.
if !isnothing(Base.find_package("KernelAbstractions"))
    @eval begin
        using KernelAbstractions

        @kernel function _ka_kronecker_kernel!(C, A, B, len_b)
            idx = @index(Global)
            # Map linear index to (i, j) pair for A[i] * B[j]
            i = div(idx - 1, len_b) + 1
            j = mod(idx - 1, len_b) + 1
            C[idx] = A[i] * B[j]
        end

        """
            _ka_kronecker!(backend_device, C, A, B)

        Launch the Kronecker product kernel on the given KernelAbstractions backend
        device (e.g., `CPU()`, `CUDADevice()`, `ROCDevice()`).

        # Arguments
        - `backend_device`: A KernelAbstractions backend (e.g., `KernelAbstractions.CPU()`).
        - `C`: Pre-allocated output vector of length `length(A) * length(B)`.
        - `A`: First input state vector.
        - `B`: Second input state vector.
        """
        function _ka_kronecker!(backend_device, C, A, B)
            len_b = length(B)
            kernel = _ka_kronecker_kernel!(backend_device, 256)
            kernel(C, A, B, len_b; ndrange=length(C))
            KernelAbstractions.synchronize(backend_device)
            return C
        end
    end
end

end # module QuantumCircuit
