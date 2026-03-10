# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>

using Test
using QuantumCircuit
using LinearAlgebra

@testset "QuantumCircuit.jl" begin

    # ── Type construction ──────────────────────────────────────────────
    @testset "Qubit" begin
        q = Qubit(1)
        @test q.index == 1
        q3 = Qubit(3)
        @test q3.index == 3
    end

    @testset "QuantumState construction" begin
        # Valid states (power-of-2 lengths)
        s1 = QuantumState(ComplexF64[1.0, 0.0])
        @test length(s1.amplitudes) == 2
        @test num_qubits(s1) == 1

        s2 = QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0])
        @test length(s2.amplitudes) == 4
        @test num_qubits(s2) == 2

        s3 = QuantumState(ComplexF64[1/sqrt(2), 1/sqrt(2)])
        @test num_qubits(s3) == 1
        @test s3.amplitudes[1] ≈ 1/sqrt(2)

        # Invalid: non-power-of-2
        @test_throws ArgumentError QuantumState(ComplexF64[1.0, 0.0, 0.0])
        @test_throws ArgumentError QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0, 0.0])
    end

    @testset "QuantumCircuitObj" begin
        circ = QuantumCircuitObj(3)
        @test circ.num_qubits == 3
        @test isempty(circ.gates)

        gate = QuantumGate("H", HADAMARD, [Qubit(1)])
        circ2 = QuantumCircuitObj(2, [gate])
        @test length(circ2.gates) == 1
        @test circ2.gates[1].name == "H"
    end

    @testset "QuantumGate construction" begin
        h = QuantumGate("H", HADAMARD, [Qubit(1)])
        @test h.name == "H"
        @test h.matrix ≈ HADAMARD
        @test length(h.target_qubits) == 1

        x = QuantumGate("X", PAULI_X, [Qubit(2)])
        @test x.name == "X"
        @test x.matrix ≈ PAULI_X
    end

    # ── Gate constants ─────────────────────────────────────────────────
    @testset "Gate matrices are unitary" begin
        for (name, mat) in [("H", HADAMARD), ("X", PAULI_X), ("Y", PAULI_Y), ("Z", PAULI_Z)]
            # U * U† = I for unitary matrices
            product = mat * mat'
            @test product ≈ I atol=1e-12
        end
    end

    @testset "Gate matrix properties" begin
        # Pauli matrices are Hermitian (self-adjoint)
        @test PAULI_X ≈ PAULI_X'
        @test PAULI_Y ≈ PAULI_Y'
        @test PAULI_Z ≈ PAULI_Z'

        # Hadamard is Hermitian
        @test HADAMARD ≈ HADAMARD'

        # Pauli matrices square to identity
        @test PAULI_X * PAULI_X ≈ I atol=1e-12
        @test PAULI_Y * PAULI_Y ≈ I atol=1e-12
        @test PAULI_Z * PAULI_Z ≈ I atol=1e-12
    end

    # ── apply_gate ─────────────────────────────────────────────────────
    @testset "apply_gate: Hadamard on |0⟩" begin
        state_0 = QuantumState(ComplexF64[1.0, 0.0])
        h_gate = QuantumGate("H", HADAMARD, [Qubit(1)])
        result = apply_gate(state_0, h_gate)
        # H|0⟩ = (|0⟩ + |1⟩)/√2
        @test result.amplitudes[1] ≈ 1/sqrt(2) atol=1e-12
        @test result.amplitudes[2] ≈ 1/sqrt(2) atol=1e-12
    end

    @testset "apply_gate: Hadamard on |1⟩" begin
        state_1 = QuantumState(ComplexF64[0.0, 1.0])
        h_gate = QuantumGate("H", HADAMARD, [Qubit(1)])
        result = apply_gate(state_1, h_gate)
        # H|1⟩ = (|0⟩ - |1⟩)/√2
        @test result.amplitudes[1] ≈ 1/sqrt(2) atol=1e-12
        @test result.amplitudes[2] ≈ -1/sqrt(2) atol=1e-12
    end

    @testset "apply_gate: Pauli-X (NOT gate)" begin
        state_0 = QuantumState(ComplexF64[1.0, 0.0])
        x_gate = QuantumGate("X", PAULI_X, [Qubit(1)])
        result = apply_gate(state_0, x_gate)
        # X|0⟩ = |1⟩
        @test result.amplitudes[1] ≈ 0.0 atol=1e-12
        @test result.amplitudes[2] ≈ 1.0 atol=1e-12

        # X|1⟩ = |0⟩
        state_1 = QuantumState(ComplexF64[0.0, 1.0])
        result2 = apply_gate(state_1, x_gate)
        @test result2.amplitudes[1] ≈ 1.0 atol=1e-12
        @test result2.amplitudes[2] ≈ 0.0 atol=1e-12
    end

    @testset "apply_gate: Pauli-Z phase flip" begin
        plus_state = QuantumState(ComplexF64[1/sqrt(2), 1/sqrt(2)])
        z_gate = QuantumGate("Z", PAULI_Z, [Qubit(1)])
        result = apply_gate(plus_state, z_gate)
        # Z(|0⟩+|1⟩)/√2 = (|0⟩-|1⟩)/√2
        @test result.amplitudes[1] ≈ 1/sqrt(2) atol=1e-12
        @test result.amplitudes[2] ≈ -1/sqrt(2) atol=1e-12
    end

    @testset "apply_gate: two-qubit system, gate on first qubit" begin
        # |00⟩ state
        state_00 = QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0])
        x_on_1 = QuantumGate("X", PAULI_X, [Qubit(1)])
        result = apply_gate(state_00, x_on_1)
        # X⊗I |00⟩ = |10⟩
        @test result.amplitudes[3] ≈ 1.0 atol=1e-12
        @test sum(abs2.(result.amplitudes)) ≈ 1.0 atol=1e-12
    end

    @testset "apply_gate: two-qubit system, gate on second qubit" begin
        state_00 = QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0])
        x_on_2 = QuantumGate("X", PAULI_X, [Qubit(2)])
        result = apply_gate(state_00, x_on_2)
        # I⊗X |00⟩ = |01⟩
        @test result.amplitudes[2] ≈ 1.0 atol=1e-12
        @test sum(abs2.(result.amplitudes)) ≈ 1.0 atol=1e-12
    end

    @testset "apply_gate: preserves normalisation" begin
        state = QuantumState(ComplexF64[1/sqrt(3), sqrt(2/3)])
        h_gate = QuantumGate("H", HADAMARD, [Qubit(1)])
        result = apply_gate(state, h_gate)
        @test sum(abs2.(result.amplitudes)) ≈ 1.0 atol=1e-12
    end

    @testset "apply_gate: error on multi-qubit gate" begin
        state = QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0])
        cnot_like = QuantumGate("CNOT", Matrix{ComplexF64}(I, 4, 4), [Qubit(1), Qubit(2)])
        @test_throws ArgumentError apply_gate(state, cnot_like)
    end

    @testset "apply_gate: error on out-of-range qubit" begin
        state = QuantumState(ComplexF64[1.0, 0.0])  # 1-qubit
        gate = QuantumGate("X", PAULI_X, [Qubit(2)])  # qubit 2 doesn't exist
        @test_throws BoundsError apply_gate(state, gate)
    end

    # ── Double Hadamard identity ───────────────────────────────────────
    @testset "apply_gate: H² = I (double Hadamard)" begin
        state = QuantumState(ComplexF64[0.6 + 0.0im, 0.8 + 0.0im])
        h_gate = QuantumGate("H", HADAMARD, [Qubit(1)])
        result = apply_gate(apply_gate(state, h_gate), h_gate)
        @test result.amplitudes ≈ state.amplitudes atol=1e-12
    end

    # ── measurement ────────────────────────────────────────────────────
    @testset "measure: deterministic |0⟩" begin
        state_0 = QuantumState(ComplexF64[1.0, 0.0])
        outcome, collapsed = measure(state_0)
        @test outcome == 0
        @test collapsed.amplitudes[1] ≈ 1.0 atol=1e-12
        @test collapsed.amplitudes[2] ≈ 0.0 atol=1e-12
    end

    @testset "measure: deterministic |1⟩" begin
        state_1 = QuantumState(ComplexF64[0.0, 1.0])
        outcome, collapsed = measure(state_1)
        @test outcome == 1
        @test collapsed.amplitudes[2] ≈ 1.0 atol=1e-12
    end

    @testset "measure: superposition produces valid outcomes" begin
        plus_state = QuantumState(ComplexF64[1/sqrt(2), 1/sqrt(2)])
        outcomes = Int[]
        for _ in 1:100
            o, _ = measure(plus_state)
            push!(outcomes, o)
        end
        # Should produce both 0 and 1 with high probability
        @test 0 in outcomes
        @test 1 in outcomes
        @test all(o -> o in [0, 1], outcomes)
    end

    @testset "measure: collapsed state is normalised" begin
        state = QuantumState(ComplexF64[1/sqrt(2), 1/sqrt(2)])
        _, collapsed = measure(state)
        @test sum(abs2.(collapsed.amplitudes)) ≈ 1.0 atol=1e-12
    end

    @testset "measure: two-qubit system" begin
        # |00⟩ state — always measures 0
        state_00 = QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0])
        outcome, collapsed = measure(state_00)
        @test outcome == 0
    end

    # ── tensor_product ─────────────────────────────────────────────────
    @testset "tensor_product: |0⟩ ⊗ |0⟩ = |00⟩" begin
        s0 = QuantumState(ComplexF64[1.0, 0.0])
        result = tensor_product(s0, s0)
        @test length(result.amplitudes) == 4
        @test result.amplitudes[1] ≈ 1.0 atol=1e-12
        @test all(x -> abs(x) < 1e-12, result.amplitudes[2:end])
    end

    @testset "tensor_product: |0⟩ ⊗ |1⟩ = |01⟩" begin
        s0 = QuantumState(ComplexF64[1.0, 0.0])
        s1 = QuantumState(ComplexF64[0.0, 1.0])
        result = tensor_product(s0, s1)
        @test result.amplitudes[2] ≈ 1.0 atol=1e-12
    end

    @testset "tensor_product: |1⟩ ⊗ |0⟩ = |10⟩" begin
        s0 = QuantumState(ComplexF64[1.0, 0.0])
        s1 = QuantumState(ComplexF64[0.0, 1.0])
        result = tensor_product(s1, s0)
        @test result.amplitudes[3] ≈ 1.0 atol=1e-12
    end

    @testset "tensor_product: preserves normalisation" begin
        a = QuantumState(ComplexF64[1/sqrt(2), 1/sqrt(2)])
        b = QuantumState(ComplexF64[1/sqrt(3), sqrt(2/3) + 0im])
        result = tensor_product(a, b)
        @test sum(abs2.(result.amplitudes)) ≈ 1.0 atol=1e-12
    end

    @testset "tensor_product: dimension scaling" begin
        s1 = QuantumState(ComplexF64[1.0, 0.0])          # 1 qubit
        s2 = QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0]) # 2 qubits
        result = tensor_product(s1, s2)
        @test num_qubits(result) == 3
        @test length(result.amplitudes) == 8
    end

    # ── state_evolve ───────────────────────────────────────────────────
    @testset "state_evolve: zero time step is identity" begin
        state = QuantumState(ComplexF64[1/sqrt(2), 1/sqrt(2)])
        H = ComplexF64[1 0; 0 -1]  # Pauli-Z
        evolved = state_evolve(state, H, 0.0)
        @test evolved.amplitudes ≈ state.amplitudes atol=1e-12
    end

    @testset "state_evolve: preserves normalisation" begin
        state = QuantumState(ComplexF64[1.0, 0.0])
        H = ComplexF64[0 1; 1 0]  # Pauli-X as Hamiltonian
        evolved = state_evolve(state, H, 0.5)
        @test sum(abs2.(evolved.amplitudes)) ≈ 1.0 atol=1e-12
    end

    @testset "state_evolve: eigenstate under diagonal H" begin
        # |0⟩ is eigenstate of Pauli-Z with eigenvalue +1
        # exp(-i*Z*t)|0⟩ = exp(-it)|0⟩ (global phase)
        state = QuantumState(ComplexF64[1.0, 0.0])
        H = ComplexF64[1 0; 0 -1]  # Pauli-Z
        t = 0.3
        evolved = state_evolve(state, H, t)
        # Should remain |0⟩ up to global phase
        @test abs2(evolved.amplitudes[1]) ≈ 1.0 atol=1e-12
        @test abs2(evolved.amplitudes[2]) ≈ 0.0 atol=1e-12
    end

    @testset "state_evolve: Rabi oscillation under Pauli-X" begin
        # Under H = σx, |0⟩ evolves to cos(t)|0⟩ - i*sin(t)|1⟩
        state = QuantumState(ComplexF64[1.0, 0.0])
        H = ComplexF64[0 1; 1 0]  # Pauli-X
        t = π / 4
        evolved = state_evolve(state, H, t)
        @test abs2(evolved.amplitudes[1]) ≈ cos(t)^2 atol=1e-10
        @test abs2(evolved.amplitudes[2]) ≈ sin(t)^2 atol=1e-10
    end

    @testset "state_evolve: dimension mismatch error" begin
        state = QuantumState(ComplexF64[1.0, 0.0])
        H_wrong = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
        @test_throws DimensionMismatch state_evolve(state, H_wrong, 0.1)
    end

    @testset "state_evolve: two-qubit system" begin
        state = QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0])
        H = kron(ComplexF64[0 1; 1 0], Matrix{ComplexF64}(I, 2, 2))  # X⊗I
        evolved = state_evolve(state, H, 0.1)
        @test sum(abs2.(evolved.amplitudes)) ≈ 1.0 atol=1e-12
    end

    # ── Point-to-point integration ─────────────────────────────────────
    @testset "Point-to-point: gate sequence" begin
        # Apply H then X to |0⟩
        state = QuantumState(ComplexF64[1.0, 0.0])
        h_gate = QuantumGate("H", HADAMARD, [Qubit(1)])
        x_gate = QuantumGate("X", PAULI_X, [Qubit(1)])

        after_h = apply_gate(state, h_gate)
        after_hx = apply_gate(after_h, x_gate)

        # X*H|0⟩ = X*(|0⟩+|1⟩)/√2 = (|1⟩+|0⟩)/√2
        @test after_hx.amplitudes[1] ≈ 1/sqrt(2) atol=1e-12
        @test after_hx.amplitudes[2] ≈ 1/sqrt(2) atol=1e-12
    end

    @testset "Point-to-point: tensor then gate" begin
        # Create |00⟩ via tensor product, then apply X to second qubit
        s0 = QuantumState(ComplexF64[1.0, 0.0])
        state_00 = tensor_product(s0, s0)
        x_on_2 = QuantumGate("X", PAULI_X, [Qubit(2)])
        result = apply_gate(state_00, x_on_2)
        # Should be |01⟩
        @test result.amplitudes[2] ≈ 1.0 atol=1e-12
    end

    # ── End-to-end ─────────────────────────────────────────────────────
    @testset "End-to-end: Bell state preparation" begin
        # Bell state: H on qubit 1 of |00⟩ gives (|0⟩+|1⟩)/√2 ⊗ |0⟩
        # (CNOT not available as single-qubit gate, but we can verify partial prep)
        s0 = QuantumState(ComplexF64[1.0, 0.0])
        h_gate = QuantumGate("H", HADAMARD, [Qubit(1)])
        after_h = apply_gate(s0, h_gate)

        # Tensor with |0⟩ to get 2-qubit register
        two_qubit = tensor_product(after_h, s0)
        @test num_qubits(two_qubit) == 2

        # Verify amplitudes: (|00⟩ + |10⟩)/√2
        @test two_qubit.amplitudes[1] ≈ 1/sqrt(2) atol=1e-12  # |00⟩
        @test abs(two_qubit.amplitudes[2]) < 1e-12              # |01⟩
        @test two_qubit.amplitudes[3] ≈ 1/sqrt(2) atol=1e-12  # |10⟩
        @test abs(two_qubit.amplitudes[4]) < 1e-12              # |11⟩
    end

    @testset "End-to-end: evolve then measure" begin
        # Start in |0⟩, evolve under X Hamiltonian for π/2 (complete flip)
        state = QuantumState(ComplexF64[1.0, 0.0])
        H = ComplexF64[0 1; 1 0]  # Pauli-X
        evolved = state_evolve(state, H, π / 2)

        # After π/2 evolution under X, should be close to -i|1⟩
        @test abs2(evolved.amplitudes[2]) ≈ 1.0 atol=1e-10

        # Measure should deterministically give 1
        outcome, _ = measure(evolved)
        @test outcome == 1
    end

    @testset "End-to-end: circuit execution pipeline" begin
        # Build a circuit, execute gates sequentially
        circ = QuantumCircuitObj(2, [
            QuantumGate("H", HADAMARD, [Qubit(1)]),
            QuantumGate("X", PAULI_X, [Qubit(2)]),
        ])

        state = QuantumState(ComplexF64[1.0, 0.0, 0.0, 0.0])
        for gate in circ.gates
            state = apply_gate(state, gate)
        end

        @test sum(abs2.(state.amplitudes)) ≈ 1.0 atol=1e-12
        @test num_qubits(state) == 2
    end

    # ── Benchmarks ─────────────────────────────────────────────────────
    @testset "Performance: single-qubit gate application" begin
        state = QuantumState(ComplexF64[1.0, 0.0])
        h_gate = QuantumGate("H", HADAMARD, [Qubit(1)])

        t = @elapsed begin
            for _ in 1:10_000
                state = apply_gate(state, h_gate)
            end
        end
        @test t < 10.0  # 10k single-qubit gates should be fast
    end

    @testset "Performance: tensor product scaling" begin
        s = QuantumState(ComplexF64[1.0, 0.0])

        t = @elapsed begin
            state = s
            for _ in 1:10  # Build up to 2^10 = 1024 amplitudes
                state = tensor_product(state, s)
            end
        end
        @test num_qubits(state) == 10
        @test t < 5.0
    end

    @testset "Performance: state evolution" begin
        n = 4  # 4-qubit system
        dim = 2^n
        state = QuantumState(vcat(ComplexF64[1.0], zeros(ComplexF64, dim - 1)))
        H = Hermitian(randn(ComplexF64, dim, dim))
        H_mat = Matrix{ComplexF64}(H)

        t = @elapsed begin
            for _ in 1:100
                state_evolve(state, H_mat, 0.01)
            end
        end
        @test t < 10.0
    end
end
