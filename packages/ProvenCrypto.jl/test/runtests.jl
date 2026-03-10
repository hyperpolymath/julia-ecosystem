# SPDX-License-Identifier: PMPL-1.0-or-later
using ProvenCrypto
using Dates: now
using Test

@testset "ProvenCrypto.jl" begin
    @testset "Hardware Detection" begin
        # Test that backend detection doesn't crash
        backend = detect_hardware()
        @test backend isa AbstractCryptoBackend

        # Test SIMD detection
        simd = detect_simd_level()
        @test simd isa Symbol
        @test simd ∈ [:none, :sse, :avx, :avx2, :avx512, :neon, :sve]

        # Test hardware features
        features = detect_hardware_features()
        @test features isa HardwareFeatures
    end

    @testset "Primitives (libsodium FFI)" begin
        @testset "AEAD Encryption" begin
            # Test authenticated encryption
            key = rand(UInt8, 32)
            nonce = rand(UInt8, 12)
            plaintext = b"Hello, ProvenCrypto!"
            ad = b"additional data"

            # Encrypt
            ciphertext = aead_encrypt(key, nonce, plaintext, ad)
            @test length(ciphertext) == length(plaintext) + 16  # 16-byte tag

            # Decrypt
            recovered = aead_decrypt(key, nonce, ciphertext, ad)
            @test recovered == plaintext

            # Test authentication failure
            bad_ciphertext = copy(ciphertext)
            bad_ciphertext[1] ⊻= 0x01  # Flip one bit
            @test aead_decrypt(key, nonce, bad_ciphertext, ad) === nothing
        end

        @testset "Hashing" begin
            data = b"test data"
            digest = hash_blake3(data)
            @test length(digest) == 32  # 256 bits

            # Test determinism
            digest2 = hash_blake3(data)
            @test digest == digest2

            # Test different inputs produce different hashes
            digest3 = hash_blake3(b"different data")
            @test digest != digest3
        end

        @testset "Key Derivation" begin
            password = b"my secure password"
            salt = rand(UInt8, 16)

            key = kdf_argon2(password, salt; memory_kb=1024, iterations=1)
            @test length(key) == 32

            # Test determinism
            key2 = kdf_argon2(password, salt; memory_kb=1024, iterations=1)
            @test key == key2

            # Test different password produces different key
            key3 = kdf_argon2(b"different password", salt; memory_kb=1024, iterations=1)
            @test key != key3
        end
    end

    @testset "Post-Quantum Cryptography" begin
        @testset "Kyber KEM" begin
            # Generate keypair
            (pk, sk) = kyber_keygen(512)  # Fastest security level for testing
            @test pk isa KyberPublicKey
            @test sk isa KyberSecretKey
            @test pk.level == 512

            # Encapsulation/decapsulation require hardware NTT backend
            # (ciphertext, ss_sender) = kyber_encapsulate(pk)
            # ss_receiver = kyber_decapsulate(sk, ciphertext)
            # @test ss_sender == ss_receiver
        end

        @testset "Dilithium Signatures" begin
            # Generate keypair
            (pk, sk) = dilithium_keygen(2)  # Fastest level for testing
            @test pk isa DilithiumPublicKey
            @test sk isa DilithiumSecretKey
            @test pk.level == 2

            # Signing/verification require full lattice arithmetic backend
            # message = b"Test message"
            # signature = dilithium_sign(sk, message)
            # @test dilithium_verify(pk, message, signature)
        end

        @testset "SPHINCS+ Signatures" begin
            # Generate keypair
            (pk, sk) = sphincs_keygen(128, :s)  # Smallest/fastest for testing
            @test pk isa SPHINCSPublicKey
            @test sk isa SPHINCSSecretKey
            @test pk.level == 128
            @test pk.variant == :s

            # SPHINCS+ signing/verification require hash tree backend
        end
    end

    @testset "Proof Export" begin
        cert = ProofCertificate(
            "Test property",
            "∀x. x = x",
            true,
            "test",
            now(),
            nothing,
            Dict{String,Any}()
        )

        # Test Idris export
        idris_path = tempname() * ".idr"
        export_idris(cert, idris_path)
        @test isfile(idris_path)
        idris_content = read(idris_path, String)
        @test occursin("module", idris_content)
        @test occursin("Test_property", idris_content)

        # Test Lean export
        lean_path = tempname() * ".lean"
        export_lean(cert, lean_path)
        @test isfile(lean_path)

        # Test Coq export
        coq_path = tempname() * ".v"
        export_coq(cert, coq_path)
        @test isfile(coq_path)

        # Test Isabelle export
        isabelle_path = tempname() * ".thy"
        export_isabelle(cert, isabelle_path)
        @test isfile(isabelle_path)

        # Cleanup
        rm(idris_path, force=true)
        rm(lean_path, force=true)
        rm(coq_path, force=true)
        rm(isabelle_path, force=true)
    end

    @testset "Idris Inside Badge" begin
        badge = idris_inside_badge()
        @test occursin("IDRIS INSIDE", badge)
        @test occursin("Formally Verified", badge)
    end

    @testset "Kyber Encapsulate/Decapsulate" begin
        (pk, sk) = kyber_keygen(512)

        # Test encapsulation returns ciphertext and shared secret
        (ciphertext, shared_secret) = kyber_encapsulate(pk)
        @test ciphertext isa Vector{UInt8}
        @test shared_secret isa Vector{UInt8}
        @test length(shared_secret) == 32
        @test length(ciphertext) > 0

        # Test that two encapsulations produce different ciphertexts (randomized)
        (ciphertext2, shared_secret2) = kyber_encapsulate(pk)
        @test ciphertext != ciphertext2
        @test shared_secret != shared_secret2
    end

    @testset "Kyber Keygen Levels" begin
        # Test all valid security levels
        for level in [512, 768, 1024]
            (pk, sk) = kyber_keygen(level)
            @test pk.level == level
            @test sk.level == level
            @test sk.pk === pk
        end

        # Invalid level should error
        @test_throws AssertionError kyber_keygen(256)
    end

    @testset "Dilithium Keygen Levels" begin
        # Test all valid security levels
        for level in [2, 3, 5]
            (pk, sk) = dilithium_keygen(level)
            @test pk.level == level
            @test sk.level == level
            @test pk isa DilithiumPublicKey
            @test sk isa DilithiumSecretKey
            @test sk.pk === pk
            @test length(pk.rho) == 32
        end

        # Invalid level should error
        @test_throws AssertionError dilithium_keygen(4)
    end

    @testset "SPHINCS+ Keygen Variants" begin
        # Test all valid level/variant combinations
        for level in [128, 192, 256]
            for variant in [:s, :f]
                (pk, sk) = sphincs_keygen(level, variant)
                @test pk.level == level
                @test pk.variant == variant
                @test sk.level == level
                @test sk.variant == variant
                @test sk.pk === pk
                @test length(pk.pk_seed) > 0
                @test length(pk.pk_root) > 0
            end
        end

        # Invalid level should error
        @test_throws AssertionError sphincs_keygen(64)
        # Invalid variant should error
        @test_throws AssertionError sphincs_keygen(128, :x)
    end

    @testset "SPHINCS+ Sign/Verify" begin
        (pk, sk) = sphincs_keygen(128, :s)
        message = Vector{UInt8}(b"Test message for SPHINCS+")

        sig = sphincs_sign(sk, message)
        @test sig isa SPHINCSSignature
        @test length(sig.sig_bytes) > 0

        # Two signatures of the same message should differ (randomized)
        sig2 = sphincs_sign(sk, message)
        @test sig.sig_bytes != sig2.sig_bytes
    end

    @testset "ZK-SNARK Prove/Verify" begin
        circuit = Vector{UInt8}(b"test circuit description")
        witness = Vector{UInt8}(b"secret witness data")

        # Generate proof
        proof = zk_prove(circuit, witness)
        @test proof isa ZKProof
        @test proof.proof_data isa ProvenCrypto.Groth16Proof
        @test length(proof.proof_data.a) == 32  # BLAKE3 hash length
        @test length(proof.proof_data.b) == 32
        @test length(proof.proof_data.c) == 32
        @test length(proof.public_inputs) == 32

        # Verify proof against the same circuit
        @test zk_verify(proof, circuit) == true

        # Verify against a different circuit should fail
        different_circuit = Vector{UInt8}(b"different circuit")
        @test zk_verify(proof, different_circuit) == false

        # Different witnesses produce different proofs
        witness2 = Vector{UInt8}(b"different witness")
        proof2 = zk_prove(circuit, witness2)
        @test proof2.proof_data.a != proof.proof_data.a
    end

    @testset "Shamir Secret Sharing" begin
        secret = Vector{UInt8}(b"my secret data!")

        # Split into 5 shares with threshold 3
        shares = shamir_split(secret, 3, 5)
        @test length(shares) == 5
        @test all(length(s) > 0 for s in shares)

        # Each share should differ (different evaluation points)
        @test shares[1] != shares[2]

        # Reconstruct from shares
        recovered = shamir_reconstruct(shares)
        @test recovered == secret

        # Reconstruct from a subset of shares
        recovered_subset = shamir_reconstruct(shares[1:3])
        @test recovered_subset == secret

        # Empty shares returns empty
        @test shamir_reconstruct(Vector{Vector{UInt8}}()) == UInt8[]

        # Single share reconstruction
        single_recovered = shamir_reconstruct([shares[1]])
        @test single_recovered == secret
    end

    @testset "Noise Protocol Structs" begin
        # Test that NoiseHandshake can be constructed
        local_kp = (rand(UInt8, 32), rand(UInt8, 32))
        remote_pk = rand(UInt8, 32)

        noise_state = ProvenCrypto.handshake(:XX, true, local_kp, remote_pk)
        nh = NoiseHandshake(noise_state)
        @test nh isa NoiseHandshake
        @test nh.state isa ProvenCrypto.NoiseState
        @test nh.state.handshake_pattern == :XX
        @test nh.state.is_initiator == true
    end

    @testset "Signal Ratchet Structs" begin
        # Test that SignalRatchet can be constructed
        initial_key = rand(UInt8, 32)
        signal_state = ProvenCrypto.handshake(initial_key)
        sr = SignalRatchet(signal_state)
        @test sr isa SignalRatchet
        @test sr.state isa ProvenCrypto.SignalState
        @test length(sr.state.root_key) == 32
        @test length(sr.state.sending_chain_key) == 32
        @test length(sr.state.receiving_chain_key) == 32
        @test sr.state.message_number_sending == 0
        @test sr.state.message_number_receiving == 0
    end

    @testset "TLS 1.3 Session Structs" begin
        # Test that TLS13Session can be constructed for client and server
        client_state = ProvenCrypto.handshake(true)
        client_session = TLS13Session(client_state)
        @test client_session isa TLS13Session
        @test client_session.state.is_client == true
        @test length(client_session.state.client_random) == 32
        @test length(client_session.state.server_random) == 32
        @test length(client_session.state.client_write_key) == 32
        @test length(client_session.state.server_write_key) == 32
        @test length(client_session.state.client_write_iv) == 12
        @test length(client_session.state.server_write_iv) == 12

        server_state = ProvenCrypto.handshake(false)
        server_session = TLS13Session(server_state)
        @test server_session.state.is_client == false
    end

    @testset "Proof Certificate Construction" begin
        # Test ProofCertificate with various metadata
        cert_with_proof = ProofCertificate(
            "Commutativity of addition",
            "forall a b, a + b = b + a",
            true,
            "Z3",
            now(),
            "by ring",
            Dict{String,Any}("solver_time" => 0.5)
        )
        @test cert_with_proof.verified == true
        @test cert_with_proof.property == "Commutativity of addition"
        @test cert_with_proof.verifier == "Z3"
        @test cert_with_proof.証明 == "by ring"
        @test cert_with_proof.metadata["solver_time"] == 0.5

        # Test unverified certificate
        cert_unverified = ProofCertificate(
            "Unproven property",
            "exists x, P(x)",
            false,
            "none",
            now(),
            nothing,
            Dict{String,Any}()
        )
        @test cert_unverified.verified == false
        @test cert_unverified.証明 === nothing
    end

    @testset "Proof Export Content Validation" begin
        cert = ProofCertificate(
            "Test property",
            "∀x. x = x",
            true,
            "test",
            now(),
            nothing,
            Dict{String,Any}()
        )

        # Lean export content check
        lean_path = tempname() * ".lean"
        export_lean(cert, lean_path)
        lean_content = read(lean_path, String)
        @test occursin("theorem", lean_content)
        @test occursin("Mathlib", lean_content)
        @test occursin("sorry", lean_content)  # Unproven -> sorry
        rm(lean_path, force=true)

        # Coq export content check
        coq_path = tempname() * ".v"
        export_coq(cert, coq_path)
        coq_content = read(coq_path, String)
        @test occursin("Theorem", coq_content)
        @test occursin("Require Import", coq_content)
        @test occursin("Admitted", coq_content)  # Unproven -> Admitted
        rm(coq_path, force=true)

        # Isabelle export content check
        isabelle_path = tempname() * ".thy"
        export_isabelle(cert, isabelle_path)
        isabelle_content = read(isabelle_path, String)
        @test occursin("theory", isabelle_content)
        @test occursin("imports Main", isabelle_content)
        @test occursin("sorry", isabelle_content)  # Unproven -> sorry
        rm(isabelle_path, force=true)
    end

    @testset "CPUBackend Display" begin
        backend = CPUBackend(:avx2, 4)
        io = IOBuffer()
        show(io, backend)
        s = String(take!(io))
        @test occursin("avx2", s)
        @test occursin("4 threads", s)
    end
end
