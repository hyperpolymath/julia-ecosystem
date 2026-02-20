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

            # TODO: Test encapsulation/decapsulation when NTT is implemented
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

            # TODO: Test signing/verification when full implementation done
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

            # TODO: Test signing/verification when implementation done
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
end
