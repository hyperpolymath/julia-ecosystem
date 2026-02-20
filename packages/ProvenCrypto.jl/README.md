# ProvenCrypto.jl

[![Project Topology](https://img.shields.io/badge/Project-Topology-9558B2)](TOPOLOGY.md)
[![Completion Status](https://img.shields.io/badge/Completion-90%25-green)](TOPOLOGY.md)

```
╔═══════════════════════════╗
║    IDRIS INSIDE   🟦      ║
║   Formally Verified       ║
║  Dependent Type Checked   ║
╚═══════════════════════════╝
```

Formally verified cryptographic protocols and post-quantum primitives for Julia.

## Features

### Post-Quantum Cryptography
- **Kyber**: KEM (Key Encapsulation Mechanism) - NIST PQC winner
- **Dilithium**: Digital signatures - NIST PQC winner
- **SPHINCS+**: Hash-based signatures (conservative security)

### Protocol Implementations
- **Noise Protocol Framework**: Modern secure channel (WireGuard, Lightning)
- **Signal Protocol**: Double Ratchet for messaging
- **TLS 1.3**: Reference implementation (educational)

### Zero-Knowledge Proofs
- **zk-SNARKs**: Groth16, PLONK, Halo2
- **zk-STARKs**: Transparent, post-quantum secure

### Threshold Cryptography
- **Shamir Secret Sharing**: M-of-N key recovery
- **Distributed Key Generation**: Multi-party computation

### Hardware Acceleration
- **GPU**: CUDA (NVIDIA), ROCm (AMD), Metal (Apple)
- **NPU/TPU**: Intel oneAPI, Google TPU, Apple Neural Engine
- **Crypto Instructions**: AES-NI, SHA extensions, Intel QAT
- **Secure Enclaves**: Intel SGX, AMD SEV, ARM TrustZone
- **Multi-platform**: x86, ARM, RISC-V, Apple Silicon

### Formal Verification
- **SMT Integration**: Z3, CVC5, Yices, MathSAT
- **Proof Assistants**: Idris 2, Lean 4, Coq, Isabelle/HOL
- **Property Verification**: Correctness, security properties
- **Proof Export**: Long-term formalization

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hyperpolymath/ProvenCrypto.jl")
```

### System Dependencies

#### libsodium (required for primitives)
```bash
# macOS
brew install libsodium

# Ubuntu/Debian
sudo apt install libsodium-dev

# Fedora
sudo dnf install libsodium-devel

# Arch
sudo pacman -S libsodium
```

## Usage

### Post-Quantum Key Exchange (Kyber)

```julia
using ProvenCrypto

# Generate keypair
(pk, sk) = kyber_keygen(768)  # AES-192 equivalent security

# Sender: Encapsulate shared secret
(ciphertext, shared_secret_sender) = kyber_encapsulate(pk)

# Receiver: Decapsulate
shared_secret_receiver = kyber_decapsulate(sk, ciphertext)

@assert shared_secret_sender == shared_secret_receiver
```

### Post-Quantum Signatures (Dilithium)

```julia
using ProvenCrypto

# Generate signing keypair
(pk, sk) = dilithium_keygen(3)  # AES-192 equivalent security

# Sign message
message = b"Hello, post-quantum world!"
signature = dilithium_sign(sk, message)

# Verify
is_valid = dilithium_verify(pk, message, signature)
@assert is_valid
```

### Authenticated Encryption (libsodium FFI)

```julia
using ProvenCrypto

# Generate key and nonce
key = rand(UInt8, 32)
nonce = rand(UInt8, 12)  # Must be unique per message!

# Encrypt
plaintext = b"Secret message"
ciphertext = aead_encrypt(key, nonce, plaintext)

# Decrypt
recovered = aead_decrypt(key, nonce, ciphertext)
@assert recovered == plaintext
```

### Hardware-Accelerated Operations

```julia
using ProvenCrypto

# Detect available hardware
backend = detect_hardware()
println(backend)
# Output: MetalBackend(M3 + Neural Engine)
#     or: CUDABackend(device=0, CC=8.9)
#     or: CPUBackend(avx512, 16 threads)

# Operations automatically use best backend
features = detect_hardware_features()
print_hardware_report(features)
```

## Security Warnings

### ⚠️ Production Use

**This library is for research and educational purposes.**

For production systems, use:
- **Symmetric crypto**: libsodium (via FFI wrappers in this library)
- **Classical asymmetric**: OpenSSL FIPS module
- **Memory-hard KDFs**: Argon2 C library (via FFI in this library)

### ⚠️ Not FIPS-Certified

Pure Julia implementations are NOT FIPS 140-2/3 certified. For compliance-critical systems, use FIPS-certified libraries via FFI.

### ✅ What's Safe

- FFI wrappers to proven libraries (libsodium, BoringSSL)
- Post-quantum reference implementations (research, interoperability)
- Protocol verification and formal analysis
- Standards compliance testing

## Containerized Execution

For maximum security isolation, run cryptographic operations in the verified container:

```bash
# Build container
cd verified-container-spec/examples/proven-crypto-runner
podman build -t proven-crypto-runner -f Containerfile

# Run with svalinn/vordr security policy
svalinn run --policy svalinn-policy.json \
    -v ./ProvenCrypto.jl:/crypto/provencrypto:ro \
    proven-crypto-runner keygen

# Interactive REPL
podman run -it -v ./ProvenCrypto.jl:/crypto/provencrypto:ro \
    proven-crypto-runner repl
```

Security features:
- Process isolation (PID, network, mount, IPC, UTS, user namespaces)
- Seccomp filters (syscall allowlist)
- Resource limits (1 CPU, 2GB RAM, 512 processes)
- No capabilities (runs with minimal privileges)
- Reproducible builds (Guix + Nix fallback)

## Formal Verification

Export verification certificates to proof assistants:

```julia
using ProvenCrypto

# Create verification certificate
cert = ProofCertificate(
    property="Kyber decapsulation correctness",
    specification="∀pk,sk,c,ss. decapsulate(sk, fst(encapsulate(pk))) = snd(encapsulate(pk))",
    verified=true,
    verifier="SMT-Z3",
    timestamp=now(),
    證明=nothing,
    metadata=Dict()
)

# Export to Idris 2
export_idris(cert, "proofs/kyber_correctness.idr")

# Export to Lean 4
export_lean(cert, "proofs/kyber_correctness.lean")

# Export to Coq
export_coq(cert, "proofs/kyber_correctness.v")

# Display Idris Inside badge
println(idris_inside_badge())
```

## Architecture

### Layer 1: Verified Primitives (FFI)
- libsodium: Authenticated encryption, hashing
- BoringSSL: Classical asymmetric crypto
- Argon2: Memory-hard KDF

### Layer 2: Protocols (Pure Julia)
- Noise, Signal, TLS 1.3 implementations
- Uses Layer 1 primitives via FFI

### Layer 3: Post-Quantum (Pure Julia + Verification)
- Kyber, Dilithium, SPHINCS+
- Hardware-accelerated (NTT via GPU/TPU/NPU)
- Formal verification claims

## Development

### Running Tests

```bash
julia --project -e 'using Pkg; Pkg.test("ProvenCrypto")'
```

### Benchmarks

```bash
julia --project benchmark/benchmarks.jl
```

### Building Documentation

```bash
julia --project docs/make.jl
```

## License

PMPL-1.0-or-later (Polymathematical Meta-Public License)

## Author

Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

## References

- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Kyber Specification](https://pq-crystals.org/kyber/)
- [Dilithium Specification](https://pq-crystals.org/dilithium/)
- [SPHINCS+ Specification](https://sphincs.org/)
- [Noise Protocol Framework](https://noiseprotocol.org/)
- [Signal Protocol](https://signal.org/docs/)
- [libsodium](https://libsodium.org/)
- [Idris 2](https://www.idris-lang.org/)
