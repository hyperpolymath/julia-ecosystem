# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    ProvenCrypto

Formally verified cryptographic protocols and post-quantum primitives for Julia.

# Features
- Post-quantum cryptography (Kyber, Dilithium, SPHINCS+)
- Protocol implementations (TLS 1.3, Noise Protocol, Signal)
- Zero-knowledge proof systems (zk-SNARKs, zk-STARKs)
- Threshold cryptography and MPC primitives
- Hardware acceleration (GPU, NPU, TPU, DSP, FPGA)
- Multi-platform support (x86, ARM, RISC-V, Apple Silicon)
- Formal verification integration (SMT, Idris, Lean, Coq)

# Architecture
- Layer 1: Verified primitives (FFI to libsodium, BoringSSL)
- Layer 2: Protocol implementations (pure Julia)
- Layer 3: Post-quantum algorithms (pure Julia + formal verification)

# Safety
**⚠️ WARNING: Research and educational use only.**
This library is NOT FIPS-certified and should not be used for
security-critical production systems without thorough review.

For production use, prefer:
- Symmetric crypto: libsodium via FFI
- Classical asymmetric: OpenSSL FIPS module
- Memory-hard KDFs: Argon2 C library

# Hardware Support
- NVIDIA GPUs: CUDA backend
- AMD GPUs: ROCm backend (AMDGPU.jl)
- Apple Silicon: Metal backend + Neural Engine
- Intel GPUs/NPUs: oneAPI backend
- TPUs: Google Cloud TPU via XLA
- FPGAs: OpenCL backend (platform-specific)
- DSPs: Platform-specific acceleration

# Verification
All critical functions include formal verification claims.
Use `@prove` macro from SMTLib integration for property verification.
Export proofs to Idris, Lean, Coq, or Isabelle for long-term formalization.

# License
PMPL-1.0-or-later (Polymathematical Meta-Public License)
"""
module ProvenCrypto

using Dates
using JSON
using Libdl
using LinearAlgebra
using Random
using SHA
using Statistics

# Core exports
export AbstractCryptoBackend,
       CPUBackend,
       detect_hardware,
       detect_simd_level,
       detect_hardware_features,
       HardwareFeatures,

       # Primitives (FFI wrappers)
       aead_encrypt,
       aead_decrypt,
       hash_blake3,
       kdf_argon2,

       # Post-quantum
       KyberPublicKey,
       KyberSecretKey,
       kyber_keygen,
       kyber_encapsulate,
       kyber_decapsulate,
       DilithiumPublicKey,
       DilithiumSecretKey,
       dilithium_keygen,
       dilithium_sign,
       dilithium_verify,
       SPHINCSPublicKey,
       SPHINCSSecretKey,
       sphincs_keygen,
       sphincs_sign,
       sphincs_verify,

       # Protocols
       NoiseHandshake,
       SignalRatchet,
       TLS13Session,

       # Zero-knowledge
       ZKProof,
       zk_prove,
       zk_verify,

       # Threshold
       shamir_split,
       shamir_reconstruct,

       # Verification
       ProofCertificate,
       idris_inside_badge,
       export_idris,
       export_lean,
       export_coq,
       export_isabelle

# Include core modules
include("backends/hardware.jl")
include("backends/advanced_hardware.jl")
include("primitives/ffi_wrappers.jl")
include("postquantum/kyber.jl")
include("postquantum/dilithium.jl")
include("postquantum/sphincs.jl")
include("protocols/noise.jl")
include("protocols/signal.jl")
include("protocols/tls13.jl")
include("zkproofs/zksnark.jl")
include("zkproofs/zkstark.jl")
include("threshold/shamir.jl")
include("verification/proof_export.jl")

# Hardware detection on module load
const HARDWARE_BACKEND = Ref{AbstractCryptoBackend}()

function __init__()
    init_ffi()

    # Load coprocessor extensions conditionally based on AcceleratorGate detection.
    # These do not have external Julia package triggers, so we include them
    # when their respective backends are detected via environment flags.
    _load_coprocessor_extensions()

    HARDWARE_BACKEND[] = detect_hardware()
    @info "ProvenCrypto.jl initialized" backend=HARDWARE_BACKEND[]
end

"""
    _load_coprocessor_extensions()

Conditionally load coprocessor backend extensions based on AcceleratorGate
availability detection. Each extension is loaded when its corresponding
AXIOM_*_AVAILABLE environment flag is set, or unconditionally if the
extension file exists and the backend type is defined in AcceleratorGate.
"""
function _load_coprocessor_extensions()
    ext_dir = joinpath(dirname(@__DIR__), "ext")

    # Map: (extension filename, AcceleratorGate availability function)
    coprocessor_exts = [
        ("ProvenCryptoTPUExt.jl",    :tpu_available),
        ("ProvenCryptoFPGAExt.jl",   :fpga_available),
        ("ProvenCryptoCryptoExt.jl", :crypto_available),
        ("ProvenCryptoNPUExt.jl",    :npu_available),
        ("ProvenCryptoDSPExt.jl",    :dsp_available),
        ("ProvenCryptoVPUExt.jl",    :vpu_available),
        ("ProvenCryptoQPUExt.jl",    :qpu_available),
        ("ProvenCryptoPPUExt.jl",    :ppu_available),
        ("ProvenCryptoMathExt.jl",   :math_available),
    ]

    for (filename, avail_fn) in coprocessor_exts
        ext_path = joinpath(ext_dir, filename)
        if isfile(ext_path)
            try
                # Load if the coprocessor is detected as available
                if getfield(AcceleratorGate, avail_fn)()
                    include(ext_path)
                    @debug "Loaded coprocessor extension" file=filename
                end
            catch e
                @warn "Failed to load coprocessor extension" file=filename exception=e
            end
        end
    end
end

end # module
