# SONNET-TASKS: ProvenCrypto.jl

**Date:** 2026-02-12
**Auditor:** Claude Opus 4.6
**Honest Completion:** ~15%

The README and CHANGELOG claim a functioning post-quantum cryptography library with
hardware acceleration, protocol implementations, zero-knowledge proofs, threshold
cryptography, and formal verification export. In reality, every single cryptographic
algorithm is a stub returning placeholder data. The only code that actually works is:
(1) hardware backend detection (CPU SIMD level), (2) libsodium FFI wrappers for
AEAD/hashing/KDF (if libsodium is installed), and (3) proof export to files (but the
spec-to-prover translation functions are hardcoded examples, not real translators).
Everything else -- Kyber, Dilithium, SPHINCS+, Noise, Signal, TLS 1.3, zk-SNARKs,
zk-STARKs, Shamir, GPU backends -- is empty structs and functions returning empty
arrays or identity values.

There is also a module load crash: two `__init__()` functions compete, three declared
extensions have no source files, the advanced hardware module is never included but
tests reference it, and template placeholders `{{PROJECT}}` are never replaced in 13 files.

---

## GROUND RULES FOR SONNET

1. Do NOT just add more stubs. Every function body you write must do real computation.
2. Do NOT skip verification blocks. Every task must be testable with the commands given.
3. Do NOT claim completion without running the tests. Julia must load the module and
   pass all tests without errors.
4. If a task says "implement X per the NIST spec," you must follow the actual spec, not
   invent a simplified version.
5. Constant-time operations are required where noted. Using `rand()` as a placeholder
   for cryptographic sampling is a security vulnerability.
6. The ABI/FFI files (Idris, Zig) are RSR template boilerplate -- they still have
   `ProvenCrypto` placeholders. Either customize them for ProvenCrypto or remove them.

---

## TASK 1: Fix Module Load Crash -- Dual `__init__()` and Missing Include

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/ProvenCrypto.jl` (lines 118-121)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/primitives/ffi_wrappers.jl` (lines 237-239)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/backends/advanced_hardware.jl` (entire file -- never included)

**Problem:**
Two `function __init__()` definitions exist. The one in `ffi_wrappers.jl` (line 237)
calls `__init_libsodium__()`, and the one in `ProvenCrypto.jl` (line 118) calls
`detect_hardware()`. Julia modules can only have one `__init__()`. The second definition
silently overwrites the first. Whichever file is `include()`d last wins, meaning either
libsodium never loads or hardware detection never runs.

Additionally, `src/backends/advanced_hardware.jl` defines `HardwareFeatures`,
`detect_hardware_features()`, and `print_hardware_report()` but is never `include()`d in
the main module. The test file (`test/runtests.jl` lines 17-18) calls these functions,
so the test suite will crash with `UndefVarError`.

**What to do:**
1. Remove the `__init__()` from `ffi_wrappers.jl`. Rename it to `init_libsodium()` or
   similar.
2. In the main `ProvenCrypto.jl` `__init__()`, call both `init_libsodium()` and
   `detect_hardware()`.
3. Add `include("backends/advanced_hardware.jl")` to `ProvenCrypto.jl` after line 102
   (after `include("backends/hardware.jl")`).
4. Export `HardwareFeatures`, `detect_hardware_features`, and `print_hardware_report`
   from the module.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto
# Module loaded without crash
println("Module loaded OK")
# Hardware detection ran
println("Backend: ", ProvenCrypto.HARDWARE_BACKEND[])
# Advanced hardware features available
features = detect_hardware_features()
println("Features type: ", typeof(features))
@assert features isa HardwareFeatures
println("TASK 1 PASS")
'
```

---

## TASK 2: Fix Missing Extensions (3 Declared but No Source Files)

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/Project.toml` (lines 24, 27, 28)
- Missing: `ext/ProvenCryptoAMDGPUExt.jl`
- Missing: `ext/ProvenCryptoOneAPIExt.jl`
- Missing: `ext/ProvenCryptoSMTExt.jl`

**Problem:**
`Project.toml` declares five extensions:
```toml
ProvenCryptoAMDGPUExt = ["AMDGPU"]
ProvenCryptoCUDAExt = ["CUDA"]
ProvenCryptoMetalExt = ["Metal"]
ProvenCryptoOneAPIExt = ["oneAPI"]
ProvenCryptoSMTExt = ["SMTLib"]
```
Only three files exist (`ext/ProvenCryptoCUDAExt.jl`, `ext/ProvenCryptoMetalExt.jl`,
`ext/ProvenCryptoROCmExt.jl`). Note that the ROCm extension file exists but is named
`ProvenCryptoROCmExt.jl` while the Project.toml declares `ProvenCryptoAMDGPUExt`. This
is a mismatch -- Julia looks for a module named `ProvenCryptoAMDGPUExt` but the file
defines `module ProvenCryptoROCmExt`.

The `ProvenCryptoOneAPIExt.jl` and `ProvenCryptoSMTExt.jl` files do not exist at all.

**What to do:**
1. Rename `ext/ProvenCryptoROCmExt.jl` to `ext/ProvenCryptoAMDGPUExt.jl` and change
   the internal `module ProvenCryptoROCmExt` to `module ProvenCryptoAMDGPUExt`.
2. Create `ext/ProvenCryptoOneAPIExt.jl` with at minimum the `oneapi_available()` override
   and stub backend methods (matching the pattern in MetalExt).
3. Create `ext/ProvenCryptoSMTExt.jl` with `@prove` macro integration for SMT-LIB solvers.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
# Check all declared extensions have matching files
for ext in ProvenCryptoAMDGPUExt ProvenCryptoCUDAExt ProvenCryptoMetalExt ProvenCryptoOneAPIExt ProvenCryptoSMTExt; do
  if [ -f "ext/${ext}.jl" ]; then
    echo "OK: ext/${ext}.jl exists"
    grep -q "module ${ext}" "ext/${ext}.jl" && echo "  module name matches" || echo "  ERROR: module name mismatch"
  else
    echo "MISSING: ext/${ext}.jl"
  fi
done
```

---

## TASK 3: Implement CPU NTT Backend (Blocks All Post-Quantum Algorithms)

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/backends/hardware.jl` (lines 247-257)

**Problem:**
Four critical CPU backend functions are placeholders that return identity or garbage:

- `backend_ntt_transform(::CPUBackend, poly, modulus)` at line 248: returns `poly` unchanged.
  NTT (Number Theoretic Transform) is the core operation for all lattice-based crypto.
  Without it, Kyber and Dilithium produce mathematically wrong results.

- `backend_polynomial_multiply(::CPUBackend, a, b, modulus)` at line 252: returns `a`
  unchanged, ignoring `b` entirely.

- `backend_sampling(::CPUBackend, distribution, params...)` at line 256: calls `randn()`
  which returns a Float64 instead of a properly sampled integer from the specified
  distribution.

- `backend_ntt_inverse_transform` at kyber.jl line 213: global stub returns `poly`
  unchanged.

**What to do:**
1. Implement Cooley-Tukey NTT for `backend_ntt_transform` over `Z_q` where `q=3329`
   (Kyber) or `q=8380417` (Dilithium). Use primitive roots of unity for each modulus.
2. Implement inverse NTT for `backend_ntt_inverse_transform`. Move it from the kyber.jl
   stub to `hardware.jl` as a proper `CPUBackend` method.
3. Implement polynomial multiplication via NTT: `a*b = INTT(NTT(a) .* NTT(b))` mod q.
4. Implement constant-time centered binomial distribution sampling for `:cbd` and
   uniform sampling for `:uniform`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto
backend = CPUBackend(:avx2, 1)

# NTT round-trip test: INTT(NTT(x)) == x
q = 3329
x = rand(0:q-1, 256)
x_ntt = ProvenCrypto.backend_ntt_transform(backend, x, q)
x_recovered = ProvenCrypto.backend_ntt_inverse_transform(backend, x_ntt, q)
@assert x_recovered == x "NTT round-trip failed"

# Polynomial multiplication test: (1+x) * (1+x) mod q
a = zeros(Int, 256); a[1] = 1; a[2] = 1
b = zeros(Int, 256); b[1] = 1; b[2] = 1
c = ProvenCrypto.backend_polynomial_multiply(backend, a, b, q)
@assert c[1] == 1  # constant term
@assert c[2] == 2  # x coefficient
@assert c[3] == 1  # x^2 coefficient

println("TASK 3 PASS")
'
```

---

## TASK 4: Implement Kyber KEM Helper Functions

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/postquantum/kyber.jl` (lines 169-213)

**Problem:**
Seven helper functions are stubs returning garbage:

1. `kyber_gen_matrix` (line 170): Returns `rand(Int, k, n) .% q` instead of
   deterministically generating matrix A from seed via SHAKE-128 (XOF).
2. `kyber_sample_cbd` (line 176): Returns `rand(Int, k, n) .- (eta / 2)` instead of
   constant-time centered binomial distribution sampling.
3. `kyber_encode` (line 182): Returns `zeros(Int, length(msg)*8)` -- wrong dimensions,
   wrong values.
4. `kyber_decode` (line 188): Returns `UInt8[]` -- empty array, loses all data.
5. `kyber_compress` (line 194): Returns `UInt8[]` -- empty, makes ciphertext vanish.
6. `kyber_decompress` (line 200): Returns zeros -- makes decapsulation impossible.
7. `encode_pk` (line 206): Returns `UInt8[]` -- empty serialization breaks hashing.

The top-level functions (`kyber_keygen`, `kyber_encapsulate`, `kyber_decapsulate`) have
correct structure but produce wrong results because every helper returns garbage.

**What to do:**
Implement each function per FIPS 203 (ML-KEM) / the Kyber specification:
1. `kyber_gen_matrix`: Use SHAKE-128 (or SHA3) to expand seed into matrix A. Julia's SHA
   package has SHA3 support.
2. `kyber_sample_cbd`: Implement CBD_eta sampling per Algorithm 2 of the Kyber spec.
3. `kyber_encode`/`kyber_decode`: Implement Compress_d and Decompress_d per Kyber spec
   Section 4.2.1.
4. `kyber_compress`/`kyber_decompress`: Implement byte-level compression per spec.
5. `encode_pk`: Serialize `(t, rho)` as per spec byte encoding.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto

# Full Kyber round-trip
(pk, sk) = kyber_keygen(512)
(ciphertext, ss_sender) = kyber_encapsulate(pk)
ss_receiver = kyber_decapsulate(sk, ciphertext)

@assert length(ss_sender) == 32 "Shared secret must be 32 bytes"
@assert length(ss_receiver) == 32 "Recovered secret must be 32 bytes"
@assert length(ciphertext) > 0 "Ciphertext must not be empty"

# Key sizes per spec
@assert pk.level == 512

println("Kyber round-trip: shared secrets match = ", ss_sender == ss_receiver)
println("TASK 4 PASS")
'
```

---

## TASK 5: Implement Dilithium Signature Helper Functions

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/postquantum/dilithium.jl` (lines 211-254)

**Problem:**
Nine helper functions are stubs:

1. `dilithium_expand_matrix` (line 212): `rand(Int, k*n, l*n) .% q` -- non-deterministic,
   wrong dimensions (should be k-by-l of n-polynomials, not a single flat matrix).
2. `dilithium_sample_eta` (line 216): `rand(Int, rows, n) .- eta` -- not CBD, not
   constant-time, wrong distribution.
3. `dilithium_sample_y` (line 220): `rand(Int, l, n) .% gamma1` -- should use
   rejection sampling from `[-gamma1+1, gamma1]`.
4. `dilithium_sample_in_ball` (line 224): Returns `zeros(Int, n)` -- must sample a
   polynomial with exactly `tau` nonzero coefficients in {-1, +1}.
5. `dilithium_high_bits` (line 235): Returns `w` unchanged -- must extract HighBits per spec.
6. `dilithium_low_bits` (line 239): Returns `w` unchanged -- must extract LowBits per spec.
7. `dilithium_make_hint` (line 243): Returns `Bool[]` -- must compute MakeHint per spec.
8. `dilithium_use_hint` (line 248): Returns `w` unchanged -- must apply UseHint per spec.
9. `encode_vector` (line 252): Returns `UInt8[]` -- must serialize polynomial vector.

Additionally, `encode_pk` is called for `DilithiumPublicKey` (lines 108, 180) but only
has a method for `KyberPublicKey` (kyber.jl line 206). This is a `MethodError` crash.

**What to do:**
1. Add an `encode_pk(pk::DilithiumPublicKey)` method in dilithium.jl.
2. Implement all nine helpers per FIPS 204 (ML-DSA) / Dilithium specification.
3. The HighBits/LowBits/MakeHint/UseHint functions are defined in Dilithium spec
   Section 3.1 -- follow those definitions exactly.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto

# Full Dilithium sign/verify round-trip
(pk, sk) = dilithium_keygen(2)
message = Vector{UInt8}("Test message for Dilithium")
signature = dilithium_sign(sk, message)

@assert signature isa DilithiumSignature
@assert length(signature.c_tilde) == 32
@assert length(signature.z) > 0

is_valid = dilithium_verify(pk, message, signature)
@assert is_valid "Valid signature must verify"

# Tampered message must fail
tampered = Vector{UInt8}("Tampered message")
@assert !dilithium_verify(pk, tampered, signature) "Tampered message must not verify"

println("TASK 5 PASS")
'
```

---

## TASK 6: Implement SPHINCS+ Helper Functions

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/postquantum/sphincs.jl` (lines 150-185)

**Problem:**
Six helper functions are stubs returning empty arrays or zeros:

1. `sphincs_compute_root` (line 151): Returns `rand(UInt8, params.n)` -- must compute
   Merkle hypertree root using WOTS+ and tree hashing.
2. `sphincs_parse_digest` (line 156): Returns `(0, 0)` -- must split digest into tree
   index and leaf index per spec.
3. `sphincs_fors_sign` (line 161): Returns `UInt8[]` -- must implement FORS (Forest of
   Random Subsets) signing.
4. `sphincs_fors_verify` (line 167): Returns `UInt8[]` -- must verify FORS and return
   the FORS public key.
5. `sphincs_ht_sign` (line 173): Returns `UInt8[]` -- must implement HyperTree signing
   with WOTS+ chains.
6. `sphincs_ht_verify` (line 180): Returns `UInt8[]` -- must verify HyperTree path
   and return reconstructed root.

**What to do:**
Implement each function per the SPHINCS+ specification (NIST SP 800-208):
1. Implement WOTS+ one-time signature scheme as the base.
2. Implement Merkle tree construction and authentication path generation.
3. Implement FORS few-time signature scheme.
4. Implement the full HyperTree structure.
5. Use hash_blake3 or SHA-256 as the tweakable hash function.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto

# Full SPHINCS+ sign/verify round-trip
(pk, sk) = sphincs_keygen(128, :f)  # Fast variant for testing
message = Vector{UInt8}("Test message for SPHINCS+")

signature = sphincs_sign(sk, message)
@assert length(signature.sig_bytes) > 0 "Signature must not be empty"

is_valid = sphincs_verify(pk, message, signature)
@assert is_valid "Valid signature must verify"

# Tampered message must fail
tampered = Vector{UInt8}("Tampered")
@assert !sphincs_verify(pk, tampered, signature) "Tampered must not verify"

println("TASK 6 PASS")
'
```

---

## TASK 7: Implement Protocol Stubs (Noise, Signal, TLS 1.3)

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/protocols/noise.jl` (entire file -- 15 lines, empty struct)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/protocols/signal.jl` (entire file -- 14 lines, empty struct)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/protocols/tls13.jl` (entire file -- 16 lines, empty struct)

**Problem:**
All three protocol implementations are empty structs with no fields and no methods:
```julia
struct NoiseHandshake end      # noise.jl line 11
struct SignalRatchet end       # signal.jl line 10
struct TLS13Session end        # tls13.jl line 12
```
These types are exported from the main module but have zero functionality. The README
and module docstring list them as features.

**What to do:**
For each protocol, implement at minimum:

**Noise Protocol (noise.jl):**
- `NoiseHandshake` struct with: pattern (XX, IK, NK), static keypair, ephemeral keypair,
  handshake state, cipher state.
- `noise_initiator()` / `noise_responder()` constructors.
- `noise_write_message()` / `noise_read_message()` for handshake messages.
- `noise_split()` to derive transport keys after handshake.
- Use `aead_encrypt`/`aead_decrypt` from ffi_wrappers for the symmetric crypto.

**Signal Protocol (signal.jl):**
- `SignalRatchet` struct with: root key, chain key, message keys, DH ratchet state.
- `signal_init_sender()` / `signal_init_receiver()`.
- `signal_ratchet_encrypt()` / `signal_ratchet_decrypt()`.
- Implement the Double Ratchet algorithm per the Signal spec.

**TLS 1.3 (tls13.jl):**
- `TLS13Session` struct with: handshake state, cipher suite, keys.
- `tls13_client_hello()` / `tls13_server_hello()`.
- Key schedule derivation using HKDF.
- This is educational/reference only (per the file's own docstring).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto

# Noise: basic handshake
hs_init = NoiseHandshake  # Should have constructor with pattern
@assert fieldcount(NoiseHandshake) > 0 "NoiseHandshake must have fields"

# Signal: basic ratchet
@assert fieldcount(SignalRatchet) > 0 "SignalRatchet must have fields"

# TLS 1.3: basic session
@assert fieldcount(TLS13Session) > 0 "TLS13Session must have fields"

println("TASK 7 PASS")
'
```

---

## TASK 8: Implement Zero-Knowledge Proof Systems

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/zkproofs/zksnark.jl` (entire file -- 28 lines)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/zkproofs/zkstark.jl` (entire file -- 14 lines)

**Problem:**
`zk_prove` (line 18) returns `ZKProof(UInt8[], UInt8[])` -- empty proof data.
`zk_verify` (line 23) always returns `false`.
`zkstark.jl` has no code at all beyond a docstring comment.

**What to do:**
1. In `zksnark.jl`:
   - Define a `Circuit` type (R1CS constraint system: matrices A, B, C).
   - Define a `Witness` type (assignment of values satisfying constraints).
   - Implement a simplified Groth16 prover: generate proof elements (A, B, C points).
   - Implement verifier: pairing check e(A,B) = e(alpha,beta) * e(C,delta).
   - At minimum, support boolean circuit verification.

2. In `zkstark.jl`:
   - Define `STARKProof` struct.
   - Implement polynomial commitment via Merkle tree (FRI protocol).
   - Implement `stark_prove()` and `stark_verify()`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto

# zk-SNARK: prove knowledge of x such that x^2 = 9
# (simplified circuit)
proof = zk_prove(nothing, nothing)  # Replace with real circuit/witness
@assert length(proof.proof_data) > 0 "Proof must contain data"

valid = zk_verify(proof, nothing)
@assert typeof(valid) == Bool

println("TASK 8 PASS")
'
```

---

## TASK 9: Implement Shamir Secret Sharing

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/threshold/shamir.jl` (lines 14-23)

**Problem:**
`shamir_split` (line 14) returns `[UInt8[] for _ in 1:num_shares]` -- a list of empty
byte arrays. No polynomial evaluation, no GF(2^8) arithmetic, no secret encoding.

`shamir_reconstruct` (line 19) returns `UInt8[]` -- empty, ignoring all shares.

**What to do:**
1. Implement Shamir's Secret Sharing over GF(256) (or a large prime field):
   - `shamir_split`: Generate random polynomial of degree `threshold-1` with constant
     term = secret byte. Evaluate at `num_shares` distinct points.
   - `shamir_reconstruct`: Use Lagrange interpolation to recover the constant term.
2. Handle multi-byte secrets by splitting each byte independently.
3. Validate: `threshold <= num_shares`, `threshold >= 2`, `num_shares <= 255`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto

secret = Vector{UInt8}("SuperSecretKey!!")
shares = shamir_split(secret, 3, 5)

# Must produce 5 non-empty shares
@assert length(shares) == 5
for s in shares
    @assert length(s) > 0 "Share must not be empty"
end

# Reconstruct from any 3 shares
recovered = shamir_reconstruct(shares[1:3])
@assert recovered == secret "3-of-5 reconstruction must recover secret"

# Reconstruct from different 3 shares
recovered2 = shamir_reconstruct(shares[3:5])
@assert recovered2 == secret "Different 3-of-5 must also work"

# 2 shares must NOT be enough (if implemented with threshold check)
# This depends on implementation -- at minimum verify 3-of-5 works

println("TASK 9 PASS")
'
```

---

## TASK 10: Implement Real Proof-Export Spec Translators

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/verification/proof_export.jl` (lines 228-251)

**Problem:**
All four translation functions return hardcoded example strings regardless of input:

```julia
translate_spec_to_idris_type(spec) = "(a : Nat) -> (b : Nat) -> a + b = b + a"  # line 230
translate_spec_to_lean(spec)       = "forall (a b : N), a + b = b + a"           # line 234
translate_spec_to_coq(spec)        = "forall (a b : nat), a + b = b + a"         # line 238
translate_spec_to_isabelle(spec)   = "\\<forall>a b::nat. a + b = b + a"         # line 242
```

The `spec` argument is completely ignored. Any specification string produces the same
commutativity theorem in the output file.

**What to do:**
1. Define a small specification language or parse a subset of SMT-LIB2 syntax.
2. Translate quantifiers (`forall`, `exists`), arithmetic, equality, and basic types.
3. At minimum, handle:
   - Universal quantification: `forall x : T. P(x)`
   - Equality: `a = b`
   - Implication: `P => Q`
   - Basic types: `Nat`, `Int`, `Bool`, `ByteVector`
4. If full SMT-LIB parsing is too complex, at minimum pass through the spec string
   with appropriate syntax adjustments for each target language instead of ignoring it.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto

# Two different specs must produce different output
spec1 = "forall x : Nat. x + 0 = x"
spec2 = "forall a b : Nat. a * b = b * a"

idris1 = ProvenCrypto.translate_spec_to_idris_type(spec1)
idris2 = ProvenCrypto.translate_spec_to_idris_type(spec2)
@assert idris1 != idris2 "Different specs must produce different translations"

lean1 = ProvenCrypto.translate_spec_to_lean(spec1)
lean2 = ProvenCrypto.translate_spec_to_lean(spec2)
@assert lean1 != lean2 "Different specs must produce different Lean translations"

println("TASK 10 PASS")
'
```

---

## TASK 11: Replace Template Placeholders in ABI/FFI Files

**Files (13 files with `{{PROJECT}}` or `{{project}}` placeholders):**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/abi/Types.idr` (3 occurrences: `{{PROJECT}}.ABI.Types`, etc.)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/abi/Layout.idr` (2 occurrences: `{{PROJECT}}.ABI.Layout`, imports)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/abi/Foreign.idr` (14 occurrences: module name, all `%foreign` declarations)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ffi/zig/build.zig` (6 occurrences: library name, header, benchmark)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ffi/zig/src/main.zig` (19 occurrences: all export functions)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ffi/zig/test/integration_test.zig` (44 occurrences: all extern declarations and test calls)
- Plus occurrences in: `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `ABI-FFI-README.md`, `.github/workflows/quality.yml`, `ci.yml`, `release.yml`

**Problem:**
184 total `{{...}}` placeholder occurrences across 13 files. These are RSR template
markers that should have been replaced when the repo was created from the template. The
Idris modules will not compile, the Zig code will not build, and the workflows reference
a nonexistent project name.

**What to do:**
1. Replace `{{PROJECT}}` with `ProvenCrypto` (capitalized, for module names and titles).
2. Replace `{{project}}` with `provencrypto` (lowercase, for library names and C symbols).
3. Replace `{{OWNER}}` with `hyperpolymath`.
4. Replace `{{FORGE}}` with `github.com`.
5. Replace `{{REPO}}` with `ProvenCrypto.jl`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
count=$(grep -r '{{' --include='*.idr' --include='*.zig' --include='*.yml' --include='*.md' --include='*.adoc' --include='*.res' -l . 2>/dev/null | wc -l)
if [ "$count" -eq 0 ]; then
  echo "TASK 11 PASS: No template placeholders remain"
else
  echo "TASK 11 FAIL: $count files still contain {{ placeholders"
  grep -r '{{' --include='*.idr' --include='*.zig' --include='*.yml' --include='*.md' --include='*.adoc' --include='*.res' -l .
fi
```

---

## TASK 12: Fix SPDX License Headers (AGPL-3.0 Must Be PMPL-1.0-or-later)

**Files with wrong license:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/examples/SafeDOMExample.res` line 1: `AGPL-3.0-or-later`
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ffi/zig/build.zig` line 2: `AGPL-3.0-or-later`
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ffi/zig/src/main.zig` line 6: `AGPL-3.0-or-later`
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ffi/zig/test/integration_test.zig` line 2: `AGPL-3.0-or-later`
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/docs/CITATIONS.adoc` line 13: `license = {AGPL-3.0-or-later}`

**Problem:**
Per the CLAUDE.md license policy, hyperpolymath original code must use
`PMPL-1.0-or-later`. AGPL-3.0 is the old license and must never be used.

**What to do:**
1. Replace all `AGPL-3.0-or-later` SPDX identifiers with `PMPL-1.0-or-later`.
2. Update `docs/CITATIONS.adoc` BibTeX entry from `AGPL-3.0-or-later` to `PMPL-1.0-or-later`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
agpl_count=$(grep -r 'AGPL' --include='*.jl' --include='*.zig' --include='*.idr' --include='*.res' --include='*.adoc' . 2>/dev/null | wc -l)
if [ "$agpl_count" -eq 0 ]; then
  echo "TASK 12 PASS: No AGPL references remain"
else
  echo "TASK 12 FAIL: $agpl_count lines still reference AGPL"
  grep -rn 'AGPL' --include='*.jl' --include='*.zig' --include='*.idr' --include='*.res' --include='*.adoc' .
fi
```

---

## TASK 13: Fix GPU Extension Backends (All Are Fallback-Only Placeholders)

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ext/ProvenCryptoCUDAExt.jl` (lines 12-14)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ext/ProvenCryptoMetalExt.jl` (lines 25-62)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/ext/ProvenCryptoROCmExt.jl` (lines 25-62)

**Problem:**
All GPU backend methods are placeholders that immediately fall back to CPU:

```julia
# MetalExt.jl line 27
@warn "Metal lattice multiplication not yet implemented; falling back to CPU"
return ProvenCrypto.backend_lattice_multiply(ProvenCrypto.CPUBackend(:neon, ...), A, x)
```

Same pattern for `backend_ntt_transform`, `backend_polynomial_multiply`, and
`backend_sampling` in both Metal and ROCm extensions. The CUDA extension (line 13)
has an empty function body:

```julia
function ProvenCrypto.backend_lattice_multiply(backend::ProvenCrypto.CUDABackend, args...)
    # CUDA-specific implementation
end
```

This silently returns `nothing` instead of a matrix, which will crash any caller.

**What to do:**
1. For CUDA: Implement lattice multiplication using `CUDA.jl` `CuArray` operations.
   Use `CUDA.@cuda` kernel for NTT if performance matters, or use cuBLAS for matrix ops.
2. For Metal: Implement using `Metal.jl` MtlArray operations.
3. For ROCm: Implement using `AMDGPU.jl` ROCArray operations.
4. At minimum, each backend_lattice_multiply must return a valid matrix/vector result,
   not `nothing`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
# Verify no empty function bodies in extensions
for ext in ext/ProvenCrypto*.jl; do
  empty=$(grep -c '^\s*end\s*$' "$ext")
  funcs=$(grep -c 'function ProvenCrypto\.' "$ext")
  echo "$ext: $funcs functions, checking for empty bodies..."
  # Check that no function has an immediately-following end with only a comment between
  grep -Pzo 'function[^)]+\)\n\s*#[^\n]*\n\s*end' "$ext" && echo "  WARNING: empty function body found" || echo "  OK"
done
```

---

## TASK 14: Add Missing .machine_readable/ Directory and SCM Files

**Files (all missing):**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/.machine_readable/STATE.scm`
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/.machine_readable/ECOSYSTEM.scm`
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/.machine_readable/META.scm`

**Problem:**
Per CLAUDE.md checkpoint file protocol, every hyperpolymath repo MUST have SCM files in
`.machine_readable/`. This directory does not exist at all. The ROADMAP.adoc is still
the RSR template text ("YOUR Template Repo Roadmap") and the CITATIONS.adoc references
"RSR-template-repo" instead of "ProvenCrypto.jl".

**What to do:**
1. Create `.machine_readable/` directory.
2. Create `STATE.scm` with: metadata, project-context (ProvenCrypto.jl -- post-quantum
   crypto library for Julia), current-position (v0.1.1, mostly stubs), route-to-mvp,
   blockers (all algorithms are placeholders), critical-next-actions.
3. Create `ECOSYSTEM.scm` with: relationship to `proven` (Idris library), `hypatia`
   (CI/CD), `verisimdb` (vulnerability DB).
4. Create `META.scm` with: architecture decisions (pure Julia + FFI), license (PMPL),
   design rationale.
5. Update `ROADMAP.adoc` from template text to actual ProvenCrypto milestones.
6. Update `docs/CITATIONS.adoc` from "RSR-template-repo" to "ProvenCrypto.jl".

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
for f in .machine_readable/STATE.scm .machine_readable/ECOSYSTEM.scm .machine_readable/META.scm; do
  if [ -f "$f" ]; then
    echo "OK: $f exists ($(wc -l < "$f") lines)"
  else
    echo "MISSING: $f"
  fi
done
# Verify ROADMAP is customized
grep -q "ProvenCrypto" ROADMAP.adoc && echo "ROADMAP customized" || echo "ROADMAP still template"
# Verify CITATIONS is customized
grep -q "ProvenCrypto" docs/CITATIONS.adoc && echo "CITATIONS customized" || echo "CITATIONS still template"
```

---

## TASK 15: Fix Changelog Lies and Add Missing Preferences.jl Dependency

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/CHANGELOG.md` (line 22)
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/Project.toml`

**Problem:**
The CHANGELOG v0.1.1 entry (line 22) states:
> "Dependencies: Added Preferences.jl for user-configurable fallback behavior"

But `Preferences` is not listed in `Project.toml` `[deps]` or `[weakdeps]`. It is not
used anywhere in the codebase. This is a false claim.

The README also references a `benchmark/benchmarks.jl` file and `docs/make.jl` file
that do not exist.

**What to do:**
Either:
- (A) Add `Preferences` to `[deps]` in Project.toml and implement the configurable
  fallback behavior the changelog claims exists, OR
- (B) Remove the false claim from the changelog.

Also:
- Create `benchmark/benchmarks.jl` (at least a stub using BenchmarkTools.jl) or remove
  the README reference.
- Create `docs/make.jl` (using Documenter.jl) or remove the README reference.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
# Check Preferences is either used or the claim is removed
if grep -q 'Preferences' CHANGELOG.md; then
  grep -q 'Preferences' Project.toml && echo "OK: Preferences in Project.toml" || echo "FAIL: Changelog claims Preferences but not in Project.toml"
else
  echo "OK: False Preferences claim removed from changelog"
fi
# Check referenced files exist
[ -f benchmark/benchmarks.jl ] && echo "OK: benchmark exists" || echo "MISSING: benchmark/benchmarks.jl (referenced in README)"
[ -f docs/make.jl ] && echo "OK: docs/make.jl exists" || echo "MISSING: docs/make.jl (referenced in README)"
```

---

## TASK 16: Fix Test Suite to Actually Test Crypto Operations

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/test/runtests.jl` (lines 74-111)

**Problem:**
The post-quantum test sections only test struct construction, not actual crypto operations:

```julia
# test/runtests.jl lines 76-85
(pk, sk) = kyber_keygen(512)
@test pk isa KyberPublicKey     # Only tests type, not correctness
@test sk isa KyberSecretKey
@test pk.level == 512
# TODO: Test encapsulation/decapsulation when NTT is implemented
# (ciphertext, ss_sender) = kyber_encapsulate(pk)   <-- COMMENTED OUT
```

The `TODO` comments at lines 82, 95, and 109 indicate the developers knew these tests
were incomplete. Once Tasks 3-6 are complete, the commented-out tests must be enabled
and expanded.

**What to do:**
1. Uncomment all `# TODO` test blocks (lines 82-85, 95-98, 109).
2. Add edge case tests: wrong key sizes, empty messages, corrupted ciphertexts.
3. Add round-trip tests for each algorithm at all security levels.
4. Add test for `shamir_split`/`shamir_reconstruct` (currently no test exists at all).
5. Add tests for `zk_prove`/`zk_verify` (currently no test exists).
6. Add tests for protocol structs once they have real fields (Task 7).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e 'using Pkg; Pkg.test("ProvenCrypto")'
# All tests must pass with zero TODO comments in test output
grep -c 'TODO' test/runtests.jl
# Should be 0
```

---

## TASK 17: Fix `kdf_argon2` Fallback (SHA-256 Is Not PBKDF2)

**Files:**
- `/var/mnt/eclipse/repos/ProvenCrypto.jl/src/primitives/ffi_wrappers.jl` (line 218)

**Problem:**
When libsodium is unavailable, the fallback is:
```julia
return SHA.sha256(vcat(password, salt))  # Placeholder; replace with PBKDF2
```
This is not a KDF. It ignores the `memory_kb`, `iterations`, `parallelism`, and
`key_length` parameters entirely. SHA-256(password || salt) is trivially brute-forceable
and the comment admits it is a placeholder.

Additionally, the function name is `hash_blake3` (line 176) but it actually calls
libsodium's `crypto_generichash` which is BLAKE2b, not BLAKE3. The docstring and
function name are misleading.

**What to do:**
1. Rename `hash_blake3` to `hash_blake2b` or add a real BLAKE3 implementation (the
   BLAKE3 reference implementation exists in Julia via `BLAKE3.jl`).
2. Replace the `kdf_argon2` fallback with actual PBKDF2 using Julia's `OpenSSL_jll` or
   implement PBKDF2-HMAC-SHA256 in pure Julia (it is a simple HMAC iteration).
3. Respect all parameters (`iterations`, `key_length`) in the fallback path.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl
julia --project -e '
using ProvenCrypto

# Test that KDF respects key_length parameter
password = Vector{UInt8}("password")
salt = rand(UInt8, 16)

key32 = kdf_argon2(password, salt; memory_kb=1024, iterations=1, key_length=32)
@assert length(key32) == 32

key64 = kdf_argon2(password, salt; memory_kb=1024, iterations=1, key_length=64)
@assert length(key64) == 64

# Different iterations should produce different keys (if libsodium available)
# or at least different iteration counts in PBKDF2 fallback

println("TASK 17 PASS")
'
```

---

## FINAL VERIFICATION

Run this complete verification to confirm all tasks are done:

```bash
cd /var/mnt/eclipse/repos/ProvenCrypto.jl

echo "=== FINAL VERIFICATION ==="

# 1. Module loads without crash
julia --project -e '
using ProvenCrypto
println("1. Module loads: OK")
println("   Backend: ", ProvenCrypto.HARDWARE_BACKEND[])
'

# 2. No template placeholders
count=$(grep -r "{{" --include="*.jl" --include="*.idr" --include="*.zig" --include="*.yml" . 2>/dev/null | wc -l)
echo "2. Template placeholders remaining: $count (want 0)"

# 3. No AGPL references
agpl=$(grep -r "AGPL" --include="*.jl" --include="*.zig" --include="*.idr" --include="*.res" . 2>/dev/null | wc -l)
echo "3. AGPL references remaining: $agpl (want 0)"

# 4. All declared extensions have files
for ext in ProvenCryptoAMDGPUExt ProvenCryptoCUDAExt ProvenCryptoMetalExt ProvenCryptoOneAPIExt ProvenCryptoSMTExt; do
  [ -f "ext/${ext}.jl" ] && echo "4. Extension $ext: OK" || echo "4. Extension $ext: MISSING"
done

# 5. SCM files exist
for f in .machine_readable/STATE.scm .machine_readable/ECOSYSTEM.scm .machine_readable/META.scm; do
  [ -f "$f" ] && echo "5. $f: OK" || echo "5. $f: MISSING"
done

# 6. Test suite passes
echo "6. Running test suite..."
julia --project -e 'using Pkg; Pkg.test("ProvenCrypto")' 2>&1 | tail -5

# 7. No TODO/Placeholder in helper functions (crypto implementations)
stub_count=$(grep -c "Placeholder\|# TODO:" src/postquantum/*.jl src/protocols/*.jl src/zkproofs/*.jl src/threshold/*.jl 2>/dev/null)
echo "7. Remaining stubs/TODOs in crypto code: $stub_count (want 0)"

# 8. Kyber round-trip
julia --project -e '
using ProvenCrypto
(pk, sk) = kyber_keygen(768)
(ct, ss1) = kyber_encapsulate(pk)
ss2 = kyber_decapsulate(sk, ct)
println("8. Kyber round-trip: ", ss1 == ss2 ? "PASS" : "FAIL")
'

# 9. Dilithium sign/verify
julia --project -e '
using ProvenCrypto
(pk, sk) = dilithium_keygen(2)
msg = Vector{UInt8}("test")
sig = dilithium_sign(sk, msg)
println("9. Dilithium sign/verify: ", dilithium_verify(pk, msg, sig) ? "PASS" : "FAIL")
'

# 10. Shamir round-trip
julia --project -e '
using ProvenCrypto
secret = Vector{UInt8}("secret")
shares = shamir_split(secret, 3, 5)
recovered = shamir_reconstruct(shares[1:3])
println("10. Shamir round-trip: ", recovered == secret ? "PASS" : "FAIL")
'

echo "=== FINAL VERIFICATION COMPLETE ==="
```
