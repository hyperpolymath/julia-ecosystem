<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# ProvenCrypto.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / HARDWARE           │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ libsodium  │ ◀──▶ │ BoringSSL  │     │
                        │  │ (Verified) │      │ (Classical)│     │
                        │  └─────▲──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        │  ┌─────┴──────┐      ┌─────┴──────┐     │
                        │  │ GPU / TPU  │      │ Secure     │     │
                        │  │ (Accel.)   │      │ Enclaves   │     │
                        │  └─────▲──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Post-Quantum     │      │ Protocols│ │
                        │  │ (Kyber/Dilith.)  │ ───▶ │ (Noise)  │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Zero-          │      │ Threshold│ │
                        │  │   Knowledge      │ ───▶ │ Crypto   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Formal         │      │ Hardware │ │
                        │  │   Verification   │ ◀──▶ │ Dispatch │ │
                        │  └──────────────────┘      └──────────┘ │
                        └──────────────────────┬──────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────┐
                        │         REPO INFRASTRUCTURE             │
                        │  .machine_readable/ (state)             │
                        │  .github/workflows/ (RSR Gate)          │
                        │  Project.toml                           │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
POST-QUANTUM
  Kyber (KEM)                       ██████████ 100%    NIST PQC compliant
  Dilithium (Signatures)            ██████████ 100%    NIST PQC compliant
  SPHINCS+                          ██████████ 100%    Hash-based signatures

PROTOCOLS & PRIMITIVES
  Noise Protocol Framework          ██████████ 100%    Modern secure channels
  Signal Protocol                   ████████░░  80%    Double Ratchet implemented
  libsodium FFI                     ██████████ 100%    Verified primitives bridge

VERIFICATION & HARDWARE
  Formal Proofs (Idris2/Lean)       ████████░░  80%    Correctness certificates
  Hardware Acceleration             ██████████ 100%    GPU/TPU/NPU detection
  Secure Enclaves                   ██████░░░░  60%    SGX/SEV in progress

REPO INFRASTRUCTURE
  .machine_readable/ (STATE.scm)    ██████░░░░  60%    Needs repo-specific update
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █████████░  ~90%   Feature Complete (Research)
```

## Key Dependencies

```
libsodium FFI ──────► Protocols (Noise) ──────► Hardware Accel
                                                   │
Post-Quantum ───────► Formal Proofs ──────────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
