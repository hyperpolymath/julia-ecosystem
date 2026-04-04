<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk> -->

# TOPOLOGY.md — julia-ecosystem

## Purpose

Central monorepo unifying 20+ hyperpolymath Julia libraries into cohesive post-disciplinary research and verified computing ecosystem. Spans formal logic, cryptography, historical dynamics, labor organizing, and domain-specific applications.

## Module Map

```
julia-ecosystem/
├── logic/
│   ├── Axiom.jl               # Formal logic and set theory
│   ├── ProvenCrypto.jl        # Cryptographic verification
│   └── ... (logic libraries)
├── dynamics/
│   ├── Cliodynamics.jl        # Historical dynamics modeling
│   ├── Cliometrics.jl         # Historical measurement
│   └── ... (dynamics libraries)
├── organizing/
│   ├── TradeUnionist.jl       # Labor organizing toolkit
│   └── ... (organizing tools)
├── application/
│   └── ... (domain-specific apps)
├── README.adoc                # Ecosystem overview
└── Project.toml               # Monorepo manifest
```

## Data Flow

```
[Research Question] ──► [Pick Library] ──► [Computation] ──► [Verified Results]
```

## Key Invariants

- All libraries share common Julia ecosystem infrastructure
- Cross-library dependencies managed via shared Project.toml
- Unified testing, CI/CD, and documentation
- Post-disciplinary focus: bridges multiple domains
