# Julia General Registry Submissions
# SPDX-License-Identifier: PMPL-1.0-or-later
#
# For each package, you'll need:
#   - repo: the Git URL (HTTPS for registry)
#   - subdir: path from repo root to the package directory
#   - name, uuid, version: from Project.toml
#   - description: for the registry listing
#
# Common fields for all packages:
#   repo  = "https://github.com/hyperpolymath/julia-ecosystem"
#   subdir = "packages/<DirName>"

## Registration command template (via LocalRegistry or Registrator):
##
##   using LocalRegistry
##   register(
##       "packages/<DirName>",
##       registry = "path/to/General",
##       repo = "https://github.com/hyperpolymath/julia-ecosystem",
##       subdir = "packages/<DirName>"
##   )
##
## Or via JuliaRegistrator bot comment on a GitHub commit:
##   @JuliaRegistrator register subdir=packages/<DirName>

---

## 1. AcceleratorGate.jl
- **name**: AcceleratorGate
- **uuid**: 59f742c7-270d-4a76-95d4-a853ae16cc71
- **version**: 0.1.0
- **subdir**: packages/AcceleratorGate.jl
- **description**: Shared coprocessor dispatch infrastructure for Julia packages. Provides a unified backend type hierarchy (GPU, TPU, NPU, FPGA, QPU, DSP, VPU, PPU, Math, Crypto) with automatic detection, platform-aware selection, resource-aware memory tracking, and self-healing fallback hooks.

## 2. Axiology.jl
- **name**: Axiology
- **uuid**: 868b87ec-ec5d-47a0-ab5f-c8a8ecbd97bd
- **version**: 0.1.0
- **subdir**: packages/Axiology.jl
- **description**: Value theory integration for machine learning models. Provides frameworks for embedding ethical constraints, preference orderings, and axiological assessments into model training and evaluation pipelines.

## 3. Axiom.jl
- **name**: Axiom
- **uuid**: bbd403f8-dcc5-405a-84eb-8de9d358675c
- **version**: 0.2.0
- **subdir**: packages/Axiom.jl
- **description**: Provably correct machine learning framework. Bridges formal verification and ML with property-based testing, SMT-backed invariant checking, and coprocessor-accelerated inference across GPU, TPU, NPU, FPGA, QPU, and other backends via AcceleratorGate.

## 4. BowtieRisk.jl
- **name**: BowtieRisk
- **uuid**: f4857c2c-646e-44f8-8901-46c9ace85fa6
- **version**: 0.1.0
- **subdir**: packages/BowtieRisk.jl
- **description**: Bow-tie risk analysis with barrier assessment and Monte Carlo simulation. Models hazards, threats, top events, consequences, and barriers (preventive/mitigative) with support for escalation factors, barrier degradation, and dependency handling.

## 5. Causals.jl
- **name**: Causals
- **uuid**: c4a8b6d2-f9e3-4c1a-b8d7-9f2e3c4d5e6f
- **version**: 0.1.0
- **subdir**: packages/Causals.jl
- **description**: Causal inference and Applied Information Economics (AIE). Implements do-calculus, Bayesian network inference, propensity score estimation, and Hubbard's Value of Information framework with coprocessor-accelerated backends for large-scale causal discovery.

## 6. Cladistics.jl
- **name**: Cladistics
- **uuid**: e3663be0-4771-44aa-b6d0-43b3a6d82e58
- **version**: 0.1.0
- **subdir**: packages/Cladistics.jl
- **description**: Phylogenetic analysis and cladistics. Implements parsimony-based tree reconstruction, character matrix operations, maximum parsimony scoring, tree rearrangement (SPR/TBR/NNI), consensus methods, and GPU-accelerated likelihood computation for large datasets.

## 7. Cliodynamics.jl
- **name**: Cliodynamics
- **uuid**: 8d2f3e70-4c6b-5e9c-a3d1-2f8e9c0b1d2e
- **version**: 1.0.0
- **subdir**: packages/Cliodynamics.jl
- **description**: Mathematical modeling and statistical analysis of historical dynamics. Implements Peter Turchin's cliodynamics research program: demographic-structural theory, secular cycles, elite overproduction, Political Stress Indicator (PSI), and state breakdown prediction using differential equation models.

## 8. Cliometrics.jl
- **name**: Cliometrics
- **uuid**: 6c8e9c60-3b5a-4d8b-9f2a-1e7f8a9b0c1d
- **version**: 0.1.0
- **subdir**: packages/Cliometrics.jl
- **description**: Quantitative economic history analysis. Applies economic theory and econometric methods to historical data: GDP reconstruction, price series deflation, demographic transition modeling, trade flow analysis, and institutional quality metrics with coprocessor-accelerated computation.

## 9. Exnovation.jl
- **name**: Exnovation
- **uuid**: eb535ea2-a284-4d5f-a499-9e884733dc08
- **version**: 0.1.0
- **subdir**: packages/Exnovation.jl
- **description**: Systematic phase-out and discontinuation planning for legacy practices, products, and technologies. Provides scoring matrices, impact assessment, stakeholder analysis, transition pathway generation, and monitoring dashboards for managed exnovation processes.

## 10. FirmwareAudit.jl
- **name**: FirmwareAudit
- **uuid**: e6a7b8c9-d0e1-4f2a-ab3c-4d5e6f7a8b9c
- **version**: 0.1.0
- **subdir**: packages/FirmwareAudit.jl
- **description**: Firmware image auditing and vulnerability scanning. Performs entropy analysis, string extraction, header format identification (ELF, PE, Mach-O, U-Boot, Intel HEX), hash verification, and known-CVE matching against an embedded vendor vulnerability database.

## 11. HackenbushGames.jl
- **name**: HackenbushGames
- **uuid**: 01ec8bc2-77c0-4797-a5fe-db76a6b99454
- **version**: 0.1.0
- **subdir**: packages/HackenbushGames.jl
- **description**: Combinatorial game theory implementation for Hackenbush games. Supports Red-Blue, Green, and multi-color Hackenbush with surreal number evaluation, game addition, canonical form reduction, and GPU/coprocessor-accelerated game tree search.

## 12. HardwareResilience.jl
- **name**: HardwareResilience
- **uuid**: d5f6a7b8-c9d0-4e1f-9a2b-3c4d5e6f7a8b
- **version**: 0.1.0
- **subdir**: packages/HardwareResilience.jl
- **description**: Hardware resilience detection and monitoring for Linux systems. Detects ECC memory, RAID arrays, thermal zones, watchdog timers, and redundant power supplies, producing a comprehensive resilience assessment with a supervised execution guardian for safety-critical workloads.

## 13. Hyperpolymath.jl
- **name**: Hyperpolymath
- **uuid**: a0b1c2d3-e4f5-6a7b-8c9d-0e1f2a3b4c5d
- **version**: 0.1.0
- **subdir**: packages/Hyperpolymath.jl
- **description**: Meta-package aggregating the hyperpolymath Julia ecosystem. Imports and re-exports all domain packages spanning logic/verification, security/forensics, history/social science, organising/action, mathematics/play, orchestration/meta, and the metal layer.

## 14. InvestigativeJournalist.jl
- **name**: InvestigativeJournalist
- **uuid**: 379a7c0a-3675-4b32-b948-cb4760c6e442
- **version**: 0.1.0
- **subdir**: packages/InvestigativeJournalist.jl
- **description**: Digital forensics and investigative analysis toolkit. Provides evidence chain management, claim tracking, source credibility scoring, timeline reconstruction, network analysis of actors, and structured output for investigative reporting workflows.

## 15. JuliaForChildren (JuliaKids.jl)
- **name**: JuliaForChildren
- **uuid**: c1c96f90-3ae4-433d-aa60-06e66792fdf1
- **version**: 0.1.0
- **subdir**: packages/JuliaKids.jl
- **description**: Educational Julia programming toolkit for children aged 7-14. Provides simplified interfaces for turtle graphics, Minecraft modding, KSP mission planning, game development, robotics, and collaborative coding with accessibility-first design and screen reader support.

## 16. JuliaPackageSpitter (JuliaPackage-Reuse-Audit.jl)
- **name**: JuliaPackageSpitter
- **uuid**: 772df90b-d426-497b-8682-0a765d4f8c0b
- **version**: 0.1.0
- **subdir**: packages/JuliaPackage-Reuse-Audit.jl
- **description**: Automated Julia package scaffolding and reuse auditing. Generates compliant package structures from configurable PackageSpec templates and audits existing packages for code reuse opportunities across the ecosystem.

## 17. KnotTheory.jl
- **name**: KnotTheory
- **uuid**: 215268c9-7579-426e-8b7c-a3dc27acd339
- **version**: 0.1.0
- **subdir**: packages/KnotTheory.jl
- **description**: Mathematical knot theory library implementing planar diagram representations, polynomial invariants (Jones, Alexander, HOMFLY-PT, Kauffman bracket), Reidemeister move simplification, braid word conversion, and Seifert circle computation with a built-in knot table.

## 18. Lithoglyph.jl
- **name**: Lithoglyph
- **uuid**: f1e2d3c4-b5a6-4b7c-8d9e-0f1a2b3c4d5e
- **version**: 0.1.0
- **subdir**: packages/Lithoglyph.jl
- **description**: Julia bindings for the LithoGlyph database engine. Provides a client for registering and searching glyphs (symbolic data with tags and provenance) in the federated LithoGlyph store, plus an FFI bridge to the core Zig/Forth normaliser.

## 19. LowLevel.jl
- **name**: LowLevel
- **uuid**: a1b2c3d4-e5f6-4a1b-8c2d-3e4f5a6b7c8d
- **version**: 0.1.0
- **subdir**: packages/LowLevel.jl
- **description**: Low-level system introspection and hardware detection for Julia. Provides CPU architecture detection (x86_64, ARM, RISC-V, MIPS, PowerPC), SIMD capability probing, cache hierarchy analysis, and platform-specific feature flags.

## 20. MacroPower.jl
- **name**: MacroPower
- **uuid**: b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e
- **version**: 0.1.0
- **subdir**: packages/MacroPower.jl
- **description**: Macroeconomic power analysis and modelling through trigger-action automation workflows. Define workflows with conditional triggers and executable actions using the @workflow macro, then run them with run_workflow for policy simulation and scenario analysis.

## 21. MinixSDK.jl
- **name**: MinixSDK
- **uuid**: d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a
- **version**: 0.1.0
- **subdir**: packages/MinixSDK.jl
- **description**: Research SDK for targeting MINIX 3 from Julia. Provides cross-compilation scaffolding, microkernel service generation, IPC message passing primitives, and driver skeleton templates for exploring MINIX's message-based architecture from Julia.

## 22. PolyglotFormalisms.jl
- **name**: PolyglotFormalisms
- **uuid**: 8fd979ee-625c-447d-87f1-33af4d789de5
- **version**: 1.1.0
- **subdir**: packages/PolyglotFormalisms.jl
- **description**: Cross-language formal methods library implementing the aLib Common Library specification. Provides arithmetic, logic, set theory, and algebraic operations with formal proofs and verification certificates exportable to Idris, Lean, Coq, and Isabelle.

## 23. PostDisciplinary.jl
- **name**: PostDisciplinary
- **uuid**: f1a9a0dc-9df1-4c08-8f01-4f9031796370
- **version**: 0.1.0
- **subdir**: packages/PostDisciplinary.jl
- **description**: Post-disciplinary research integration framework. Connects insights across disciplines using knowledge graphs, memetic evolution models, boundary objects, and VeriSimDB-backed provenance tracking for transdisciplinary research projects.

## 24. PRComms.jl
- **name**: PRComms
- **uuid**: 2dde2a48-bffb-456d-9be4-0a16c25066d3
- **version**: 0.1.0
- **subdir**: packages/PRComms.jl
- **description**: Public relations and communications management toolkit. Provides release lifecycle management, stakeholder mapping, message framing analysis, media outlet targeting, boundary objects for cross-team alignment, and campaign effectiveness tracking.

## 25. ProvenCrypto.jl
- **name**: ProvenCrypto
- **uuid**: 33678010-b125-405f-b046-d17447b3c4c1
- **version**: 0.1.0
- **subdir**: packages/ProvenCrypto.jl
- **description**: Formally verified cryptographic protocols and post-quantum primitives. Implements Kyber KEM, Dilithium/SPHINCS+ signatures, ZK-SNARKs, Shamir secret sharing, Noise protocol, Signal ratchet, and TLS 1.3 with proof export to Idris 2, Lean 4, Coq, and Isabelle/HOL.

## 26. QuantumCircuit.jl
- **name**: QuantumCircuit
- **uuid**: b3d4e5f6-a7b8-4c9d-ae0f-1a2b3c4d5e6f
- **version**: 0.1.0
- **subdir**: packages/QuantumCircuit.jl
- **description**: Quantum circuit simulation and gate-level computation. Provides qubit registers, standard gates (Hadamard, Pauli, CNOT, Toffoli, phase, T), measurement, Bell state preparation, circuit composition, and coprocessor-accelerated state vector simulation via AcceleratorGate.

## 27. ShellIntegration.jl
- **name**: ShellIntegration
- **uuid**: c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f
- **version**: 0.1.0
- **subdir**: packages/ShellIntegration.jl
- **description**: Capability-restricted shell execution from Julia. Provides sandboxed command execution with configurable allow/deny lists, timeout enforcement, output capture, and audit logging for safe system interaction from Julia workflows.

## 28. SiliconCore.jl
- **name**: SiliconCore
- **uuid**: c4e5f6a7-b8c9-4d0e-8f1a-2b3c4d5e6f7a
- **version**: 0.1.0
- **subdir**: packages/SiliconCore.jl
- **description**: Cross-platform CPU feature detection and hardware capability analysis. Probes Linux, macOS, Windows, and BSD systems for SIMD instruction sets (SSE through AVX-512, NEON, SVE2, RVV), cache hierarchy, core topology, and platform classification across x86_64, aarch64, and RISC-V.

## 29. Skein.jl
- **name**: Skein
- **uuid**: e8a1f3d0-7c42-4e9a-b5d1-3a7f8c2e1d0b
- **version**: 0.1.0
- **subdir**: packages/Skein.jl
- **description**: Skein relation computation and knot polynomial evaluation. Implements skein module algebra, Kauffman bracket via skein relations, Jones polynomial computation, and bulk import/export for KnotInfo-style datasets with GPU-accelerated polynomial arithmetic.

## 30. SMTLib.jl
- **name**: SMTLib
- **uuid**: 7d3f9a2c-8b4e-5c1f-a6d0-9e8f7b2c3d4e
- **version**: 0.1.0
- **subdir**: packages/SMTLib.jl
- **description**: Lightweight Julia interface to SMT solvers (Z3, CVC5) via SMT-LIB2 format. Provides a complete pipeline from Julia expressions to SMT-LIB2 scripts, solver invocation, model parsing, and unsatisfiable core extraction with GPU-accelerated batch solving.

## 31. SoftwareSovereign.jl
- **name**: SoftwareSovereign
- **uuid**: cc72cefe-1ea1-4255-93cd-1af1078aa475
- **version**: 0.1.0
- **subdir**: packages/SoftwareSovereign.jl
- **description**: Software sovereignty and supply chain analysis. Provides dependency auditing, license compliance checking, SBOM generation, provenance verification, and sovereignty scoring for assessing digital autonomy and reducing vendor lock-in risk.

## 32. TradeUnionist.jl
- **name**: TradeUnionist
- **uuid**: 28827ff7-c05d-49d1-8ea0-4ff47f2d6875
- **version**: 0.1.0
- **subdir**: packages/TradeUnionist.jl
- **description**: Trade union organising and collective bargaining toolkit. Provides membership management, cost proposal modelling, geospatial branch mapping (haversine distance), campaign branding, ballot management, and collective agreement tracking.

## 33. ViableSystems.jl
- **name**: ViableSystems
- **uuid**: a6c07668-559f-42e4-88f9-b00ef4c02498
- **version**: 0.1.0
- **subdir**: packages/ViableSystems.jl
- **description**: Viable System Model (VSM) implementation based on Stafford Beer's cybernetics framework. Models Systems 1-5 (operations, coordination, control, intelligence, policy), recursive structure, variety management, and boundary objects for organisational diagnosis.

## 34. ZeroProb.jl
- **name**: ZeroProb
- **uuid**: f9e8c2e0-8b4a-4d5f-9a3c-1e2d3c4b5a6f
- **version**: 0.1.0
- **subdir**: packages/ZeroProb.jl
- **description**: Zero-probability event handling and black swan analysis. Provides frameworks for reasoning about measure-zero events in finance, risk management, betting systems, and scientific edge cases where standard probability models break down.
