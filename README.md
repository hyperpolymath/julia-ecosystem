<!--
SPDX-License-Identifier: CC-BY-SA-4.0
SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
-->

[![OpenSSF Best Practices](https://img.shields.io/badge/OpenSSF-Best_Practices-green?logo=opensourcesecurity)](https://www.bestpractices.dev/en/projects/new?repo_url=https://github.com/hyperpolymath/julia-ecosystem)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-blue.svg)](https://www.mozilla.org/MPL/2.0/)
<embed
src="https://api.thegreenwebfoundation.org/greencheckimage/github.com"
data-link="https://www.thegreenwebfoundation.org/green-web-check/?url=github.com" />
image:<a href="https://img.shields.io/badge/Julia-1.10+-9558B2?logo=julia"
data-link="https://julialang.org/">Julia</a>

**A unified framework for post-disciplinary research, organizing, and
verified computing.**

<div id="toc">

</div>

# Overview

This is the central monorepo for the **Hyperpolymath Julia Ecosystem**.
It consolidates 20+ specialized libraries into a single, cohesive
research and development environment. From formal logic and cryptography
to historical dynamics and labor organizing, this ecosystem provides the
"Post-Disciplinary Glue" to tackle complex global challenges.

# Repository Map

## 🧠 Logic & Verification

- <a href="packages/Axiom.jl" class="jl">Axiom</a>: Provably correct
  machine learning and formal verification.

- <a href="packages/SMTLib.jl" class="jl">SMTLib</a>: Julia interface
  for SMT solvers (Z3, CVC5).

- <a href="packages/PolyglotFormalisms.jl"
  class="jl">PolyglotFormalisms</a>: Formally verified cross-language
  common library.

- <a href="packages/ZeroProb.jl" class="jl">ZeroProb</a>: Reasoning
  about measure-zero events and black swans.

## 🛡️ Security & Forensics

- <a href="packages/ProvenCrypto.jl" class="jl">ProvenCrypto</a>:
  Formally verified PQC and cryptographic protocols.

- <a href="packages/InvestigativeJournalism.jl"
  class="jl">InvestigativeJournalism</a>: High-intelligence forensics
  and secure evidence lockers.

## 🏛️ History & Social Science

- <a href="packages/Cliodynamics.jl" class="jl">Cliodynamics</a>:
  Mathematical modeling of historical dynamics (DST).

- <a href="packages/Cliometrics.jl" class="jl">Cliometrics</a>:
  Quantitative economic history and convergence analysis.

- <a href="packages/Axiology.jl" class="jl">Axiology</a>: Formal value
  theory and ethical alignment for ML.

- <a href="packages/ViableSystems.jl" class="jl">ViableSystems</a>:
  Organizational cybernetics (VSM) and Soft Systems Methodology (SSM).

## ✊ Organizing & Action

- <a href="packages/TradeUnionism.jl" class="jl">TradeUnionism</a>:
  Data-driven labor organizing and spatial power mapping.

- <a href="packages/PRComms.jl" class="jl">PRComms</a>: High-integrity
  strategic communications and crisis management.

- <a href="packages/Exnovation.jl" class="jl">Exnovation</a>: Systematic
  phase-out of legacy practices and structures.

- <a href="packages/BowtieRisk.jl" class="jl">BowtieRisk</a>: Structured
  hazard analysis and barrier modeling.

## 🎨 Mathematics & Play

- <a href="packages/JuliaKids.jl" class="jl">JuliaKids</a>: Joyful
  visual coding for children with Minecraft/KSP interop.

- <a href="packages/KnotTheory.jl" class="jl">KnotTheory</a>:
  Computational knot theory and invariants.

- <a href="packages/Skein.jl" class="jl">Skein</a>: Persistence layer
  for knot-theoretic data.

- <a href="packages/HackenbushGames.jl" class="jl">HackenbushGames</a>:
  Combinatorial game theory toolkit.

- <a href="packages/Cladistics.jl" class="jl">Cladistics</a>:
  Phylogenetic analysis and evolutionary relationships.

## ⚙️ Orchestration & Meta

- <a href="packages/PostDisciplinary.jl" class="jl">PostDisciplinary</a>:
  The universal graph linking all disciplinary modules.

- <a href="packages/JuliaPackage-Reuse-Audit.jl"
  class="jl">JuliaPackageSpitter</a>: Automated scaffolding for new
  ecosystem libraries.

- <a href="packages/MacroPower.jl" class="jl">MacroPower</a>: Low-code
  automation and workflow engine.

- <a href="packages/ShellIntegration.jl" class="jl">ShellIntegration</a>:
  Unified shell interface (PowerShell & Valence).

- <a href="packages/MinixSDK.jl" class="jl">MinixSDK</a>: Research
  foundation for Julia-to-MINIX 3 microkernel development.

- <a href="packages/SoftwareSovereign.jl" class="jl">SoftwareSovereign</a>:
  Universal software policy engine and license-aware discovery.

## 🔌 The Metal Layer (Subdivided LowLevel)

- <a href="packages/LowLevel.jl" class="jl">LowLevel</a>: The
  meta-orchestrator for high-integrity hardware control.

- <a href="packages/SiliconCore.jl" class="jl">SiliconCore</a>:
  Multi-arch Assembly, CPUID, and manual memory arenas.

- <a href="packages/AcceleratorGate.jl" class="jl">AcceleratorGate</a>:
  GPU, NPU, and TPU driver dispatch.

- <a href="packages/QuantumCircuit.jl" class="jl">QuantumCircuit</a>:
  QPU abstraction and reversible computing.

- <a href="packages/HardwareResilience.jl"
  class="jl">HardwareResilience</a>: Self-healing, diagnostics, and
  fault-recovery.

- <a href="packages/FirmwareAudit.jl" class="jl">FirmwareAudit</a>:
  BIOS, UEFI, ACPI, and RAID telemetry.

# Development in the Monorepo

To work on a specific package within this repo:

```bash
# Example: Testing PRComms.jl
cd packages/PRComms.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Registration

When registering these packages, use the `subdir` argument:
`@JuliaRegistrator` `register` `subdirectory=packages/PackageName.jl`

# License

All packages in this ecosystem are licensed under the
**Palimpsest-MPL-1.0 License** (MPL-2.0). See the LICENSE file in the
root and in each package subdirectory for details.

------------------------------------------------------------------------

*Synthesis is the ultimate discipline.*
