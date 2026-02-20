<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Hyperpolymath.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          JULIA REPL / USER SPACE        │
                        ├─────────────────────────────────────────┤
                        │                                         │
                        │           ┌──────────────────┐          │
                        │           │ Hyperpolymath.jl │          │
                        │           │  (Metapackage)   │          │
                        │           └────────┬─────────┘          │
                        │                    │                    │
                        └────────────────────┼────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │          THE ECOSYSTEM BUNDLE           │
                        ├────────────────────┬────────────────────┤
                        │  ┌─────────────┐   │   ┌─────────────┐  │
                        │  │ Formalism / │   │   │  Historical │  │
                        │  │ Logic Layer │   │   │  Dynamics   │  │
                        │  └─────┬───────┘   │   └─────┬───────┘  │
                        │        │           │         │          │
                        │  ┌─────▼───────┐   │   ┌─────▼───────┐  │
                        │  │ Journalism /│   │   │  Social /   │  │
                        │  │  Forensics  │   │   │  Political  │  │
                        │  └─────┬───────┘   │   └─────┬───────┘  │
                        │        │           │         │          │
                        │  ┌─────▼───────┐   │   ┌─────▼───────┐  │
                        │  │  Education /│   │   │ Orchestrat. │  │
                        │  │   Gaming    │   │   │  & Bridges  │  │
                        │  └─────┬───────┘   │   └─────┬───────┘  │
                        │        │           │         │          │
                        └────────┴───────────┴─────────┴──────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │          HARDWARE / LOW-LEVEL           │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ LowLevel.jl│ ───▶ │ SiliconCore│     │
                        │  └────────────┘      └────────────┘     │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
ECOSYSTEM LAYERS
  Formalism (Axiom/SMT/ZeroProb)    ██████████ 100%    Core logic bundle complete
  Historical (Cliodyn/Metrics)      ██████████ 100%    Research-grade modules
  Social (TradeUnion/PR/Exnovate)   ██████████ 100%    Operational toolsets
  Orchestration (PostDisc/Macro)    ██████████ 100%    Glue logic re-exported

INTEGRATION
  Re-export logic                   ██████████ 100%    using Hyperpolymath works
  Project.toml dependencies         ██████████ 100%    All 29 packages linked

REPO INFRASTRUCTURE
  .github/workflows/ (CI)           ██████████ 100%    Metapackage health check

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ██████████ 100%    Metapackage Complete
```

## Key Dependencies

```
SiliconCore ──────► LowLevel ──────► Disciplinary Modules
                                            │
PostDisciplinary ◀──────────────────────────┘
      │
Hyperpolymath.jl ──────► User REPL ──────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
