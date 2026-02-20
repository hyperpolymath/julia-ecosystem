<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# PostDisciplinary.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │  Cliodyn.  │ ───▶ │  Axiology  │     │
                        │  │ (History)  │      │  (Values)  │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        │  ┌─────▼──────┐      ┌─────┴──────┐     │
                        │  │ Investigative     │   Axiom    │     │
                        │  │ (Journalism) ───▶ │  (Proofs)  │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Research Project │      │Universal │ │
                        │  │  (Orchestrator)  │ ───▶ │ Ontology │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Cross-Theory     │      │ Synthesis│ │
                        │  │  Verification    │ ───▶ │ Engine   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Meta-Analysis    │      │ Impact   │ │
                        │  │   (Logic)        │ ◀──▶ │ Tracking │ │
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
CORE ORCHESTRATION
  Research Project Orchestrator     ████████░░  80%    Project structures functional
  Universal Ontology                ██████░░░░  60%    Initial mappings defined
  Cross-Theory Verification         ████░░░░░░  40%    Axiom bridge in progress

SYNTHESIS & ANALYSIS
  Synthesis Engine                  ████████░░  80%    Report generation logic
  Meta-Analysis Logic               ██████░░░░  60%    Statistical aggregation
  Consensus Layer                   ████░░░░░░  40%    Multi-prover agreement

DOMAIN MODULES
  Memetics                          ██████░░░░  60%    Idea spread tracking
  Methodology Frameworks            ████████░░  80%    Standard research patterns
  Impact Tracking                   ██████░░░░  60%    Value-at-risk indicators

INFRASTRUCTURE
  Verisim Storage Bridge            ██████░░░░  60%    Persistence logic
  .machine_readable/ (STATE.a2ml)   ██████████ 100%    Updated

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ██████░░░░  ~60%   Initial Orchestration Layer
```

## Key Dependencies

```
Universal Ontology ──────► Research Project ──────► Cross-Theory Verification
                                                     │
Synthesis Engine ◀──────── Meta-Analysis ◀──────────┘
      │
Impact Tracking ──────► Synthesis Report ─────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
