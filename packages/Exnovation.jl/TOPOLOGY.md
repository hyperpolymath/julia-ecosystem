<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Exnovation.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ DataFrames │ ───▶ │   Plots    │     │
                        │  │ (Analysis) │      │ (Visualiz.)│     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Exnovation Item  │      │ Assessment│ │
                        │  │  (Structures)    │ ───▶ │ Engine   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Intelligent    │      │ Decision │ │
                        │  │     Failure      │ ───▶ │ Pipeline │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Debiasing      │      │ Portfolio│ │
                        │  │    Actions       │ ◀──▶ │ Scoring  │ │
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
CORE MODELS
  Exnovation Item                   ██████████ 100%    Legacy system structures
  Drivers & Barriers                ██████████ 100%    Full Enum support (incl. Political)
  Assessment Engine                 ██████████ 100%    Scoring & recommendation logic

PIPELINE & STRATEGY
  Intelligent Failure               ██████████ 100%    Readiness scoring
  Decision Pipeline                 ██████████ 100%    Stage-gates implemented
  Portfolio Scoring                 ██████████ 100%    Budget allocation logic
  Debiasing Actions                 ██████████ 100%    Cognitive/Structural prompts

INFRASTRUCTURE
  API Documentation                 ██████████ 100%    Documenter.jl complete
  Tests (32 passing)                ██████████ 100%    Full RSR compliance
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v1.0.0

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ██████████ 100%    Production Phase (Complete)
```

## Key Dependencies

```
Exnovation Item ──────► Assessment Engine ──────► Decision Pipeline
                                                     │
Intelligent Failure ─────────────────────────────────┘
                                                     │
Debiasing Actions ◀───── Portfolio Scoring ◀─────────┘
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
