<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Cliometrics.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │  Maddison  │ ───▶ │   Polity   │     │
                        │  │ (Datasets) │      │ (Instit.)  │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Historical Data  │      │ Growth   │ │
                        │  │  (Cleaning)      │ ───▶ │Accounting│ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Convergence    │      │ Causal   │ │
                        │  │    Analysis      │ ───▶ │Inference │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Institutional  │      │ Result   │ │
                        │  │     Analysis     │ ◀──▶ │ Analyzer │ │
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
CORE ANALYSIS
  Historical Data Cleaning          ██████████ 100%    Cleaning & interpolation
  Growth Accounting                 ██████████ 100%    Solow residual, decomposition
  Convergence Analysis              ██████████ 100%    Beta-convergence complete
  Institutional Analysis            ██████████ 100%    Quality indices & change

INFERENCE & TOOLS
  Causal Inference                  ██████████ 100%    DiD & Counterfactuals
  Data Loading                      ██████████ 100%    Maddison & Penn compatible
  Sigma-convergence                 ░░░░░░░░░░   0%    Planned v0.2.0

REPO INFRASTRUCTURE
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v0.1.0
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance
  ABI/FFI Standards                 █████████░  90%    Implementation baseline

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ████████░░  ~85%   Beta Phase (Refining Docs)
```

## Key Dependencies

```
Historical Data ──────► Growth Accounting ──────► Convergence Analysis
                                                      │
Institutional Analysis ──────► Causal Inference ◀─────┘
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
