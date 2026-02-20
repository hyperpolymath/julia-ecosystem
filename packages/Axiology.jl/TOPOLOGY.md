<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Axiology.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │    MLJ     │ ───▶ │  ECHIDNA   │     │
                        │  │ (Framework)│      │  (Provers) │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Value Hierarchy  │      │  Satisfy │ │
                        │  │  (Types)         │ ───▶ │  Check   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Optimization     │      │ Verification│
                        │  │   (Maximize)     │ ◀──▶ │  Engine     │
                        │  └────────┬─────────┘      └──────────┘ │
                        │           │                             │
                        │  ┌────────▼─────────┐      ┌──────────┐ │
                        │  │  Pareto          │      │ Proof    │ │
                        │  │  Frontier        │ ───▶ │ Handling │ │
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
CORE LOGIC
  Specification                     ██████████ 100%    Complete
  Type Definitions                  ██████████ 100%    5 concrete types defined
  Julia Implementation              ███████░░░  70%    Core functions working

INTEGRATION
  ML Integration (MLJ/Flux)         ░░░░░░░░░░   0%    Not started
  Formal Verification (ECHIDNA)     █░░░░░░░░░   5%    Stub exists

REPO INFRASTRUCTURE
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ██████░░░░  ~65%   Core value system functional
```

## Key Dependencies

```
Type Definitions ──────► Satisfy Check ──────► Optimization (Maximize)
                                                   │
                                         ┌─────────┴─────────┐
                                         ▼                   ▼
                                 ML Integration      Formal Verification
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
