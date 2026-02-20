<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# SMTLib.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │     Z3     │      │    cvc5    │     │
                        │  │ (Optimizer)│      │  (Solver)  │     │
                        │  └─────▲──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        │  ┌─────┴──────┐      ┌─────┴──────┐     │
                        │  │    Yices   │      │   MathSAT  │     │
                        │  │  (Solver)  │      │  (Solver)  │     │
                        │  └─────▲──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │    @smt Macro    │      │ Solver   │ │
                        │  │      (DSL)       │ ───▶ │Discovery │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Expression     │      │ SMT-LIB2 │ │
                        │  │   Generation     │ ───▶ │ Generator│ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ S-Expression     │      │ Result & │ │
                        │  │    Parser        │ ◀─── │ Model    │ │
                        │  └────────┬─────────┘      └──────────┘ │
                        │           │                             │
                        │  ┌────────▼─────────┐      ┌──────────┐ │
                        │  │   Theory         │      │ Increment│ │
                        │  │   Helpers        │      │ Context  │ │
                        │  └──────────────────┘      └──────────┘ │
                        └──────────────────────┬──────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────┐
                        │         REPO INFRASTRUCTURE             │
                        │  .machine_readable/ (state)             │
                        │  .github/workflows/ (RSR Gate)          │
                        │  scripts/ (readiness)                   │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
CORE LOGIC
  Solver Discovery                  ██████████ 100%    z3, cvc5, yices detected
  SMT-LIB2 Generation               ██████████ 100%    Full theory support
  S-Expression Parser               ██████████ 100%    Recursive from_smtlib parser

FEATURES
  @smt Macro                        ██████████ 100%    Concise constraint building
  Quantifiers (forall/exists)       ██████████ 100%    First-order reasoning
  Optimization (maximize/minimize)  ██████████ 100%    Z3 νZ integration
  Incremental Solving (push/pop)    ██████████ 100%    Stack tracking implemented
  Theory Helpers                    ██████████ 100%    BV, FP, Array, Regex sorts

REPO INFRASTRUCTURE
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v0.1.0
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance
  Readiness Scripts                 ██████████ 100%    readiness-check.sh implemented

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █████████░  ~93%   Stable, near production
```

## Key Dependencies

```
Solver Discovery ──────► Context Management ──────► Expr Generation
                                                        │
Parser (from_smtlib) ◀───── Result Handling ◀──────────┘
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
