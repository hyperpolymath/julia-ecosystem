<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Causals.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ Statistics │      │   Graphs   │     │
                        │  │ (JuliaPkg) │      │ (JuliaPkg) │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │   Dempster-      │      │ Bradford │ │
                        │  │   Shafer         │      │ Hill     │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Causal DAG     │      │ Granger  │ │
                        │  │   (Graph)        │ ───▶ │ Causality│ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Propensity       │      │ Do       │ │
                        │  │   Score          │      │ Calculus │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Counterfactual   │      │ Unified  │ │
                        │  │   Reasoning      │      │ API      │ │
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
CAUSAL METHODS
  Dempster-Shafer                   ██████████ 100%    Belief functions complete
  Bradford Hill                     ██████████ 100%    9-criterion assessment
  Causal DAG                        ██████████ 100%    d-separation, identification
  Granger Causality                 ██████████ 100%    VAR models, F-tests
  Propensity Score                  ██████████ 100%    Matching, IPW, robust
  Do-Calculus                       ██████████ 100%    Intervention framework
  Counterfactuals                   ██████████ 100%    Necessity/Sufficiency

INFRASTRUCTURE & DOCS
  Unified API                       ██████████ 100%    Integrated interface
  Tests (105 passing)               ██████████ 100%    Core coverage
  Documentation                     ██████████ 100%    10 pages complete
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v0.2.0

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █████████░  ~95%   Core Implementation Complete
```

## Key Dependencies

```
Causal DAG ───────────► Do-Calculus ──────────► Counterfactuals
                            ▲
Propensity Score ───────────┘
                            │
Granger ─────────────► Unified API ◀────────── Bradford Hill
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
