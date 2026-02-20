<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# ZeroProb.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │Distributns.│ ───▶ │   Plots    │     │
                        │  │ (Library)  │      │ (Visualiz.)│     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ ZeroProb Types   │      │ Density  │ │
                        │  │ (Core System)    │ ───▶ │  Ratios  │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Hausdorff        │      │ Epsilon  │ │
                        │  │ Measures         │ ◀──▶ │ Neighb.  │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Paradox          │      │ Black    │ │
                        │  │ Demonstrations   │ ───▶ │ Swans    │ │
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
CORE MEASURES
  Density Ratio                     ██████████ 100%    Continuous relevance scoring
  Hausdorff Measure                 ██████████ 100%    Fractional dimension support
  Epsilon-Neighborhood              ██████████ 100%    Convergence rate estimation

PEDAGOGY & PARADOXES
  Classical Paradoxes               ██████████ 100%    Cantor, Bertrand, Banach-Tarski
  Continuum Paradox                 ██████████ 100%    Core demonstration logic
  Visual Paradoxes                  ██████████ 100%    Pedagogical plots

APPLICATIONS
  Black Swan Events                 ██████████ 100%    High-impact tail risk
  Insurance/Quantum Events          ██████████ 100%    Specialized event types
  Market Crash Models               ██████████ 100%    Financial edge cases

REPO INFRASTRUCTURE
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v0.2.0
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █████████░  ~95%   Stable Mathematical Toolkit
```

## Key Dependencies

```
ZeroProb Types ──────► Density Ratios ──────► Paradox Demonstrations
                                                   │
Hausdorff Measure ───► Epsilon-Neighborhood ───────┤
                                                   │
Black Swans ─────────► Application Models ────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
