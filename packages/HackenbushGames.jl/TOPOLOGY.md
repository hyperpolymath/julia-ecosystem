<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# HackenbushGames.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ GraphViz   │ ◀─── │ Combinator │     │
                        │  │  (DOT)     │      │ Game Theory│     │
                        │  └─────▲──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Hackenbush Graph │      │ Valid    │ │
                        │  │  (Structures)    │ ───▶ │  Moves   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Evaluator      │      │ Game     │ │
                        │  │ (Red-Blue/Green) │ ───▶ │ Forms    │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Arithmetic     │      │ Visual   │ │
                        │  │ (Dyadics/Nimbers)│ ◀──▶ │ Helpers  │ │
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
CORE STRUCTURES
  Hackenbush Graph                  ██████████ 100%    Edge/Node grounding complete
  Valid Moves                       ██████████ 100%    Move enumeration working
  Pruning Logic                     ██████████ 100%    Disconnected component removal

EVALUATION ENGINE
  Red-Blue Stalks                   ██████████ 100%    Dyadic rational arithmetic
  Green Grundy Numbers              ██████████ 100%    Impartial game evaluation
  Game Form {L|R}                   ██████████ 100%    Canonical forms & simplification
  Nim-Sum / Mex                     ██████████ 100%    Combinatorial helpers

INFRASTRUCTURE & DOCS
  ASCII Visualization               ██████████ 100%    Terminal-friendly output
  GraphViz Export                   ██████████ 100%    DOT format support
  Tests (35 passing)                ██████████ 100%    Full core coverage
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v1.0.0

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ██████████ 100%    Production Phase (Complete)
```

## Key Dependencies

```
Hackenbush Graph ──────► Move Enumeration ──────► Game Evaluation
                                                     │
Dyadic Arithmetic ──────► Red-Blue Stalks ───────────┤
                                                     │
Nimber Helpers ───────► Green Grundy ──────────────┘
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
