<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Cladistics.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │  Graphs.jl │ ───▶ │  BioJulia  │     │
                        │  │ (Tree Lib) │      │ (Context)  │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Distance Methods │      │ Character│ │
                        │  │ (UPGMA/NJ)       │ ───▶ │ Methods  │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Tree Analysis  │      │ Parsimony│ │
                        │  │   (RF/Rooting)   │ ◀──▶ │ Engine   │ │
                        │  └────────┬─────────┘      └──────────┘ │
                        │           │                             │
                        │  ┌────────▼─────────┐      ┌──────────┐ │
                        │  │  Bootstrap       │      │ Newick   │ │
                        │  │  Support         │ ───▶ │ Export   │ │
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
DISTANCE METHODS
  Distance Metrics                  ██████████ 100%    JC69, K2P, etc.
  UPGMA                             ██████████ 100%    Ultrametric tree construction
  Neighbor-Joining                  ██████████ 100%    Non-clock rate handling

CHARACTER METHODS
  Maximum Parsimony                 ████████░░  80%    Heuristic search in progress
  Fitch Algorithm                   ██████████ 100%    Score calculation working
  Parsimony-Informative Sites       ██████████ 100%    Filtering logic complete

TREE ANALYSIS
  Bootstrap Analysis                ████████░░  80%    Replication logic complete
  Robinson-Foulds Distance          ██████████ 100%    Topology comparison
  Newick Export                     ██████████ 100%    Standard output format

REPO INFRASTRUCTURE
  .machine_readable/ (STATE.scm)    ██████░░░░  60%    Needs repo-specific update
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ███████░░░  ~75%   Functional Alpha
```

## Key Dependencies

```
Distance Metrics ──────► UPGMA/NJ ──────► Tree Analysis
                                               │
Character States ──────► Parsimony ──────► Bootstrap Support
                                               │
                                         Newick Export
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
