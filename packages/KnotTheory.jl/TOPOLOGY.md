<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# KnotTheory.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │  SnapPy    │ ───▶ │ KnotAtlas  │     │
                        │  │ (Visualiz.)│      │ (Database) │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Planar Diagram   │      │  Knot    │ │
                        │  │ (Structures)     │ ───▶ │  Table   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Polynomial     │      │ Classical│ │
                        │  │   Invariants     │ ◀──▶ │Invariants│ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Seifert        │      │ Simplifi-│ │
                        │  │   Theory         │ ───▶ │ cation   │ │
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
  Planar Diagram (PD Code)          ██████████ 100%    Oriented crossing model
  DT/Dowker Codes                   ██████████ 100%    Conversion logic complete
  Knot Table                        ██████████ 100%    15 standard knots (up to 7_7)

INVARIANTS
  Polynomial (Jones/Alex/etc)       ██████████ 100%    Skein relations & Fox calculus
  Classical (Writhe/Link/etc)       ██████████ 100%    Basic properties implemented
  Seifert Theory                    ██████████ 100%    Matrix & circles decomposition

TRANSFORMS & INTEROP
  Reidemeister Simplification       ██████████ 100%    R1, R2, R3 stability check
  Braid Word (TANGLE)               ██████████ 100%    Planar <-> Braid conversion
  JSON I/O                          ██████████ 100%    Serialization support

INFRASTRUCTURE
  Tests (285 passing)                ██████████ 100%    Comprehensive coverage
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v0.2.0

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █████████░  ~95%   Stable Implementation
```

## Key Dependencies

```
Planar Diagram ──────► Seifert Theory ──────► Polynomial Invariants
                                                   │
Knot Table ──────────► Search & Lookup ────────────┤
                                                   │
Braid Words ─────────► PD Code ────────────► Simplification
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
