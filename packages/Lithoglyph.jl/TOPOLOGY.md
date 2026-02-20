<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Lithoglyph.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          LITHOGLYPH FEDERATION          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ Remote Node│ ◀──▶ │ Local Core │     │
                        │  │ (HTTP/API) │      │ (Zig/Forth)│     │
                        │  └─────▲──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           CLIENT LAYER                  │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ LithoglyphClient │      │ FFI      │ │
                        │  │   (Logic)        │ ───▶ │ Bridge   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Symbolic       │      │ Search   │ │
                        │  │   Storage        │ ◀──▶ │ Engine   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Glyph Model    │      │ Metadata │ │
                        │  │   (Provenance)   │ ───▶ │ Manager  │ │
                        │  └──────────────────┘      └──────────┘ │
                        └──────────────────────┬──────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────┐
                        │         REPO INFRASTRUCTURE             │
                        │  .github/workflows/ (RSR Gate)          │
                        │  Project.toml                           │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
CORE CLIENT
  Lithoglyph Client                 ██████████ 100%    Struct and connection logic
  Glyph Model                       ██████████ 100%    Provenance & hashing fields
  FFI Bridge (Zig/Forth)            ██░░░░░░░░  20%    DL handle placeholder

FEATURES
  Register Glyph                    ████░░░░░░  40%    Stub println, needs POST
  Federated Search                  ████░░░░░░  40%    Mock results, needs query
  Axiomatic Provenance              ░░░░░░░░░░   0%    Idris2 link planned

REPO INFRASTRUCTURE
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ████░░░░░░  ~40%   Initial Client Interface
```

## Key Dependencies

```
Lithoglyph Core ─────► FFI Bridge ──────────► Normalization
                                                 │
Client Config ───────► Lithoglyph Client ──────┤
                                                 │
Glyph Schema ────────► Register / Search ─────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
