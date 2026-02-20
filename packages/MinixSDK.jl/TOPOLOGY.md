<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# MinixSDK.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          TARGET OS / KERNEL             │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │  MINIX 3   │ ◀──▶ │ Reincarna. │     │
                        │  │ (Microkern)│      │ (Server)   │     │
                        │  └─────▲──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           DEVELOPMENT LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Cross-Compiler   │      │ Service  │ │
                        │  │   (Lowering)     │ ───▶ │ Generator│ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   IPC Wrappers   │      │ Driver   │ │
                        │  │   (C-Headers)    │ ───▶ │ Templates│ │
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
CORE SDK
  Cross-Compiler Stubs              ██░░░░░░░░  20%    Julia to C lowering logic
  Service Boilerplate               ██░░░░░░░░  20%    MINIX 3 driver headers
  IPC Wrappers                      █░░░░░░░░░  10%    Research phase

TARGETS
  Fault-Tolerant Drivers            ░░░░░░░░░░   0%    Planned
  Verified Micro-Services           ░░░░░░░░░░   0%    Planned
  C-Code Generation                 ██░░░░░░░░  20%    Template-based generation

REPO INFRASTRUCTURE
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █░░░░░░░░░  ~15%   Research Prototype
```

## Key Dependencies

```
Julia Logic ──────► Cross-Compiler ──────► MINIX C Service
                                              │
IPC Definitions ───► IPC Wrappers ───────────┤
                                              │
Driver Specs ──────► Boilerplate Gen ────────┘
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
