<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# SiliconCore.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          HARDWARE / BARE METAL          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ CPUID/SMX  │ ───▶ │ Multi-Arch │     │
                        │  │ (Registers)│      │ Intrinsics │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           CORE LAYER                    │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Arch Detection   │      │ ASM      │ │
                        │  │   (Logic)        │ ───▶ │ Kernels  │ │
                        │  └──────────────────┘      └──────────┘ │
                        └──────────────────────┬──────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────┐
                        │         REPO INFRASTRUCTURE             │
                        │  Project.toml                           │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
CORE PRIMITIVES
  Arch Detection                    ██░░░░░░░░  20%    Sys.ARCH wrapper
  ASM Kernels                       █░░░░░░░░░  10%    Vector add placeholder

REPO INFRASTRUCTURE
  Project.toml                      ██████████ 100%    Metadata complete

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █░░░░░░░░░  ~10%   Initial Foundation
```

## Key Dependencies

```
Hardware Features ───► Arch Detection ──────► LowLevel.jl
                                               │
ASM Intrinsics ──────► ASM Kernels ────────────┘
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
