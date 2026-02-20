<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# HardwareResilience.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          HARDWARE KERNELS               │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ CPU Kernel │ ◀──▶ │ GPU Kernel │     │
                        │  │ (Executed) │      │ (Executed) │     │
                        │  └─────┬──────┘      └─────┬──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           RESILIENCE LAYER              │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Kernel Guardian  │      │ Error    │ │
                        │  │   (Monitor)      │ ───▶ │ Handler  │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Self-Healing   │ ───▶ │ Fallback │ │
                        │  │    Logic         │      │ Dispatch │ │
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
CORE RESILIENCE
  Kernel Guardian                   ██░░░░░░░░  20%    Struct defined
  Monitor Logic                     █░░░░░░░░░  10%    Basic try-catch stub
  Self-Healing                      █░░░░░░░░░  10%    Placeholder prints

REPO INFRASTRUCTURE
  Project.toml                      ░░░░░░░░░░   0%    Missing (Folder only)

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █░░░░░░░░░  ~10%   Initial Stub
```

## Key Dependencies

```
Kernel Execution ──────► Kernel Guardian ──────► Self-Healing
                                                   │
Error Detection ───────► Error Handling ───────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
