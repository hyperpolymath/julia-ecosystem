<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# MacroPower.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / TRIGGERS           │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ File System│      │    Time    │     │
                        │  │ (Watcher)  │      │ (Scheduler)│     │
                        │  └─────┬──────┘      └─────┬──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │   @workflow      │      │ Trigger  │ │
                        │  │     Macro        │ ───▶ │ Registry │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Workflow       │      │  Action  │ │
                        │  │    Engine        │ ───▶ │ Execution│ │
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
CORE ENGINE
  @workflow Macro                   ██░░░░░░░░  20%    Initial stub implementation
  Workflow Engine                   ██░░░░░░░░  20%    Basic loop defined
  Trigger/Action Types              ██████████ 100%    Structs defined

MODULES
  Time Triggers                     ░░░░░░░░░░   0%    Planned
  File System Triggers              ░░░░░░░░░░   0%    Planned
  Action Registry                   ░░░░░░░░░░   0%    Planned

REPO INFRASTRUCTURE
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ██░░░░░░░░  ~20%   Initial Scaffold
```

## Key Dependencies

```
Macro Parser ──────► Workflow Model ──────► Trigger Monitor
                                               │
Execution Engine ◀────── Action Dispatch ◀─────┘
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
