<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# TradeUnionist.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / STAKEHOLDERS       │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │  Employers │ ◀──▶ │   Members  │     │
                        │  │(Workplaces)│      │ (Engagement)     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │   Workplace      │      │ Member   │ │
                        │  │   Mapping        │ ───▶ │Engagement│ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Grievance      │      │ Bargaining│ │
                        │  │   Pipeline       │ ◀──▶ │ Support  │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Mobilization   │      │ Metrics  │ │
                        │  │   Planning       │ ───▶ │ & Analyt.│ │
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
CORE ORGANIZING
  Workplace Mapping                 ██████████ 100%    Site & Dept structures
  Member Engagement                 ████████░░  80%    Conversation lifecycle
  Organizing Metrics                ██████░░░░  60%    Engagement heatmaps

LABOR RELATIONS
  Grievance Pipeline                ████████░░  80%    Intake & evidence logging
  Bargaining Support                ████████░░  80%    Clause costing & comparison
  Contract Enforcement              ██████░░░░  60%    Deadline tracking

STRATEGY & PLANNING
  Mobilization Planning             ████████░░  80%    Action templates
  Events Management                 ██████░░░░  60%    Logistics tracking
  Comms Bridge                      ██████░░░░  60%    Outreach automation

INFRASTRUCTURE
  Verisim Storage Bridge            ██████░░░░  60%    Secure persistence
  .machine_readable/ (STATE.a2ml)   ██████████ 100%    Updated

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ███████░░░  ~70%   Functional Prototype
```

## Key Dependencies

```
Workplace Mapping ──────► Member Engagement ──────► Mobilization
                                                   │
Grievance Pipeline ──────► Contract Knowledge ─────┤
                                                   │
Bargaining Support ──────► Strategy Planning ─────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
