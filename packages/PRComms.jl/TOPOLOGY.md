<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# PRComms.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / CHANNELS           │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ News Outlets │ ◀──▶ │ Social Media│     │
                        │  │ (Journalists)│      │ (Audiences) │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │   Messaging      │      │ Newsroom │ │
                        │  │   (Pillars)      │ ───▶ │ Workflow │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Campaign       │      │ Crisis   │ │
                        │  │   Coordination   │ ◀──▶ │ Mode     │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Analytics      │      │ Strategy │ │
                        │  │   (Sentiment)    │ ───▶ │ Planning │ │
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
CORE OPERATIONS
  Message Architecture              ██████████ 100%    Pillars & proof points
  Newsroom Workflow                 ████████░░  80%    Draft/Review/Embargo logic
  Media Contact Management          ██████░░░░  60%    Database schema implemented

RESPONSE & PLANNING
  Crisis Mode Playbooks             ████████░░  80%    Escalation trees functional
  Campaign Coordination             ██████░░░░  60%    Asset tracking in progress
  Strategy Planning                 ████████░░  80%    Multi-channel templates

INSIGHTS & INTEROP
  Analytics & Surveys               ██████░░░░  60%    Sentiment scoring basics
  Boundary Objects                  ████████░░  80%    Stakeholder interfaces
  .machine_readable/ (STATE.a2ml)   ██████████ 100%    Updated

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ███████░░░  ~70%   Operational Beta
```

## Key Dependencies

```
Message Pillars ──────► Newsroom Workflow ──────► Publication
                                                   │
Strategy Planning ──────► Campaign Coord ────────┤
                                                   │
Crisis Playbooks ───────► Risk Gating ───────────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
