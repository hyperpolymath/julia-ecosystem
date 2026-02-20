<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# JuliaForChildren.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / PLATFORMS          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ Minecraft  │ ───▶ │    KSP     │     │
                        │  │ (Integration)     │(Integration)     │
                        │  └─────┬──────┘      └─────┬──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Guided Lessons   │      │ Coding   │ │
                        │  │ (Curriculum)     │ ───▶ │ Sandbox  │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   Drawing & Art  │      │ Game     │ │
                        │  │   (CairoMakie)   │ ◀──▶ │ Engine   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Achievement      │      │ Robot    │ │
                        │  │ Tracking         │ ───▶ │ Control  │ │
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
CORE LEARNING
  Guided Lessons                    ████████░░  80%    Curriculum basics implemented
  Coding Sandbox                    ██████████ 100%    Safe environment functional
  Achievement Tracking              ██████░░░░  60%    Badges & streaks in progress

PLAYFUL MODULES
  Drawing & Art (Luxor/Makie)       ██████████ 100%    Visual feedback engine
  Robot Control                     ████████░░  80%    Basic commands implemented
  Game Engine                       ████████░░  80%    Gamebolt integration
  Minecraft/KSP Integration         ██████░░░░  60%    External bridges functional

INFRASTRUCTURE
  Instructor Mode                   ████░░░░░░  40%    Dashboard planned
  LLM Buddy                         ██████░░░░  60%    AI-assisted learning
  .machine_readable/ (STATE.a2ml)   ██████████ 100%    Updated

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ███████░░░  ~70%   Playable Alpha
```

## Key Dependencies

```
Curriculum Design ──────► Guided Lessons ──────► Achievement Tracking
                                                   │
Visual Engine ──────► Coding Sandbox ───────────┤
                                                   │
External Bridges ──────► Game/Robot Integration ──┘
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
