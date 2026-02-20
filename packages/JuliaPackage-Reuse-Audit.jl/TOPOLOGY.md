<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# JuliaPackageSpitter.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │ RSR Templ. │ ───▶ │ CI Actions │     │
                        │  │ (Base)     │      │ (Marketpl.)│     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Package Generat. │      │ Template │ │
                        │  │   (Logic)        │ ───▶ │ Processor│ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   CI Profile     │      │ ABI/FFI  │ │
                        │  │   (Strictness)   │ ───▶ │ Scaffolder│ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │   LLM Handoff    │      │ Registry │ │
                        │  │   (Tasks Gen)    │ ───▶ │ Integrat.│ │
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
SCAFFOLDING ENGINE
  Package Generator                 ██████████ 100%    Directory structure creation
  Template Processor                ██████████ 100%    Place-holder replacement logic
  CI Profile Selection              ██████████ 100%    Minimal/Standard/Strict

FEATURES
  LLM Handoff (SONNET-TASKS)        ██████████ 100%    Automated task briefing
  ABI/FFI Scaffolding               ████████░░  80%    Idris2/Zig baseline templates
  Smart Templating                  █████████░  90%    Boilerplate anchors

INFRASTRUCTURE
  Generator Templates               ██████████ 100%    Comprehensive ecosystem base
  Tests (12 passing)                ████████░░  80%    Integration tests verified
  .machine_readable/ (STATE.a2ml)   ██████████ 100%    Updated

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █████████░  ~90%   Feature Complete
```

## Key Dependencies

```
RSR Templates ──────► Template Processor ──────► Package Generator
                                                   │
CI Profiles ─────────► CI Scaffolder ────────────┤
                                                   │
LLM Briefing ────────► SONNET-TASKS ───────────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
