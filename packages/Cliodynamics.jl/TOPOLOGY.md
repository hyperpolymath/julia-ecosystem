<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Cliodynamics.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │   Seshat   │ ───▶ │  Turing.jl │     │
                        │  │ (Databank) │      │ (Bayesian) │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │ Population       │      │ Elite    │ │
                        │  │ Dynamics (DST)   │ ───▶ │ Dynamics │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │ Political Stress │      │ Secular  │ │
                        │  │  Indicator (PSI) │ ───▶ │ Cycles   │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │  Spatial Models  │      │ Fitting  │ │
                        │  │  (Diffusion)     │ ◀──▶ │ Engine   │ │
                        │  └──────────────────┘      └──────────┘ │
                        └──────────────────────┬──────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────┐
                        │         REPO INFRASTRUCTURE             │
                        │  .machine_readable/ (state)             │
                        │  .github/workflows/ (RSR Gate)          │
                        │  Project.toml, Manifest.toml            │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
CORE MODELS
  Population Dynamics (DST)         ██████████ 100%    Malthusian & DST complete
  Elite Dynamics (EOI)              ██████████ 100%    Overproduction index
  Political Stress (PSI)            ██████████ 100%    Composite stress indicators
  Secular Cycles                    ██████████ 100%    Phase detection & analysis

EXTENSIONS & DATA
  Spatial Models                    ██████████ 100%    Diffusion & territorial competition
  Model Fitting                     ██████████ 100%    Optim.jl parameter estimation
  Bayesian Inference                ██████████ 100%    Turing.jl integration
  Seshat Integration                ██████████ 100%    Historical data loading

INFRASTRUCTURE
  Plots.jl Recipes                  ██████████ 100%    5 visualization types
  Documentation (Documenter.jl)     ██████████ 100%    10 pages complete
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v1.0.0
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            ██████████ 100%    Production Release (v1.0.0)
```

## Key Dependencies

```
Population Dynamics ───► Elite Dynamics ───► Political Stress
                                                 │
Secular Cycles ◀─────────────────────────────────┘
      │
Spatial Models ──────► Fitting Engine ──────► Bayesian Inference
                                                 │
                                         COMPLETE (v1.0.0)
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
