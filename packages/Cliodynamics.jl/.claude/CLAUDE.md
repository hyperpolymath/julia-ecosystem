# Cliodynamics.jl - Project Instructions

## Overview

Cliodynamics.jl is a Julia package for quantitative modeling of historical dynamics, implementing Peter Turchin's cliodynamic theories including demographic-structural theory, elite overproduction indices, secular cycle analysis, spatial models, and Bayesian inference.

## Project Structure

- **src/Cliodynamics.jl** - Single-file module implementation (deliberate architectural decision — ADR-002)
- **ext/CliodynamicsPlotsExt.jl** - Plots.jl recipe extension (5 plot types)
- **ext/CliodynamicsTuringExt.jl** - Turing.jl Bayesian inference extension
- **test/runtests.jl** - 124 tests across 18 testsets
- **examples/** - Usage examples (basic_usage.jl, historical_analysis.jl, publication_examples.jl)
- **docs/** - Documenter.jl documentation (10 pages)
- **data/seshat_sample.csv** - Synthetic Seshat-format historical data
- **ffi/zig/** - Zig FFI implementation following Idris2 ABI
- **src/abi/** - Idris2 ABI definitions with formal proofs

## Building and Testing

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests (124 tests)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run examples
julia --project=. examples/basic_usage.jl
julia --project=. examples/historical_analysis.jl
julia --project=. examples/publication_examples.jl

# Build documentation
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Code Style

### Julia Conventions
- Follow official Julia style guide
- Use lowercase with underscores for function names
- Use CamelCase for types
- Document all exported functions with docstrings
- Include `@testset` for all test groups
- Use `Base.@kwdef` for structs with keyword constructors

### Mathematical Functions
- ODE models return DifferentialEquations.jl solutions (interpolate via `sol(t)[i]`)
- Parameters follow academic literature conventions
- Use keyword arguments for optional parameters
- Validate inputs at function boundaries

## Architecture

### Single-File Design (ADR-002)
The core implementation is in `src/Cliodynamics.jl`. Optional features use Julia package extensions:
- `CliodynamicsPlotsExt` (loaded with `using Plots`) — plot recipes
- `CliodynamicsTuringExt` (loaded with `using Turing`) — Bayesian inference

### 34 Exported Functions
- **Models**: malthusian_model, demographic_structural_model, state_capacity_model, collective_action_problem
- **Indicators**: elite_overproduction_index, political_stress_indicator, instability_probability, conflict_intensity, crisis_threshold, instability_events, population_pressure, carrying_capacity_estimate
- **Cycles**: secular_cycle_analysis, detect_cycle_phases
- **Fitting**: fit_malthusian, fit_demographic_structural, estimate_parameters
- **Data**: load_seshat_csv, prepare_seshat_data
- **Spatial**: spatial_instability_diffusion, territorial_competition_model, frontier_formation_index
- **Bayesian** (extension): bayesian_malthusian, bayesian_dst, bayesian_model_comparison
- **Utilities**: moving_average, detrend, normalize_timeseries
- **Types**: MalthusianParams, DemographicStructuralParams, StateCapacityParams, SecularCyclePhase, InstabilityEvent

## Dependencies

### Core (required)
- Julia 1.10+
- DifferentialEquations.jl 7 — ODE solvers
- DataFrames.jl 1 — tabular data
- Optim.jl 1 — parameter estimation
- LinearAlgebra, Statistics (stdlib)

### Optional (weak dependencies)
- RecipesBase.jl 1 — plot recipes (via Plots.jl)
- Turing.jl 0.30-0.35 — Bayesian inference
- Distributions.jl 0.25 — probability distributions
- MCMCChains.jl 6 — MCMC chain analysis

## Testing

124 tests across 18 testsets covering:
- All mathematical models (correctness, edge cases)
- Spatial models (diffusion, competition, frontier)
- Model fitting and parameter estimation
- Seshat data loading and preparation
- Utility functions

## Common Workflows

### Adding a New Model
1. Add function to `src/Cliodynamics.jl`
2. Add `export` in module header
3. Add comprehensive docstring
4. Add `@testset` in `test/runtests.jl`
5. Add `@docs` block in relevant `docs/src/*.md`
6. Update STATE.scm
7. Run tests to verify

### Updating Documentation
- Update docstrings in source
- Update `docs/src/*.md` pages
- Update STATE.scm for completion tracking
- Update examples/ if API changes
- Rebuild docs: `julia --project=docs docs/make.jl`

## License

PMPL-1.0-or-later (Palimpsest License)

All files must have SPDX header:
```julia
# SPDX-License-Identifier: PMPL-1.0-or-later
```

## Notes for AI Agents

- This is a **Julia package**, not Rust/Elixir/ReScript
- The single-file design is intentional (don't suggest splitting)
- Mathematical models follow academic literature (cite Turchin if modifying)
- Test coverage is critical — all exported functions must have tests
- Examples should work standalone without Plots dependency
- Bayesian features require `using Turing` (package extension)
- Spatial models are in core (no extra dependencies beyond DifferentialEquations)
