# Cliodynamics.jl

*Mathematical modeling and statistical analysis of historical dynamics.*

## What is Cliodynamics?

Cliodynamics is the scientific study of historical dynamics — applying mathematical models and quantitative methods to understand long-term patterns in social complexity, state formation, demographic cycles, elite dynamics, and political instability.

This package implements frameworks from Peter Turchin's cliodynamics research program, providing Julia tools for:

- **Population dynamics** — Malthusian models, demographic-structural theory
- **Elite dynamics** — Overproduction indices, intra-elite competition
- **Political instability** — Stress indicators, conflict intensity, instability probability
- **Secular cycles** — 150-300 year oscillation detection and phase classification
- **State formation** — Capacity models, collective action problems
- **Spatial models** — Multi-region interaction, instability diffusion
- **Model fitting** — Parameter estimation with bootstrap confidence intervals
- **Data integration** — Seshat Global History Databank support

## Installation

```julia
using Pkg
Pkg.add("Cliodynamics")
```

## Quick Start

```julia
using Cliodynamics
using DataFrames

# Model Malthusian population dynamics
params = MalthusianParams(r=0.02, K=1000.0, N0=100.0)
sol = malthusian_model(params, tspan=(0.0, 200.0))
println("Population at t=200: ", round(sol(200.0)[1], digits=1))

# Calculate elite overproduction
data = DataFrame(
    year = 1800:1900,
    population = collect(100_000:1000:200_000),
    elites = [1000 + 10*i + 5*i^1.5 for i in 0:100]
)
eoi = elite_overproduction_index(data)
println("Final EOI: ", round(eoi.eoi[end], digits=3))
```

See the [Tutorial](@ref) for a comprehensive walkthrough.

## References

- Turchin, P. (2003). *Historical Dynamics: Why States Rise and Fall*. Princeton University Press.
- Turchin, P. (2016). *Ages of Discord*. Beresta Books.
- Turchin, P. & Nefedov, S. A. (2009). *Secular Cycles*. Princeton University Press.
- Turchin, P. (2023). *End Times*. Penguin Press.
