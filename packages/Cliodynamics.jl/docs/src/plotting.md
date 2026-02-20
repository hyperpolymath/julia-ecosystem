# Plotting

Cliodynamics.jl provides plot recipes via a package extension that loads automatically when `Plots.jl` (or `RecipesBase`) is available.

## Usage

```julia
using Cliodynamics
using Plots  # Triggers extension loading
```

## Available Recipes

### Political Stress Indicator

```julia
psi_result = political_stress_indicator(data)
plot(psi_result, Val(:psi))
```

Shows PSI composite line with MMP, EMP, and SFD component breakdown.

### Elite Overproduction Index

```julia
eoi_result = elite_overproduction_index(data)
plot(eoi_result, Val(:eoi))
```

Shows EOI with zero baseline and filled area.

### Secular Cycle Decomposition

```julia
analysis = secular_cycle_analysis(timeseries, window=30)
plot(analysis, Val(:secular_cycle))
```

Two-panel layout showing trend and cycle components.

### Cycle Phase Timeline

```julia
phases = detect_cycle_phases(data)
plot(phases, Val(:phases))
```

Scatter plot with phases color-coded: green (Expansion), yellow (Stagflation), red (Crisis), blue (Depression).

### Conflict Intensity

```julia
intensity = conflict_intensity(events, window=10)
plot(intensity, Val(:conflict))
```

Filled area plot of conflict intensity over time.
