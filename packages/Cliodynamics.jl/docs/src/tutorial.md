# Tutorial

This tutorial walks through the core features of Cliodynamics.jl, from basic population models to full historical analysis pipelines.

## Population Dynamics

The simplest model is Malthusian logistic growth:

```julia
using Cliodynamics

params = MalthusianParams(r=0.02, K=1000.0, N0=100.0)
sol = malthusian_model(params, tspan=(0.0, 200.0))

# Interpolate at any time point
println("t=0:   ", round(sol(0.0)[1], digits=1))
println("t=100: ", round(sol(100.0)[1], digits=1))
println("t=200: ", round(sol(200.0)[1], digits=1))
```

## Demographic-Structural Theory

The DST model couples population, elites, and state capacity in a system of ODEs:

```julia
params = DemographicStructuralParams(
    r=0.015, K=1000.0, w=2.0, δ=0.03, ε=0.001,
    N0=500.0, E0=10.0, S0=100.0
)
sol = demographic_structural_model(params, tspan=(0.0, 300.0))

# Three state variables: Population, Elites, State capacity
state = sol(150.0)
println("N=$(round(state[1],digits=1)), E=$(round(state[2],digits=1)), S=$(round(state[3],digits=1))")
```

## Elite Overproduction

Compute when elite numbers outpace available positions:

```julia
using DataFrames

data = DataFrame(
    year = 1800:1900,
    population = collect(100_000:1000:200_000),
    elites = [1000 + 10*i + 5*i^1.5 for i in 0:100]
)

eoi = elite_overproduction_index(data)
# eoi.eoi contains the index values (positive = overproduction)
```

## Political Stress Indicator

The PSI combines three destabilizing forces:

```julia
stress_data = DataFrame(
    year = 1800:1900,
    real_wages = 100.0 .- collect(0:100).^1.2 ./ 10,
    elite_ratio = 0.01 .+ collect(0:100) ./ 5000,
    state_revenue = 1000.0 .- collect(0:100).^1.5 ./ 5
)

psi = political_stress_indicator(stress_data)
# psi.psi = composite, psi.mmp, psi.emp, psi.sfd = components
```

## Secular Cycle Analysis

Detect long-term oscillations and classify phases:

```julia
# Detect cycles in time series data
timeseries = 100.0 .+ 50.0 .* sin.(2π .* (1:300) ./ 100) .+ 2 .* randn(300)
analysis = secular_cycle_analysis(Float64.(timeseries), window=30)
println("Period: ", analysis.period, " years")
println("Amplitude: ", round(analysis.amplitude, digits=2))
```

## Seshat Data Integration

Load and analyze historical data from the Seshat Global History Databank:

```julia
raw = load_seshat_csv("data/seshat_sample.csv")
roman = prepare_seshat_data(raw, polity="RomPrinworlds")

# Compute EOI across all Roman periods
roman_all = filter(row -> occursin("Rom", string(row.polity)), prepare_seshat_data(raw))
sort!(roman_all, :year)
eoi = elite_overproduction_index(roman_all)
```

## Model Fitting

Recover parameters from observed data:

```julia
# Fit Malthusian model
years = collect(0.0:10.0:100.0)
population = [50.0 * exp(0.03 * t) for t in years]
result = fit_malthusian(years, population, r_init=0.01, K_init=600.0)
println("Fitted r=$(round(result.params.r, digits=4))")

# Generic estimation with confidence intervals
model_fn(p, t) = p[1] .* exp.(p[2] .* (t .- t[1]))
est = estimate_parameters(model_fn, population, years, [50.0, 0.02], n_bootstrap=200)
println("95% CI for r: [$(round(est.ci_lower[2], digits=5)), $(round(est.ci_upper[2], digits=5))]")
```

## Plotting

When Plots.jl is loaded, plot recipes activate automatically:

```julia
using Plots

# PSI with component breakdown
plot(psi, Val(:psi))

# Elite overproduction
plot(eoi, Val(:eoi))

# Secular cycle decomposition
plot(analysis, Val(:secular_cycle))
```
