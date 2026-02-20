# Model Fitting

## Malthusian Fitting

Recover growth rate and carrying capacity from observed population data:

```@docs
fit_malthusian
```

## Demographic-Structural Fitting

Fit the full DST model to historical time series with population, elite, and state capacity data:

```@docs
fit_demographic_structural
```

## Generic Parameter Estimation

Fit arbitrary models with bootstrap confidence intervals:

```@docs
estimate_parameters
```

### Example

```julia
using Cliodynamics

# Define a custom growth model
model_fn(p, t) = p[1] .* exp.(p[2] .* (t .- t[1]))

# Observed data
years = collect(0.0:10.0:100.0)
observed = 100.0 .* exp.(0.02 .* years) .+ randn(length(years)) .* 5.0

# Estimate with confidence intervals
result = estimate_parameters(model_fn, observed, years, [80.0, 0.01], n_bootstrap=200)

println("A = $(round(result.params[1], digits=1)) [$(round(result.ci_lower[1], digits=1)), $(round(result.ci_upper[1], digits=1))]")
println("r = $(round(result.params[2], digits=5)) [$(round(result.ci_lower[2], digits=5)), $(round(result.ci_upper[2], digits=5))]")
```
