# Causals.jl

Comprehensive causal inference toolkit for Julia.

## Overview

Causals.jl unifies multiple approaches to causal reasoning:

- **Dempster-Shafer theory** - Combine uncertain expert opinions
- **Bradford Hill criteria** - Assess causality in observational studies
- **Causal DAGs** - Graphical models and identification
- **Granger causality** - Time series causal analysis
- **Propensity scores** - Observational study adjustment
- **Do-calculus** - Pearl's intervention framework
- **Counterfactuals** - "What if" reasoning

## Why Causals.jl?

Existing Julia causal packages are fragmented. Causals.jl provides:

✓ Complete coverage of major causal methods
✓ Production-quality implementations
✓ Comprehensive documentation
✓ Active maintenance

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hyperpolymath/Causals.jl")
```

## Quick Start

### Combining Expert Evidence

```julia
using Causals

# Two experts provide evidence about hypotheses
frame = [:A, :B, :C]

expert1 = MassAssignment(frame, Dict(
    Set([:A]) => 0.6,
    Set([:A, :B, :C]) => 0.4  # uncertainty
))

expert2 = MassAssignment(frame, Dict(
    Set([:A, :B]) => 0.7,
    Set([:A, :B, :C]) => 0.3
))

# Combine using Dempster's rule
combined = combine_dempster(expert1, expert2)

# Get belief interval
lower, upper = uncertainty(combined, Set([:A]))
```

### Granger Causality

```julia
# Test if X Granger-causes Y
causes, F_stat, p_value, lag = granger_test(x_series, y_series)

if causes
    println("X Granger-causes Y with lag $lag")
end
```

### Causal DAG Analysis

```julia
# Build causal graph
g = CausalGraph([:X, :Y, :Z])
add_edge!(g, :Z, :X)  # Z → X
add_edge!(g, :Z, :Y)  # Z → Y
add_edge!(g, :X, :Y)  # X → Y

# Check if Z blocks backdoor path
@assert backdoor_criterion(g, :X, :Y, Set([:Z]))
```

## Modules

```@contents
Pages = ["dempster_shafer.md", "bradford_hill.md", "causal_dag.md",
         "granger.md", "propensity.md", "do_calculus.md", "counterfactuals.md"]
Depth = 1
```

## License

PMPL-1.0-or-later (MPL-2.0 compatible)
