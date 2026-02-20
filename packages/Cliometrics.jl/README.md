# Cliometrics.jl

[![Project Topology](https://img.shields.io/badge/Project-Topology-9558B2)](TOPOLOGY.md)
[![Completion Status](https://img.shields.io/badge/Completion-85%25-green)](TOPOLOGY.md)

[![License](https://img.shields.io/badge/license-PMPL--1.0--or--later-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/julia-1.6+-purple.svg)](https://julialang.org)

A Julia library for quantitative economic history analysis.

## Overview

Cliometrics applies economic theory and quantitative methods to the study of historical economic phenomena. This package provides tools for:

- **Historical Data Analysis**: Load, clean, and analyze historical economic datasets
- **Growth Accounting**: Decompose economic growth into capital, labor, and TFP contributions
- **Convergence Analysis**: Test for economic convergence across regions and time periods
- **Institutional Analysis**: Quantify and analyze the role of institutions in economic development
- **Counterfactual Modeling**: Estimate treatment effects and alternative historical scenarios

## Installation

```julia
using Pkg
Pkg.add("Cliometrics")
```

## Quick Start

```julia
using Cliometrics
using DataFrames

# Load historical GDP data
data = load_historical_data("maddison_historical_gdp.csv")

# Calculate growth rates
growth_rates = calculate_growth_rates(data, :real_gdp_per_capita)

# Perform growth decomposition
decomposition = decompose_growth(
    data,
    output=:gdp,
    capital=:capital_stock,
    labor=:labor_force,
    alpha=0.35  # Capital share
)

# Test for convergence
convergence = convergence_analysis(
    country_data,
    :initial_gdp_1950,
    :growth_rate_1950_2000
)

# Create institutional quality index
quality_index = institutional_quality_index(
    institutions_data,
    [:rule_of_law, :property_rights, :contract_enforcement],
    weights=[0.4, 0.3, 0.3]
)
```

## Features

### Growth Analysis ✅
- Geometric and arithmetic growth rate calculations
- Solow residual (TFP) estimation
- Growth accounting decomposition
- Long-run growth trend analysis *(planned for v0.2.0)*

### Convergence Testing ✅
- Beta-convergence analysis
- Sigma-convergence testing *(planned for v0.2.0)*
- Conditional convergence estimation *(planned for v0.2.0)*
- Half-life calculations

### Institutional Analysis ✅
- Composite institutional quality indices
- Institutional change measurement
- Relationship between institutions and growth

### Data Tools ✅
- Historical time series cleaning
- Missing value interpolation
- Outlier detection and handling *(planned for v0.2.0)*
- Cross-country data alignment *(planned for v0.2.0)*

### Causal Inference ✅
- Counterfactual scenario modeling
- Difference-in-differences estimation (DiD)
- Treatment effect analysis

## Examples

### Example 1: Industrial Revolution Growth Analysis

```julia
using Cliometrics

# Load data from Broadberry et al. British Economic Growth 1270-1870
uk_data = load_historical_data("broadberry_uk_gdp.csv")

# Calculate pre and post-industrial revolution growth
pre_industrial = filter(row -> 1700 <= row.year < 1780, uk_data)
industrial = filter(row -> 1780 <= row.year <= 1870, uk_data)

pre_growth = mean(calculate_growth_rates(pre_industrial, :gdp_per_capita))
post_growth = mean(calculate_growth_rates(industrial, :gdp_per_capita))

println("Pre-Industrial Revolution: $(round(pre_growth*100, digits=2))% per year")
println("Industrial Revolution: $(round(post_growth*100, digits=2))% per year")
```

### Example 2: Great Divergence Analysis

```julia
# Compare Western Europe vs China 1500-1800
divergence_data = DataFrame(
    year = 1500:50:1800,
    western_europe_gdp = [1200, 1300, 1450, 1650, 1900, 2200, 2600],
    china_gdp = [1100, 1150, 1200, 1250, 1280, 1300, 1320]
)

comparison = compare_historical_trajectories(
    divergence_data,
    ["Western Europe", "China"],
    variable=:gdp_per_capita
)
```

### Example 3: Institutions and Growth

```julia
# Acemoglu & Robinson Why Nations Fail analysis
institutions = DataFrame(
    country = ["USA", "Haiti", "South Korea", "North Korea"],
    inclusive_institutions = [0.9, 0.3, 0.8, 0.1],
    gdp_per_capita_1960 = [15000, 2000, 1200, 1100],
    gdp_per_capita_2020 = [65000, 1800, 42000, 1300]
)

institutions.growth_rate = (institutions.gdp_per_capita_2020 ./
                           institutions.gdp_per_capita_1960) .^ (1/60) .- 1

# Regression of growth on institutions
using GLM
model = lm(@formula(growth_rate ~ inclusive_institutions), institutions)
```

## Methodology

This package implements standard cliometric methods including:

- **Growth Accounting**: Following Solow (1957) and subsequent literature
- **Convergence Tests**: Based on Barro & Sala-i-Martin (1992)
- **Institutional Indices**: Inspired by Acemoglu et al. (2001)
- **Historical National Accounts**: Compatible with Maddison Project format

## Data Sources

Compatible with major historical datasets:
- Maddison Project Database
- Penn World Table (historical extensions)
- Broadberry et al. historical national accounts
- Polity IV (institutional data)
- V-Dem (institutional indicators)

## Citation

If you use this package in research, please cite:

```bibtex
@software{cliometrics_jl,
  author = {Jewell, Jonathan D.A.},
  title = {Cliometrics.jl: Quantitative Economic History in Julia},
  year = {2026},
  url = {https://github.com/hyperpolymath/Cliometrics.jl}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Palimpsest License (PMPL-1.0-or-later). See [LICENSE](LICENSE) for details.

## References

- Solow, R. M. (1957). "Technical Change and the Aggregate Production Function." *Review of Economics and Statistics*, 39(3), 312-320.
- Barro, R. J., & Sala-i-Martin, X. (1992). "Convergence." *Journal of Political Economy*, 100(2), 223-251.
- Acemoglu, D., Johnson, S., & Robinson, J. A. (2001). "The Colonial Origins of Comparative Development." *American Economic Review*, 91(5), 1369-1401.
- Crafts, N., & Toniolo, G. (Eds.). (1996). *Economic Growth in Europe Since 1945*. Cambridge University Press.
- Maddison, A. (2007). *Contours of the World Economy 1-2030 AD: Essays in Macro-Economic History*. Oxford University Press.
