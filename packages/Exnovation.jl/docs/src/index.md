# Exnovation.jl

**Exnovation.jl** is a decision-support library for exnovation: the strategic retirement
of legacy systems. It helps organizations assess whether to exnovate (retire), pilot, or
keep existing infrastructure using quantitative drivers, barriers, and impact models.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hyperpolymath/Exnovation.jl")
```

## Quick Start

```julia
using Exnovation

item = ExnovationItem(:LegacyCRM, "Legacy CRM system", "Sales operations")

drivers = [
    Driver(:SecurityRisk, 0.7, "Legacy stack has known vulnerabilities"),
    Driver(:Sustainability, 0.4, "Cloud move reduces footprint"),
]

barriers = [
    Barrier(Cognitive, 0.5, "Sunk-cost framing in past investments"),
    Barrier(Behavioral, 0.3, "Habits and routines tied to old workflows"),
]

criteria = DecisionCriteria(0.3, 0.3, 0.2, 0.2)

assessment = ExnovationAssessment(
    item,
    drivers,
    barriers,
    criteria,
    1_200_000.0,  # sunk_cost
    350_000.0,    # forward_value
    900_000.0,    # replacement_value
    0.4,          # strategic_fit (lower is worse)
    0.6,          # performance (lower is worse)
    0.7,          # risk (higher is worse)
)

score = exnovation_score(assessment)
println(score)
println(recommendation(assessment))
```

## Examples

The repository includes two complete examples:

- **`examples/01_basic_usage.jl`**: Basic exnovation assessment workflow
- **`examples/02_portfolio_management.jl`**: Portfolio prioritization and budgeting

## API Reference

See [API](api.md) for complete reference documentation.
