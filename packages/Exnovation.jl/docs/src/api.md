<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->

# API Reference

This page documents all exported types and functions in Exnovation.jl.

## Types

```@docs
ExnovationItem
Driver
Barrier
BarrierType
ExnovationAssessment
LegacyType
FailureType
IntelligentFailureCriteria
FailureAssessment
FailureSummary
ImpactModel
PortfolioItem
StageGate
```

## Functions

```@docs
assess_exnovation
normalize_score
recommendation
debiasing_actions
intelligent_failure_score
failure_summary
barrier_templates
portfolio_scores
allocate_budget
run_stage_gates
lifecycle_fit
```

## Enums

- `BarrierType`: `Cognitive`, `Emotional`, `Behavioral`, `Structural`, `Political`
- `LegacyType`: `Sustaining`, `Disruptive`
- `FailureType`: `Intelligent`, `Preventable`, `Complex`

## Index

```@index
```
