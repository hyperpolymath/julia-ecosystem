# Do-Calculus and Interventions

The DoCalculus module provides Pearl's do-calculus for reasoning about causal interventions.

## Key Concepts

- **Do-Operator**: do(X=x) represents setting X to value x (intervention)
- **Interventional Query**: P(Y | do(X)) differs from observational P(Y | X)
- **Effect Identification**: Determining if causal effect can be computed from observational data
- **Adjustment Formula**: Computing P(Y | do(X)) using backdoor adjustment
- **Do-Calculus Rules**: Three rules for simplifying interventional queries

## API Reference

```@docs
do_intervention
identify_effect
adjustment_formula
confounding_adjustment
do_calculus_rules
Query
```
