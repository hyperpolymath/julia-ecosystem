# API Reference

Complete API reference for all exported functions and types in Causals.jl.

## Dempster-Shafer Module

```@docs
MassAssignment
belief
plausibility
combine_dempster
pignistic_transform
```

## Bradford Hill Module

```@docs
BradfordHillCriteria
assess_causality
strength_of_evidence
```

## Causal DAG Module

```@docs
CausalGraph
add_edge!
remove_edge!
d_separation
ancestors
descendants
backdoor_criterion
frontdoor_criterion
markov_blanket
```

## Granger Causality Module

```@docs
granger_test
granger_causality
optimal_lag
bidirectional_granger
```

## Propensity Score Module

```@docs
propensity_score
matching
inverse_probability_weighting
stratification
doubly_robust
```

## Do-Calculus Module

```@docs
do_intervention
identify_effect
adjustment_formula
confounding_adjustment
do_calculus_rules
Query
```

## Counterfactuals Module

```@docs
counterfactual
twin_network
probability_of_necessity
probability_of_sufficiency
probability_of_necessity_and_sufficiency
Counterfactual
```

## Index

```@index
```
