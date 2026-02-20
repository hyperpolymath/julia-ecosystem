# Propensity Score Methods

The PropensityScore module provides methods for causal inference from observational data using propensity scores.

## Key Concepts

- **Propensity Score**: P(treatment=1 | covariates) - probability of receiving treatment
- **Matching**: Pair treated and control units with similar propensity scores
- **Inverse Probability Weighting (IPW)**: Weight observations by inverse propensity
- **Stratification**: Group by propensity score and estimate effects within strata
- **Doubly Robust**: Consistent if either propensity or outcome model is correct

## API Reference

```@docs
propensity_score
matching
inverse_probability_weighting
stratification
doubly_robust
```
