# Dempster-Shafer Theory

The Dempster-Shafer module provides evidence combination and belief functions for reasoning under uncertainty.

## Key Concepts

- **Frame of Discernment**: Set of mutually exclusive hypotheses
- **Mass Assignment**: Probability mass distributed over subsets of hypotheses
- **Belief Function**: Lower probability bound for a proposition
- **Plausibility Function**: Upper probability bound for a proposition
- **Dempster's Rule**: Combines evidence from independent sources

## API Reference

```@docs
MassAssignment
belief
plausibility
combine_dempster
pignistic_transform
```
