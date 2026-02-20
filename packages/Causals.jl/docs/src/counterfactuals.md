# Counterfactual Reasoning

The Counterfactuals module provides tools for counterfactual reasoning and causal responsibility.

## Key Concepts

- **Counterfactual**: "What would have happened if...?" questions
- **Structural Causal Model (SCM)**: Equations defining how variables are generated
- **Three-Step Process**: Abduction (infer noise), Action (intervene), Prediction (compute counterfactual)
- **Probability of Necessity (PN)**: Was treatment necessary for outcome?
- **Probability of Sufficiency (PS)**: Would treatment be sufficient for outcome?
- **Twin Network**: Graph representing both factual and counterfactual worlds

## API Reference

```@docs
counterfactual
twin_network
probability_of_necessity
probability_of_sufficiency
probability_of_necessity_and_sufficiency
Counterfactual
```
