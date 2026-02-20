# Causal Directed Acyclic Graphs

The CausalDAG module provides tools for representing and reasoning about causal relationships using directed acyclic graphs.

## Key Concepts

- **DAG**: Directed acyclic graph representing causal structure
- **D-Separation**: Conditional independence criterion in DAGs
- **Backdoor Criterion**: Conditions for identifying causal effects
- **Frontdoor Criterion**: Alternative identification strategy
- **Markov Blanket**: Set of variables that render a node conditionally independent

## API Reference

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
