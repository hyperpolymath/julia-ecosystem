# Examples

This page provides examples demonstrating the functionality of Causals.jl.

## Basic Usage

The `examples/01_basic_usage.jl` file demonstrates:

- **Dempster-Shafer Evidence Combination**: Combining evidence from multiple sources about medical diagnoses
- **Bradford Hill Causal Assessment**: Assessing the causal relationship between smoking and lung cancer
- **Causal DAG Operations**: Building causal graphs, testing d-separation, checking backdoor criterion

To run the basic usage example:

```julia
include("examples/01_basic_usage.jl")
```

## Advanced Analysis

The `examples/02_advanced_analysis.jl` file demonstrates:

- **Granger Causality**: Testing whether one time series helps predict another
- **Propensity Score Matching**: Estimating treatment effects from observational data with confounding
- **Do-Calculus and Interventions**: Identifying causal effects and using adjustment formulas
- **Counterfactual Reasoning**: Computing "what if" scenarios using structural causal models

To run the advanced analysis example:

```julia
include("examples/02_advanced_analysis.jl")
```

## Example Datasets

The examples use synthetic datasets to demonstrate the methods. For real-world applications, you can replace these with your own data following the same structure.
