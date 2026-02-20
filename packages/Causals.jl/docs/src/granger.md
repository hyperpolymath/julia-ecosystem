# Granger Causality

The Granger module provides Granger causality tests for time series analysis.

## Key Concepts

- **Granger Causality**: X Granger-causes Y if past values of X help predict Y beyond Y's own past
- **F-Test**: Statistical test comparing restricted and unrestricted VAR models
- **Optimal Lag**: Number of lags that minimizes information criterion (AIC)
- **Bidirectional Causality**: Testing causality in both directions

## API Reference

```@docs
granger_test
granger_causality
optimal_lag
bidirectional_granger
```
