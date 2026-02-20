# Elite Dynamics

## Elite Overproduction Index

The EOI measures when the supply of elite aspirants exceeds available elite positions:

```math
\text{EOI} = \frac{E/N}{(E/N)_{\text{baseline}}} - 1
```

Positive values indicate overproduction — more elite aspirants than the system can absorb, leading to intra-elite competition and political instability.

```@docs
elite_overproduction_index
```

## Instability Events

Extract discrete instability events from continuous indicator time series:

```@docs
InstabilityEvent
instability_events
```
