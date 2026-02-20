# State Formation

## State Capacity Model

Models state capacity as a function of population size (tax base) and elite demands:

```math
S = \tau \cdot \alpha \cdot N^\beta - \gamma \cdot E
```

where:
- ``\tau`` = tax rate
- ``\alpha`` = administrative efficiency
- ``N`` = population, ``\beta`` = returns to scale
- ``\gamma`` = elite cost coefficient, ``E`` = elite population

```@docs
StateCapacityParams
state_capacity_model
```

## Collective Action

Models the probability of successful collective action as a function of group size, benefit, and cost:

```@docs
collective_action_problem
```
