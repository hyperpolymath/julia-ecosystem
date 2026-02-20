# Political Instability

## Political Stress Indicator

The PSI is a composite index combining three destabilizing forces:

```math
\text{PSI} = 0.4 \cdot \text{MMP} + 0.4 \cdot \text{EMP} + 0.2 \cdot \text{SFD}
```

where:
- **MMP** (Mass Mobilization Potential): Popular immiseration from wage decline
- **EMP** (Elite Mobilization Potential): Elite overproduction and competition
- **SFD** (State Fiscal Distress): Revenue crisis undermining state capacity

```@docs
political_stress_indicator
```

## Instability Probability

Convert continuous stress indicators into event probabilities using a sigmoid function:

```@docs
instability_probability
```

## Conflict Intensity

Aggregate discrete instability events into a continuous conflict intensity measure over time:

```@docs
conflict_intensity
```

## Crisis Detection

```@docs
crisis_threshold
```
