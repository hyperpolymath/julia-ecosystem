# Secular Cycles

## Overview

Secular cycles are long-term oscillations (150-300 years) identified by Turchin and Nefedov in agrarian societies. Each cycle passes through four phases:

1. **Expansion**: Low population pressure, state strengthening, prosperity
2. **Stagflation**: Rising pressure, elite overproduction begins
3. **Crisis**: Political instability, state breakdown, conflict
4. **Depression/Intercycle**: Population decline, elite winnowing, recovery

## Cycle Analysis

Detect secular cycles in time series data using trend-cycle decomposition:

```@docs
secular_cycle_analysis
```

## Phase Detection

Classify each time point into one of the four secular cycle phases:

```@docs
SecularCyclePhase
detect_cycle_phases
```
