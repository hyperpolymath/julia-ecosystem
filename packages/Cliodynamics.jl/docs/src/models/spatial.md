# Spatial Models

Spatial cliodynamic models extend single-polity analysis to multi-region systems where instability, population pressure, and elite competition diffuse across borders.

!!! note "v1.0.0 Feature"
    Spatial models are available from Cliodynamics.jl v1.0.0 onwards.

## Multi-Region Interaction

Model how instability in one region propagates to neighbors:

```@docs
spatial_instability_diffusion
```

## Territorial Competition

Model state competition over territory and resources:

```@docs
territorial_competition_model
```

## Frontier Effects

Model the meta-ethnic frontier thesis — states form most readily at boundaries between culturally distinct groups:

```@docs
frontier_formation_index
```
