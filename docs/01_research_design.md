# Research Design

## Main paper positioning

This is not a generic phenotype prediction paper. The intended contribution is a robust framework for **environmental extrapolation** in single-crop G×E prediction.

## Target prediction task

For a genotype `g` and environment `e`, predict phenotype:

```math
\hat{y}_{g,e}
```

where `e` may represent an unseen location-year or future year.

## Decomposition

Use the decomposition:

```math
y_{g,e} = \mu + G_g + E_e + (G \times E)_{g,e} + \epsilon
```

ResidualGxE-Former focuses on predicting:

```math
(G \times E)_{g,e}
```

The final prediction is reconstructed as:

```math
\hat{y}_{g,e} = \hat{\mu} + \hat{G}_g + \hat{E}_e + f_\theta(g,e)
```

## Why this matters

Breeding decisions often require predicting performance in environments that have not been directly observed. Random split performance is insufficient because it can overstate real-world utility.

## Priority evaluation scenarios

1. Leave-environment-out.
2. Leave-year-out / temporal split.
3. Leave-genotype-environment-out.
4. Leave-genotype-out.
5. Random split, only as reference.

## Primary target traits

For G2F maize, start with grain yield. Add plant height, flowering time, and other available traits after the pipeline is stable.

For FIP1 wheat, use yield_adjusted first, then protein, heading_date, and plant_height.
