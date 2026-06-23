# Model Specification: ResidualGxE-Former

## Overview

ResidualGxE-Former predicts nonlinear genotype-by-environment residuals after stable main effects have been estimated.

## Modules

### 1. Mixed-model prior module

Fit main effects using training data only:

```math
y = \mu + G + E + optional\ covariates + \epsilon
```

Compute residual targets:

```math
r_{g,e} = y_{g,e} - \hat{\mu} - \hat{G}_g - \hat{E}_e
```

### 2. Genotype encoder

Initial implementation:

```text
SNP vector -> patching -> embedding -> 1D CNN -> Transformer/attention pooling -> genotype embedding
```

Important options:

- chromosome-aware patching;
- LD-like local convolution;
- marker dropout;
- optional SNP-to-gene mapping in later version.

### 3. Weather time-series encoder

Initial implementation:

```text
daily weather sequence -> variable embedding -> day/stage positional encoding -> TCN or Temporal Transformer -> stage pooling
```

The minimal working implementation can use TCN/GRU before Temporal Transformer is added.

### 4. Soil/location/management encoder

Static environment features:

```text
soil + location + management -> MLP -> static environment embedding
```

### 5. G×E fusion

Preferred variants:

- Cross-attention: genotype tokens query environment tokens.
- FiLM: environment embedding modulates genotype representation.
- Concatenation baseline.

### 6. Prediction heads

Required:

- residual prediction head.

Optional:

- direct phenotype head;
- uncertainty head;
- auxiliary trait heads.

## Loss

Recommended total loss:

```math
L = L_{Huber}(r, \hat{r}) + \lambda_1 L_{rank} + \lambda_2 L_{env-balance} + \lambda_3 L_{aux}
```

Implement Huber first. Add ranking loss and environment-balanced sampling after baseline model is stable.
