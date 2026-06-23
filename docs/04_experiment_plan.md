# Experiment Plan

## Splits

Run all models under these settings:

1. Random split.
2. Leave-genotype-out.
3. Leave-environment-out.
4. Leave-year-out.
5. Leave-genotype-environment-out.

The main paper should emphasize leave-environment-out and leave-year-out.

## Baselines

### Required

- mean baseline;
- environment mean baseline;
- genotype mean baseline;
- BayesRidge;
- Ridge/rrBLUP-like linear model;
- GBLUP/kernel ridge;
- RandomForest;
- XGBoost or LightGBM if installed;
- MLP genotype+environment;
- environment-only neural model;
- genotype-only neural model;
- concat G+E Transformer.

### Stretch

- reaction-norm GBLUP;
- Gaussian-process G×E kernel;
- factor analytic mixed model.

## Main model

ResidualGxE-Former variants:

1. TCN weather encoder + FiLM.
2. TCN weather encoder + cross-attention.
3. Temporal Transformer weather encoder + cross-attention.

## Ablations

- direct phenotype prediction instead of residual learning;
- summary weather features instead of daily sequence;
- no cross-attention;
- no FiLM;
- no rank loss;
- no environment-balanced sampling;
- no auxiliary traits.

## Metrics

Required:

- Pearson r;
- Spearman rho;
- RMSE;
- MAE;
- environment-wise Pearson;
- genotype-wise Pearson;
- SelectionGain@5%, @10%, @20%;
- NDCG@5%, @10%, @20%;
- bootstrap confidence intervals.

## Main tables

1. Dataset summary.
2. Main prediction results.
3. Selection utility.
4. Ablation study.
5. External FIP1 validation.

## Main figures

1. Pipeline architecture.
2. Data/split schematic.
3. Main results by split.
4. Selection gain curves.
5. Ablation bars.
6. Growth-stage/environment sensitivity heatmaps.
