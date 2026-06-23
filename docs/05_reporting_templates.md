# Reporting Templates

## Dataset summary table

| Dataset | Crop | Years | Environments | Genotypes | Samples | Markers | Traits |
|---|---|---:|---:|---:|---:|---:|---|
| G2F | maize |  |  |  |  |  | yield |
| FIP1 | wheat |  |  |  |  |  | yield_adjusted, protein, heading_date, plant_height |

## Main result table

| Model | Random r | Leave-G r | Leave-E r | Leave-Year r | Leave-GE r | Weighted Score |
|---|---:|---:|---:|---:|---:|---:|
| GBLUP |  |  |  |  |  |  |
| Reaction-norm GBLUP |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |
| Concat Transformer |  |  |  |  |  |  |
| ResidualGxE-Former |  |  |  |  |  |  |

## Selection utility table

| Model | SG@5% | SG@10% | SG@20% | NDCG@10% |
|---|---:|---:|---:|---:|
| GBLUP |  |  |  |  |
| XGBoost |  |  |  |  |
| ResidualGxE-Former |  |  |  |  |

## Ablation table

| Variant | Leave-E r | Leave-Year r | SG@10% | Interpretation |
|---|---:|---:|---:|---|
| Full model |  |  |  |  |
| w/o residual learning |  |  |  | tests mixed-model decomposition |
| w/o weather sequence |  |  |  | tests temporal environment representation |
| w/o cross-attention |  |  |  | tests genotype-specific environment response |
| w/o rank loss |  |  |  | tests breeding selection utility |
