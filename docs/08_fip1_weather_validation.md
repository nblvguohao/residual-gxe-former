# FIP1 Weather Validation Workflow

FIP1 processed phenotype/genotype/environment tables are available locally, but the current processed FIP1 directory does not contain `weather_daily.parquet`.

This means:

- FIP1 can support residual profile, candidate environment, and candidate genotype validation.
- FIP1 cannot yet support cross-crop stage-weather residual validation.

## Prepare User-Provided Weather

The adapter does not download external data. Provide a local CSV/TSV/Parquet file with at least:

```text
environment_id or yearsite_uid
date
one or more weather variables:
  tmax, tmin, tmean, precipitation, solar_radiation, relative_humidity, wind_speed
```

Run:

```bash
python scripts/13_prepare_fip1_weather.py \
  --weather-file data/raw/fip1/fip1_weather.csv \
  --data-dir data/processed/fip1 \
  --out-dir data/processed/fip1
```

Expected output:

```text
data/processed/fip1/weather_daily.parquet
data/processed/fip1/weather_manifest.json
```

## Rebuild FIP1 Atlas

After weather is prepared:

```bash
python scripts/10_build_residual_gxe_atlas.py \
  --data-dir data/processed/fip1 \
  --split-dir data/processed/fip1/splits \
  --residual-dir data/processed/fip1/residual_targets \
  --out-dir outputs/residual_gxe_atlas/fip1 \
  --min-n 5

python scripts/11_summarize_residual_gxe_atlas.py \
  --atlas-dir outputs/residual_gxe_atlas/fip1 \
  --out-dir outputs/residual_gxe_atlas/fip1_summary \
  --min-abs-pearson 0.05 \
  --top-n 20 \
  --min-n 5 \
  --min-runs 1

python scripts/12_compare_residual_gxe_atlases.py \
  --left-summary-dir outputs/residual_gxe_atlas/g2f_summary \
  --right-summary-dir outputs/residual_gxe_atlas/fip1_summary \
  --left-label g2f_maize \
  --right-label fip1_wheat \
  --out-dir outputs/residual_gxe_atlas/cross_dataset_g2f_fip1
```

## Interpretation Rule

Only claim cross-dataset stage-weather consistency when FIP1 has non-empty `stable_stage_weather_features.csv`.

If that table is empty because weather is unavailable, the correct manuscript wording is:

> FIP1 supports external residual-structure validation, but stage-weather residual mechanisms could not be externally validated because harmonized weather time series were unavailable.

