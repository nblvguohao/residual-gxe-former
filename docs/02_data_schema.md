# Unified Data Schema

## Core sample table

File: `phenotype.parquet`

Required columns:

```text
sample_id              unique sample identifier
genotype_id            genotype/hybrid/line identifier
environment_id         location-year or trial environment identifier
year                   integer year
location_id            location identifier
trait_id               normalized trait identifier
trait_name             original trait name
trait_family           yield, flowering, height, quality, stress, other
phenotype_value        numeric phenotype value
phenotype_unit         unit string
replicate_id           optional replicate/block ID
block_id               optional block ID
trial_id               optional trial ID
source_dataset         e.g., g2f, fip1
```

## Genotype table

Recommended storage depends on size:

- small/medium: `genotype.parquet`;
- large: Zarr/HDF5/NumPy memmap.

Required fields:

```text
genotype_id
marker_id
chromosome
position
allele_dosage          0/1/2 or imputed numeric dosage
```

Wide matrix format is also acceptable:

```text
genotype_id, marker_1, marker_2, ..., marker_M
```

but there must be a separate marker map:

```text
marker_id, chromosome, position, ref, alt
```

## Environment table

File: `environment.parquet`

Required columns:

```text
environment_id
year
location_id
latitude
longitude
altitude
planting_date
harvest_date
management_notes
source_dataset
```

## Daily weather table

File: `weather_daily.parquet`

Required columns:

```text
environment_id
date
day_after_planting
stage_label            optional; can be inferred later
tmax
tmin
tmean
precipitation
solar_radiation
relative_humidity
wind_speed
vpd
gdd
```

Not every dataset has every weather variable. Missing variables must be tracked in `data_manifest.yaml`.

## Soil table

File: `soil.parquet`

Required columns where available:

```text
environment_id
ph
organic_matter
sand
silt
clay
cec
water_holding_capacity
soil_depth
```

## Split table

Required columns:

```text
sample_id
split                 train, val, test
fold                  integer fold ID
reason                random, leave_genotype, leave_environment, leave_year, leave_ge
```

## Data manifest

File: `data_manifest.yaml`

Must include:

```yaml
dataset: g2f
created_at: null
raw_data_path: null
n_samples: null
n_genotypes: null
n_environments: null
n_years: null
traits: []
missingness:
  genotype: null
  weather: null
  soil: null
notes: []
```
