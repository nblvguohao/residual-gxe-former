# ResidualGxE-Former

Minimal public repository for leakage-safe genotype-by-environment phenotype
prediction experiments.

## Scope

This repository contains reproducible code, configuration files, tests, and
empty data/output placeholders. It does not include raw datasets, processed
datasets, manuscript files, author metadata, release archives, model
checkpoints, or generated analysis outputs.

## Layout

```text
configs/                 experiment configurations
data/raw/g2f/            place local G2F raw data here
data/raw/fip1/           place local FIP1 raw data here
data/processed/          generated processed data, not committed
docs/                    technical schema and experiment notes
outputs/                 generated results, not committed
scripts/                 command-line pipeline entry points
src/residual_gxe/        Python package
tests/                   smoke and unit tests
```

## Basic Checks

```bash
python scripts/00_check_environment.py
python scripts/run_smoke_test.py
pytest -q
python -m compileall src scripts
```

Or:

```bash
make check
```

The smoke test uses synthetic data only and must not be reported as a
scientific result.

## Data Policy

Raw datasets are not redistributed in this repository. Place locally obtained
data under `data/raw/g2f/` or `data/raw/fip1/`, then run the preparation
scripts described in `docs/02_data_schema.md` and `docs/04_experiment_plan.md`.

Generated files under `data/processed/` and `outputs/` are ignored by Git.
