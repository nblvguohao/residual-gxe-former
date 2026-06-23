from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.data.schema import validate_phenotype_table, validate_split_table
from residual_gxe.data.splits import SplitConfig, SplitType, make_split_table


def parse_args():
    parser = argparse.ArgumentParser(description="Construct deterministic benchmark splits.")
    parser.add_argument("--config", type=Path, required=False, default=None)
    parser.add_argument("--raw-dir", type=Path, required=False, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=False, default=None)
    parser.add_argument("--residual-dir", type=Path, required=False, default=None)
    parser.add_argument("--results-dir", type=Path, required=False, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    config = _load_config(args.config)
    split_cfg = config.get("splits", {})
    seeds = split_cfg.get("seeds", [1234])
    val_frac = split_cfg.get("val_fraction", 0.15)
    test_frac = split_cfg.get("test_fraction", 0.20)
    split_types: list[SplitType] = split_cfg.get(
        "split_types",
        ["random", "leave_genotype", "leave_environment", "leave_year", "leave_ge"],
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pheno_path = args.data_dir / "phenotype.parquet"
    if not pheno_path.exists():
        raise FileNotFoundError(f"Phenotype table not found: {pheno_path}")

    phenotype = read_table(pheno_path)
    validation = validate_phenotype_table(phenotype)
    if not validation.ok:
        raise ValueError(f"Phenotype validation failed: {validation.missing_columns} {validation.extra_message}")

    all_splits: list[pd.DataFrame] = []

    for split_type in split_types:
        for seed in seeds:
            cfg = SplitConfig(
                split_type=split_type,
                seed=seed,
                val_fraction=val_frac,
                test_fraction=test_frac,
            )
            try:
                split_df = make_split_table(phenotype, cfg, fold=seed)
            except ValueError as e:
                print(f"  [SKIP] {split_type} seed={seed}: {e}")
                continue

            val_result = validate_split_table(split_df)
            if not val_result.ok:
                print(f"  [WARN] {split_type} seed={seed}: {val_result.extra_message}")
                continue

            split_df["split_type"] = split_type
            split_df["seed"] = seed
            all_splits.append(split_df)

            n_train = int((split_df["split"] == "train").sum())
            n_val = int((split_df["split"] == "val").sum())
            n_test = int((split_df["split"] == "test").sum())
            print(f"  {split_type} seed={seed}: train={n_train} val={n_val} test={n_test}")

    if not all_splits:
        raise SystemExit("No valid splits were generated.")

    combined = pd.concat(all_splits, ignore_index=True)
    write_table(combined, out_dir / "splits.parquet")

    for split_type in split_types:
        subset = combined[combined["split_type"] == split_type]
        if len(subset) > 0:
            type_dir = out_dir / split_type
            type_dir.mkdir(parents=True, exist_ok=True)
            for seed_val in subset["seed"].unique():
                seed_df = subset[subset["seed"] == seed_val].drop(columns=["split_type", "seed"])
                write_table(seed_df, type_dir / f"splits_seed{seed_val}.parquet")

    print(f"\nSplits written to {out_dir / 'splits.parquet'}")
    print(f"Total split configurations: {len(all_splits)}")


if __name__ == "__main__":
    main()
