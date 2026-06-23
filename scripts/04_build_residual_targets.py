from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.data.schema import validate_phenotype_table
from residual_gxe.data.splits import load_split_table
from residual_gxe.models.mixed_model import (
    AdditiveMainEffects,
    RidgeMainEffects,
    fit_additive_main_effects,
    fit_ridge_main_effects,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build fold-specific residual targets.")
    parser.add_argument("--config", type=Path, required=False, default=None)
    parser.add_argument("--raw-dir", type=Path, required=False, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=False, default=None)
    parser.add_argument("--results-dir", type=Path, required=False, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="additive", choices=["additive", "ridge"],
                        help="Main effect estimation method: additive (simple mean) or ridge (BayesianRidge on one-hot).")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Ridge alpha (only used with --method ridge --no-bayesian).")
    parser.add_argument("--no-bayesian", action="store_true",
                        help="Use plain Ridge instead of BayesianRidge (only with --method ridge).")
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
    mm_cfg = config.get("mixed_model", {})
    fallback_geno = mm_cfg.get("fallback_for_unseen_genotype", "global_mean")
    fallback_env = mm_cfg.get("fallback_for_unseen_environment", "global_mean")
    method = args.method

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pheno_path = args.data_dir / "phenotype.parquet"
    if not pheno_path.exists():
        raise FileNotFoundError(f"Phenotype table not found: {pheno_path}")
    phenotype = read_table(pheno_path)
    validate_phenotype_table(phenotype)

    all_splits = load_split_table(args.split_dir)

    all_residuals: list[pd.DataFrame] = []
    metadata_entries: list[dict] = []

    for (split_type, seed), split_group in all_splits.groupby(["split_type", "seed"]):
        print(f"\n=== {split_type} seed={seed} ===")

        train_ids = set(split_group.loc[split_group["split"] == "train", "sample_id"])
        split_cols = ["sample_id", "split"]
        if "official_split" in split_group.columns:
            split_cols.append("official_split")
        split_pheno = phenotype.merge(split_group[split_cols], on="sample_id", how="inner")

        train_pheno = split_pheno[split_pheno["sample_id"].isin(train_ids)].copy()

        if len(train_pheno) == 0:
            print("  Skipping: empty train set")
            continue

        if method == "ridge":
            effects = fit_ridge_main_effects(
                train_pheno,
                use_bayesian=not args.no_bayesian,
                alpha=args.ridge_alpha,
            )
        else:
            effects = fit_additive_main_effects(train_pheno)

        # Compute residuals for all samples
        residuals = effects.residuals(split_pheno)
        main_pred = effects.predict_main_effects(split_pheno)

        out_cols = ["sample_id", "split"]
        if "official_split" in split_pheno.columns:
            out_cols.append("official_split")
        out = split_pheno[out_cols].copy()
        out["global_mean"] = effects.global_mean
        out["genotype_effect"] = split_pheno["genotype_id"].map(effects.genotype_effects).fillna(0.0)
        out["environment_effect"] = split_pheno["environment_id"].map(effects.environment_effects).fillna(0.0)
        out["main_prediction"] = main_pred.values
        out["residual_target"] = residuals.values
        out["phenotype_value"] = split_pheno["phenotype_value"].values
        out["split_type"] = split_type
        out["seed"] = seed

        all_residuals.append(out)

        train_n = int((out["split"] == "train").sum())
        val_n = int((out["split"] == "val").sum())
        test_n = int((out["split"] == "test").sum())

        train_resid = out.loc[out["split"] == "train", "residual_target"]
        test_resid = out.loc[out["split"] == "test", "residual_target"]
        print(f"  train={train_n} val={val_n} test={test_n}")
        print(f"  train residual mean={train_resid.mean():.4f} std={train_resid.std():.4f}")
        print(f"  test  residual mean={test_resid.mean():.4f} std={test_resid.std():.4f}")

        meta = {
            "split_type": split_type,
            "seed": int(seed),
            "method": method,
            "global_mean": effects.intercept if hasattr(effects, "intercept") else effects.global_mean,
            "n_genotype_effects": len(effects.genotype_effects) if hasattr(effects, "genotype_effects") else len(effects.genotype_coefficients),
            "n_environment_effects": len(effects.environment_effects) if hasattr(effects, "environment_effects") else len(effects.environment_coefficients),
            "fallback_for_unseen_genotype": fallback_geno,
            "fallback_for_unseen_environment": fallback_env,
            "n_train": train_n,
            "n_val": val_n,
            "n_test": test_n,
            "train_residual_mean": float(train_resid.mean()),
            "train_residual_std": float(train_resid.std()),
            "test_residual_mean": float(test_resid.mean()),
            "test_residual_std": float(test_resid.std()),
        }
        metadata_entries.append(meta)

    if all_residuals:
        combined = pd.concat(all_residuals, ignore_index=True)
        write_table(combined, out_dir / "residual_targets.parquet")
        (out_dir / "residual_metadata.json").write_text(
            json.dumps(metadata_entries, indent=2), encoding="utf-8"
        )
        print(f"\nResidual targets written to {out_dir / 'residual_targets.parquet'}")
        print(f"Total entries: {len(combined)}")
    else:
        print("No residual targets generated.")


if __name__ == "__main__":
    main()
