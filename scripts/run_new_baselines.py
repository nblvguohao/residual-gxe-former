"""Run RKHS, DNNGP, Reaction Norm GBLUP on G2F deployment-shift splits."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.data.splits import load_split_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import (
    fit_rkhs_fast,
    fit_dnngp,
    fit_reaction_norm_gblup_efficient,
    BaselineResult,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run new baselines (RKHS, DNNGP, Reaction Norm GBLUP)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split-type", type=str, default=None,
                   help="Only run this split type (e.g., leave_environment)")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pheno = read_table(args.data_dir / "phenotype.parquet")
    splits = load_split_table(args.split_dir)

    # Genotype data
    geno_path = args.data_dir / "genotype.parquet"
    geno_wide = None
    if geno_path.exists():
        from residual_gxe.training.preprocessing import build_genotype_wide
        geno_wide = build_genotype_wide(read_table(geno_path))

    all_metrics = []
    metrics_path = args.out_dir / "metrics.csv"
    if metrics_path.exists():
        all_metrics = pd.read_csv(metrics_path).to_dict(orient="records")

    for (split_type, seed), group in splits.groupby(["split_type", "seed"]):
        if args.split_type and split_type != args.split_type:
            continue

        train_ids = set(group.loc[group["split"] == "train", "sample_id"])
        test_ids = set(group.loc[group["split"] == "test", "sample_id"])

        print(f"\n{'='*50}")
        print(f"  {split_type}  seed={seed}")
        print(f"{'='*50}")

        # Skip existing
        existing = [
            r for r in all_metrics
            if r.get("split_type") == split_type and int(r.get("seed", -1)) == int(seed)
        ]
        existing_models = {r["model"] for r in existing}
        if {"rkhs_fast", "dnngp", "reaction_norm_gblup_efficient"} <= existing_models:
            print("  All 3 models exist, skipping")
            continue

        train_pheno = pheno[pheno["sample_id"].isin(train_ids)].copy()
        test_pheno = pheno[pheno["sample_id"].isin(test_ids)].copy()

        if len(train_pheno) == 0 or len(test_pheno) == 0:
            print("  Empty split, skip")
            continue

        # Build feature matrices
        marker_cols = [c for c in geno_wide.columns if c != "genotype_id"]
        train_idx = train_pheno.merge(geno_wide[["genotype_id"] + marker_cols], on="genotype_id", how="left")
        test_idx = test_pheno.merge(geno_wide[["genotype_id"] + marker_cols], on="genotype_id", how="left")

        X_train = train_idx[marker_cols].fillna(0.0).to_numpy(dtype=np.float64)
        X_test = test_idx[marker_cols].fillna(0.0).to_numpy(dtype=np.float64)
        y_train = train_pheno["phenotype_value"].to_numpy(dtype=np.float64)
        y_test = test_pheno["phenotype_value"].to_numpy(dtype=np.float64)

        # Environmental index for Reaction Norm GBLUP
        env_mean = train_pheno.groupby("environment_id")["phenotype_value"].mean()
        train_env_idx = train_pheno["environment_id"].map(env_mean).fillna(env_mean.mean()).values
        test_env_idx = test_pheno["environment_id"].map(env_mean).fillna(env_mean.mean()).values

        print(f"  train={len(train_pheno)} test={len(test_pheno)} markers={len(marker_cols)}")

        # ---- RKHS ----
        if "rkhs_fast" not in existing_models:
            t0 = time.perf_counter()
            try:
                result = fit_rkhs_fast(X_train, y_train, X_test)
                elapsed = time.perf_counter() - t0
                m = metrics_dict(y_test, result.predictions)
                all_metrics.append({"split_type": split_type, "seed": int(seed), "model": "rkhs_fast", "target": "phenotype", **m})
                print(f"  [rkhs_fast] p={m['pearson']:.4f} r={m['rmse']:.4f} ({elapsed:.0f}s)")
            except Exception as e:
                print(f"  [rkhs_fast] FAILED: {e}")

        # ---- DNNGP ----
        if "dnngp" not in existing_models:
            t0 = time.perf_counter()
            try:
                result = fit_dnngp(
                    X_train, y_train, X_test,
                    hidden_dim=256, dropout=0.3, n_blocks=3,
                    epochs=200, batch_size=256,
                )
                elapsed = time.perf_counter() - t0
                m = metrics_dict(y_test, result.predictions)
                all_metrics.append({"split_type": split_type, "seed": int(seed), "model": "dnngp", "target": "phenotype", **m})
                print(f"  [dnngp] p={m['pearson']:.4f} r={m['rmse']:.4f} ({elapsed:.0f}s)")
            except Exception as e:
                print(f"  [dnngp] FAILED: {e}")

        # ---- Reaction Norm GBLUP ----
        if "reaction_norm_gblup_efficient" not in existing_models:
            t0 = time.perf_counter()
            try:
                result = fit_reaction_norm_gblup_efficient(
                    X_train, y_train, X_test,
                    env_index_train=train_env_idx,
                    env_index_test=test_env_idx,
                )
                elapsed = time.perf_counter() - t0
                m = metrics_dict(y_test, result.predictions)
                all_metrics.append({"split_type": split_type, "seed": int(seed), "model": "reaction_norm_gblup_efficient", "target": "phenotype", **m})
                print(f"  [rn_gblup] p={m['pearson']:.4f} r={m['rmse']:.4f} ({elapsed:.0f}s)")
            except Exception as e:
                print(f"  [rn_gblup] FAILED: {e}")

    # Save
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        write_table(df, args.out_dir / "metrics.csv")
        (args.out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str))
        print(f"\nSaved {len(df)} records to {args.out_dir / 'metrics.csv'}")

        # Summary
        for model in df["model"].unique():
            sub = df[df["model"] == model]
            if len(sub) > 0:
                print(f"\n{model}:")
                for s in sorted(sub["split_type"].unique()):
                    ss = sub[sub["split_type"] == s]
                    print(f"  {s}: p={ss['pearson'].mean():.4f}+/-{ss['pearson'].std():.4f} (n={len(ss)})")


if __name__ == "__main__":
    main()
