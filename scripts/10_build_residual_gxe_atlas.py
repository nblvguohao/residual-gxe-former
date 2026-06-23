from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.data.splits import load_split_table
from residual_gxe.evaluation.metrics import safe_pearson, safe_spearman


WEATHER_COLS = [
    "tmax",
    "tmin",
    "tmean",
    "precipitation",
    "solar_radiation",
    "relative_humidity",
    "wind_speed",
    "vpd",
    "gdd",
]

STAGES = [
    ("early", 0, 30),
    ("mid", 31, 90),
    ("late", 91, 160),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build training-fold residual GxE biological atlas.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=False, default=None)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--discovery-split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--min-n", type=int, default=8)
    return parser.parse_args()


def _stage_label(day: float) -> str | None:
    if pd.isna(day):
        return None
    for label, low, high in STAGES:
        if low <= float(day) <= high:
            return label
    return None


def _prepare_weather_stage_features(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    if "environment_id" not in weather.columns:
        raise ValueError("weather_daily.parquet must contain environment_id")

    usable_cols = [c for c in WEATHER_COLS if c in weather.columns]
    if not usable_cols:
        return pd.DataFrame({"environment_id": pd.unique(weather["environment_id"].astype(str))})

    weather["environment_id"] = weather["environment_id"].astype(str)
    if "day_after_planting" not in weather.columns or weather["day_after_planting"].isna().all():
        sort_cols = ["environment_id"] + (["date"] if "date" in weather.columns else [])
        weather = weather.sort_values(sort_cols)
        weather["day_after_planting"] = weather.groupby("environment_id").cumcount()

    weather["stage"] = weather["day_after_planting"].map(_stage_label)
    weather = weather[weather["stage"].notna()].copy()
    if len(weather) == 0:
        return pd.DataFrame({"environment_id": pd.unique(weather["environment_id"])})

    rows = []
    for (env_id, stage), sub in weather.groupby(["environment_id", "stage"]):
        row = {"environment_id": str(env_id), "stage": stage, "n_weather_days": int(len(sub))}
        for col in usable_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            if col == "precipitation":
                row[f"{col}_sum"] = float(values.sum(skipna=True))
            row[f"{col}_mean"] = float(values.mean(skipna=True))
        rows.append(row)

    stage_long = pd.DataFrame(rows)
    if stage_long.empty:
        return pd.DataFrame({"environment_id": pd.unique(weather["environment_id"])})
    wide = stage_long.pivot(index="environment_id", columns="stage")
    wide.columns = [f"{stage}_{name}" for name, stage in wide.columns]
    wide = wide.reset_index()
    return wide


def _association_rows(df: pd.DataFrame, feature_cols: list[str], min_n: int) -> list[dict]:
    rows = []
    for split_type, split_df in df.groupby("split_type"):
        for seed, seed_df in split_df.groupby("seed"):
            for col in feature_cols:
                sub = seed_df[["residual_target", col]].dropna()
                if len(sub) < min_n:
                    continue
                rows.append({
                    "split_type": split_type,
                    "seed": int(seed),
                    "feature": col,
                    "n": int(len(sub)),
                    "pearson": safe_pearson(sub["residual_target"], sub[col]),
                    "spearman": safe_spearman(sub["residual_target"], sub[col]),
                    "abs_pearson": abs(safe_pearson(sub["residual_target"], sub[col])),
                    "discovery_scope": "training_fold_only",
                })
    return rows


def _genotype_stability(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    for (split_type, seed, genotype_id), sub in df.groupby(["split_type", "seed", "genotype_id"]):
        if len(sub) < min_n:
            continue
        rows.append({
            "split_type": split_type,
            "seed": int(seed),
            "genotype_id": genotype_id,
            "n": int(len(sub)),
            "mean_residual": float(sub["residual_target"].mean()),
            "std_residual": float(sub["residual_target"].std()),
            "mean_phenotype": float(sub["phenotype_value"].mean()),
            "residual_stability_score": float(-sub["residual_target"].std()),
        })
    return pd.DataFrame(rows)


def _environment_profiles(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    for (split_type, seed, env_id), sub in df.groupby(["split_type", "seed", "environment_id"]):
        if len(sub) < min_n:
            continue
        rows.append({
            "split_type": split_type,
            "seed": int(seed),
            "environment_id": env_id,
            "year": int(sub["year"].iloc[0]) if "year" in sub.columns and pd.notna(sub["year"].iloc[0]) else None,
            "location_id": sub["location_id"].iloc[0] if "location_id" in sub.columns else None,
            "n": int(len(sub)),
            "mean_residual": float(sub["residual_target"].mean()),
            "std_residual": float(sub["residual_target"].std()),
            "mean_phenotype": float(sub["phenotype_value"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phenotype = read_table(args.data_dir / "phenotype.parquet")
    residuals = read_table(args.residual_dir / "residual_targets.parquet")
    weather_path = args.data_dir / "weather_daily.parquet"
    weather = read_table(weather_path) if weather_path.exists() else pd.DataFrame()

    required = {"sample_id", "residual_target", "split_type", "seed"}
    missing = required - set(residuals.columns)
    if missing:
        raise ValueError(f"Residual target table missing required columns: {sorted(missing)}")
    if "split" not in residuals.columns:
        if args.split_dir is None:
            raise ValueError("Residual target table has no split column; provide --split-dir to merge splits.parquet")
        splits = load_split_table(args.split_dir)
        residuals = residuals.merge(
            splits[["sample_id", "split_type", "seed", "split"]],
            on=["sample_id", "split_type", "seed"],
            how="left",
        )
        if residuals["split"].isna().any():
            raise ValueError("Could not assign split labels to all residual rows from split-dir")

    meta_cols = [c for c in ["sample_id", "genotype_id", "environment_id", "year", "location_id", "phenotype_value"] if c in phenotype.columns]
    merged = residuals.merge(phenotype[meta_cols], on="sample_id", how="left", suffixes=("", "_pheno"))
    if "phenotype_value_pheno" in merged.columns and "phenotype_value" not in merged.columns:
        merged["phenotype_value"] = merged["phenotype_value_pheno"]

    discovery = merged[merged["split"] == args.discovery_split].copy()
    if len(discovery) == 0:
        raise ValueError(f"No rows found for discovery split: {args.discovery_split}")

    weather_features = _prepare_weather_stage_features(weather) if len(weather) > 0 else pd.DataFrame()
    if len(weather_features) > 0 and "environment_id" in weather_features.columns:
        discovery["environment_id"] = discovery["environment_id"].astype(str)
        discovery = discovery.merge(weather_features, on="environment_id", how="left")

    feature_cols = [
        c for c in discovery.columns
        if any(c.startswith(f"{stage}_") for stage, _, _ in STAGES)
        and c not in {"stage"}
        and pd.api.types.is_numeric_dtype(discovery[c])
    ]

    assoc = pd.DataFrame(_association_rows(discovery, feature_cols, args.min_n))
    if assoc.empty:
        assoc = pd.DataFrame(columns=[
            "split_type", "seed", "feature", "n", "pearson", "spearman",
            "abs_pearson", "discovery_scope",
        ])
    if len(assoc) > 0:
        assoc = assoc.sort_values(["abs_pearson", "n"], ascending=[False, False])
    write_table(assoc, args.out_dir / "stage_weather_residual_associations.csv")

    geno_stability = _genotype_stability(discovery, args.min_n)
    if geno_stability.empty:
        geno_stability = pd.DataFrame(columns=[
            "split_type", "seed", "genotype_id", "n", "mean_residual",
            "std_residual", "mean_phenotype", "residual_stability_score",
        ])
    if len(geno_stability) > 0:
        geno_stability = geno_stability.sort_values(["std_residual", "n"], ascending=[True, False])
    write_table(geno_stability, args.out_dir / "genotype_residual_stability.csv")

    env_profiles = _environment_profiles(discovery, args.min_n)
    if env_profiles.empty:
        env_profiles = pd.DataFrame(columns=[
            "split_type", "seed", "environment_id", "year", "location_id",
            "n", "mean_residual", "std_residual", "mean_phenotype",
        ])
    if len(env_profiles) > 0:
        env_profiles = env_profiles.sort_values(["mean_residual"], ascending=True)
    write_table(env_profiles, args.out_dir / "environment_residual_profiles.csv")

    manifest = {
        "schema_version": 1,
        "script": "scripts/10_build_residual_gxe_atlas.py",
        "data_dir": str(args.data_dir),
        "split_dir": str(args.split_dir) if args.split_dir else None,
        "residual_dir": str(args.residual_dir),
        "discovery_split": args.discovery_split,
        "discovery_scope": "training_fold_only" if args.discovery_split == "train" else args.discovery_split,
        "min_n": int(args.min_n),
        "n_discovery_rows": int(len(discovery)),
        "n_stage_weather_features": int(len(feature_cols)),
        "outputs": {
            "stage_weather_residual_associations": str(args.out_dir / "stage_weather_residual_associations.csv"),
            "genotype_residual_stability": str(args.out_dir / "genotype_residual_stability.csv"),
            "environment_residual_profiles": str(args.out_dir / "environment_residual_profiles.csv"),
        },
        "notes": [
            "Use train split for discovery claims. Val/test summaries should be labelled as validation only.",
            "Associations are descriptive and require biological validation before mechanistic claims.",
        ],
    }
    (args.out_dir / "atlas_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
