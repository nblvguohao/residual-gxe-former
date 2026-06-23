"""Run ablation models (single-modality & partial-fusion) and Cropformer baseline.

Supports:
  --model genotype_only | weather_only | static_env_only | geno_env | cropformer | dnngp
  --split-type leave_environment | leave_ge | leave_year | leave_genotype | random
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.data.splits import load_split_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.deep import (
    ResidualGxEFormer,
    GenotypeOnlyModel,
    WeatherOnlyModel,
    StaticEnvOnlyModel,
    GenoEnvModel,
    CropformerSimple,
)
from residual_gxe.models.baselines import DNNGP, fit_dnngp, BaselineResult
from residual_gxe.training.preprocessing import FoldPreprocessor, build_genotype_wide
from residual_gxe.training.trainer import (
    MultiModalDataset,
    predict,
    train_model,
    torch_device_report,
)

MODEL_CHOICES = [
    "genotype_only", "weather_only", "static_env_only",
    "geno_env", "cropformer", "dnngp", "residual_gxe_former",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation models and baselines.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    p.add_argument("--split-type", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--metrics-file", type=str, default=None,
                   help="Override metrics output path (for parallel runs)")
    return p.parse_args()


def build_model(model_name: str, n_markers: int, weather_dim: int, static_dim: int,
                hidden_dim: int = 128, patch_size: int = 64, dropout: float = 0.15):
    """Factory for ablation models."""
    if model_name == "genotype_only":
        return GenotypeOnlyModel(n_markers, hidden_dim=hidden_dim, patch_size=patch_size, dropout=dropout)
    elif model_name == "weather_only":
        return WeatherOnlyModel(weather_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_name == "static_env_only":
        return StaticEnvOnlyModel(static_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_name == "geno_env":
        return GenoEnvModel(n_markers, static_dim, hidden_dim=hidden_dim, patch_size=patch_size, dropout=dropout)
    elif model_name == "cropformer":
        return CropformerSimple(n_markers, env_dim=static_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_name == "residual_gxe_former":
        return ResidualGxEFormer(n_markers, weather_dim, static_dim,
                                 hidden_dim=hidden_dim, patch_size=patch_size,
                                 dropout=dropout, use_film=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    args = parse_args()
    model_name = args.model
    quick = args.quick
    hidden_dim = 32 if quick else 128
    epochs = 5 if quick else 50
    # H200 has 143GB VRAM — use large batches for fast training
    if model_name == "cropformer":
        batch_size = 512
    elif model_name == "dnngp":
        batch_size = 512
    else:
        batch_size = 1024
    lr = 5e-4
    patience = 12
    device = args.device

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    pheno = read_table(args.data_dir / "phenotype.parquet")
    genotype = read_table(args.data_dir / "genotype.parquet") if (args.data_dir / "genotype.parquet").exists() else None
    environment = read_table(args.data_dir / "environment.parquet") if (args.data_dir / "environment.parquet").exists() else None
    weather = read_table(args.data_dir / "weather_daily.parquet") if (args.data_dir / "weather_daily.parquet").exists() else None
    splits = load_split_table(args.split_dir)
    geno_wide = build_genotype_wide(genotype)

    all_metrics = []
    metrics_path = args.out_dir / "metrics.csv"
    if metrics_path.exists():
        all_metrics = pd.read_csv(metrics_path).to_dict(orient="records")

    # ---- DNNGP path (uses separate training loop) ----
    if model_name == "dnngp":
        marker_cols = [c for c in geno_wide.columns if c != "genotype_id"]
        for (split_type, seed), group in splits.groupby(["split_type", "seed"]):
            if args.split_type and split_type != args.split_type:
                continue
            # Skip existing
            existing = [
                r for r in all_metrics
                if str(r.get("split_type")) == str(split_type)
                and int(r.get("seed", -1)) == int(seed)
                and str(r.get("model")) == "dnngp"
            ]
            if args.skip_existing and existing:
                print(f"  [{model_name}] {split_type} seed={seed}: skipping existing")
                continue

            train_ids = set(group.loc[group["split"] == "train", "sample_id"])
            test_ids = set(group.loc[group["split"] == "test", "sample_id"])
            train_pheno = pheno[pheno["sample_id"].isin(train_ids)]
            test_pheno = pheno[pheno["sample_id"].isin(test_ids)]
            if len(train_pheno) == 0 or len(test_pheno) == 0:
                continue

            train_idx = train_pheno.merge(geno_wide[["genotype_id"] + marker_cols], on="genotype_id", how="left")
            test_idx = test_pheno.merge(geno_wide[["genotype_id"] + marker_cols], on="genotype_id", how="left")
            X_train = train_idx[marker_cols].to_numpy(dtype=np.float32, na_value=0.0)
            X_test = test_idx[marker_cols].to_numpy(dtype=np.float32, na_value=0.0)
            y_train = train_pheno["phenotype_value"].to_numpy(dtype=np.float32)
            y_test = test_pheno["phenotype_value"].to_numpy(dtype=np.float32)

            print(f"  [{model_name}] {split_type} seed={seed}: train={len(train_pheno)} test={len(test_pheno)} markers={len(marker_cols)}")
            t0 = time.perf_counter()
            try:
                result = fit_dnngp(X_train, y_train, X_test, hidden_dim=256, dropout=0.3, n_blocks=3,
                                   epochs=120, batch_size=512, seed=int(seed))
                elapsed = time.perf_counter() - t0
                m = metrics_dict(y_test, result.predictions)
                all_metrics.append({"split_type": split_type, "seed": int(seed), "model": "dnngp", "target": "phenotype", **m})
                print(f"  [dnngp] pearson={m['pearson']:.4f} rmse={m['rmse']:.4f} ({elapsed:.0f}s)")
            except Exception as e:
                print(f"  [dnngp] FAILED: {e}")

    else:
        # ---- Deep learning ablation models (use FoldPreprocessor pipeline) ----
        for (split_type, seed), group in splits.groupby(["split_type", "seed"]):
            if args.split_type and split_type != args.split_type:
                continue

            # Skip existing
            existing = [
                r for r in all_metrics
                if str(r.get("split_type")) == str(split_type)
                and int(r.get("seed", -1)) == int(seed)
                and str(r.get("model")) == model_name
            ]
            if args.skip_existing and existing:
                print(f"  [{model_name}] {split_type} seed={seed}: skipping existing")
                continue

            # Remove old entries for this model+split+seed before re-running
            all_metrics = [
                r for r in all_metrics
                if not (str(r.get("split_type")) == str(split_type)
                        and int(r.get("seed", -1)) == int(seed)
                        and str(r.get("model")) == model_name)
            ]

            train_ids = set(group.loc[group["split"] == "train", "sample_id"])
            val_ids = set(group.loc[group["split"] == "val", "sample_id"])
            test_ids = set(group.loc[group["split"] == "test", "sample_id"])

            train_pheno = pheno[pheno["sample_id"].isin(train_ids)].copy()
            val_pheno = pheno[pheno["sample_id"].isin(val_ids)].copy()
            test_pheno = pheno[pheno["sample_id"].isin(test_ids)].copy()

            if len(train_pheno) == 0 or len(test_pheno) == 0:
                print(f"  [{model_name}] {split_type} seed={seed}: empty split, skip")
                continue

            y_train = train_pheno["phenotype_value"].to_numpy(dtype=np.float32)
            y_val = val_pheno["phenotype_value"].to_numpy(dtype=np.float32)
            y_test = test_pheno["phenotype_value"].to_numpy(dtype=np.float32)

            # Build features
            print(f"  [{model_name}] {split_type} seed={seed}: building features...", flush=True)
            preprocessor = FoldPreprocessor(max_markers=5000, weather_seq_len=30, weather_mode="daily")
            preprocessor.fit(train_pheno, geno_wide, environment, weather)

            X_geno_train, X_weather_train, X_env_train = preprocessor.transform(train_pheno, geno_wide, environment, weather)
            X_geno_val, X_weather_val, X_env_val = preprocessor.transform(val_pheno, geno_wide, environment, weather)
            X_geno_test, X_weather_test, X_env_test = preprocessor.transform(test_pheno, geno_wide, environment, weather)

            n_markers = X_geno_train.shape[1]
            weather_dim = X_weather_train.shape[2]
            static_dim = X_env_train.shape[1]

            print(f"  [{model_name}] {split_type} seed={seed}: n_markers={n_markers} weather_dim={weather_dim} static_dim={static_dim} "
                  f"train={len(train_pheno)} val={len(val_pheno)} test={len(test_pheno)}", flush=True)

            # Build datasets
            train_ds = MultiModalDataset(X_geno_train, X_weather_train, X_env_train, y_train)
            val_ds = MultiModalDataset(X_geno_val, X_weather_val, X_env_val, y_val) if len(val_pheno) > 0 else None
            test_ds = MultiModalDataset(X_geno_test, X_weather_test, X_env_test, y_test)

            # Create model
            model = build_model(model_name, n_markers, weather_dim, static_dim,
                                hidden_dim=hidden_dim, dropout=0.15)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  [{model_name}] params={n_params:,}", flush=True)

            start_time = time.perf_counter()
            result = train_model(model, train_ds, val_ds, epochs=epochs, batch_size=batch_size,
                                 lr=lr, weight_decay=1e-4, early_stopping_patience=patience,
                                 gradient_clip_norm=1.0, rank_weight=0.05, device=device,
                                 phenotype_weight=0.5, log_every=50)
            model.eval()
            y_pred = predict(model, test_ds, batch_size=batch_size * 2, device=device)
            elapsed = time.perf_counter() - start_time

            m = metrics_dict(y_test, y_pred)
            all_metrics.append({"split_type": split_type, "seed": int(seed), "model": model_name,
                                "target": "phenotype", "n_params": n_params, "time_s": elapsed, **m})
            print(f"  [{model_name}] {split_type} seed={seed}: pearson={m['pearson']:.4f} rmse={m['rmse']:.4f} "
                  f"({elapsed:.0f}s, best_epoch={result['best_epoch']})", flush=True)

    # Save
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        metrics_out = Path(args.metrics_file) if args.metrics_file else (args.out_dir / "metrics.csv")
        write_table(df, metrics_out)
        (metrics_out.with_suffix(".json")).write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved {len(df)} records to {metrics_out}")

        # Summary per model
        for m_name in df["model"].unique():
            sub = df[df["model"] == m_name]
            if len(sub) > 0:
                print(f"\n{m_name}:")
                for s in sorted(sub["split_type"].unique()):
                    ss = sub[sub["split_type"] == s]
                    if len(ss) > 0:
                        print(f"  {s}: pearson={ss['pearson'].mean():.4f}+/-{ss['pearson'].std():.4f} (n={len(ss)})")


if __name__ == "__main__":
    main()
