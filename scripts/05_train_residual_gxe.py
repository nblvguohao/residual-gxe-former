from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.data.splits import load_split_table
from residual_gxe.evaluation.artifacts import (
    build_prediction_frame,
    build_runtime_frame,
    write_prediction_artifact,
    write_runtime_artifact,
)
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.deep import ResidualGxEFormer
from residual_gxe.training.preprocessing import FoldPreprocessor, build_genotype_wide
from residual_gxe.training.trainer import (
    MultiModalDataset,
    predict,
    train_model,
    torch_device_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResidualGxE-Former.")
    parser.add_argument("--config", type=Path, required=False, default=None)
    parser.add_argument("--raw-dir", type=Path, required=False, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=False, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Small model and few epochs for smoke testing")
    parser.add_argument("--split-type", type=str, default=None, help="Only run on this split type")
    parser.add_argument("--skip-existing", action="store_true", help="Skip split/seed runs with existing predictions and metrics.")
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
    model_cfg = config.get("model", {})
    preprocess_cfg = config.get("preprocessing", {})
    weather_cfg = preprocess_cfg.get("weather", {}) or {}
    train_cfg = config.get("training", {})

    quick = args.quick
    split_type_filter = args.split_type
    epochs = 5 if quick else train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 64)
    lr = train_cfg.get("learning_rate", 5e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    patience = train_cfg.get("early_stopping_patience", 15)
    gradient_clip_norm = train_cfg.get("gradient_clip_norm", 1.0)
    num_workers = train_cfg.get("num_workers", 0)
    log_every = train_cfg.get("log_every", 20)
    amp = train_cfg.get("amp", False)
    hidden_dim = 32 if quick else model_cfg.get("genotype_encoder", {}).get("embedding_dim", 128)
    patch_size = model_cfg.get("genotype_encoder", {}).get("patch_size", 64)
    max_markers = 200 if quick else model_cfg.get("genotype_encoder", {}).get("max_markers", 5000)
    rank_weight = train_cfg.get("loss", {}).get("rank_loss_weight", 0.05)
    phenotype_weight = train_cfg.get("loss", {}).get("phenotype_loss_weight", 0.5)
    device = train_cfg.get("device", "auto")
    use_main_effect_input = bool(model_cfg.get("use_main_effect_input", False))
    weather_mode = str(weather_cfg.get("mode", "daily"))
    weather_seq_len = int(weather_cfg.get("seq_len", 30))
    weather_standardize = str(preprocess_cfg.get("weather_standardization", "")).lower() == "train_fold"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    pheno_path = args.data_dir / "phenotype.parquet"
    if not pheno_path.exists():
        raise FileNotFoundError(f"Phenotype table not found: {pheno_path}")
    phenotype = read_table(pheno_path)

    genotype = None
    geno_path = args.data_dir / "genotype.parquet"
    if geno_path.exists():
        genotype = read_table(geno_path)

    environment = None
    env_path = args.data_dir / "environment.parquet"
    if env_path.exists():
        environment = read_table(env_path)

    weather = None
    weather_path = args.data_dir / "weather_daily.parquet"
    if weather_path.exists():
        weather = read_table(weather_path)

    all_splits = load_split_table(args.split_dir)

    residuals_path = args.residual_dir / "residual_targets.parquet"
    use_residuals = residuals_path.exists()
    if use_residuals:
        all_residuals = read_table(residuals_path)

    geno_wide = build_genotype_wide(genotype)

    all_metrics: list[dict] = []
    run_manifests: list[dict] = []
    runtime_records: list[dict] = []
    metrics_path = out_dir / "metrics.csv"
    if metrics_path.exists():
        all_metrics = pd.read_csv(metrics_path).to_dict(orient="records")
    manifests_path = out_dir / "run_manifests.json"
    if manifests_path.exists():
        run_manifests = json.loads(manifests_path.read_text(encoding="utf-8"))
    runtime_path = out_dir / "runtime.csv"
    if runtime_path.exists():
        runtime_records = pd.read_csv(runtime_path).to_dict(orient="records")

    for (split_type, seed), split_group in all_splits.groupby(["split_type", "seed"]):
        if split_type_filter and split_type != split_type_filter:
            continue
        print(f"\n{'='*60}")
        print(f"  {split_type} seed={seed}")
        print(f"{'='*60}")

        expected_prediction = out_dir / "predictions" / str(split_type) / f"seed{seed}" / "predictions.parquet"
        existing_metric_targets = {
            str(row.get("target"))
            for row in all_metrics
            if str(row.get("split_type")) == str(split_type)
            and int(row.get("seed", -1)) == int(seed)
            and str(row.get("model")) == "ResidualGxEFormer"
        }
        if args.skip_existing and expected_prediction.exists() and "phenotype" in existing_metric_targets:
            print(f"  Skipping existing ResidualGxEFormer run: {split_type} seed={seed}")
            continue

        all_metrics = [
            row
            for row in all_metrics
            if not (
                str(row.get("split_type")) == str(split_type)
                and int(row.get("seed", -1)) == int(seed)
                and str(row.get("model")) == "ResidualGxEFormer"
            )
        ]
        run_manifests = [
            row
            for row in run_manifests
            if not (str(row.get("split_type")) == str(split_type) and int(row.get("seed", -1)) == int(seed))
        ]
        runtime_records = [
            row
            for row in runtime_records
            if not (
                str(row.get("split_type")) == str(split_type)
                and int(row.get("seed", -1)) == int(seed)
                and str(row.get("model")) == "ResidualGxEFormer"
            )
        ]

        train_ids = set(split_group.loc[split_group["split"] == "train", "sample_id"])
        val_ids = set(split_group.loc[split_group["split"] == "val", "sample_id"])
        test_ids = set(split_group.loc[split_group["split"] == "test", "sample_id"])

        train_pheno = phenotype[phenotype["sample_id"].isin(train_ids)].copy()
        val_pheno = phenotype[phenotype["sample_id"].isin(val_ids)].copy()
        test_pheno = phenotype[phenotype["sample_id"].isin(test_ids)].copy()

        if len(train_pheno) == 0 or len(test_pheno) == 0:
            print("  Skipping: empty train or test set")
            continue

        # Always predict phenotype directly (Plan B).
        # Main effects (G_effect + E_effect) are extracted from residual targets when
        # available and passed as auxiliary input features via use_main_effect_input.
        y_train = train_pheno["phenotype_value"].to_numpy(dtype=np.float32)
        y_val = val_pheno["phenotype_value"].to_numpy(dtype=np.float32)
        y_test = test_pheno["phenotype_value"].to_numpy(dtype=np.float32)

        has_main_effects = False
        if use_residuals:
            res_group = all_residuals[
                (all_residuals["split_type"] == split_type) & (all_residuals["seed"] == seed)
            ]
            if len(res_group) > 0:
                train_targets = train_pheno.merge(res_group[["sample_id", "main_prediction"]], on="sample_id", how="left")
                val_targets = val_pheno.merge(res_group[["sample_id", "main_prediction"]], on="sample_id", how="left")
                test_targets = test_pheno.merge(res_group[["sample_id", "main_prediction"]], on="sample_id", how="left")
                train_main = train_targets["main_prediction"].fillna(0).to_numpy(dtype=np.float32)
                val_main = val_targets["main_prediction"].fillna(0).to_numpy(dtype=np.float32)
                test_main = test_targets["main_prediction"].fillna(0).to_numpy(dtype=np.float32)
                has_main_effects = True
            else:
                train_main = np.zeros_like(y_train)
                val_main = np.zeros_like(y_val)
                test_main = np.zeros_like(y_test)
        else:
            train_main = np.zeros_like(y_train)
            val_main = np.zeros_like(y_val)
            test_main = np.zeros_like(y_test)

        # Build features
        if geno_wide is None:
            print("  Skipping: no genotype data available")
            continue

        env_feats = environment

        print(
            f"  building fold features: weather_mode={weather_mode} "
            f"use_main_effect_input={use_main_effect_input and has_main_effects} "
            f"phenotype_direct=True",
            flush=True,
        )
        preprocessor = FoldPreprocessor(
            max_markers=max_markers,
            marker_strategy=preprocess_cfg.get("marker_strategy", "variance"),
            weather_seq_len=weather_seq_len,
            weather_mode=weather_mode,
            weather_standardize=weather_standardize,
        )
        feature_start = time.perf_counter()
        preprocessor.fit(train_pheno, geno_wide, env_feats, weather)
        print(
            f"  preprocessor fit done in {time.perf_counter() - feature_start:.1f}s; "
            f"markers={len(preprocessor.marker_cols)} weather_cols={len(preprocessor.weather_cols)}",
            flush=True,
        )
        X_geno_train, X_weather_train, X_env_train = preprocessor.transform(
            train_pheno,
            geno_wide,
            env_feats,
            weather,
            main_effects=train_main if use_main_effect_input and has_main_effects else None,
        )
        print(f"  train feature transform done in {time.perf_counter() - feature_start:.1f}s", flush=True)
        if preprocessor.weather_standardize and X_weather_train.size:
            flat = X_weather_train.reshape(-1, X_weather_train.shape[-1])
            center = np.nanmean(flat, axis=0)
            scale = np.nanstd(flat, axis=0)
            scale = np.where((scale == 0) | np.isnan(scale), 1.0, scale)
            center = np.where(np.isnan(center), 0.0, center)
            preprocessor.weather_center = center.astype(float).tolist()
            preprocessor.weather_scale = scale.astype(float).tolist()
            X_weather_train = preprocessor._standardize_weather(X_weather_train)
        X_geno_val, X_weather_val, X_env_val = preprocessor.transform(
            val_pheno,
            geno_wide,
            env_feats,
            weather,
            main_effects=val_main if use_main_effect_input and has_main_effects else None,
        )
        print(f"  val feature transform done in {time.perf_counter() - feature_start:.1f}s", flush=True)
        X_geno_test, X_weather_test, X_env_test = preprocessor.transform(
            test_pheno,
            geno_wide,
            env_feats,
            weather,
            main_effects=test_main if use_main_effect_input and has_main_effects else None,
        )
        print(f"  test feature transform done in {time.perf_counter() - feature_start:.1f}s", flush=True)

        n_markers = X_geno_train.shape[1]
        weather_dim = X_weather_train.shape[2]
        static_dim = X_env_train.shape[1]

        print(f"  n_markers={n_markers} weather_dim={weather_dim} static_dim={static_dim}")
        print(f"  train={len(train_pheno)} val={len(val_pheno)} test={len(test_pheno)}")

        # Build datasets — always predict phenotype directly
        train_ds = MultiModalDataset(X_geno_train, X_weather_train, X_env_train, y_train)
        val_ds = MultiModalDataset(X_geno_val, X_weather_val, X_env_val, y_val) if len(val_pheno) > 0 else None
        test_ds = MultiModalDataset(X_geno_test, X_weather_test, X_env_test, y_test)

        # Create model — always single-head phenotype prediction (Plan B)
        model = ResidualGxEFormer(
            n_markers=n_markers,
            weather_dim=weather_dim,
            static_env_dim=static_dim,
            hidden_dim=hidden_dim,
            patch_size=min(patch_size, n_markers),
            dropout=0.15,
            multi_task=False,
            gated_residual=False,
            use_film=bool(model_cfg.get("fusion", {}).get("use_film", True)),
        )

        # Train
        start_time = time.perf_counter()
        result = train_model(
            model,
            train_ds,
            val_ds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=patience,
            gradient_clip_norm=gradient_clip_norm,
            rank_weight=rank_weight,
            device=device,
            phenotype_weight=phenotype_weight,
            num_workers=num_workers,
            log_every=log_every,
            amp=amp,
        )

        # Predict — always direct phenotype prediction (Plan B)
        y_pred_phenotype = predict(model, test_ds, batch_size=batch_size * 2, device=device)
        elapsed = time.perf_counter() - start_time

        # Evaluate phenotype prediction
        m_phenotype = metrics_dict(test_pheno["phenotype_value"].to_numpy(), y_pred_phenotype)

        record = {
            "split_type": split_type,
            "seed": int(seed),
            "model": "ResidualGxEFormer",
            "target": "phenotype",
            **m_phenotype,
        }
        all_metrics.append(record)
        print(f"  [phenotype] pearson={m_phenotype['pearson']:.4f} rmse={m_phenotype['rmse']:.4f}")

        # Compute residual metrics for diagnostic purposes (if residuals available)
        if has_main_effects:
            y_pred_residual_diag = y_pred_phenotype - test_main
            m_residual_diag = metrics_dict(y_test - test_main, y_pred_residual_diag) if has_main_effects else {}
        else:
            m_residual_diag = {}

        # Save predictions for this split
        split_out = out_dir / "predictions" / split_type / f"seed{seed}"
        pred_df = build_prediction_frame(
            test_pheno["sample_id"].values,
            test_pheno["phenotype_value"].values,
            y_pred_phenotype,
            split_type=split_type,
            seed=int(seed),
            model="ResidualGxEFormer",
            target="phenotype",
            extra_columns={
                "y_true_phenotype": test_pheno["phenotype_value"].values,
                "y_pred_phenotype": y_pred_phenotype,
                "main_prediction": test_main,
            },
        )
        prediction_path = write_prediction_artifact(pred_df, split_out)

        # Save model checkpoint
        ckpt_dir = out_dir / "checkpoints" / split_type / f"seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        (ckpt_dir / "training_history.json").write_text(json.dumps(result["history"], indent=2), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "script": "scripts/05_train_residual_gxe.py",
            "split_type": split_type,
            "seed": int(seed),
            "target_mode": "phenotype",
            "data_dir": str(args.data_dir),
            "split_dir": str(args.split_dir),
            "residual_dir": str(args.residual_dir),
            "n_train": int(len(train_pheno)),
            "n_val": int(len(val_pheno)),
            "n_test": int(len(test_pheno)),
            "model": {
                "n_markers": int(n_markers),
                "weather_dim": int(weather_dim),
                "static_env_dim": int(static_dim),
                "hidden_dim": int(hidden_dim),
                "patch_size": int(min(patch_size, n_markers)),
                "multi_task": False,
                "gated_residual": False,
                "use_main_effect_input": bool(use_main_effect_input and has_main_effects),
                "direct_phenotype_weight": 0.0,
            },
            "training": {
                "epochs_requested": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "early_stopping_patience": int(patience),
                "rank_loss_weight": float(rank_weight),
                "device": device,
                "device_report": torch_device_report(device),
                "gradient_clip_norm": float(gradient_clip_norm),
                "num_workers": int(num_workers),
                "log_every": int(log_every),
                "amp": bool(amp),
                "best_epoch": int(result["best_epoch"]),
                "best_val_loss": float(result["best_val_loss"]),
            },
            "preprocessing": preprocessor.to_manifest(),
            "outputs": {
                "predictions": str(prediction_path),
                "checkpoint": str(ckpt_dir / "model.pt"),
            },
        }
        manifest_dir = out_dir / "manifests" / split_type / f"seed{seed}"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        preprocessor.write_manifest(manifest_dir / "preprocessing_manifest.yaml")
        (manifest_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        run_manifests.append(manifest)
        runtime_records.append({
            "model": "ResidualGxEFormer",
            "split_type": split_type,
            "seed": int(seed),
            "target": "phenotype",
            "time_s": float(elapsed),
            "n_train": int(len(train_pheno)),
            "n_val": int(len(val_pheno)),
            "n_test": int(len(test_pheno)),
            "n_markers": int(n_markers),
            "weather_dim": int(weather_dim),
            "static_env_dim": int(static_dim),
            "epochs_requested": int(epochs),
            "best_epoch": int(result["best_epoch"]),
        })

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        write_table(metrics_df, out_dir / "metrics.csv")
        (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf-8")
        (out_dir / "run_manifests.json").write_text(
            json.dumps(run_manifests, indent=2, default=str), encoding="utf-8"
        )
        write_runtime_artifact(build_runtime_frame(runtime_records), out_dir)
        print(f"\nMetrics written to {out_dir / 'metrics.csv'}")

        # Summary — phenotype prediction only
        subset = metrics_df[metrics_df["target"] == "phenotype"]
        if len(subset) > 0:
            print("\n[phenotype] Mean across splits:")
            for metric in ["pearson", "rmse", "selection_gain_at_10pct"]:
                vals = subset[metric].dropna()
                if len(vals) > 0:
                    print(f"  {metric}: {vals.mean():.4f} +/- {vals.std():.4f}")
    else:
        print("No metrics generated.")


if __name__ == "__main__":
    main()
