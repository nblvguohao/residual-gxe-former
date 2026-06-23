from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
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
from residual_gxe.models.deep import (
    CrossAttentionFusion,
    GenotypeEncoder,
    ResidualGxEFormer,
    StaticEnvEncoder,
    WeatherEncoder,
)
from residual_gxe.training.preprocessing import FoldPreprocessor, build_genotype_wide
from residual_gxe.training.trainer import (
    MultiModalDataset,
    predict,
    train_model,
    torch_device_report,
)


class ConcatFusion(nn.Module):
    """Simple concatenation fusion baseline (no cross-attention, no FiLM)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, genotype_emb: torch.Tensor, weather_tokens: torch.Tensor, static_env_emb: torch.Tensor) -> torch.Tensor:
        env_pooled = weather_tokens.mean(dim=1) if weather_tokens.dim() == 3 else weather_tokens
        fused = self.proj(torch.cat([genotype_emb, env_pooled + static_env_emb], dim=-1))
        return self.norm(fused)


class DirectPredictFormer(ResidualGxEFormer):
    """Same architecture but head is used for direct phenotype prediction (no residual)."""
    pass


def _build_ablation_model(
    variant: dict,
    n_markers: int,
    weather_dim: int,
    static_env_dim: int,
    hidden_dim: int,
    patch_size: int,
) -> nn.Module:
    """Build a model variant for ablation study. Always uses real feature dims."""
    use_cross_attn = variant.get("cross_attention", True)
    use_film = variant.get("film", True)

    model = ResidualGxEFormer(
        n_markers=n_markers,
        weather_dim=weather_dim,
        static_env_dim=static_env_dim,
        hidden_dim=hidden_dim,
        patch_size=min(patch_size, n_markers),
        dropout=0.15,
        use_film=use_film,
    )

    if not use_cross_attn and not use_film:
        model.fusion = ConcatFusion(hidden_dim=hidden_dim)

    return model


def _zero_env_inputs(X_w: np.ndarray, X_env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return zeroed copies of environment features for ablation."""
    return np.zeros_like(X_w), np.zeros_like(X_env)


def _zero_geno_input(X_geno: np.ndarray) -> np.ndarray:
    """Return zeroed genotype features for ablation."""
    return np.zeros_like(X_geno)


DEFAULT_ABLATIONS = [
    {"name": "full", "residual_learning": True, "weather_sequence": True, "cross_attention": True, "film": True, "rank_loss": True, "use_genotype": True, "use_environment": True},
    {"name": "no_residual", "residual_learning": False, "weather_sequence": True, "cross_attention": True, "film": True, "rank_loss": True, "use_genotype": True, "use_environment": True},
    {"name": "no_weather_seq", "residual_learning": True, "weather_sequence": False, "cross_attention": True, "film": True, "rank_loss": True, "use_genotype": True, "use_environment": True},
    {"name": "concat_only", "residual_learning": True, "weather_sequence": True, "cross_attention": False, "film": False, "rank_loss": True, "use_genotype": True, "use_environment": True},
    {"name": "no_rank_loss", "residual_learning": True, "weather_sequence": True, "cross_attention": True, "film": True, "rank_loss": False, "use_genotype": True, "use_environment": True},
    {"name": "genotype_only", "residual_learning": True, "weather_sequence": True, "cross_attention": True, "film": True, "rank_loss": True, "use_genotype": True, "use_environment": False},
    {"name": "environment_only", "residual_learning": True, "weather_sequence": True, "cross_attention": True, "film": True, "rank_loss": True, "use_genotype": False, "use_environment": True},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--config", type=Path, required=False, default=None)
    parser.add_argument("--raw-dir", type=Path, required=False, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=False, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Small model and few epochs for smoke testing")
    parser.add_argument("--split-type", type=str, default=None, help="Only run on this split type (e.g. leave_environment)")
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
    ablations = config.get("ablations", DEFAULT_ABLATIONS)
    preprocess_cfg = config.get("preprocessing", {})
    weather_cfg = preprocess_cfg.get("weather", {}) or {}
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    quick = args.quick
    epochs = 5 if quick else train_cfg.get("epochs", 50)
    batch_size = train_cfg.get("batch_size", 64)
    lr = train_cfg.get("learning_rate", 5e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    patience = train_cfg.get("early_stopping_patience", 10)
    gradient_clip_norm = train_cfg.get("gradient_clip_norm", 1.0)
    num_workers = train_cfg.get("num_workers", 0)
    log_every = train_cfg.get("log_every", 20)
    amp = train_cfg.get("amp", False)
    hidden_dim = 32 if quick else model_cfg.get("hidden_dim", 128)
    patch_size = model_cfg.get("patch_size", 64)
    max_markers = 200 if quick else model_cfg.get("max_markers", 5000)
    device = train_cfg.get("device", "auto")
    weather_mode = str(weather_cfg.get("mode", "daily"))
    weather_seq_len = int(weather_cfg.get("seq_len", 30))
    weather_standardize = str(preprocess_cfg.get("weather_standardization", "")).lower() == "train_fold"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load shared data
    pheno_path = args.data_dir / "phenotype.parquet"
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
    all_residuals = read_table(args.residual_dir / "residual_targets.parquet")

    geno_wide = build_genotype_wide(genotype)

    if geno_wide is None:
        raise SystemExit("No genotype data available")

    # Determine which splits to run
    if args.split_type:
        split_types_to_run = [args.split_type]
    else:
        split_types_to_run = sorted(all_splits["split_type"].unique())

    all_metrics: list[dict] = []
    run_manifests: list[dict] = []
    runtime_records: list[dict] = []

    for ablation in ablations:
        name = ablation["name"]
        use_residual = ablation.get("residual_learning", True)
        use_rank = ablation.get("rank_loss", True)
        rank_w = 0.05 if use_rank else 0.0

        print(f"\n{'#'*60}")
        print(f"  ABLATION: {name}")
        print(f"  residual={use_residual} weather_seq={ablation.get('weather_sequence',True)} cross_attn={ablation.get('cross_attention',True)} film={ablation.get('film',True)} rank_loss={use_rank}")
        print(f"{'#'*60}")

        for (split_type, seed), split_group in all_splits.groupby(["split_type", "seed"]):
            if split_type not in split_types_to_run:
                continue

            print(f"\n  --- {split_type} seed={seed} ---")

            train_ids = set(split_group.loc[split_group["split"] == "train", "sample_id"])
            val_ids = set(split_group.loc[split_group["split"] == "val", "sample_id"])
            test_ids = set(split_group.loc[split_group["split"] == "test", "sample_id"])

            train_pheno = phenotype[phenotype["sample_id"].isin(train_ids)].copy()
            val_pheno = phenotype[phenotype["sample_id"].isin(val_ids)].copy()
            test_pheno = phenotype[phenotype["sample_id"].isin(test_ids)].copy()

            if len(train_pheno) == 0 or len(test_pheno) == 0:
                continue

            # Targets
            res_group = all_residuals[
                (all_residuals["split_type"] == split_type) & (all_residuals["seed"] == seed)
            ]
            if use_residual and len(res_group) > 0:
                train_t = train_pheno.merge(res_group[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
                test_t = test_pheno.merge(res_group[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
                y_train = train_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
                y_test = test_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
                test_main = test_t["main_prediction"].fillna(0).to_numpy(dtype=np.float32)
            else:
                y_train = train_pheno["phenotype_value"].to_numpy(dtype=np.float32)
                y_test = test_pheno["phenotype_value"].to_numpy(dtype=np.float32)
                test_main = np.zeros_like(y_test)

            val_t = val_pheno.merge(res_group[["sample_id", "residual_target"]], on="sample_id", how="left") if use_residual and len(res_group) > 0 else val_pheno.copy()
            y_val = val_t["residual_target"].fillna(0).to_numpy(dtype=np.float32) if use_residual else val_pheno["phenotype_value"].to_numpy(dtype=np.float32)

            # Build features
            preprocessor = FoldPreprocessor(
                max_markers=max_markers,
                marker_strategy=preprocess_cfg.get("marker_strategy", "variance"),
                weather_seq_len=weather_seq_len,
                weather_mode=weather_mode,
                weather_standardize=weather_standardize,
            )
            X_geno_train, X_w_train, X_env_train = preprocessor.fit_transform(
                train_pheno, geno_wide, environment, weather
            )
            X_geno_test, X_w_test, X_env_test = preprocessor.transform(
                test_pheno, geno_wide, environment, weather
            )
            X_geno_val, X_w_val, X_env_val = preprocessor.transform(
                val_pheno, geno_wide, environment, weather
            )

            # Apply ablation modifications to feature arrays
            if not ablation.get("weather_sequence", True):
                X_w_train, _ = _zero_env_inputs(X_w_train, X_env_train)
                X_w_test, _ = _zero_env_inputs(X_w_test, X_env_test)
                X_w_val, _ = _zero_env_inputs(X_w_val, X_env_val)

            if not ablation.get("use_environment", True):
                X_w_train, X_env_train = _zero_env_inputs(X_w_train, X_env_train)
                X_w_test, X_env_test = _zero_env_inputs(X_w_test, X_env_test)
                X_w_val, X_env_val = _zero_env_inputs(X_w_val, X_env_val)

            if not ablation.get("use_genotype", True):
                X_geno_train = _zero_geno_input(X_geno_train)
                X_geno_test = _zero_geno_input(X_geno_test)
                X_geno_val = _zero_geno_input(X_geno_val)

            n_markers = X_geno_train.shape[1]
            weather_dim = X_w_train.shape[2]
            static_dim = X_env_train.shape[1]

            # Build model
            model = _build_ablation_model(
                ablation, n_markers, weather_dim, static_dim, hidden_dim, patch_size=patch_size,
            )

            train_ds = MultiModalDataset(X_geno_train, X_w_train, X_env_train, y_train)
            val_ds = MultiModalDataset(X_geno_val, X_w_val, X_env_val, y_val) if len(val_pheno) > 0 else None
            test_ds = MultiModalDataset(X_geno_test, X_w_test, X_env_test, y_test)

            start_time = time.perf_counter()
            train_result = train_model(
                model, train_ds, val_ds,
                epochs=epochs, batch_size=batch_size, lr=lr,
                weight_decay=weight_decay, early_stopping_patience=patience,
                gradient_clip_norm=gradient_clip_norm,
                rank_weight=rank_w, device=device,
                num_workers=num_workers, log_every=log_every, amp=amp,
            )

            y_pred = predict(model, test_ds, batch_size=batch_size * 2, device=device)
            y_pred_pheno = y_pred + test_main
            elapsed = time.perf_counter() - start_time

            m = metrics_dict(test_pheno["phenotype_value"].to_numpy(), y_pred_pheno)
            m["model"] = "ResidualGxEFormer"
            m["ablation"] = name
            m["split_type"] = split_type
            m["seed"] = int(seed)
            all_metrics.append(m)
            print(f"    phenotype pearson={m['pearson']:.4f} rmse={m['rmse']:.4f}")

            pred_df = build_prediction_frame(
                test_pheno["sample_id"].to_numpy(),
                test_pheno["phenotype_value"].to_numpy(),
                y_pred_pheno,
                split_type=split_type,
                seed=int(seed),
                model="ResidualGxEFormer",
                target="phenotype",
                extra_columns={
                    "ablation": name,
                    "y_true_residual_or_direct": y_test,
                    "y_pred_residual_or_direct": y_pred,
                    "main_prediction": test_main,
                },
            )
            pred_path = write_prediction_artifact(
                pred_df,
                out_dir / "predictions" / name / str(split_type) / f"seed{seed}",
            )

            manifest = {
                "schema_version": 1,
                "script": "scripts/06_run_ablation.py",
                "ablation": name,
                "split_type": split_type,
                "seed": int(seed),
                "n_train": int(len(train_pheno)),
                "n_val": int(len(val_pheno)),
                "n_test": int(len(test_pheno)),
                "target_mode": "residual" if use_residual else "phenotype",
                "feature_dim": {
                    "genotype": int(n_markers),
                    "weather": int(weather_dim),
                    "static_env": int(static_dim),
                },
                "preprocessing": preprocessor.to_manifest(),
                "outputs": {
                    "predictions": str(pred_path),
                },
                "training": {
                    "epochs_requested": int(epochs),
                    "batch_size": int(batch_size),
                    "learning_rate": float(lr),
                    "weight_decay": float(weight_decay),
                    "early_stopping_patience": int(patience),
                    "gradient_clip_norm": float(gradient_clip_norm),
                    "rank_loss_weight": float(rank_w),
                    "device": device,
                    "device_report": torch_device_report(device),
                    "num_workers": int(num_workers),
                    "log_every": int(log_every),
                    "amp": bool(amp),
                    "best_epoch": int(train_result["best_epoch"]),
                    "best_val_loss": float(train_result["best_val_loss"]),
                },
            }
            manifest_dir = out_dir / "manifests" / name / split_type / f"seed{seed}"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            (manifest_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8"
            )
            run_manifests.append(manifest)
            runtime_records.append({
                "model": "ResidualGxEFormer",
                "ablation": name,
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
                "best_epoch": int(train_result["best_epoch"]),
            })

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        write_table(metrics_df, out_dir / "ablation_metrics.csv")
        (out_dir / "ablation_metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf-8")
        (out_dir / "run_manifests.json").write_text(json.dumps(run_manifests, indent=2, default=str), encoding="utf-8")
        write_runtime_artifact(build_runtime_frame(runtime_records), out_dir)
        print(f"\nAblation metrics written to {out_dir / 'ablation_metrics.csv'}")

        print("\nSummary (phenotype pearson r, mean across splits):")
        for name, grp in metrics_df.groupby("ablation"):
            p = grp["pearson"].dropna()
            if len(p) > 0:
                print(f"  {name:20s}: {p.mean():.4f} +/- {p.std():.4f}")
    else:
        print("No metrics generated.")


if __name__ == "__main__":
    main()
