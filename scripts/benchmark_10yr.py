"""Benchmark on full 10-year G2F competition data (164K samples, 2,425 markers)."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import (
    fit_gblup_efficient, fit_random_forest, fit_ridge_rrblup_like,
    mean_baseline, fit_xgboost, fit_lightgbm,
)
from residual_gxe.models.deep import ResidualGxEFormer
from residual_gxe.models.mixed_model import fit_additive_main_effects
from residual_gxe.training.trainer import (
    MultiModalDataset, _build_feature_arrays, train_model, predict,
)

EPOCHS = 40
HIDDEN_DIM = 64
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEEDS = 1  # fast mode

print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}, Hidden: {HIDDEN_DIM}")

OUT = ROOT / "data" / "processed" / "g2f"

# Load data
pheno = read_table(OUT / "phenotype.parquet")
geno_raw = read_table(OUT / "genotype.parquet")
env_raw = read_table(OUT / "environment.parquet")
weather_raw = read_table(OUT / "weather_daily.parquet")
splits_all = read_table(OUT / "splits" / "splits.parquet")
residuals = read_table(OUT / "residual_targets" / "residual_targets.parquet")

# Build genotype wide matrix
print("Building genotype matrix...")
t0 = time.time()
geno_wide = geno_raw.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean()).reset_index()
n_markers = len([c for c in geno_wide.columns if c != "genotype_id"])
print(f"  {geno_wide.shape[0]} genotypes x {n_markers} markers ({time.time()-t0:.1f}s)")

all_results = []
split_types = sorted(splits_all["split_type"].unique())
print(f"Split types: {split_types}")

for split_type in split_types:
    st_splits = splits_all[splits_all["split_type"] == split_type]
    seeds = sorted(st_splits["seed"].unique())

    for seed in seeds[:N_SEEDS]:
        sg = st_splits[st_splits["seed"] == seed]
        train_ids = set(sg.loc[sg["split"] == "train", "sample_id"])
        test_ids = set(sg.loc[sg["split"] == "test", "sample_id"])
        val_ids = set(sg.loc[sg["split"] == "val", "sample_id"])

        train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
        test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()
        val_p = pheno[pheno["sample_id"].isin(val_ids)].copy()

        print(f"\n{'='*50}")
        print(f"  {split_type} seed={seed}  train={len(train_p)} test={len(test_p)}")
        print(f"{'='*50}")

        # Build features (genotype only for baselines, full for deep)
        Xg_tr, Xw_tr, Xe_tr = _build_feature_arrays(train_p, geno_wide, env_raw, weather_raw)
        Xg_te, Xw_te, Xe_te = _build_feature_arrays(test_p, geno_wide, env_raw, weather_raw)
        y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
        y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)

        X_tr = np.column_stack([Xg_tr, Xe_tr])
        X_te = np.column_stack([Xg_te, Xe_te])

        # ---- Baselines ----
        for name, fn in [
            ("GlobalMean", lambda: mean_baseline(train_p, test_p)),
            ("Ridge", lambda: fit_ridge_rrblup_like(X_tr, y_tr, X_te)),
            ("GBLUP_Efficient", lambda: fit_gblup_efficient(Xg_tr, y_tr, Xg_te)),
        ]:
            try:
                t0_s = time.time()
                result = fn()
                dt = time.time() - t0_s
                m = metrics_dict(y_te, result.predictions)
                m.update(model=name, split_type=split_type, seed=int(seed), time_s=round(dt, 1))
                all_results.append(m)
                print(f"  {name:20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  {name:20s}: FAILED - {e}")

        try:
            import xgboost
            t0_s = time.time()
            r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=200)
            dt = time.time() - t0_s
            m = metrics_dict(y_te, r.predictions)
            m.update(model="XGBoost", split_type=split_type, seed=int(seed), time_s=round(dt,1))
            all_results.append(m)
            print(f"  {'XGBoost':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")
        except Exception as e:
            pass

        # ---- AdditiveMainEffects ----
        effects = fit_additive_main_effects(train_p)
        main_pred = effects.predict_main_effects(test_p)
        m = metrics_dict(y_te, main_pred.values)
        m.update(model="AdditiveMainEffects", split_type=split_type, seed=int(seed), time_s=0.0)
        all_results.append(m)
        print(f"  {'AdditiveMainEffects':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

        # ---- ResidualGxEFormer ----
        res_group = residuals[(residuals["split_type"] == split_type) & (residuals["seed"] == seed)]
        if len(res_group) > 0:
            train_t = train_p.merge(res_group[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
            test_t = test_p.merge(res_group[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
            y_tr_r = train_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
            y_te_r = test_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
            test_main = test_t["main_prediction"].fillna(0).to_numpy(dtype=np.float32)

            try:
                t0_s = time.time()
                sdim = Xe_tr.shape[1]
                model = ResidualGxEFormer(
                    n_markers=n_markers, weather_dim=Xw_tr.shape[2],
                    static_env_dim=sdim, hidden_dim=HIDDEN_DIM,
                    patch_size=min(32, n_markers), dropout=0.15,
                )
                train_ds = MultiModalDataset(Xg_tr, Xw_tr, Xe_tr, y_tr_r)
                test_ds = MultiModalDataset(Xg_te, Xw_te, Xe_te, y_te_r)
                train_model(model, train_ds, None, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-4, early_stopping_patience=12, rank_weight=0.05, device=DEVICE)
                y_pred = predict(model, test_ds, batch_size=BATCH_SIZE*2, device=DEVICE) + test_main
                dt = time.time() - t0_s
                m = metrics_dict(y_te, y_pred)
                m.update(model="ResidualGxEFormer", split_type=split_type, seed=int(seed), time_s=round(dt,1))
                all_results.append(m)
                print(f"  {'ResidualGxEFormer':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  {'ResidualGxEFormer':20s}: FAILED - {e}")

# ---- Summary ----
df = pd.DataFrame(all_results)
out_dir = ROOT / "outputs" / "benchmark_10yr"
out_dir.mkdir(parents=True, exist_ok=True)
write_table(df, out_dir / "results.csv")
(out_dir / "results.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

print(f"\n{'='*80}")
print("BENCHMARK SUMMARY (Pearson r by model, mean across splits)")
print(f"{'='*80}")
print(f"{'Model':<25s} {'Pearson':>8s} {'RMSE':>8s} {'SelGain@10%':>12s} {'Time':>8s}")
print("-"*65)
for model in df["model"].unique():
    grp = df[df["model"] == model]
    p = grp["pearson"].dropna().mean()
    r = grp["rmse"].dropna().mean()
    sg = grp["selection_gain_at_10pct"].dropna().mean()
    t = grp["time_s"].mean()
    print(f"{model:<25s} {p:>8.4f} {r:>8.2f} {sg:>12.2f} {t:>7.1f}s")

# Per split type summary
print(f"\n{'='*80}")
print("Per Split Type:")
for st in split_types:
    print(f"\n  {st}:")
    st_df = df[df["split_type"] == st]
    for model in st_df["model"].unique():
        grp = st_df[st_df["model"] == model]
        p = grp["pearson"].dropna().mean()
        r = grp["rmse"].dropna().mean()
        print(f"    {model:<25s}: pearson={p:6.4f}  rmse={r:6.2f}")
