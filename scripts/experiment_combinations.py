"""Systematic combination experiments on 10-year G2F data."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import fit_xgboost, mean_baseline
from residual_gxe.models.deep import ResidualGxEFormer
from residual_gxe.models.mixed_model import fit_additive_main_effects
from residual_gxe.training.trainer import (
    MultiModalDataset, _build_feature_arrays, train_model, predict,
)

OUT = ROOT / "data" / "processed" / "g2f"
EPOCHS, HIDDEN_DIM, BATCH = 30, 64, 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Hidden: {HIDDEN_DIM}")

# ---- Load data ----
pheno = read_table(OUT / "phenotype.parquet")
geno_raw = read_table(OUT / "genotype.parquet")
env_raw = read_table(OUT / "environment.parquet")
weather_raw = read_table(OUT / "weather_daily.parquet")
ec_data = pd.read_csv(ROOT / "data" / "raw" / "g2f" / "competition_2024" / "6_Training_EC_Data_2014_2023.csv")
splits_all = read_table(OUT / "splits" / "splits.parquet")
residuals = read_table(OUT / "residual_targets" / "residual_targets.parquet")

# ---- Helper: build genotype matrices ----
def build_geno_wide(n_markers: int) -> pd.DataFrame:
    all_m = geno_raw["marker_id"].unique()
    if n_markers >= len(all_m):
        sub = geno_raw
    else:
        marker_var = geno_raw.groupby("marker_id")["allele_dosage"].var().sort_values(ascending=False)
        top = marker_var.index[:n_markers]
        sub = geno_raw[geno_raw["marker_id"].isin(top)]
    gw = sub.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
    return gw.fillna(gw.mean()).reset_index()

print("Building genotype matrices (2.4K, 5K, 10K)...")
g2k = build_geno_wide(2425)
g5k = build_geno_wide(5000)
g10k = build_geno_wide(10000)

def nm(gw): return len([c for c in gw.columns if c != "genotype_id"])
print(f"  2.4K: {nm(g2k)} markers | 5K: {nm(g5k)} | 10K: {nm(g10k)}")

# ---- Helper: weather aggregation by growth stage ----
def build_weather_agg(pheno_df: pd.DataFrame, gw: pd.DataFrame) -> np.ndarray:
    """Aggregate weather into 3 growth stages: early(0-30d), mid(30-90d), late(90d+)."""
    Xg, Xw, Xe = _build_feature_arrays(pheno_df, gw, env_raw, weather_raw, max_markers=nm(gw))
    n = Xw.shape[0]
    # Xw: [n, 30, 6] — split into 3 stages: [0:10], [10:20], [20:30]
    stage1 = Xw[:, :10, :].mean(axis=1)  # early
    stage2 = Xw[:, 10:20, :].mean(axis=1)  # mid
    stage3 = Xw[:, 20:30, :].mean(axis=1)  # late
    Xw_agg = np.column_stack([stage1, stage2, stage3])  # [n, 18]
    return np.column_stack([Xg, Xe, Xw_agg])

# ---- Helper: add EC features ----
def build_with_ec(pheno_df: pd.DataFrame, gw: pd.DataFrame) -> np.ndarray:
    Xg, Xw, Xe = _build_feature_arrays(pheno_df, gw, env_raw, weather_raw, max_markers=nm(gw))
    # EC: environmental covariates per environment
    ec_num = ec_data.select_dtypes(include=[np.number])
    ec_feat = ec_num.set_index(ec_data["Env"]).reindex(pheno_df["environment_id"]).fillna(0).to_numpy(dtype=np.float32)
    return np.column_stack([Xg, Xe, ec_feat])

# ---- Helper: weather aggregated + EC ----
def build_weather_agg_ec(pheno_df: pd.DataFrame, gw: pd.DataFrame) -> np.ndarray:
    Xg, Xw, Xe = _build_feature_arrays(pheno_df, gw, env_raw, weather_raw, max_markers=nm(gw))
    stage1 = Xw[:, :10, :].mean(axis=1)
    stage2 = Xw[:, 10:20, :].mean(axis=1)
    stage3 = Xw[:, 20:30, :].mean(axis=1)
    Xw_agg = np.column_stack([stage1, stage2, stage3])
    ec_num = ec_data.select_dtypes(include=[np.number])
    ec_feat = ec_num.set_index(ec_data["Env"]).reindex(pheno_df["environment_id"]).fillna(0).to_numpy(dtype=np.float32)
    return np.column_stack([Xg, Xe, Xw_agg, ec_feat])

# ---- Prepare leave_year split ----
splits_ly = splits_all[(splits_all["split_type"] == "leave_year") & (splits_all["seed"] == 1234)]
train_ids = set(splits_ly[splits_ly["split"] == "train"]["sample_id"])
test_ids = set(splits_ly[splits_ly["split"] == "test"]["sample_id"])
train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()
y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)
print(f"\nleave_year split: train={len(train_p)} test={len(test_p)}")

# ---- Residual targets ----
res_g = residuals[(residuals["split_type"] == "leave_year") & (residuals["seed"] == 1234)]
train_t = train_p.merge(res_g[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
test_t = test_p.merge(res_g[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
y_tr_r = train_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
test_main = test_t["main_prediction"].fillna(0).to_numpy(dtype=np.float32)

all_results = []

# ============================================================
# EXPERIMENT 1: Feature Combinations (XGBoost, fast)
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT 1: Feature Engineering Impact (XGBoost, 2.4K markers)")
print(f"{'='*70}")

for feat_name, feat_fn in [
    ("G_only", lambda p: np.column_stack([
        _build_feature_arrays(p, g2k, None, None, max_markers=2425)[0],
        np.zeros((len(p), 1), dtype=np.float32)])),
    ("G+W_raw", lambda p: np.column_stack([
        _build_feature_arrays(p, g2k, env_raw, weather_raw, max_markers=2425)[0],
        _build_feature_arrays(p, g2k, env_raw, weather_raw, max_markers=2425)[2]])),
    ("G+W_agg3", lambda p: build_weather_agg(p, g2k)),
    ("G+W+EC", lambda p: build_with_ec(p, g2k)),
    ("G+W_agg3+EC", lambda p: build_weather_agg_ec(p, g2k)),
]:
    try:
        X_tr_f = feat_fn(train_p)
        X_te_f = feat_fn(test_p)
        t0 = time.time()
        r = fit_xgboost(X_tr_f, y_tr, X_te_f, n_estimators=300)
        dt = time.time() - t0
        m = metrics_dict(y_te, r.predictions)
        m.update(experiment="E1_features", variant=feat_name, time_s=round(dt, 1),
                 n_features=X_tr_f.shape[1])
        all_results.append(m)
        print(f"  {feat_name:18s}: feats={X_tr_f.shape[1]:5d}  pearson={m['pearson']:.4f}  rmse={m['rmse']:.2f}  ({dt:.1f}s)")
    except Exception as e:
        print(f"  {feat_name:18s}: FAILED - {type(e).__name__}: {e}")

# ============================================================
# EXPERIMENT 2: Marker Count (XGBoost, G+W features)
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT 2: Marker Count Impact (XGBoost, G+W)")
print(f"{'='*70}")

for label, gw in [("2.4K markers", g2k), ("5K markers", g5k), ("10K markers", g10k)]:
    nmk = nm(gw)
    Xg_tr, Xw_tr, Xe_tr = _build_feature_arrays(train_p, gw, env_raw, weather_raw, max_markers=nmk)
    Xg_te, Xw_te, Xe_te = _build_feature_arrays(test_p, gw, env_raw, weather_raw, max_markers=nmk)
    X_tr = np.column_stack([Xg_tr, Xe_tr])
    X_te = np.column_stack([Xg_te, Xe_te])
    try:
        t0 = time.time()
        r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
        m = metrics_dict(y_te, r.predictions)
        m.update(experiment="E2_markers", variant=label, time_s=round(time.time()-t0, 1), n_markers=nmk)
        all_results.append(m)
        print(f"  {label:18s}: markers={nmk:5d}  pearson={m['pearson']:.4f}  rmse={m['rmse']:.2f}")
    except Exception as e:
        print(f"  {label:18s}: FAILED - {e}")

# ============================================================
# EXPERIMENT 3: Architecture variants on best feature set
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT 3: Architecture Comparison (2.4K, G+W)")
print(f"{'='*70}")

Xg_tr, Xw_tr, Xe_tr = _build_feature_arrays(train_p, g2k, env_raw, weather_raw, max_markers=2425)
Xg_te, Xw_te, Xe_te = _build_feature_arrays(test_p, g2k, env_raw, weather_raw, max_markers=2425)
X_tr_s = np.column_stack([Xg_tr, Xe_tr])
X_te_s = np.column_stack([Xg_te, Xe_te])
n_markers, wdim, sdim = Xg_tr.shape[1], Xw_tr.shape[2], Xe_tr.shape[1]

# Baseline: AdditiveMainEffects
effects = fit_additive_main_effects(train_p)
main_pred = effects.predict_main_effects(test_p)
m = metrics_dict(y_te, main_pred.values)
m.update(experiment="E3_arch", variant="AdditiveMainEffects", time_s=0)
all_results.append(m)
print(f"  {'AdditiveMainEffects':25s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# Baseline: XGBoost
r = fit_xgboost(X_tr_s, y_tr, X_te_s, n_estimators=300)
m = metrics_dict(y_te, r.predictions)
m.update(experiment="E3_arch", variant="XGBoost_G+W", time_s=0)
all_results.append(m)
print(f"  {'XGBoost_G+W':25s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# Deep model variants
for variant_label, use_residual, multi_task in [
    ("RxEFormer_residual", True, False),
    ("RxEFormer_direct", False, False),
    ("RxEFormer_multitask", True, True),
]:
    try:
        t0 = time.time()
        model = ResidualGxEFormer(
            n_markers=n_markers, weather_dim=wdim, static_env_dim=sdim,
            hidden_dim=HIDDEN_DIM, patch_size=min(32, n_markers), dropout=0.15,
            multi_task=multi_task,
        )

        if use_residual:
            yt = y_tr_r
            yte = y_tr_r  # for test dataset construction
        else:
            yt = y_tr
            yte = y_tr

        if multi_task:
            yt_dual = np.column_stack([y_tr, y_tr_r])  # [pheno, resid]
            train_ds = MultiModalDataset(Xg_tr, Xw_tr, Xe_tr, yt_dual)
            test_ds = MultiModalDataset(Xg_te, Xw_te, Xe_te, y_tr_r)  # use residual target for eval
        else:
            train_ds = MultiModalDataset(Xg_tr, Xw_tr, Xe_tr, yt)
            test_ds = MultiModalDataset(Xg_te, Xw_te, Xe_te, y_tr_r if use_residual else y_tr)

        train_model(model, train_ds, None, epochs=EPOCHS, batch_size=BATCH, lr=3e-4,
                    early_stopping_patience=10, rank_weight=0.05, device=DEVICE,
                    phenotype_weight=0.5)

        y_pred = predict(model, test_ds, batch_size=BATCH*2, device=DEVICE)
        if use_residual:
            y_pred_final = y_pred + test_main
        else:
            y_pred_final = y_pred

        dt = time.time() - t0
        m = metrics_dict(y_te, y_pred_final)
        m.update(experiment="E3_arch", variant=variant_label, time_s=round(dt, 1))
        all_results.append(m)
        print(f"  {variant_label:25s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} ({dt:.1f}s)")
    except Exception as e:
        import traceback
        print(f"  {variant_label:25s}: FAILED - {e}")
        traceback.print_exc()

# ============================================================
# SUMMARY
# ============================================================
out_dir = ROOT / "outputs" / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(all_results).to_csv(out_dir / "combinations.csv", index=False)
(out_dir / "combinations.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

for exp in ["E1_features", "E2_markers", "E3_arch"]:
    sub = [r for r in all_results if r.get("experiment") == exp]
    if not sub: continue
    exp_names = {"E1_features": "Feature Engineering", "E2_markers": "Marker Count", "E3_arch": "Architecture"}
    print(f"\n{'='*70}")
    print(f"{exp_names[exp]}: {'leave_year' if exp != 'E3_arch' else 'leave_year'} split")
    print(f"{'Variant':<28s} {'Pearson':>8s} {'RMSE':>8s} {'Time':>8s}")
    print("-"*55)
    for r in sorted(sub, key=lambda x: -(x.get("pearson", -999) or -999)):
        print(f"  {r['variant']:<26s} {r.get('pearson', float('nan')):>8.4f} {r.get('rmse', 99):>8.2f} {r.get('time_s', 0):>7.1f}s")
