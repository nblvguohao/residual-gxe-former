"""Fair comparison: same markers, same splits, all models."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import (
    fit_gblup_efficient, fit_random_forest,
    fit_ridge_rrblup_like, mean_baseline, fit_xgboost, fit_lightgbm,
)
from residual_gxe.models.deep import ResidualGxEFormer
from residual_gxe.models.mixed_model import fit_additive_main_effects
from residual_gxe.training.trainer import (
    MultiModalDataset, _build_feature_arrays, train_model, predict,
)

MARKERS = 10000
MARKERS_BASELINE = 5000  # fewer markers for sklearn baselines (speed)
EPOCHS = 30
HIDDEN_DIM = 64
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_TYPES = ["leave_genotype", "leave_environment", "random"]
N_SEEDS = 1  # run 1 seed per split type for speed

print(f"Device: {DEVICE}")
print(f"Markers: {MARKERS} (deep) / {MARKERS_BASELINE} (baselines), Epochs: {EPOCHS}, Hidden: {HIDDEN_DIM}")

# ---- Load data ----
pheno = read_table(ROOT / "data/processed/g2f/phenotype.parquet")
geno_raw = read_table(ROOT / "data/processed/g2f/genotype.parquet")
env_raw = read_table(ROOT / "data/processed/g2f/environment.parquet")
weather_raw = None
wp = ROOT / "data/processed/g2f/weather_daily.parquet"
if wp.exists():
    weather_raw = read_table(wp)
splits_all = read_table(ROOT / "data/processed/g2f/splits/splits.parquet")
residuals = read_table(ROOT / "data/processed/g2f/residual_targets/residual_targets.parquet")

# Build genotype wide matrix (once, shared by all models)
print("Building genotype matrix...")
t0 = time.time()
all_markers = geno_raw["marker_id"].unique()
if MARKERS >= len(all_markers):
    selected = all_markers
else:
    # Fast variance-based selection: pivot a small random sample first,
    # then select top MARKERS by variance in the wide matrix
    rng = np.random.default_rng(42)
    subset_markers = rng.choice(all_markers, size=min(30000, len(all_markers)), replace=False)
    geno_subset = geno_raw[geno_raw["marker_id"].isin(subset_markers)]
    geno_wide_sample = geno_subset.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
    marker_var = geno_wide_sample.var().sort_values(ascending=False)
    selected = marker_var.index[:MARKERS].tolist()
    print(f"  Selected top {MARKERS}/{len(marker_var)} markers by variance (range: {marker_var.iloc[0]:.3f} - {marker_var.iloc[MARKERS-1]:.3f})")
geno_sub = geno_raw[geno_raw["marker_id"].isin(selected)]
geno_wide = geno_sub.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean()).reset_index()
print(f"  {geno_wide.shape[0]} genotypes x {geno_wide.shape[1]-1} markers ({time.time()-t0:.1f}s)")

all_results = []

for split_type in SPLIT_TYPES:
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

        # Build features
        Xg_tr, Xw_tr, Xe_tr = _build_feature_arrays(train_p, geno_wide, env_raw, weather_raw, max_markers=MARKERS)
        Xg_te, Xw_te, Xe_te = _build_feature_arrays(test_p, geno_wide, env_raw, weather_raw, max_markers=MARKERS)
        y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
        y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)

        # Feature matrix for sklearn baselines: concat genotype + env (fewer markers for speed)
        Xg_tr_b, Xw_tr_b, Xe_tr_b = _build_feature_arrays(train_p, geno_wide, env_raw, weather_raw, max_markers=MARKERS_BASELINE)
        Xg_te_b, Xw_te_b, Xe_te_b = _build_feature_arrays(test_p, geno_wide, env_raw, weather_raw, max_markers=MARKERS_BASELINE)
        X_tr = np.column_stack([Xg_tr_b, Xe_tr_b])
        X_te = np.column_stack([Xg_te_b, Xe_te_b])

        # ---- Baselines ----
        baseline_fns = [
            ("GlobalMean", lambda: mean_baseline(train_p, test_p)),
            ("Ridge", lambda: fit_ridge_rrblup_like(X_tr, y_tr, X_te)),
            ("RandomForest", lambda: fit_random_forest(X_tr, y_tr, X_te, n_estimators=200)),
            ("GBLUP_Efficient", lambda: fit_gblup_efficient(Xg_tr_b, y_tr, Xg_te_b)),
        ]

        # Add optional models
        try:
            import xgboost
            baseline_fns.append(("XGBoost", lambda: fit_xgboost(X_tr, y_tr, X_te, n_estimators=200)))
        except ImportError:
            pass
        try:
            import lightgbm
            baseline_fns.append(("LightGBM", lambda: fit_lightgbm(X_tr, y_tr, X_te, n_estimators=200)))
        except ImportError:
            pass

        # GBLUP already in baseline_fns above

        for name, fn in baseline_fns:
            try:
                t0 = time.time()
                result = fn()
                dt = time.time() - t0
                m = metrics_dict(y_te, result.predictions)
                m["model"] = name
                m["split_type"] = split_type
                m["seed"] = int(seed)
                m["time_s"] = round(dt, 1)
                all_results.append(m)
                print(f"  {name:20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.1f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  {name:20s}: FAILED — {str(e)[:80]}")

        # ---- Residual target (additive main effects) ----
        effects = fit_additive_main_effects(train_p)
        main_pred = effects.predict_main_effects(test_p)
        m = metrics_dict(y_te, main_pred.values)
        m["model"] = "AdditiveMainEffects"
        m["split_type"] = split_type
        m["seed"] = int(seed)
        m["time_s"] = 0.0
        all_results.append(m)
        print(f"  {'AdditiveMainEffects':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.1f}")

        # ---- ResidualGxEFormer (residual learning) ----
        res_group = residuals[(residuals["split_type"] == split_type) & (residuals["seed"] == seed)]
        if len(res_group) > 0:
            train_t = train_p.merge(res_group[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
            test_t = test_p.merge(res_group[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
            val_t = val_p.merge(res_group[["sample_id", "residual_target"]], on="sample_id", how="left")
            y_tr_r = train_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
            y_te_r = test_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
            y_val_r = val_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
            test_main = test_t["main_prediction"].fillna(0).to_numpy(dtype=np.float32)

            n_markers = Xg_tr.shape[1]
            weather_dim = Xw_tr.shape[2]
            base_static_dim = Xe_tr.shape[1]

            # Build main effect features for use with _build_feature_arrays
            train_main = train_t["main_prediction"].fillna(0).to_numpy(dtype=np.float32)

            for variant_name, use_residual, use_rank, use_main_feat in [
                ("ResidualGxEFormer_full", True, True, False),
                ("ResidualGxEFormer_noResid", False, True, False),
                ("ResidualGxEFormer_noRank", True, False, False),
                ("ResidualGxEFormer_mainEffects", True, True, True),
            ]:
                try:
                    t0 = time.time()
                    if use_main_feat:
                        Xg_tr_v, Xw_tr_v, Xe_tr_v = _build_feature_arrays(train_p, geno_wide, env_raw, weather_raw, max_markers=MARKERS, main_effects=train_main)
                        Xg_te_v, Xw_te_v, Xe_te_v = _build_feature_arrays(test_p, geno_wide, env_raw, weather_raw, max_markers=MARKERS, main_effects=test_main)
                        sdim = Xe_tr_v.shape[1]
                    else:
                        Xg_tr_v, Xw_tr_v, Xe_tr_v = Xg_tr, Xw_tr, Xe_tr
                        Xg_te_v, Xw_te_v, Xe_te_v = Xg_te, Xw_te, Xe_te
                        sdim = static_dim

                    model = ResidualGxEFormer(n_markers=n_markers, weather_dim=weather_dim, static_env_dim=sdim, hidden_dim=HIDDEN_DIM, patch_size=64, dropout=0.15)

                    if use_residual:
                        yt, yv = y_tr_r, y_val_r
                    else:
                        yt, yv = y_tr, val_p["phenotype_value"].to_numpy(dtype=np.float32)
                    train_ds = MultiModalDataset(Xg_tr_v, Xw_tr_v, Xe_tr_v, yt)
                    val_ds = MultiModalDataset(
                        *[_build_feature_arrays(val_p, geno_wide, env_raw, weather_raw, max_markers=MARKERS)[i] for i in range(3)],
                        yv,
                    ) if len(val_p) > 0 else None
                    test_ds = MultiModalDataset(Xg_te_v, Xw_te_v, Xe_te_v, y_te_r if use_residual else y_te)

                    rank_w = 0.05 if use_rank else 0.0
                    train_model(model, train_ds, val_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-4, early_stopping_patience=12, rank_weight=rank_w, device=DEVICE)
                    y_pred = predict(model, test_ds, batch_size=BATCH_SIZE*2, device=DEVICE)

                    if use_residual:
                        y_pred_final = y_pred + test_main
                    else:
                        y_pred_final = y_pred

                    dt = time.time() - t0
                    m = metrics_dict(y_te, y_pred_final)
                    m["model"] = variant_name
                    m["split_type"] = split_type
                    m["seed"] = int(seed)
                    m["time_s"] = round(dt, 1)
                    all_results.append(m)
                    print(f"  {variant_name:20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.1f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  {variant_name:20s}: FAILED — {str(e)[:80]}")

# ---- Summary ----
df = pd.DataFrame(all_results)
out = ROOT / "outputs" / "comparison"
out.mkdir(parents=True, exist_ok=True)
write_table(df, out / "comparison_results.csv")
(out / "comparison_results.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

print(f"\n{'='*80}")
print("SUMMARY: Pearson r by model (mean across all splits)")
print(f"{'='*80}")
print(f"{'Model':<30s} {'Pearson':>8s} {'RMSE':>8s} {'SelGain@10%':>12s} {'Time':>8s}")
print("-"*70)
for model, grp in df.groupby("model"):
    p = grp["pearson"].dropna().mean()
    r = grp["rmse"].dropna().mean()
    sg = grp["selection_gain_at_10pct"].dropna().mean()
    t = grp["time_s"].mean()
    print(f"{model:<30s} {p:>8.4f} {r:>8.1f} {sg:>12.2f} {t:>7.1f}s")
