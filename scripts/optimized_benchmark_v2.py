"""Optimized benchmark v2: cache features, focus on leave_year + leave_environment."""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import fit_xgboost, mean_baseline
from residual_gxe.models.mixed_model import fit_additive_main_effects
from residual_gxe.training.trainer import _build_feature_arrays

OUT = ROOT / "data/processed/g2f"
N_SEEDS = 2

pheno = read_table(OUT / "phenotype.parquet")
genotype = read_table(OUT / "genotype.parquet")
env_raw = read_table(OUT / "environment.parquet")
weather_raw = read_table(OUT / "weather_daily.parquet")
splits = read_table(OUT / "splits/splits.parquet")

# Build genotype wide (use existing 2.4K)
geno_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean()).reset_index()
n_markers = len([c for c in geno_wide.columns if c != "genotype_id"])
print(f"Markers: {n_markers}")

all_results = []

for st in ["leave_year", "leave_environment"]:  # most important
    st_splits = splits[splits["split_type"] == st]
    for seed in sorted(st_splits["seed"].unique())[:N_SEEDS]:
        sg = st_splits[st_splits["seed"] == seed]
        train_ids = set(sg[sg["split"] == "train"]["sample_id"])
        test_ids = set(sg[sg["split"] == "test"]["sample_id"])
        train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
        test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()
        y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
        y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)

        print(f"\n{'='*50}")
        print(f"  {st} seed={seed} train={len(train_p)} test={len(test_p)}")

        # Build base features ONCE
        t0 = time.time()
        Xg_tr, Xw_tr, Xe_tr = _build_feature_arrays(train_p, geno_wide, env_raw, weather_raw, max_markers=n_markers)
        Xg_te, Xw_te, Xe_te = _build_feature_arrays(test_p, geno_wide, env_raw, weather_raw, max_markers=n_markers)

        # Weather aggregation
        def agg_weather(Xw):
            s1, s2, s3 = Xw[:,:10,:].mean(1), Xw[:,10:20,:].mean(1), Xw[:,20:30,:].mean(1)
            return np.column_stack([s1, s2, s3])

        Xw_tr_agg = agg_weather(Xw_tr)
        Xw_te_agg = agg_weather(Xw_te)

        # Lag features
        train_p_copy = train_p.copy()
        train_p_copy["prev_year"] = train_p_copy["year"] - 1
        lag_map = {}
        for (loc, yr), grp in train_p_copy.groupby(["location_id", "year"]):
            lag_map[(loc, yr)] = grp["phenotype_value"].mean()

        lag_tr = np.array([lag_map.get((r["location_id"], r["year"]-1), 0.0) for _, r in train_p.iterrows()], dtype=np.float32)
        lag_te = np.array([lag_map.get((r["location_id"], r["year"]-1), 0.0) for _, r in test_p.iterrows()], dtype=np.float32)

        print(f"  Features built: {time.time()-t0:.1f}s")

        # Variant 1: G+W_agg3
        X1_tr = np.column_stack([Xg_tr, Xe_tr, Xw_tr_agg])
        X1_te = np.column_stack([Xg_te, Xe_te, Xw_te_agg])

        # Variant 2: G+W_agg3+lag
        X2_tr = np.column_stack([Xg_tr, Xe_tr, Xw_tr_agg, lag_tr])
        X2_te = np.column_stack([Xg_te, Xe_te, Xw_te_agg, lag_te])

        for feat_name, X_tr, X_te in [
            ("G+W_agg3", X1_tr, X1_te),
            ("G+W_agg3+lag", X2_tr, X2_te),
        ]:
            t0 = time.time()
            r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
            dt = time.time() - t0
            m = metrics_dict(y_te, r.predictions)
            m.update(model=feat_name, split_type=st, seed=int(seed), time_s=round(dt,1))
            all_results.append(m)
            print(f"  {feat_name:20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} ({dt:.1f}s)")

        # Baselines
        effects = fit_additive_main_effects(train_p)
        main_pred = effects.predict_main_effects(test_p)
        m = metrics_dict(y_te, main_pred.values)
        m.update(model="AdditiveMainEffects", split_type=st, seed=int(seed), time_s=0)
        all_results.append(m)
        print(f"  {'AdditiveMainEffects':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# Summary
df = pd.DataFrame(all_results)
out_dir = ROOT / "outputs/optimized"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "results_v2.csv", index=False)

print(f"\n{'='*60}")
print("OPTIMIZED RESULTS")
print(f"{'='*60}")
for st in ["leave_year", "leave_environment"]:
    print(f"\n{st}:")
    st_df = df[df["split_type"] == st]
    for model in st_df["model"].unique():
        grp = st_df[st_df["model"] == model]
        print(f"  {model:<25s}: pearson={grp['pearson'].mean():.4f} rmse={grp['rmse'].mean():.2f}")
