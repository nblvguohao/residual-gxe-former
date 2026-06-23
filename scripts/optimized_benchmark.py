"""Optimized benchmark: weather aggregation + lag features + environment clustering."""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import fit_xgboost, fit_ridge_rrblup_like, mean_baseline
from residual_gxe.models.mixed_model import fit_additive_main_effects

OUT = ROOT / "data/processed/g2f"
N_SEEDS = 2
print(f"Optimized benchmark: weather agg + lag + env cluster")

# ---- Load ----
pheno = read_table(OUT / "phenotype.parquet")
genotype = read_table(OUT / "genotype.parquet")
env_raw = read_table(OUT / "environment.parquet")
weather_raw = read_table(OUT / "weather_daily.parquet")
splits = read_table(OUT / "splits/splits.parquet")

# Genotype wide matrix
geno_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean()).reset_index()
n_markers = len([c for c in geno_wide.columns if c != "genotype_id"])

# ---- Feature builders ----
def build_weather_agg_features(pheno_df):
    """Weather aggregated by 3 growth stages, flattened with genotype+env."""
    from residual_gxe.training.trainer import _build_feature_arrays
    Xg, Xw, Xe = _build_feature_arrays(pheno_df, geno_wide, env_raw, weather_raw, max_markers=n_markers)
    stage1 = Xw[:, :10, :].mean(axis=1)
    stage2 = Xw[:, 10:20, :].mean(axis=1)
    stage3 = Xw[:, 20:30, :].mean(axis=1)
    return np.column_stack([Xg, Xe, stage1, stage2, stage3])

def build_lag_features(pheno_df, train_pheno):
    """Add lag yield (previous year same location mean)."""
    from residual_gxe.training.trainer import _build_feature_arrays
    Xg, Xw, Xe = _build_feature_arrays(pheno_df, geno_wide, env_raw, weather_raw, max_markers=n_markers)
    stage1 = Xw[:, :10, :].mean(axis=1)
    stage2 = Xw[:, 10:20, :].mean(axis=1)
    stage3 = Xw[:, 20:30, :].mean(axis=1)
    base = np.column_stack([Xg, Xe, stage1, stage2, stage3])

    # Lag: for each env, compute mean yield in previous year
    train_pheno = train_pheno.copy()
    train_pheno["prev_year"] = train_pheno["year"] - 1
    lag_map = {}
    for (loc, yr), grp in train_pheno.groupby(["location_id", "year"]):
        lag_map[(loc, yr)] = grp["phenotype_value"].mean()

    lag_values = np.zeros(len(pheno_df), dtype=np.float32)
    for i, (_, row) in enumerate(pheno_df.iterrows()):
        key = (row["location_id"], row["year"] - 1)
        lag_values[i] = lag_map.get(key, 0.0)

    return np.column_stack([base, lag_values])

def build_env_cluster_features(pheno_df, train_pheno, n_clusters=5):
    """Add environment cluster ID as one-hot."""
    from residual_gxe.training.trainer import _build_feature_arrays
    Xg, Xw, Xe = _build_feature_arrays(pheno_df, geno_wide, env_raw, weather_raw, max_markers=n_markers)
    stage1 = Xw[:, :10, :].mean(axis=1)
    stage2 = Xw[:, 10:20, :].mean(axis=1)
    stage3 = Xw[:, 20:30, :].mean(axis=1)
    base = np.column_stack([Xg, Xe, stage1, stage2, stage3])

    # Cluster environments based on weather profile
    from sklearn.cluster import KMeans
    env_weather = {}
    for env_id, grp in weather_raw.groupby("environment_id"):
        cols = ["tmax","tmin","precipitation","solar_radiation","relative_humidity"]
        vals = grp[cols].mean().fillna(0).to_numpy()
        env_weather[env_id] = vals

    env_ids = list(env_weather.keys())
    env_mat = np.array([env_weather[e] for e in env_ids])
    km = KMeans(n_clusters=min(n_clusters, len(env_ids)), random_state=42, n_init=5)
    clusters = km.fit_predict(env_mat)
    env_to_cluster = {e: c for e, c in zip(env_ids, clusters)}

    cluster_ids = np.array([env_to_cluster.get(e, 0) for e in pheno_df["environment_id"]])
    cluster_onehot = np.eye(min(n_clusters, len(env_ids)))[cluster_ids]

    return np.column_stack([base, cluster_onehot])

# ---- Run benchmark ----
all_results = []
split_types = sorted(splits["split_type"].unique())

for st in split_types:
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

        # Feature variants
        for feat_name, feat_fn in [
            ("G+W_raw", lambda p: build_weather_agg_features(p).shape),  # placeholder
        ]:
            pass

        # Build all features at once
        t0_build = time.time()
        X_tr_raw = build_weather_agg_features(train_p)
        X_te_raw = build_weather_agg_features(test_p)
        X_tr_lag = build_lag_features(train_p, train_p)
        X_te_lag = build_lag_features(test_p, train_p)
        X_tr_clust = build_env_cluster_features(train_p, train_p)
        X_te_clust = build_env_cluster_features(test_p, train_p)

        print(f"  Feature building: {time.time()-t0_build:.1f}s")
        print(f"    raw_agg: {X_tr_raw.shape[1]} | +lag: {X_tr_lag.shape[1]} | +cluster: {X_tr_clust.shape[1]}")

        # Test each feature set with XGBoost
        for feat_name, X_tr, X_te in [
            ("G+W_agg3", X_tr_raw, X_te_raw),
            ("G+W_agg3+lag", X_tr_lag, X_te_lag),
            ("G+W_agg3+cluster", X_tr_clust, X_te_clust),
        ]:
            try:
                t0 = time.time()
                r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
                dt = time.time() - t0
                m = metrics_dict(y_te, r.predictions)
                m.update(model=feat_name, split_type=st, seed=int(seed), time_s=round(dt,1))
                all_results.append(m)
                print(f"  {feat_name:22s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} ({dt:.1f}s)")
            except Exception as e:
                print(f"  {feat_name:22s}: FAILED - {e}")

        # Baselines
        effects = fit_additive_main_effects(train_p)
        main_pred = effects.predict_main_effects(test_p)
        m = metrics_dict(y_te, main_pred.values)
        m.update(model="AdditiveMainEffects", split_type=st, seed=int(seed), time_s=0)
        all_results.append(m)

# ---- Summary ----
df = pd.DataFrame(all_results)
out_dir = ROOT / "outputs/optimized"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "results.csv", index=False)

print(f"\n{'='*70}")
print("OPTIMIZED BENCHMARK SUMMARY")
print(f"{'='*70}")
for st in split_types:
    print(f"\n{st}:")
    st_df = df[df["split_type"] == st]
    for model in st_df["model"].unique():
        grp = st_df[st_df["model"] == model]
        p = grp["pearson"].mean()
        r = grp["rmse"].mean()
        print(f"  {model:<25s}: pearson={p:.4f} rmse={r:.2f}")

# Overall
print(f"\n{'Model':<25s} {'Pearson':>8s} {'RMSE':>8s}")
print("-"*45)
for model in df["model"].unique():
    grp = df[df["model"] == model]
    print(f"  {model:<25s} {grp['pearson'].mean():>8.4f} {grp['rmse'].mean():>8.2f}")
