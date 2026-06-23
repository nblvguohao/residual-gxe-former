"""Final benchmark: 50K markers + weather aggregation (fast feature building)."""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import fit_xgboost
from residual_gxe.models.mixed_model import fit_additive_main_effects

OUT = ROOT / "data/processed/g2f"
print("Loading data...")
t0 = time.time()

pheno = read_table(OUT / "phenotype.parquet")
weather = read_table(OUT / "weather_daily.parquet")
splits = read_table(OUT / "splits/splits.parquet")

# Load genotype (50K if exists, else 2.4K)
g50k_path = OUT / "genotype_50k_wide.parquet"
if g50k_path.exists():
    geno_wide = pd.read_parquet(g50k_path)
    print(f"Using 50K markers: {geno_wide.shape}")
else:
    genotype = read_table(OUT / "genotype.parquet")
    geno_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
    geno_wide = geno_wide.fillna(geno_wide.mean()).reset_index()
    print(f"Using 2.4K markers: {geno_wide.shape}")

print(f"Data loaded: {time.time()-t0:.1f}s")

# ---- Pre-compute weather stage aggregates per environment (FAST!) ----
print("Pre-computing weather aggregates per environment...")
t0 = time.time()

env_weather_agg = {}
for env_id, grp in weather.groupby("environment_id"):
    grp_sorted = grp.sort_values("date")
    cols = ["tmax", "tmin", "tmean", "precipitation", "solar_radiation", "relative_humidity"]
    vals = grp_sorted[cols].fillna(0).to_numpy(dtype=np.float32)
    n = len(vals)
    if n >= 30:
        s1 = vals[:10].mean(axis=0)
        s2 = vals[10:20].mean(axis=0)
        s3 = vals[20:30].mean(axis=0)
    elif n >= 20:
        s1 = vals[:n//3].mean(axis=0)
        s2 = vals[n//3:2*n//3].mean(axis=0)
        s3 = vals[2*n//3:].mean(axis=0)
    elif n >= 10:
        mid = n // 2
        s1 = vals[:mid].mean(axis=0)
        s2 = vals[mid:].mean(axis=0)
        s3 = np.zeros(6, dtype=np.float32)
    elif n > 0:
        s1 = vals.mean(axis=0)
        s2 = np.zeros(6, dtype=np.float32)
        s3 = np.zeros(6, dtype=np.float32)
    else:
        s1 = s2 = s3 = np.zeros(6, dtype=np.float32)
    env_weather_agg[env_id] = np.concatenate([s1, s2, s3])  # 18-dim

print(f"  {len(env_weather_agg)} environments ({time.time()-t0:.1f}s)")

# ---- Build genotype index once ----
marker_cols = [c for c in geno_wide.columns if c != "genotype_id"]
geno_idx = {gid: i for i, gid in enumerate(geno_wide["genotype_id"])}
geno_mat = geno_wide[marker_cols].to_numpy(dtype=np.float32)  # [n_genos, n_markers]
geno_mean = geno_mat.mean(axis=0)

# ---- Fast feature builder (numpy indexing, no merge!) ----
def build_features_fast(pheno_df, add_lag=False, train_pheno=None):
    """Build features using pre-computed numpy arrays — O(n) memory."""
    n = len(pheno_df)

    # Genotype: direct numpy indexing
    Xg = np.zeros((n, len(marker_cols)), dtype=np.float32)
    for i, gid in enumerate(pheno_df["genotype_id"]):
        idx = geno_idx.get(gid)
        Xg[i] = geno_mat[idx] if idx is not None else geno_mean

    # Weather: lookup pre-computed aggregates
    Xw = np.zeros((n, 18), dtype=np.float32)
    for i, env_id in enumerate(pheno_df["environment_id"]):
        Xw[i] = env_weather_agg.get(env_id, np.zeros(18, dtype=np.float32))

    # Static env features
    Xe = np.zeros((n, 4), dtype=np.float32)
    Xe[:, 0] = pheno_df["year"].to_numpy(dtype=np.float32)
    loc_counts = pheno_df["location_id"].value_counts()
    Xe[:, 1] = pheno_df["location_id"].map(loc_counts).fillna(0).to_numpy(dtype=np.float32)
    env_counts = pheno_df["environment_id"].value_counts()
    Xe[:, 2] = pheno_df["environment_id"].map(env_counts).fillna(0).to_numpy(dtype=np.float32)

    base = np.column_stack([Xg, Xe, Xw])

    # Lag feature
    if add_lag and train_pheno is not None:
        tp = train_pheno.copy()
        lag_map = {}
        for (loc, yr), grp in tp.groupby(["location_id", "year"]):
            lag_map[(loc, yr)] = grp["phenotype_value"].mean()
        lag_vals = np.array([
            lag_map.get((r["location_id"], r["year"]-1), 0.0)
            for _, r in pheno_df.iterrows()
        ], dtype=np.float32)
        base = np.column_stack([base, lag_vals])

    return base


# ---- Run benchmark ----
all_results = []

for st in ["leave_year", "leave_environment"]:
    st_splits = splits[splits["split_type"] == st]
    for seed in sorted(st_splits["seed"].unique())[:1]:
        sg = st_splits[st_splits["seed"] == seed]
        train_ids = set(sg[sg["split"] == "train"]["sample_id"])
        test_ids = set(sg[sg["split"] == "test"]["sample_id"])
        train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
        test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()
        y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
        y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)

        print(f"\n{'='*50}")
        print(f"  {st} seed={seed} train={len(train_p)} test={len(test_p)}")

        # Build features (no lag)
        t0 = time.time()
        X_tr = build_features_fast(train_p, add_lag=False, train_pheno=train_p)
        X_te = build_features_fast(test_p, add_lag=False, train_pheno=train_p)
        build_t = time.time() - t0
        n_feat = X_tr.shape[1]
        print(f"  Features: {n_feat} dims ({build_t:.1f}s)")

        # XGBoost
        t0 = time.time()
        r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
        dt = time.time() - t0
        m = metrics_dict(y_te, r.predictions)
        m.update(model=f"XGBoost_{n_feat}feat", split_type=st, seed=int(seed), time_s=round(dt,1))
        all_results.append(m)
        print(f"  XGBoost:        pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} ({dt:.1f}s)")

        # XGBoost with lag
        X_tr_lag = build_features_fast(train_p, add_lag=True, train_pheno=train_p)
        X_te_lag = build_features_fast(test_p, add_lag=True, train_pheno=train_p)
        t0 = time.time()
        r = fit_xgboost(X_tr_lag, y_tr, X_te_lag, n_estimators=300)
        m = metrics_dict(y_te, r.predictions)
        m.update(model="XGBoost+lag", split_type=st, seed=int(seed), time_s=round(time.time()-t0,1))
        all_results.append(m)
        print(f"  XGBoost+lag:    pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

        # Baseline
        effects = fit_additive_main_effects(train_p)
        main_pred = effects.predict_main_effects(test_p)
        m = metrics_dict(y_te, main_pred.values)
        m.update(model="AdditiveMainEffects", split_type=st, seed=int(seed), time_s=0)
        all_results.append(m)
        print(f"  AdditiveMain:   pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# ---- Compare 2.4K vs 50K on same task ----
if g50k_path.exists():
    # Also run with 2.4K for direct comparison
    genotype_ref = read_table(OUT / "genotype.parquet")
    geno_2k = genotype_ref.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
    geno_2k = geno_2k.fillna(geno_2k.mean()).reset_index()

    st_splits = splits[(splits["split_type"] == "leave_year") & (splits["seed"] == 1234)]
    train_ids = set(st_splits[st_splits["split"] == "train"]["sample_id"])
    test_ids = set(st_splits[st_splits["split"] == "test"]["sample_id"])
    train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
    test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()
    y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
    y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)

    for label, gw in [("2.4K", geno_2k), ("50K", geno_wide)]:
        nm = len([c for c in gw.columns if c != "genotype_id"])
        t0 = time.time()
        X_tr = build_features_fast(train_p, gw, add_lag=False)
        X_te = build_features_fast(test_p, gw, add_lag=False)
        bt = time.time() - t0
        r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
        m = metrics_dict(y_te, r.predictions)
        m.update(model=f"XGBoost_{label}_markers", split_type="leave_year", seed=1234, time_s=round(time.time()-t0-bt,1))
        all_results.append(m)
        print(f"\n  {label} markers: n_feat={nm} pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} (build={bt:.1f}s)")

# ---- Summary ----
df = pd.DataFrame(all_results)
out_dir = ROOT / "outputs" / "final"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "results.csv", index=False)

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
for model in df["model"].unique():
    grp = df[df["model"] == model]
    p = grp["pearson"].mean()
    r = grp["rmse"].mean()
    print(f"  {model:<30s}: pearson={p:.4f} rmse={r:.2f}")
