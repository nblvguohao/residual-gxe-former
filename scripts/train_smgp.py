"""SMGP: Structured Multi-modal G×E Prediction framework.

Components:
  Layer 1: PCA genomic embeddings (512-dim) from 50K markers
  Layer 2: Causal physiological indices (15-dim) from weather
  Layer 3: Structured XGBoost with feature groups

Baselines: competition markers + weather aggregation, additive effects
"""
from __future__ import annotations

import sys, time
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
env_raw = read_table(OUT / "environment.parquet")
weather_raw = read_table(OUT / "weather_daily.parquet")
physio = read_table(OUT / "physiological_indices.parquet")
embeddings = pd.read_parquet(OUT / "genotype_embeddings_pca512.parquet")
splits = read_table(OUT / "splits/splits.parquet")
print(f"  Loaded: {time.time()-t0:.1f}s")

# Competition genotype matrix
genotype = read_table(OUT / "genotype.parquet")
geno_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean()).reset_index()

# Embedding columns
emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
emb_idx = embeddings.set_index("genotype_id")

# Causal physiological features
CAUSAL_F = ["et0_cum_late","vpd_mean_mid","gdd_max_mid","gdd_cum_mid","gdd_mean_mid",
            "gdd_cum_late","vpd_mean_early","vpd_max_mid","drought_stress_cum_mid","gdd_cum_early",
            "precip_cum_mid","wb_mean_mid","heat_stress_cum_mid","precip_cum_early","vpd_cum_early"]
physio_idx = physio.set_index("environment_id")

def build_smgp_features(pheno_df, weather_raw, env_raw):
    """Build SMGP features: embeddings + physio + weather agg."""
    n = len(pheno_df)

    # Layer 1: Genomic embeddings
    X_emb = np.zeros((n, len(emb_cols)), dtype=np.float32)
    for i, gid in enumerate(pheno_df["genotype_id"]):
        if gid in emb_idx.index:
            X_emb[i] = emb_idx.loc[gid, emb_cols].to_numpy(dtype=np.float32)

    # Layer 2: Physiological indices
    X_physio = np.zeros((n, len(CAUSAL_F)), dtype=np.float32)
    for i, env_id in enumerate(pheno_df["environment_id"]):
        if env_id in physio_idx.index:
            vals = physio_idx.loc[env_id, CAUSAL_F]
            if isinstance(vals, pd.Series):
                vals = vals.fillna(0).to_numpy(dtype=np.float32)
            else:
                vals = np.array([vals], dtype=np.float32)
            X_physio[i] = vals

    # Layer 3: Static env features
    X_env = np.zeros((n, 6), dtype=np.float32)
    X_env[:, 0] = pheno_df["year"].to_numpy(dtype=np.float32)
    loc_counts = pheno_df["location_id"].value_counts()
    X_env[:, 1] = pheno_df["location_id"].map(loc_counts).fillna(0).to_numpy(dtype=np.float32)
    env_counts = pheno_df["environment_id"].value_counts()
    X_env[:, 2] = pheno_df["environment_id"].map(env_counts).fillna(0).to_numpy(dtype=np.float32)

    # Weather aggregation (as additional environment features)
    from residual_gxe.training.trainer import _build_feature_arrays
    _, Xw, Xe = _build_feature_arrays(pheno_df, geno_wide, env_raw, weather_raw, max_markers=2425)
    s1, s2, s3 = Xw[:,:10,:].mean(1), Xw[:,10:20,:].mean(1), Xw[:,20:30,:].mean(1)
    X_env[:, 3:6] = Xe[:, :3]  # add env features from weather builder

    X_w = np.column_stack([s1, s2, s3, X_env])

    return np.column_stack([X_emb, X_physio, X_w])

def build_baseline_features(pheno_df):
    """Baseline: competition markers + weather aggregation."""
    from residual_gxe.training.trainer import _build_feature_arrays
    Xg, Xw, Xe = _build_feature_arrays(pheno_df, geno_wide, env_raw, weather_raw, max_markers=2425)
    s1, s2, s3 = Xw[:,:10,:].mean(1), Xw[:,10:20,:].mean(1), Xw[:,20:30,:].mean(1)
    return np.column_stack([Xg, Xe, s1, s2, s3])

# ---- Benchmark ----
all_results = []
for st in ["leave_year", "leave_environment"]:
    st_splits = splits[splits["split_type"] == st]
    seed = sorted(st_splits["seed"].unique())[0]
    sg = st_splits[st_splits["seed"] == seed]
    train_ids = set(sg[sg["split"]=="train"]["sample_id"])
    test_ids = set(sg[sg["split"]=="test"]["sample_id"])
    train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
    test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()
    y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
    y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)

    print(f"\n{'='*50}")
    print(f"  {st} seed={seed} train={len(train_p)} test={len(test_p)}")

    for label, build_fn in [
        ("SMGP (PCA512+Physio+W)", lambda p: build_smgp_features(p, weather_raw, env_raw)),
        ("Competition 2425+W_agg3", build_baseline_features),
    ]:
        t0 = time.time()
        X_tr = build_fn(train_p)
        X_te = build_fn(test_p)
        bt = time.time() - t0
        r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
        m = metrics_dict(y_te, r.predictions)
        m.update(model=label, split_type=st, seed=int(seed),
                 n_feat=X_tr.shape[1], time_s=round(time.time()-t0,1), build_s=round(bt,1))
        all_results.append(m)
        print(f"  {label:30s}: feat={X_tr.shape[1]:5d} pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} ({bt:.1f}s)")

    # Additive baseline
    effects = fit_additive_main_effects(train_p)
    main_pred = effects.predict_main_effects(test_p)
    m = metrics_dict(y_te, main_pred.values)
    m.update(model="AdditiveMainEffects", split_type=st, seed=int(seed), n_feat=0, time_s=0, build_s=0)
    all_results.append(m)
    print(f"  {'AdditiveMainEffects':30s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# Summary
df = pd.DataFrame(all_results)
out_dir = ROOT.parent / "outputs" / "smgp"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "results_v1.csv", index=False)

print(f"\n{'='*60}")
print("SMGP v1 RESULTS")
for st in ["leave_year", "leave_environment"]:
    print(f"\n{st}:")
    for model in df[df["split_type"]==st]["model"].unique():
        grp = df[(df["split_type"]==st)&(df["model"]==model)]
        print(f"  {model:<30s}: pearson={grp['pearson'].mean():.4f} rmse={grp['rmse'].mean():.2f} feat={grp['n_feat'].mean():.0f}")
