"""Causal-aware prediction model: physiological indices instead of raw weather."""
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

# Load data
pheno = read_table(OUT / "phenotype.parquet")
genotype = read_table(OUT / "genotype.parquet")
physio = read_table(OUT / "physiological_indices.parquet")
splits = read_table(OUT / "splits/splits.parquet")

# Genotype wide matrix
geno_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean()).reset_index()
marker_cols = [c for c in geno_wide.columns if c != "genotype_id"]

# Weather aggregation (baseline)
weather_raw = read_table(OUT / "weather_daily.parquet")
from residual_gxe.training.trainer import _build_feature_arrays

# ---- Causal feature set ----
# Top physiological indices correlated with yield (from causal discovery)
# Top physiological indices from causal discovery (only columns in physio indices file)
CAUSAL_FEATURES = [
    "et0_cum_late", "vpd_mean_mid", "gdd_max_mid", "gdd_cum_mid",
    "gdd_mean_mid", "gdd_cum_late", "vpd_mean_early", "vpd_max_mid",
    "drought_stress_cum_mid", "gdd_cum_early",
]
# Verify all features exist
physio_check = read_table(OUT / "physiological_indices.parquet")
for f in CAUSAL_FEATURES:
    if f not in physio_check.columns:
        print(f"WARNING: {f} not in physiological indices!")
available_features = [f for f in CAUSAL_FEATURES if f in physio_check.columns]
print(f"Using {len(available_features)} causal physiological features")

def build_physio_features(pheno_df):
    """Build features: genotype + physiological indices."""
    # Genotype
    Xg = pheno_df[["genotype_id"]].merge(geno_wide, on="genotype_id", how="left")
    Xg = Xg[marker_cols].fillna(0.0).to_numpy(dtype=np.float32)

    # Physiological indices (per environment)
    physio_idx = physio.set_index("environment_id")
    Xp = np.zeros((len(pheno_df), len(available_features)), dtype=np.float32)
    for i, env_id in enumerate(pheno_df["environment_id"]):
        if env_id in physio_idx.index:
            vals = physio_idx.loc[env_id, available_features]
            if isinstance(vals, pd.Series):
                vals = vals.fillna(0).to_numpy(dtype=np.float32)
            else:
                vals = np.array([vals], dtype=np.float32)
            Xp[i] = vals

    # Static env features
    Xe = np.zeros((len(pheno_df), 3), dtype=np.float32)
    Xe[:, 0] = pheno_df["year"].to_numpy(dtype=np.float32)
    loc_counts = pheno_df["location_id"].value_counts()
    Xe[:, 1] = pheno_df["location_id"].map(loc_counts).fillna(0).to_numpy(dtype=np.float32)
    env_counts = pheno_df["environment_id"].value_counts()
    Xe[:, 2] = pheno_df["environment_id"].map(env_counts).fillna(0).to_numpy(dtype=np.float32)

    return np.column_stack([Xg, Xe, Xp])

def build_raw_weather_features(pheno_df):
    """Build features: genotype + raw weather aggregation."""
    Xg, Xw, Xe = _build_feature_arrays(pheno_df, geno_wide, None, weather_raw, max_markers=len(marker_cols))
    s1, s2, s3 = Xw[:,:10,:].mean(1), Xw[:,10:20,:].mean(1), Xw[:,20:30,:].mean(1)
    Xw_agg = np.column_stack([s1, s2, s3])
    return np.column_stack([Xg, Xe, Xw_agg])

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

        for feat_name, build_fn in [
            ("W_agg3 (baseline)", build_raw_weather_features),
            ("Physio_indices (causal)", build_physio_features),
        ]:
            t0 = time.time()
            X_tr = build_fn(train_p)
            X_te = build_fn(test_p)
            bt = time.time() - t0

            r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
            m = metrics_dict(y_te, r.predictions)
            m.update(model=feat_name, split_type=st, seed=int(seed),
                     time_s=round(time.time()-t0,1), n_feat=X_tr.shape[1])
            all_results.append(m)
            print(f"  {feat_name:30s}: feat={X_tr.shape[1]:5d} pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} ({bt:.1f}s)")

        # Additive baseline
        effects = fit_additive_main_effects(train_p)
        main_pred = effects.predict_main_effects(test_p)
        m = metrics_dict(y_te, main_pred.values)
        m.update(model="AdditiveMainEffects", split_type=st, seed=int(seed), time_s=0, n_feat=0)
        all_results.append(m)
        print(f"  {'AdditiveMainEffects':30s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# ---- Summary ----
df = pd.DataFrame(all_results)
out_dir = ROOT.parent / "outputs" / "causal"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "prediction_results.csv", index=False)

print(f"\n{'='*60}")
print("CAUSAL PREDICTION RESULTS")
print(f"{'='*60}")
for st in ["leave_year", "leave_environment"]:
    print(f"\n{st}:")
    st_df = df[df["split_type"] == st]
    for model in st_df["model"].unique():
        grp = st_df[st_df["model"] == model]
        print(f"  {model:<30s}: pearson={grp['pearson'].mean():.4f} rmse={grp['rmse'].mean():.2f} feat={grp['n_feat'].mean():.0f}")
