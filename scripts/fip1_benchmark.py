"""FIP1 wheat cross-species benchmark."""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import fit_xgboost, fit_ridge_rrblup_like, mean_baseline
from residual_gxe.models.mixed_model import fit_additive_main_effects
from residual_gxe.training.trainer import _build_feature_arrays

OUT = ROOT / "data/processed/fip1"
pheno = read_table(OUT / "phenotype.parquet")
genotype = read_table(OUT / "genotype.parquet")
splits_all = read_table(OUT / "splits/official_splits.parquet")
splits_all["split_type"] = "official"
splits_all["seed"] = 0

# Build wide genotype matrix from array column
print("Building FIP1 genotype matrix...")
t0 = time.time()
arrays = []
gids = []
for _, row in genotype.iterrows():
    codes = row["marker_biallelic_codes"]
    if codes is None or (isinstance(codes, float) and np.isnan(codes)):
        continue
    codes = np.asarray(codes, dtype=np.float32)
    if len(codes) == 0: continue
    arrays.append(codes)
    gids.append(row["genotype_id"])
X = np.vstack([a for a in arrays])
n_markers = X.shape[1]
marker_cols = [f"m{i:05d}" for i in range(n_markers)]
geno_wide = pd.DataFrame(X, columns=marker_cols)
geno_wide.insert(0, "genotype_id", gids)
print(f"  {len(gids)} genotypes × {n_markers} markers ({time.time()-t0:.1f}s)")

# Select top 5000 markers by variance
marker_var = geno_wide[marker_cols].var().sort_values(ascending=False)
top_markers = marker_var.index[:5000].tolist()
print(f"  Using top 5000 markers")

all_results = []

for st in sorted(splits_all["split_type"].unique()):
    st_splits = splits_all[splits_all["split_type"] == st]
    for seed in sorted(st_splits["seed"].unique())[:1]:
        sg = st_splits[st_splits["seed"] == seed]
        train_ids = set(sg[sg["split"]=="train"]["sample_id"])
        test_ids = set(sg[sg["split"]=="test"]["sample_id"])
        train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
        test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()
        y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
        y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)

        print(f"\n{'='*50}")
        print(f"  {st} seed={seed} train={len(train_p)} test={len(test_p)}")

        # Build genotype features
        Xg_tr = train_p[["genotype_id"]].merge(geno_wide[["genotype_id"]+top_markers], on="genotype_id", how="left")
        Xg_tr = Xg_tr[top_markers].fillna(0.0).to_numpy(dtype=np.float32)
        Xg_te = test_p[["genotype_id"]].merge(geno_wide[["genotype_id"]+top_markers], on="genotype_id", how="left")
        Xg_te = Xg_te[top_markers].fillna(0.0).to_numpy(dtype=np.float32)

        # Env features (year + location frequency)
        Xe_tr = np.zeros((len(train_p), 2), dtype=np.float32)
        Xe_tr[:, 0] = train_p["year"].to_numpy(dtype=np.float32)
        Xe_te = np.zeros((len(test_p), 2), dtype=np.float32)
        Xe_te[:, 0] = test_p["year"].to_numpy(dtype=np.float32)

        X_tr = np.column_stack([Xg_tr, Xe_tr])
        X_te = np.column_stack([Xg_te, Xe_te])

        for name, fn in [
            ("GlobalMean", lambda: mean_baseline(train_p, test_p)),
            ("Ridge_rrBLUP", lambda: fit_ridge_rrblup_like(X_tr, y_tr, X_te)),
            ("XGBoost_5K", lambda: fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)),
        ]:
            try:
                t0 = time.time()
                r = fn()
                m = metrics_dict(y_te, r.predictions)
                m.update(model=name, dataset="FIP1_wheat", split_type=st, seed=int(seed), time_s=round(time.time()-t0,1))
                all_results.append(m)
                print(f"  {name:20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")
            except Exception as e:
                print(f"  {name:20s}: FAILED - {e}")

        # Additive baseline
        effects = fit_additive_main_effects(train_p)
        main_pred = effects.predict_main_effects(test_p)
        m = metrics_dict(y_te, main_pred.values)
        m.update(model="AdditiveMainEffects", dataset="FIP1_wheat", split_type=st, seed=int(seed), time_s=0)
        all_results.append(m)
        print(f"  {'AdditiveMainEffects':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# Summary
df = pd.DataFrame(all_results)
out_dir = ROOT / "outputs" / "fip1"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "benchmark.csv", index=False)

print(f"\n{'='*60}")
print("FIP1 CROSS-SPECIES BENCHMARK")
for st in sorted(df["split_type"].unique()):
    print(f"\n{st}:")
    for model in df[df["split_type"]==st]["model"].unique():
        grp = df[(df["split_type"]==st)&(df["model"]==model)]
        print(f"  {model:<25s}: pearson={grp['pearson'].mean():.4f} rmse={grp['rmse'].mean():.2f}")

# Also print G2F comparison
print(f"\n{'='*60}")
print("G2F (maize) vs FIP1 (wheat) comparison:")
print(f"  G2F leave_year best:   XGBoost 0.629")
for st in sorted(df["split_type"].unique()):
    xgb = df[(df["split_type"]==st)&(df["model"]=="XGBoost_5K")]
    if len(xgb) > 0:
        print(f"  FIP1 {st:20s}: XGBoost {xgb['pearson'].mean():.4f}")
