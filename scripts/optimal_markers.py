"""Find optimal N: variance-selected markers + weather aggregation."""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import fit_xgboost

OUT = ROOT / "data/processed/g2f"

# Load data
pheno = read_table(OUT / "phenotype.parquet")
env_raw = read_table(OUT / "environment.parquet")
weather_raw = read_table(OUT / "weather_daily.parquet")
splits = read_table(OUT / "splits/splits.parquet")

# 50K markers (wide)
geno_50k = pd.read_parquet(OUT / "genotype_50k_wide.parquet")
all_markers = [c for c in geno_50k.columns if c != "genotype_id"]
marker_var = geno_50k[all_markers].var().sort_values(ascending=False)
print(f"50K markers loaded. Variance range: [{marker_var.iloc[0]:.3f}, {marker_var.iloc[-1]:.3f}]")

# 2.4K competition markers (for comparison)
geno_2k = genotype = read_table(OUT / "genotype.parquet")
geno_2k_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_2k_wide = geno_2k_wide.fillna(geno_2k_wide.mean()).reset_index()
print(f"2.4K competition markers")

# Prepare leave_year split
splits_ly = splits[(splits["split_type"]=="leave_year")&(splits["seed"]==1234)]
train_ids = set(splits_ly[splits_ly["split"]=="train"]["sample_id"])
test_ids = set(splits_ly[splits_ly["split"]=="test"]["sample_id"])
train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()
y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)

# Pre-build weather aggregation (shared across all marker counts)
from residual_gxe.training.trainer import _build_feature_arrays
print("Pre-building weather features...")
t0 = time.time()
_, Xw_tr, Xe_tr = _build_feature_arrays(train_p, geno_2k_wide, env_raw, weather_raw, max_markers=2425)
_, Xw_te, Xe_te = _build_feature_arrays(test_p, geno_2k_wide, env_raw, weather_raw, max_markers=2425)
s1_tr, s2_tr, s3_tr = Xw_tr[:,:10,:].mean(1), Xw_tr[:,10:20,:].mean(1), Xw_tr[:,20:30,:].mean(1)
s1_te, s2_te, s3_te = Xw_te[:,:10,:].mean(1), Xw_te[:,10:20,:].mean(1), Xw_te[:,20:30,:].mean(1)
W_tr = np.column_stack([Xe_tr, s1_tr, s2_tr, s3_tr])
W_te = np.column_stack([Xe_te, s1_te, s2_te, s3_te])
print(f"  Weather features: {W_tr.shape[1]} dims ({time.time()-t0:.1f}s)")

# Test different marker counts
all_results = []
marker_sizes = [500, 1000, 2000, 3000, 5000, 10000, 20000]

for n_select in marker_sizes:
    if n_select > len(all_markers):
        continue
    top_markers = marker_var.index[:n_select].tolist()
    gm = geno_50k[["genotype_id"] + top_markers]

    t0 = time.time()
    # Genotype features
    Xg_tr = train_p[["genotype_id"]].merge(gm, on="genotype_id", how="left")
    Xg_tr = Xg_tr[top_markers].fillna(0.0).to_numpy(dtype=np.float32)
    Xg_te = test_p[["genotype_id"]].merge(gm, on="genotype_id", how="left")
    Xg_te = Xg_te[top_markers].fillna(0.0).to_numpy(dtype=np.float32)

    # Combine
    X_tr = np.column_stack([Xg_tr, W_tr])
    X_te = np.column_stack([Xg_te, W_te])
    bt = time.time() - t0

    # XGBoost
    r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
    m = metrics_dict(y_te, r.predictions)
    m.update(model=f"Top{n_select}_markers", n_markers=n_select, n_feat=X_tr.shape[1],
             time_s=round(time.time()-t0,1), build_s=round(bt,1))
    all_results.append(m)

    rank = marker_var.iloc[n_select-1] if n_select <= len(marker_var) else 0
    print(f"  Top {n_select:5d} markers ({n_select+W_tr.shape[1]:5d} feats): pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} ({bt:.1f}s) min_var={rank:.3f}")

# Baseline: competition 2.4K markers
t0 = time.time()
Xg_tr = train_p[["genotype_id"]].merge(geno_2k_wide, on="genotype_id", how="left")
markers_2k = [c for c in geno_2k_wide.columns if c != "genotype_id"]
Xg_tr = Xg_tr[markers_2k].fillna(0.0).to_numpy(dtype=np.float32)
Xg_te = test_p[["genotype_id"]].merge(geno_2k_wide, on="genotype_id", how="left")
Xg_te = Xg_te[markers_2k].fillna(0.0).to_numpy(dtype=np.float32)
X_tr_c = np.column_stack([Xg_tr, W_tr])
X_te_c = np.column_stack([Xg_te, W_te])
r = fit_xgboost(X_tr_c, y_tr, X_te_c, n_estimators=300)
m = metrics_dict(y_te, r.predictions)
m.update(model="Competition_2425", n_markers=2425, n_feat=X_tr_c.shape[1], time_s=round(time.time()-t0,1))
all_results.append(m)
print(f"  Competition 2425 markers: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# Summary
df = pd.DataFrame(all_results)
print(f"\n{'='*60}")
print("OPTIMAL MARKER COUNT")
print(f"{'N_markers':<12s} {'Feats':<7s} {'Pearson':>8s} {'RMSE':>8s} {'Build':>7s}")
print("-"*45)
for _, r in df.iterrows():
    print(f"  {r['n_markers']:<12d} {r['n_feat']:<7d} {r['pearson']:>8.4f} {r['rmse']:>8.2f} {r['build_s']:>6.1f}s")

out_dir = ROOT.parent / "outputs" / "optimal_markers"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "results.csv", index=False)
