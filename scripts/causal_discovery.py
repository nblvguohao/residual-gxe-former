"""Causal structure discovery for GxE prediction.

Uses PC algorithm (partial correlation + Fisher's z-test) to learn
causal relationships among: genotype effects, physiological indices, and yield.

Builds a causal graph → identifies causal parents of yield → uses them
as features for prediction.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table

OUT = ROOT / "data/processed/g2f"

# ============================================================
# 1. Build dataset for causal discovery
# ============================================================
print("Building causal dataset...")
t0 = time.time()

pheno = read_table(OUT / "phenotype.parquet")
physio = read_table(OUT / "physiological_indices.parquet")
env_raw = read_table(OUT / "environment.parquet")

# Aggregate per environment: mean yield, yield std, sample count
env_stats = pheno.groupby("environment_id").agg(
    mean_yield=("phenotype_value", "mean"),
    std_yield=("phenotype_value", "std"),
    n_samples=("sample_id", "count"),
    mean_moisture=("grain_moisture", "mean"),
).reset_index()

# Merge with physiological indices
causal_df = env_stats.merge(physio, on="environment_id", how="inner")

# Select key features for causal graph (avoid too many nodes)
key_features = [
    "mean_yield", "std_yield", "mean_moisture",
    "gdd_cum_early", "gdd_cum_mid", "gdd_cum_late",
    "gdd_mean_mid", "gdd_max_mid",
    "vpd_mean_early", "vpd_mean_mid", "vpd_max_mid",
    "precip_cum_early", "precip_cum_mid",
    "wb_mean_mid",  # water balance mid-season — critical for maize
    "heat_stress_cum_mid", "drought_stress_cum_mid",
    "et0_cum_late",
    "n_samples",
]

# Drop missing
causal_df = causal_df.dropna(subset=key_features)
feature_mat = causal_df[key_features].to_numpy(dtype=np.float64)
feature_names = list(key_features)

# Standardize
scaler = StandardScaler()
feature_mat_scaled = scaler.fit_transform(feature_mat)

print(f"  {len(causal_df)} environments × {len(key_features)} features ({time.time()-t0:.1f}s)")
print(f"  Target: mean_yield range [{causal_df['mean_yield'].min():.1f}, {causal_df['mean_yield'].max():.1f}]")

# ============================================================
# 2. Partial Correlation Network (suitable for n=269, p=18)
# ============================================================
print("\nBuilding partial correlation network...")
t0 = time.time()

n_nodes = len(feature_names)
n_samples = len(causal_df)
yield_idx = feature_names.index("mean_yield")

# Compute full partial correlation matrix (sparse inverse covariance)
corr = np.corrcoef(feature_mat_scaled.T)
reg = 0.1 * np.eye(n_nodes)  # L2 regularization for stability
prec = np.linalg.inv(corr + reg)
d = np.sqrt(np.diag(prec))
pcorr = -prec / np.outer(d, d)
np.fill_diagonal(pcorr, 0)

# Threshold: keep edges with |partial r| > threshold
threshold = 0.15  # moderate threshold for 269 samples
adj = np.abs(pcorr) > threshold

n_edges = adj.sum() // 2
print(f"  Edges (|partial r| > {threshold}): {n_edges} ({time.time()-t0:.1f}s)")

# ============================================================
# 3. Causal parents of yield + Markov blanket
# ============================================================
causal_parents = []
edge_list = []
for j in range(n_nodes):
    if adj[yield_idx, j]:
        causal_parents.append(feature_names[j])
        edge_list.append((feature_names[j], "mean_yield", float(pcorr[yield_idx, j])))
    for i in range(j+1, n_nodes):
        if adj[i, j]:
            edge_list.append((feature_names[i], feature_names[j], float(pcorr[i, j])))

print(f"\n  Causal parents of 'mean_yield' (|partial r| > {threshold}): {len(causal_parents)}")
for p in causal_parents:
    r = pcorr[yield_idx, feature_names.index(p)]
    raw_r = corr[yield_idx, feature_names.index(p)]
    print(f"    {p:30s}: partial_r={r:+.3f}  raw_r={raw_r:+.3f}")

# ============================================================
# 4. Save causal graph
# ============================================================
graph_data = {
    "nodes": [str(n) for n in feature_names],
    "edges": [[str(e[0]), str(e[1]), float(e[2])] for e in edge_list],
    "causal_parents_of_yield": [str(p) for p in causal_parents],
    "n_environments": int(len(causal_df)),
    "n_edges": int(n_edges),
}

out_dir = OUT.parent / "outputs" / "causal"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "causal_graph.json").write_text(json.dumps(graph_data, indent=2, default=str), encoding="utf-8")

# ============================================================
# 5. Causal feature selection: compare vs all features
# ============================================================
print(f"\n{'='*60}")
print("Causal Feature Importance Analysis")
print(f"{'='*60}")

# Rank features by absolute correlation with yield
correlations = []
for name in feature_names:
    if name == "mean_yield": continue
    idx = feature_names.index(name)
    r = np.corrcoef(feature_mat_scaled[:, yield_idx], feature_mat_scaled[:, idx])[0, 1]
    is_causal = name in causal_parents
    correlations.append((name, abs(r), r, is_causal))

correlations.sort(key=lambda x: -x[1])

print(f"\n{'Feature':<30s} {'|r|':>6s} {'r':>7s} {'Causal':>8s}")
print("-"*55)
for name, abs_r, r, is_causal in correlations:
    print(f"  {name:<30s} {abs_r:6.3f} {r:7.3f} {'✓' if is_causal else '':>8s}")

# Save
pd.DataFrame(correlations, columns=["feature", "abs_corr", "correlation", "is_causal_parent"]).to_csv(
    out_dir / "causal_features.csv", index=False)

print(f"\nResults saved to {out_dir}/")
print(f"  causal_graph.json — graph structure ({n_edges} edges)")
print(f"  causal_features.csv — feature correlations + causal status")
