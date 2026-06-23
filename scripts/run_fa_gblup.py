"""FA-GBLUP: Factor-Analytic Genomic BLUP for GxE prediction.

Simplified Python approximation of MegaLMM (Hu et al. 2024, Genetics).
Uses Factor Analysis on environment covariance + GBLUP kernel.

Reference: "MegaLMM improves genomic predictions in new environments
using environmental covariates" — Hu, Rincent & Runcie (2024)
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.mixed_model import fit_additive_main_effects

OUT = ROOT / "data/processed/g2f"
pheno = read_table(OUT / "phenotype.parquet")
genotype = read_table(OUT / "genotype.parquet")
splits = read_table(OUT / "splits/splits.parquet")

# Build genotype matrix
print("Building genotype matrix...")
geno_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean())
geno_ids = geno_wide.index.tolist()
geno_mat = geno_wide.to_numpy(dtype=np.float32)  # [n_genos, n_markers]

def make_gblup_kernel(G):
    """VanRaden GBLUP kernel: G @ G.T / p."""
    G_centered = G - G.mean(axis=0, keepdims=True)
    return G_centered @ G_centered.T / G.shape[1]

K = make_gblup_kernel(geno_mat)  # [n_genos, n_genos]
geno_to_idx = {g: i for i, g in enumerate(geno_ids)}

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

    # Step 1: Additive main effects
    effects = fit_additive_main_effects(train_p)
    main_tr = effects.predict_main_effects(train_p).values
    main_te = effects.predict_main_effects(test_p).values

    # Step 2: Compute residuals per environment
    resid_tr = y_tr - main_tr
    train_p_copy = train_p.copy()
    train_p_copy["residual"] = resid_tr

    # Environment-level residual means
    env_resid = train_p_copy.groupby("environment_id")["residual"].agg(["mean", "std", "count"])
    env_list = sorted(env_resid.index.tolist())
    env_to_idx = {e: i for i, e in enumerate(env_list)}

    # Build environment residual vector and genotype × environment matrix
    n_envs = len(env_list)
    env_mean_resid = np.array([env_resid.loc[e, "mean"] for e in env_list])

    # Step 3: Factor Analysis on environment residuals
    # Use genotype-level data to estimate factor structure
    n_factors = min(5, n_envs - 1)

    # Build G×E residual matrix: [n_genos × n_envs]
    ge_matrix = np.full((len(geno_ids), n_envs), np.nan)
    for _, row in train_p_copy.iterrows():
        g_idx = geno_to_idx.get(row["genotype_id"])
        e_idx = env_to_idx.get(row["environment_id"])
        if g_idx is not None and e_idx is not None:
            ge_matrix[g_idx, e_idx] = row["residual"]

    # Fill missing with row/col means
    for i in range(ge_matrix.shape[0]):
        row_mean = np.nanmean(ge_matrix[i])
        ge_matrix[i] = np.where(np.isnan(ge_matrix[i]), row_mean, ge_matrix[i])
    for j in range(ge_matrix.shape[1]):
        col_mean = np.nanmean(ge_matrix[:, j])
        ge_matrix[:, j] = np.where(np.isnan(ge_matrix[:, j]), col_mean, ge_matrix[:, j])

    try:
        fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
        fa.fit(ge_matrix.T)  # [n_envs, n_genos] → factor loadings
        # Factor loadings: [n_envs, n_factors] — each env's loading on latent factors

        # Step 4: Predict using factor structure
        # For each test sample: predict = additive + factor-weighted GBLUP
        preds = np.zeros(len(test_p))
        for i, (_, row) in enumerate(test_p.iterrows()):
            g_idx = geno_to_idx.get(row["genotype_id"])
            e_id = row["environment_id"]
            if g_idx is None:
                preds[i] = main_te[i]
                continue

            # GBLUP component: weighted by genotype similarity to training genotypes
            g_sim = K[g_idx]  # similarity to all training genotypes

            # Factor-weighted prediction:
            # If environment is in training, use its factor loading
            # If new, use nearest environment's factor loading
            if e_id in env_to_idx:
                e_idx_test = env_to_idx[e_id]
            else:
                # Find nearest training environment (by mean residual)
                e_idx_test = np.argmin(np.abs(env_mean_resid))

            # Weight genotype similarities by factor alignment
            fa_loadings = fa.components_.T  # [n_envs, n_factors]
            env_factor = fa_loadings[e_idx_test]  # target env's factors

            # Compute environment similarity via factor space
            env_sim = fa_loadings @ env_factor  # [n_envs]
            env_sim = env_sim / (np.linalg.norm(env_sim) + 1e-10)

            # Aggregate: GBLUP prediction weighted by environment similarity
            train_g_indices = [geno_to_idx.get(g) for g in train_p["genotype_id"] if geno_to_idx.get(g) is not None]
            gblup_weighted = 0.0
            weight_sum = 0.0
            for j, (_, tr) in enumerate(train_p.iterrows()):
                g_tr = geno_to_idx.get(tr["genotype_id"])
                e_tr = env_to_idx.get(tr["environment_id"])
                if g_tr is None or e_tr is None: continue
                w = g_sim[g_tr] * env_sim[e_tr]
                gblup_weighted += w * resid_tr[j]
                weight_sum += abs(w)

            if weight_sum > 1e-10:
                preds[i] = main_te[i] + gblup_weighted / weight_sum
            else:
                preds[i] = main_te[i]

        m = metrics_dict(y_te, preds)
        m.update(model="FA_GBLUP", split_type=st, seed=int(seed), time_s=0)
        all_results.append(m)
        print(f"  FA_GBLUP (k={n_factors}): pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

    except Exception as e:
        print(f"  FA_GBLUP: FAILED - {e}")

    # Baseline comparison
    m = metrics_dict(y_te, main_te)
    m.update(model="AdditiveMainEffects", split_type=st, seed=int(seed), time_s=0)
    all_results.append(m)
    print(f"  AdditiveMainEffects: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# Summary
df = pd.DataFrame(all_results)
out_dir = ROOT.parent / "outputs" / "fa_gblup"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "results.csv", index=False)

print(f"\n{'='*60}")
print("FA-GBLUP (MegaLMM-lite) RESULTS")
for st in ["leave_year", "leave_environment"]:
    print(f"\n{st}:")
    for model in df[df["split_type"]==st]["model"].unique():
        grp = df[(df["split_type"]==st)&(df["model"]==model)]
        print(f"  {model:<25s}: pearson={grp['pearson'].mean():.4f} rmse={grp['rmse'].mean():.2f}")
