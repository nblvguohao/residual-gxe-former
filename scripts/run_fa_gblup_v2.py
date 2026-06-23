"""FA-GBLUP v2: Fast vectorized Factor-Analytic GBLUP.

Approximates MegaLMM by:
  1. Fit additive effects (G + E main effects)
  2. Factor Analysis on G×E residual matrix
  3. Ridge regression with FA-transformed kernel features
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from residual_gxe.data.loaders import read_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.mixed_model import fit_additive_main_effects

OUT = ROOT / "data/processed/g2f"
pheno = read_table(OUT / "phenotype.parquet")
splits = read_table(OUT / "splits/splits.parquet")

# Genotype
genotype = read_table(OUT / "genotype.parquet")
geno_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean())
gids = geno_wide.index.tolist()
G = geno_wide.to_numpy(dtype=np.float32)
K = (G - G.mean(0)) @ (G - G.mean(0)).T / G.shape[1]  # GBLUP kernel
g2i = {g: i for i, g in enumerate(gids)}

all_results = []
for st in ["leave_year", "leave_environment"]:
    st_splits = splits[splits["split_type"]==st]
    seed = sorted(st_splits["seed"].unique())[0]
    sg = st_splits[st_splits["seed"]==seed]
    train_ids = set(sg[sg["split"]=="train"]["sample_id"])
    test_ids = set(sg[sg["split"]=="test"]["sample_id"])
    tr = pheno[pheno["sample_id"].isin(train_ids)].copy()
    te = pheno[pheno["sample_id"].isin(test_ids)].copy()
    y_tr = tr["phenotype_value"].to_numpy(dtype=np.float32)
    y_te = te["phenotype_value"].to_numpy(dtype=np.float32)

    print(f"\n{'='*50}")
    print(f"  {st} train={len(tr)} test={len(te)}")

    # Additive main effects
    eff = fit_additive_main_effects(tr)
    main_tr = eff.predict_main_effects(tr).values
    main_te = eff.predict_main_effects(te).values
    resid_tr = y_tr - main_tr

    # Build G×E residual matrix [n_genos × n_envs]
    envs = sorted(tr["environment_id"].unique())
    e2i = {e: i for i, e in enumerate(envs)}
    ge_r = np.zeros((len(gids), len(envs)))
    ge_cnt = np.zeros((len(gids), len(envs)))
    for _, row in tr.iterrows():
        gi = g2i.get(row["genotype_id"]); ei = e2i.get(row["environment_id"])
        if gi is not None and ei is not None:
            ge_r[gi, ei] += row["phenotype_value"] - eff.global_mean - eff.genotype_effects.get(row["genotype_id"],0) - eff.environment_effects.get(row["environment_id"],0)
            ge_cnt[gi, ei] += 1
    ge_r = np.where(ge_cnt > 0, ge_r / ge_cnt, 0)

    n_envs = len(envs)
    n_factors = min(5, n_envs-1)
    try:
        t0 = time.time()
        fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=500)
        F_env = fa.fit_transform(ge_r.T)  # [n_envs, n_factors]
        F_geno = fa.components_.T       # [n_genos, n_factors]
        print(f"  FA fit: {time.time()-t0:.1f}s, factors={n_factors}")

        # Build features: genotype kernel weighted by factor alignment
        # For each sample: feature = [K @ F_env[e], main_effect]
        # Build features: genotype FA scores + environment FA scores
        # F_geno: [n_genos, k] — genotype loadings on latent factors
        # F_env: [n_envs, k] — environment loadings on latent factors
        # For each sample: concat[F_geno[gi] * F_env[ei], F_geno[gi], F_env[ei]]
        def build_fa_features(pheno_df):
            n = len(pheno_df)
            k = n_factors
            X = np.zeros((n, k * 3 + 1), dtype=np.float32)
            for i, (_, row) in enumerate(pheno_df.iterrows()):
                gi = g2i.get(row["genotype_id"])
                ei = e2i.get(row["environment_id"])
                if gi is not None and ei is not None and ei < n_envs:
                    g_f = F_geno[gi]        # [k]
                    e_f = F_env[ei]          # [k]
                    X[i, :k] = g_f * e_f     # interaction
                    X[i, k:2*k] = g_f        # genotype main
                    X[i, 2*k:3*k] = e_f      # environment main
                X[i, -1] = 1.0  # bias
            return X

        X_tr_fa = build_fa_features(tr)
        X_te_fa = build_fa_features(te)

        # Ridge regression on FA features
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_fa, resid_tr)
        pred_fa = ridge.predict(X_te_fa)
        preds = main_te + pred_fa

        m = metrics_dict(y_te, preds)
        m.update(model="FA_GBLUP", split_type=st, seed=int(seed))
        all_results.append(m)
        print(f"  FA_GBLUP (k={n_factors}): pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")
    except Exception as e:
        print(f"  FA_GBLUP: FAILED - {e}")

    # Baseline
    m = metrics_dict(y_te, main_te)
    m.update(model="AdditiveMainEffects", split_type=st, seed=int(seed))
    all_results.append(m)
    print(f"  AdditiveMainEffects: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

df = pd.DataFrame(all_results)
print(f"\n{'='*60}")
for st in ["leave_year","leave_environment"]:
    print(f"\n{st}:")
    for model in df[df["split_type"]==st]["model"].unique():
        grp = df[(df["split_type"]==st)&(df["model"]==model)]
        print(f"  {model:<25s}: pearson={grp['pearson'].mean():.4f} rmse={grp['rmse'].mean():.2f}")
