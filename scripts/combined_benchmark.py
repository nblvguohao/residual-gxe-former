"""Combined benchmark: weather aggregation + physiological indices."""
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

pheno = read_table(OUT / "phenotype.parquet")
genotype = read_table(OUT / "genotype.parquet")
physio = read_table(OUT / "physiological_indices.parquet")
env_raw = read_table(OUT / "environment.parquet")
weather_raw = read_table(OUT / "weather_daily.parquet")
splits = read_table(OUT / "splits/splits.parquet")

geno_wide = genotype.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
geno_wide = geno_wide.fillna(geno_wide.mean()).reset_index()
marker_cols = [c for c in geno_wide.columns if c != "genotype_id"]

# Causal features
CAUSAL_F = ["et0_cum_late","vpd_mean_mid","gdd_max_mid","gdd_cum_mid","gdd_mean_mid",
            "gdd_cum_late","vpd_mean_early","vpd_max_mid","drought_stress_cum_mid","gdd_cum_early",
            "precip_cum_mid","wb_mean_mid","heat_stress_cum_mid","precip_cum_early","vpd_cum_early"]

def build_X(pheno_df, use_physio=True, use_weather=True):
    """Build features: geno + optional physio + optional weather agg."""
    parts = []

    # Genotype
    Xg = pheno_df[["genotype_id"]].merge(geno_wide, on="genotype_id", how="left")
    parts.append(Xg[marker_cols].fillna(0.0).to_numpy(dtype=np.float32))

    # Weather aggregation
    if use_weather:
        from residual_gxe.training.trainer import _build_feature_arrays
        _, Xw, Xe = _build_feature_arrays(pheno_df, geno_wide, env_raw, weather_raw, max_markers=len(marker_cols))
        s1, s2, s3 = Xw[:,:10,:].mean(1), Xw[:,10:20,:].mean(1), Xw[:,20:30,:].mean(1)
        parts.append(np.column_stack([Xe, s1, s2, s3]))

    # Physiological indices
    if use_physio:
        physio_idx = physio.set_index("environment_id")
        Xp = np.zeros((len(pheno_df), len(CAUSAL_F)), dtype=np.float32)
        for i, env_id in enumerate(pheno_df["environment_id"]):
            if env_id in physio_idx.index:
                vals = physio_idx.loc[env_id, CAUSAL_F]
                if isinstance(vals, pd.Series):
                    vals = vals.fillna(0).to_numpy(dtype=np.float32)
                else:
                    vals = np.array([vals], dtype=np.float32)
                Xp[i] = vals
        parts.append(Xp)

    return np.column_stack(parts)


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

    for label, use_p, use_w in [
        ("W_agg3 (baseline)", False, True),
        ("Physio (causal)", True, False),
        ("W_agg3 + Physio", True, True),
    ]:
        t0 = time.time()
        X_tr = build_X(train_p, use_physio=use_p, use_weather=use_w)
        X_te = build_X(test_p, use_physio=use_p, use_weather=use_w)
        bt = time.time() - t0
        r = fit_xgboost(X_tr, y_tr, X_te, n_estimators=300)
        m = metrics_dict(y_te, r.predictions)
        m.update(model=label, split_type=st, seed=int(seed), time_s=round(time.time()-t0,1), n_feat=X_tr.shape[1])
        all_results.append(m)
        print(f"  {label:25s}: feat={X_tr.shape[1]:5d} pearson={m['pearson']:.4f} rmse={m['rmse']:.2f} ({bt:.1f}s)")

    # Additive baseline
    effects = fit_additive_main_effects(train_p)
    main_pred = effects.predict_main_effects(test_p)
    m = metrics_dict(y_te, main_pred.values)
    m.update(model="AdditiveMainEffects", split_type=st, seed=int(seed), time_s=0, n_feat=0)
    all_results.append(m)
    print(f"  {'AdditiveMainEffects':25s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.2f}")

# Summary
df = pd.DataFrame(all_results)
out_dir = ROOT.parent / "outputs" / "combined"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "results.csv", index=False)

print(f"\n{'='*60}")
for st in ["leave_year", "leave_environment"]:
    print(f"\n{st}:")
    for model in df[df["split_type"]==st]["model"].unique():
        grp = df[(df["split_type"]==st)&(df["model"]==model)]
        print(f"  {model:<25s}: pearson={grp['pearson'].mean():.4f} rmse={grp['rmse'].mean():.2f}")
