"""Cross-dataset benchmark: train and evaluate on G2F + FIP1."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import fit_ridge_rrblup_like, mean_baseline, fit_deepgs
from residual_gxe.models.deep import ResidualGxEFormer
from residual_gxe.models.mixed_model import fit_additive_main_effects
from residual_gxe.training.trainer import (
    MultiModalDataset, _build_feature_arrays, train_model, predict,
)

EPOCHS = 30
HIDDEN_DIM = 64
BATCH_SIZE = 128
MAX_MARKERS = 5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Markers: {MAX_MARKERS}, Epochs: {EPOCHS}, Hidden: {HIDDEN_DIM}")

BENCHMARKS = [
    {
        "name": "G2F (maize)",
        "pheno_path": ROOT / "data/processed/g2f/phenotype.parquet",
        "geno_path": ROOT / "data/processed/g2f/genotype.parquet",
        "env_path": ROOT / "data/processed/g2f/environment.parquet",
        "weather_path": ROOT / "data/processed/g2f/weather_daily.parquet",
        "splits_path": ROOT / "data/processed/g2f/splits/splits.parquet",
        "residuals_path": ROOT / "data/processed/g2f/residual_targets/residual_targets.parquet",
    },
    {
        "name": "FIP1 (wheat)",
        "pheno_path": ROOT / "data/processed/fip1/phenotype.parquet",
        "geno_path": ROOT / "data/processed/fip1/genotype.parquet",
        "env_path": ROOT / "data/processed/fip1/environment.parquet",
        "weather_path": None,
        "splits_path": ROOT / "data/processed/fip1/splits/splits.parquet",
        "residuals_path": ROOT / "data/processed/fip1/residual_targets/residual_targets.parquet",
    },
]


def load_genotype_wide(geno_path: Path, max_markers: int) -> pd.DataFrame | None:
    """Load genotype and build wide matrix, handling different formats."""
    if not geno_path.exists():
        return None
    geno = read_table(geno_path)

    if "marker_biallelic_codes" in geno.columns:
        arrays = []
        gids = []
        for _, row in geno.iterrows():
            codes = row["marker_biallelic_codes"]
            if codes is None or (isinstance(codes, float) and np.isnan(codes)):
                continue
            codes = np.asarray(codes, dtype=np.float32)
            if len(codes) == 0:
                continue
            arrays.append(codes)
            gids.append(row["genotype_id"])
        if arrays:
            n_m = min(len(arrays[0]), max_markers)
            X = np.vstack([a[:n_m] for a in arrays])
            marker_cols = [f"m{i:05d}" for i in range(n_m)]
            geno_wide = pd.DataFrame(X, columns=marker_cols)
            geno_wide.insert(0, "genotype_id", gids)
            return geno_wide
    elif "marker_id" in geno.columns:
        all_m = geno["marker_id"].unique()
        if len(all_m) > max_markers:
            rng = np.random.default_rng(42)
            all_m = rng.choice(all_m, size=max_markers, replace=False)
        gsub = geno[geno["marker_id"].isin(all_m)]
        gwide = gsub.pivot_table(index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first")
        gwide = gwide.fillna(gwide.mean()).reset_index()
        return gwide

    return geno


all_results = []

for ds in BENCHMARKS:
    print(f"\n{'#'*60}")
    print(f"  DATASET: {ds['name']}")
    print(f"{'#'*60}")

    pheno = read_table(ds["pheno_path"])
    geno_wide = load_genotype_wide(ds["geno_path"], MAX_MARKERS)
    env_raw = read_table(ds["env_path"]) if ds["env_path"].exists() else None
    weather_raw = read_table(ds["weather_path"]) if ds["weather_path"] and Path(ds["weather_path"]).exists() else None
    splits_all = read_table(ds["splits_path"])

    residuals = None
    if ds["residuals_path"].exists():
        residuals = read_table(ds["residuals_path"])

    if geno_wide is None:
        print("  No genotype data, skipping")
        continue

    n_geno = geno_wide.shape[0]
    n_markers = len([c for c in geno_wide.columns if c != "genotype_id"])
    print(f"  Genotypes: {n_geno}, Markers: {n_markers}")

    split_types = sorted(splits_all["split_type"].unique())
    for split_type in split_types:
        st = splits_all[splits_all["split_type"] == split_type]
        seed = sorted(st["seed"].unique())[0]
        sg = st[st["seed"] == seed]

        train_ids = set(sg.loc[sg["split"] == "train", "sample_id"])
        test_ids = set(sg.loc[sg["split"] == "test", "sample_id"])
        train_p = pheno[pheno["sample_id"].isin(train_ids)].copy()
        test_p = pheno[pheno["sample_id"].isin(test_ids)].copy()

        if len(train_p) < 100 or len(test_p) < 50:
            continue

        print(f"\n  --- {split_type} seed={seed}  train={len(train_p)} test={len(test_p)} ---")

        Xg_tr, Xw_tr, Xe_tr = _build_feature_arrays(train_p, geno_wide, env_raw, weather_raw, max_markers=MAX_MARKERS)
        Xg_te, Xw_te, Xe_te = _build_feature_arrays(test_p, geno_wide, env_raw, weather_raw, max_markers=MAX_MARKERS)
        y_tr = train_p["phenotype_value"].to_numpy(dtype=np.float32)
        y_te = test_p["phenotype_value"].to_numpy(dtype=np.float32)
        X_tr = np.column_stack([Xg_tr, Xe_tr])
        X_te = np.column_stack([Xg_te, Xe_te])

        # Baselines
        for name, fn in [
            ("GlobalMean", lambda: mean_baseline(train_p, test_p)),
            ("Ridge", lambda: fit_ridge_rrblup_like(X_tr, y_tr, X_te)),
        ]:
            try:
                t0 = time.time()
                r = fn()
                dt = time.time() - t0
                m = metrics_dict(y_te, r.predictions)
                m["model"] = name
                m["dataset"] = ds["name"]
                m["split_type"] = split_type
                m["time_s"] = round(dt, 1)
                all_results.append(m)
                print(f"    {name:20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.1f}")
            except Exception as e:
                print(f"    {name:20s}: FAILED - {e}")

        # DeepGS
        try:
            t0 = time.time()
            r = fit_deepgs(X_tr, y_tr, X_te, epochs=100)
            dt = time.time() - t0
            m = metrics_dict(y_te, r.predictions)
            m["model"] = "DeepGS"
            m["dataset"] = ds["name"]
            m["split_type"] = split_type
            m["time_s"] = round(dt, 1)
            all_results.append(m)
            print(f"    {'DeepGS':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.1f}")
        except Exception as e:
            print(f"    {'DeepGS':20s}: FAILED - {e}")

        # AdditiveMainEffects
        effects = fit_additive_main_effects(train_p)
        main_pred = effects.predict_main_effects(test_p)
        m = metrics_dict(y_te, main_pred.values)
        m["model"] = "AdditiveMainEffects"
        m["dataset"] = ds["name"]
        m["split_type"] = split_type
        m["time_s"] = 0.0
        all_results.append(m)
        print(f"    {'AdditiveMainEffects':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.1f}")

        # ResidualGxEFormer
        if residuals is not None:
            res_group = residuals[(residuals["split_type"] == split_type) & (residuals["seed"] == seed)]
            if len(res_group) > 0:
                train_t = train_p.merge(res_group[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
                test_t = test_p.merge(res_group[["sample_id", "residual_target", "main_prediction"]], on="sample_id", how="left")
                y_tr_r = train_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
                y_te_r = test_t["residual_target"].fillna(0).to_numpy(dtype=np.float32)
                test_main = test_t["main_prediction"].fillna(0).to_numpy(dtype=np.float32)

                try:
                    t0 = time.time()
                    sdim = Xe_tr.shape[1]
                    model = ResidualGxEFormer(
                        n_markers=n_markers, weather_dim=Xw_tr.shape[2],
                        static_env_dim=sdim, hidden_dim=HIDDEN_DIM,
                        patch_size=min(64, n_markers), dropout=0.15,
                    )
                    train_ds = MultiModalDataset(Xg_tr, Xw_tr, Xe_tr, y_tr_r)
                    test_ds = MultiModalDataset(Xg_te, Xw_te, Xe_te, y_te_r)
                    train_model(model, train_ds, None, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-4, early_stopping_patience=12, rank_weight=0.05, device=DEVICE)
                    y_pred = predict(model, test_ds, batch_size=BATCH_SIZE*2, device=DEVICE) + test_main
                    dt = time.time() - t0
                    m = metrics_dict(y_te, y_pred)
                    m["model"] = "ResidualGxEFormer"
                    m["dataset"] = ds["name"]
                    m["split_type"] = split_type
                    m["time_s"] = round(dt, 1)
                    all_results.append(m)
                    print(f"    {'ResidualGxEFormer':20s}: pearson={m['pearson']:.4f} rmse={m['rmse']:.1f}")
                except Exception as e:
                    print(f"    {'ResidualGxEFormer':20s}: FAILED - {e}")

# Summary
df = pd.DataFrame(all_results)
out = ROOT / "outputs" / "cross_dataset"
out.mkdir(parents=True, exist_ok=True)
write_table(df, out / "cross_dataset_results.csv")
(out / "cross_dataset_results.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

print(f"\n{'='*80}")
print("CROSS-DATASET SUMMARY")
print(f"{'='*80}")
for dataset in df["dataset"].unique():
    ds_df = df[df["dataset"] == dataset]
    print(f"\n{dataset}:")
    for model, grp in ds_df.groupby("model"):
        p = grp["pearson"].dropna().mean()
        r = grp["rmse"].dropna().mean()
        print(f"  {model:<25s}: pearson={p:6.4f}  rmse={r:6.1f}")
