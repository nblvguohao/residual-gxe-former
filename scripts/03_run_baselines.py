from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.data.schema import validate_phenotype_table
from residual_gxe.data.splits import load_split_table
from residual_gxe.evaluation.artifacts import (
    build_prediction_frame,
    build_runtime_frame,
    write_prediction_artifact,
    write_runtime_artifact,
)
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.baselines import (
    fit_bayes_ridge,
    fit_gblup_efficient,
    fit_gblup_kernel_ridge,
    fit_lightgbm,
    fit_random_forest,
    fit_ridge_rrblup_like,
    fit_xgboost,
    mean_baseline,
)
from residual_gxe.training.preprocessing import FoldPreprocessor, build_genotype_wide


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline models.")
    parser.add_argument("--config", type=Path, required=False, default=None)
    parser.add_argument("--raw-dir", type=Path, required=False, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=False, default=None)
    parser.add_argument("--results-dir", type=Path, required=False, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Use fewer trees/markers for fast smoke testing")
    parser.add_argument("--split-type", action="append", default=None, help="Only run the specified split type. Can be repeated.")
    parser.add_argument("--seed", action="append", type=int, default=None, help="Only run the specified seed. Can be repeated.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to run.")
    parser.add_argument("--max-markers", type=int, default=None, help="Override marker count used by the fold preprocessor.")
    parser.add_argument("--rf-trees", type=int, default=None, help="Override tree count for RF/XGBoost/LightGBM.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip model/split/seed rows already present in baseline_metrics.csv.")
    return parser.parse_args()


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_genotype_matrix(genotype_path: Path, pheno_genotype_ids: set[str]) -> pd.DataFrame | None:
    """Build wide genotype matrix (samples x markers) for given genotype IDs."""
    if not genotype_path.exists():
        return None
    geno = read_table(genotype_path)
    if "marker_id" in geno.columns:
        # Long format: pivot to wide
        geno = geno[geno["genotype_id"].isin(pheno_genotype_ids)]
        if "allele_dosage" not in geno.columns:
            return None
        wide = geno.pivot_table(
            index="genotype_id",
            columns="marker_id",
            values="allele_dosage",
            aggfunc="first",
        )
        wide = wide.dropna(axis=1, how="all").fillna(wide.mean())
        return wide.reset_index()
    else:
        # Already wide format
        return geno[geno["genotype_id"].isin(pheno_genotype_ids)]


def _build_env_features(env_path: Path, pheno: pd.DataFrame) -> pd.DataFrame | None:
    """Build environment feature matrix."""
    if not env_path.exists():
        return None
    env = read_table(env_path)
    env_cols = [c for c in env.columns if c not in ("environment_id", "source_dataset", "management_notes", "planting_date", "harvest_date")]
    # One-hot encode location if present
    if "location_id" in env.columns:
        loc_dummies = pd.get_dummies(env["location_id"], prefix="loc")
        env = pd.concat([env.drop(columns=["location_id"]), loc_dummies], axis=1)
    # Keep numeric columns
    numeric_cols = ["environment_id"] + [c for c in env.columns if c != "environment_id" and env[c].dtype in ("float64", "int64", "int32", "float32", "int16", "float16", "bool", "uint8")]
    # Filter to only columns that exist
    numeric_cols = [c for c in numeric_cols if c in env.columns]
    return env[numeric_cols]


def _prepare_features(
    pheno: pd.DataFrame,
    genotype: pd.DataFrame | None,
    environment: pd.DataFrame | None,
    max_markers: int = 5000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct feature matrix X and target y from phenotype, genotype, environment."""
    y = pheno["phenotype_value"].to_numpy(dtype=float)

    feature_parts = []
    feature_names = []

    if genotype is not None:
        geno_wide = _build_genotype_matrix_from_df(genotype, set(pheno["genotype_id"]), max_markers=max_markers)
        if geno_wide is not None:
            geno_indexed = pheno[["genotype_id"]].merge(geno_wide, on="genotype_id", how="left")
            marker_cols = [c for c in geno_indexed.columns if c != "genotype_id"]
            X_geno = geno_indexed[marker_cols].fillna(0).to_numpy(dtype=float)
            feature_parts.append(X_geno)
            feature_names.extend(marker_cols)

    if environment is not None:
        env_indexed = pheno[["environment_id"]].merge(environment, on="environment_id", how="left")
        env_feat_cols = [c for c in env_indexed.columns if c != "environment_id"]
        env_feat_cols = [c for c in env_feat_cols if env_indexed[c].dtype in ("float64", "int64", "int32", "float32", "bool")]
        X_env = env_indexed[env_feat_cols].fillna(0).to_numpy(dtype=float)
        feature_parts.append(X_env)
        feature_names.extend(env_feat_cols)

    if feature_parts:
        X = np.column_stack(feature_parts) if len(feature_parts) > 1 else feature_parts[0]
    else:
        X = np.ones((len(pheno), 1))
        feature_names = ["intercept"]

    # Remove rows with NaN in X or y
    valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    return X[valid], y[valid], feature_names


def _read_existing_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _metric_exists(metrics: pd.DataFrame, model: str, split_type: str, seed: int) -> bool:
    if len(metrics) == 0:
        return False
    required = {"model", "split_type", "seed"}
    if not required <= set(metrics.columns):
        return False
    hit = metrics[
        (metrics["model"].astype(str) == str(model))
        & (metrics["split_type"].astype(str) == str(split_type))
        & (metrics["seed"].astype(int) == int(seed))
    ]
    return len(hit) > 0


def _upsert_frame(path: Path, row: dict, keys: list[str]) -> pd.DataFrame:
    existing = _read_existing_metrics(path)
    row_df = pd.DataFrame([row])
    if len(existing) > 0:
        mask = pd.Series(True, index=existing.index)
        for key in keys:
            if key in existing.columns and key in row:
                mask &= existing[key].astype(str) == str(row[key])
            else:
                mask &= False
        existing = existing.loc[~mask].copy()
        out = pd.concat([existing, row_df], ignore_index=True)
    else:
        out = row_df
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _write_json_records(csv_path: Path, json_path: Path) -> None:
    if not csv_path.exists():
        return
    records = pd.read_csv(csv_path).to_dict(orient="records")
    json_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def _build_genotype_matrix_from_df(geno: pd.DataFrame, ids: set[str], max_markers: int = 5000) -> pd.DataFrame | None:
    """Build wide genotype matrix (genotype_id x marker columns) from various formats."""
    geno_sub = geno[geno["genotype_id"].isin(ids)]
    if len(geno_sub) == 0:
        return None

    if "marker_biallelic_codes" in geno_sub.columns:
        # FIP1-style: markers as array in a single column — vectorized expansion
        arrays = []
        gids = []
        for _, row in geno_sub.iterrows():
            codes = row["marker_biallelic_codes"]
            if codes is None or (isinstance(codes, float) and np.isnan(codes)):
                continue
            codes = np.asarray(codes, dtype=float)
            if len(codes) == 0:
                continue
            arrays.append(codes)
            gids.append(row["genotype_id"])
        if not arrays:
            return None
        n_markers = min(len(arrays[0]), max_markers)
        # Stack into a single matrix, subsample markers
        if len(arrays) > 1:
            X = np.vstack([a[:n_markers] for a in arrays])
        else:
            X = arrays[0][:n_markers].reshape(1, -1)
        marker_ids = [f"m{i:05d}" for i in range(n_markers)]
        wide = pd.DataFrame(X, columns=marker_ids)
        wide.insert(0, "genotype_id", gids)
        return wide

    if "marker_id" in geno_sub.columns and "allele_dosage" in geno_sub.columns:
        geno_sub = geno_sub.copy()
        # Subsample markers if too many
        all_markers = geno_sub["marker_id"].unique()
        if len(all_markers) > max_markers:
            rng = np.random.default_rng(42)
            selected = rng.choice(all_markers, size=max_markers, replace=False)
            geno_sub = geno_sub[geno_sub["marker_id"].isin(selected)]
        wide = geno_sub.pivot_table(
            index="genotype_id", columns="marker_id", values="allele_dosage", aggfunc="first",
        )
        wide = wide.dropna(axis=1, how="all")
        means = wide.mean()
        wide = wide.fillna(means)
        return wide.reset_index()

    # Already wide format — subsample if needed
    marker_cols = [c for c in geno_sub.columns if c != "genotype_id"]
    if len(marker_cols) > max_markers:
        rng = np.random.default_rng(42)
        selected = list(rng.choice(marker_cols, size=max_markers, replace=False))
        return geno_sub[["genotype_id"] + selected]
    return geno_sub


def main():
    args = parse_args()
    config = _load_config(args.config)
    baseline_cfg = config.get("baselines", {})
    seeds = config.get("training", {}).get("seeds", [1234])

    quick = args.quick
    max_markers = args.max_markers if args.max_markers is not None else (500 if quick else 5000)
    rf_trees = args.rf_trees if args.rf_trees is not None else (50 if quick else 300)
    skip_gblup = quick
    skip_rf = False  # always include RF
    split_filter = set(args.split_type) if args.split_type else None
    seed_filter = set(args.seed) if args.seed else None
    model_filter = {m.strip() for m in args.models.split(",") if m.strip()} if args.models else None

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "baseline_metrics.csv"
    runtime_path = out_dir / "runtime.csv"
    existing_metrics = _read_existing_metrics(metrics_path)

    pheno_path = args.data_dir / "phenotype.parquet"
    if not pheno_path.exists():
        raise FileNotFoundError(f"Phenotype table not found: {pheno_path}")
    phenotype = read_table(pheno_path)
    validate_phenotype_table(phenotype)

    genotype = None
    geno_path = args.data_dir / "genotype.parquet"
    if geno_path.exists():
        genotype = read_table(geno_path)

    environment = None
    env_path = args.data_dir / "environment.parquet"
    if env_path.exists():
        environment = read_table(env_path)

    geno_wide = build_genotype_wide(genotype) if genotype is not None else None

    all_splits = load_split_table(args.split_dir)

    all_metrics: list[dict] = []
    run_manifests: list[dict] = []
    prediction_paths: list[str] = []
    runtime_records: list[dict] = []

    for (split_type, seed), split_group in all_splits.groupby(["split_type", "seed"]):
        if split_filter and str(split_type) not in split_filter:
            continue
        if seed_filter and int(seed) not in seed_filter:
            continue
        print(f"\n=== {split_type} seed={seed} ===", flush=True)

        train_ids = set(split_group.loc[split_group["split"] == "train", "sample_id"])
        test_ids = set(split_group.loc[split_group["split"] == "test", "sample_id"])

        train_pheno = phenotype[phenotype["sample_id"].isin(train_ids)].copy()
        test_pheno = phenotype[phenotype["sample_id"].isin(test_ids)].copy()

        if len(train_pheno) == 0 or len(test_pheno) == 0:
            print("  Skipping: empty train or test set")
            continue

        y_train = train_pheno["phenotype_value"].to_numpy(dtype=float)
        y_test = test_pheno["phenotype_value"].to_numpy(dtype=float)
        preprocessor_manifest: dict | None = None
        if geno_wide is not None:
            preprocessor = FoldPreprocessor(max_markers=max_markers)
            X_geno_train, _X_weather_train, X_env_train = preprocessor.fit_transform(
                train_pheno, geno_wide, environment, None
            )
            X_geno_test, _X_weather_test, X_env_test = preprocessor.transform(
                test_pheno, geno_wide, environment, None
            )
            X_train = np.column_stack([X_geno_train, X_env_train])
            X_test = np.column_stack([X_geno_test, X_env_test])
            preprocessor_manifest = preprocessor.to_manifest()
        else:
            X_train, y_train, _ = _prepare_features(train_pheno, genotype, environment, max_markers=max_markers)
            X_test, y_test, _ = _prepare_features(test_pheno, genotype, environment, max_markers=max_markers)

        if X_train.shape[1] < 1:
            print("  Skipping: no features available")
            continue

        # Simple baselines
        for name, fn in [
            ("global_mean", lambda: mean_baseline(train_pheno, test_pheno)),
        ]:
            if model_filter and name not in model_filter:
                continue
            if args.skip_existing and _metric_exists(existing_metrics, name, str(split_type), int(seed)):
                print(f"  {name}: SKIPPED (existing metric)", flush=True)
                continue
            try:
                start_time = time.perf_counter()
                result = fn()
                elapsed = time.perf_counter() - start_time
                m = metrics_dict(y_test, result.predictions)
                m["model"] = name
                m["split_type"] = split_type
                m["seed"] = seed
                all_metrics.append(m)
                pred_df = build_prediction_frame(
                    test_pheno["sample_id"].to_numpy(),
                    y_test,
                    result.predictions,
                    split_type=split_type,
                    seed=int(seed),
                    model=name,
                    target="phenotype",
                )
                pred_path = write_prediction_artifact(
                    pred_df,
                    out_dir / "predictions" / name / str(split_type) / f"seed{seed}",
                )
                prediction_paths.append(str(pred_path))
                runtime_records.append({
                    "model": name,
                    "split_type": split_type,
                    "seed": int(seed),
                    "target": "phenotype",
                    "time_s": float(elapsed),
                    "n_train": int(len(train_pheno)),
                    "n_test": int(len(test_pheno)),
                    "feature_dim": int(X_train.shape[1]),
                })
                existing_metrics = _upsert_frame(metrics_path, m, ["model", "split_type", "seed"])
                _upsert_frame(runtime_path, runtime_records[-1], ["model", "split_type", "seed", "target"])
                _write_json_records(metrics_path, out_dir / "baseline_metrics.json")
                print(f"  {name}: pearson={m['pearson']:.4f} rmse={m['rmse']:.4f}", flush=True)
            except Exception as e:
                print(f"  {name}: FAILED - {e}", flush=True)

        # Statistical baselines
        stat_models = baseline_cfg.get("statistical", ["bayes_ridge", "ridge_rrblup", "gblup_kernel_ridge"])
        for model_name in stat_models:
            if model_filter and model_name not in model_filter:
                continue
            if args.skip_existing and _metric_exists(existing_metrics, model_name, str(split_type), int(seed)):
                print(f"  {model_name}: SKIPPED (existing metric)", flush=True)
                continue
            if skip_gblup and model_name == "gblup_kernel_ridge":
                print(f"  {model_name}: SKIPPED (quick mode)", flush=True)
                continue
            try:
                start_time = time.perf_counter()
                if model_name == "bayes_ridge":
                    result = fit_bayes_ridge(X_train, y_train, X_test)
                elif model_name == "ridge_rrblup":
                    result = fit_ridge_rrblup_like(X_train, y_train, X_test)
                elif model_name == "gblup_kernel_ridge":
                    result = fit_gblup_efficient(X_train, y_train, X_test)
                else:
                    continue
                elapsed = time.perf_counter() - start_time
                m = metrics_dict(y_test, result.predictions)
                m["model"] = model_name
                m["split_type"] = split_type
                m["seed"] = seed
                all_metrics.append(m)
                pred_df = build_prediction_frame(
                    test_pheno["sample_id"].to_numpy(),
                    y_test,
                    result.predictions,
                    split_type=split_type,
                    seed=int(seed),
                    model=model_name,
                    target="phenotype",
                )
                pred_path = write_prediction_artifact(
                    pred_df,
                    out_dir / "predictions" / model_name / str(split_type) / f"seed{seed}",
                )
                prediction_paths.append(str(pred_path))
                runtime_records.append({
                    "model": model_name,
                    "split_type": split_type,
                    "seed": int(seed),
                    "target": "phenotype",
                    "time_s": float(elapsed),
                    "n_train": int(len(train_pheno)),
                    "n_test": int(len(test_pheno)),
                    "feature_dim": int(X_train.shape[1]),
                })
                if model_name == "gblup_kernel_ridge":
                    runtime_records[-1]["implementation"] = "marker_space_ridge_equivalent"
                existing_metrics = _upsert_frame(metrics_path, m, ["model", "split_type", "seed"])
                _upsert_frame(runtime_path, runtime_records[-1], ["model", "split_type", "seed", "target"])
                _write_json_records(metrics_path, out_dir / "baseline_metrics.json")
                print(f"  {model_name}: pearson={m['pearson']:.4f} rmse={m['rmse']:.4f}", flush=True)
            except Exception as e:
                print(f"  {model_name}: FAILED - {e}", flush=True)

        # ML baselines
        ml_models = baseline_cfg.get("machine_learning", ["random_forest"])
        for model_name in ml_models:
            if model_filter and model_name not in model_filter:
                continue
            if args.skip_existing and _metric_exists(existing_metrics, model_name, str(split_type), int(seed)):
                print(f"  {model_name}: SKIPPED (existing metric)", flush=True)
                continue
            if skip_rf and model_name == "random_forest":
                continue
            try:
                start_time = time.perf_counter()
                if model_name == "random_forest":
                    result = fit_random_forest(X_train, y_train, X_test, n_estimators=rf_trees)
                elif model_name in ("xgboost", "xgboost_optional"):
                    result = fit_xgboost(X_train, y_train, X_test, n_estimators=rf_trees)
                elif model_name in ("lightgbm", "lightgbm_optional"):
                    result = fit_lightgbm(X_train, y_train, X_test, n_estimators=rf_trees)
                else:
                    continue
                elapsed = time.perf_counter() - start_time
                m = metrics_dict(y_test, result.predictions)
                m["model"] = model_name
                m["split_type"] = split_type
                m["seed"] = seed
                all_metrics.append(m)
                pred_df = build_prediction_frame(
                    test_pheno["sample_id"].to_numpy(),
                    y_test,
                    result.predictions,
                    split_type=split_type,
                    seed=int(seed),
                    model=model_name,
                    target="phenotype",
                )
                pred_path = write_prediction_artifact(
                    pred_df,
                    out_dir / "predictions" / model_name / str(split_type) / f"seed{seed}",
                )
                prediction_paths.append(str(pred_path))
                runtime_records.append({
                    "model": model_name,
                    "split_type": split_type,
                    "seed": int(seed),
                    "target": "phenotype",
                    "time_s": float(elapsed),
                    "n_train": int(len(train_pheno)),
                    "n_test": int(len(test_pheno)),
                    "feature_dim": int(X_train.shape[1]),
                })
                existing_metrics = _upsert_frame(metrics_path, m, ["model", "split_type", "seed"])
                _upsert_frame(runtime_path, runtime_records[-1], ["model", "split_type", "seed", "target"])
                _write_json_records(metrics_path, out_dir / "baseline_metrics.json")
                print(f"  {model_name}: pearson={m['pearson']:.4f} rmse={m['rmse']:.4f}", flush=True)
            except ImportError:
                print(f"  {model_name}: SKIPPED (not installed)", flush=True)
            except Exception as e:
                print(f"  {model_name}: FAILED - {e}", flush=True)

        if preprocessor_manifest is not None:
            manifest = {
                "schema_version": 1,
                "script": "scripts/03_run_baselines.py",
                "split_type": split_type,
                "seed": int(seed),
                "data_dir": str(args.data_dir),
                "split_dir": str(args.split_dir),
                "n_train": int(len(train_pheno)),
                "n_test": int(len(test_pheno)),
                "feature_dim": int(X_train.shape[1]),
                "preprocessing": preprocessor_manifest,
                "prediction_artifact_count": int(len(prediction_paths)),
            }
            manifest_dir = out_dir / "manifests" / split_type / f"seed{seed}"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            (manifest_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8"
            )
            run_manifests.append(manifest)

    metrics_df = _read_existing_metrics(metrics_path)
    if len(metrics_df) > 0:
        _write_json_records(metrics_path, out_dir / "baseline_metrics.json")
        (out_dir / "run_manifests.json").write_text(
            json.dumps(run_manifests, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / "prediction_artifacts.json").write_text(
            json.dumps(prediction_paths, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nMetrics written to {metrics_path}")

        # Summary by model
        print("\nSummary (pearson r mean across splits):")
        for model, grp in metrics_df.groupby("model"):
            print(f"  {model}: {grp['pearson'].mean():.4f} ± {grp['pearson'].std():.4f}")
    else:
        print("No metrics generated.")


if __name__ == "__main__":
    main()
