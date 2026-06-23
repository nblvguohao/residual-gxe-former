from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.metrics import (
    bootstrap_metrics_ci,
    grouped_metrics,
    ndcg_at_k,
    paired_bootstrap_difference,
    regression_metrics,
    safe_spearman,
    safe_pearson,
    selection_gain_at_fraction,
)
from residual_gxe.evaluation.reporting import weighted_score


def parse_args():
    parser = argparse.ArgumentParser(description="Create manuscript-ready tables and reports.")
    parser.add_argument("--config", type=Path, required=False, default=None)
    parser.add_argument("--raw-dir", type=Path, required=False, default=None)
    parser.add_argument("--data-dir", type=Path, required=False, default=None)
    parser.add_argument("--split-dir", type=Path, required=False, default=None)
    parser.add_argument("--residual-dir", type=Path, required=False, default=None)
    parser.add_argument("--results-dir", type=Path, required=True, help="Root directory containing all results (baselines/, residual_gxe/, ablations/)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=1, help="CPU worker processes for bootstrap and paired comparisons.")
    return parser.parse_args()


def _report_progress(message: str) -> None:
    print(message, flush=True)


def _can_use_process_pool() -> bool:
    """Return whether module-level worker functions are pickle-importable.

    Pytest loads this script dynamically as ``make_reports`` in one test. On
    Python 3.13, ProcessPoolExecutor pickles functions by module name, and that
    dynamic module is not necessarily importable in worker processes. In that
    case, use the same deterministic serial path instead of failing preflight.
    """
    module = sys.modules.get(__name__)
    if module is None:
        return False
    return (
        getattr(module, "_bootstrap_prediction_task", None) is _bootstrap_prediction_task
        and getattr(module, "_paired_comparison_task", None) is _paired_comparison_task
    )


def _discover_metrics_files(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Scan results directory and load all metrics CSV/parquet files."""
    found: dict[str, pd.DataFrame] = {}
    for pattern in ["**/metrics.csv", "**/baseline_metrics.csv", "**/ablation_metrics.csv"]:
        for fpath in results_dir.glob(pattern):
            key = str(fpath.parent.relative_to(results_dir))
            try:
                df = read_table(fpath)
                found[key] = df
            except Exception:
                continue
    return found


def _discover_prediction_files(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load prediction files with normalized y_true/y_pred columns."""
    found: dict[str, pd.DataFrame] = {}
    for fpath in results_dir.glob("**/predictions.parquet"):
        key = str(fpath.parent.relative_to(results_dir)).replace("\\", "/")
        try:
            df = read_table(fpath)
        except Exception:
            continue
        if "sample_id" not in df.columns:
            continue
        df = df.copy()
        if "y_true_phenotype" in df.columns and "y_pred_phenotype" in df.columns:
            df["y_true"] = pd.to_numeric(df["y_true_phenotype"], errors="coerce")
            df["y_pred"] = pd.to_numeric(df["y_pred_phenotype"], errors="coerce")
        elif not {"y_true", "y_pred"} <= set(df.columns):
            continue
        else:
            df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
            df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
        if df[["y_true", "y_pred"]].isna().any(axis=None):
            df = df.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
        keep_cols = [
            col
            for col in ["sample_id", "y_true", "y_pred", "split_type", "seed", "model", "target"]
            if col in df.columns
        ]
        extra_cols = [
            col
            for col in df.columns
            if col not in keep_cols and col not in {"y_true_phenotype", "y_pred_phenotype"}
        ]
        df = df[keep_cols + extra_cols]
        df["prediction_key"] = key
        found[key] = df
    return found


def _discover_runtime_files(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load formal runtime/resource files and annotate their source directory."""
    found: dict[str, pd.DataFrame] = {}
    for pattern in ["**/runtime.csv", "**/runtime_resources.csv", "**/resource_usage.csv"]:
        for fpath in results_dir.glob(pattern):
            key = str(fpath.parent.relative_to(results_dir)).replace("\\", "/")
            try:
                df = read_table(fpath)
            except Exception:
                continue
            if "time_s" not in df.columns and not {"runtime_s", "elapsed_s", "wall_time_s"} & set(df.columns):
                continue
            df = df.copy()
            df["runtime_key"] = key
            df["runtime_file"] = str(fpath).replace("\\", "/")
            if "time_s" not in df.columns:
                for col in ["runtime_s", "elapsed_s", "wall_time_s"]:
                    if col in df.columns:
                        df["time_s"] = df[col]
                        break
            found[key] = df
    return found


def _write_runtime_summary(runtime_tables: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    frames = []
    for _, df in runtime_tables.items():
        frames.append(df.copy())
    if not frames:
        return pd.DataFrame()
    runtime = pd.concat(frames, ignore_index=True)
    if "model" not in runtime.columns:
        runtime["model"] = runtime["runtime_key"]
    if "split_type" not in runtime.columns:
        runtime["split_type"] = "not_recorded"
    if "seed" not in runtime.columns:
        runtime["seed"] = -1

    group_cols = [c for c in ["model", "split_type", "seed"] if c in runtime.columns]
    summary = (
        runtime.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("time_s", "count"),
            mean_time_s=("time_s", "mean"),
            median_time_s=("time_s", "median"),
            max_time_s=("time_s", "max"),
            min_time_s=("time_s", "min"),
        )
        .reset_index()
    )
    optional_cols = [c for c in ["memory_mb", "peak_memory_mb", "n_train", "n_test", "feature_dim"] if c in runtime.columns]
    for col in optional_cols:
        values = runtime.groupby(group_cols, dropna=False)[col].mean().reset_index(name=f"mean_{col}")
        summary = summary.merge(values, on=group_cols, how="left")

    write_table(summary, out_dir / "runtime_resources.csv")
    return summary


def _write_bootstrap_tables(
    predictions: dict[str, pd.DataFrame],
    out_dir: Path,
    n_bootstrap: int = 1000,
    jobs: int = 1,
) -> pd.DataFrame:
    tasks = []
    for key, df in predictions.items():
        tasks.append(
            {
                "key": key,
                "y_true": df["y_true"].to_numpy(dtype=float),
                "y_pred": df["y_pred"].to_numpy(dtype=float),
                "split_type": df["split_type"].iloc[0] if "split_type" in df.columns and len(df) else None,
                "seed": int(df["seed"].iloc[0]) if "seed" in df.columns and len(df) else None,
                "n_bootstrap": int(n_bootstrap),
            }
        )
    rows = []
    use_process_pool = jobs > 1 and len(tasks) > 1 and _can_use_process_pool()
    if jobs > 1 and len(tasks) > 1 and not use_process_pool:
        _report_progress("Bootstrap CIs: falling back to jobs=1 because report workers are not importable.")
    if use_process_pool:
        _report_progress(f"Bootstrap CIs: {len(tasks)} prediction files, {n_bootstrap} iterations, jobs={jobs}")
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_bootstrap_prediction_task, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if i == 1 or i % 10 == 0 or i == len(futures):
                    _report_progress(f"  bootstrap completed {i}/{len(futures)}")
    else:
        _report_progress(f"Bootstrap CIs: {len(tasks)} prediction files, {n_bootstrap} iterations, jobs=1")
        for i, task in enumerate(tasks, start=1):
            rows.append(_bootstrap_prediction_task(task))
            if i == 1 or i % 10 == 0 or i == len(tasks):
                _report_progress(f"  bootstrap completed {i}/{len(tasks)}")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    write_table(out, out_dir / "bootstrap_metric_ci.csv")
    return out


def _bootstrap_prediction_task(task: dict) -> pd.DataFrame:
    ci = bootstrap_metrics_ci(
        task["y_true"],
        task["y_pred"],
        n_bootstrap=task["n_bootstrap"],
        seed=1234,
    )
    ci["prediction_key"] = task["key"]
    ci["split_type"] = task["split_type"]
    ci["seed"] = task["seed"]
    return ci


def _write_grouped_diagnostics(
    predictions: dict[str, pd.DataFrame],
    phenotype: pd.DataFrame | None,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    if phenotype is None:
        return {}
    meta_cols = [c for c in ["sample_id", "environment_id", "genotype_id", "year", "location_id"] if c in phenotype.columns]
    if "sample_id" not in meta_cols:
        return {}

    env_rows = []
    geno_rows = []
    for key, df in predictions.items():
        merged = df.merge(phenotype[meta_cols], on="sample_id", how="left")
        for group_col, sink in [("environment_id", env_rows), ("genotype_id", geno_rows)]:
            if group_col not in merged.columns:
                continue
            out = grouped_metrics(merged, group_col, y_col="y_true", pred_col="y_pred", min_n=3)
            out["prediction_key"] = key
            out["split_type"] = merged["split_type"].iloc[0] if "split_type" in merged.columns and len(merged) else None
            out["seed"] = int(merged["seed"].iloc[0]) if "seed" in merged.columns and len(merged) else None
            sink.append(out)

    outputs: dict[str, pd.DataFrame] = {}
    if env_rows:
        env_df = pd.concat(env_rows, ignore_index=True)
        write_table(env_df, out_dir / "environment_wise_metrics.csv")
        outputs["environment"] = env_df
    if geno_rows:
        geno_df = pd.concat(geno_rows, ignore_index=True)
        write_table(geno_df, out_dir / "genotype_wise_metrics.csv")
        outputs["genotype"] = geno_df
    return outputs


def _write_paired_comparisons(
    predictions: dict[str, pd.DataFrame],
    out_dir: Path,
    n_bootstrap: int = 1000,
    jobs: int = 1,
) -> pd.DataFrame:
    by_split_seed: dict[tuple[str | None, int | None], list[tuple[str, pd.DataFrame]]] = {}
    for key, df in predictions.items():
        split_type = df["split_type"].iloc[0] if "split_type" in df.columns and len(df) else None
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns and len(df) else None
        by_split_seed.setdefault((split_type, seed), []).append((key, df))

    tasks = []
    for (split_type, seed), items in by_split_seed.items():
        if len(items) < 2:
            continue
        for i, (key_a, df_a) in enumerate(items):
            for key_b, df_b in items[i + 1:]:
                merged = df_a[["sample_id", "y_true", "y_pred"]].merge(
                    df_b[["sample_id", "y_pred"]],
                    on="sample_id",
                    how="inner",
                    suffixes=("_a", "_b"),
                )
                if len(merged) < 3:
                    continue
                tasks.append(
                    {
                        "split_type": split_type,
                        "seed": seed,
                        "model_a": key_a,
                        "model_b": key_b,
                        "y_true": merged["y_true"].to_numpy(dtype=float),
                        "y_pred_a": merged["y_pred_a"].to_numpy(dtype=float),
                        "y_pred_b": merged["y_pred_b"].to_numpy(dtype=float),
                        "n_bootstrap": int(n_bootstrap),
                    }
                )
    rows = []
    use_process_pool = jobs > 1 and len(tasks) > 1 and _can_use_process_pool()
    if jobs > 1 and len(tasks) > 1 and not use_process_pool:
        _report_progress("Paired comparisons: falling back to jobs=1 because report workers are not importable.")
    if use_process_pool:
        _report_progress(f"Paired comparisons: {len(tasks)} model pairs, {n_bootstrap} iterations, jobs={jobs}")
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_paired_comparison_task, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                if i == 1 or i % 10 == 0 or i == len(futures):
                    _report_progress(f"  paired comparisons completed {i}/{len(futures)}")
    else:
        _report_progress(f"Paired comparisons: {len(tasks)} model pairs, {n_bootstrap} iterations, jobs=1")
        for i, task in enumerate(tasks, start=1):
            rows.extend(_paired_comparison_task(task))
            if i == 1 or i % 10 == 0 or i == len(tasks):
                _report_progress(f"  paired comparisons completed {i}/{len(tasks)}")
    out = pd.DataFrame(rows)
    if len(out) > 0:
        write_table(out, out_dir / "paired_model_comparisons.csv")
    return out


def _paired_comparison_task(task: dict) -> list[dict]:
    metric_fns = {
        "pearson": (safe_pearson, True),
        "spearman": (safe_spearman, True),
        "rmse": (lambda yt, yp: regression_metrics(yt, yp).rmse, False),
        "mae": (lambda yt, yp: regression_metrics(yt, yp).mae, False),
        "selection_gain_at_5pct": (lambda yt, yp: selection_gain_at_fraction(yt, yp, 0.05), True),
        "selection_gain_at_10pct": (lambda yt, yp: selection_gain_at_fraction(yt, yp, 0.10), True),
        "selection_gain_at_20pct": (lambda yt, yp: selection_gain_at_fraction(yt, yp, 0.20), True),
        "ndcg_at_5pct": (lambda yt, yp: ndcg_at_k(yt, yp, max(1, int(np.ceil(len(yt) * 0.05)))), True),
        "ndcg_at_10pct": (lambda yt, yp: ndcg_at_k(yt, yp, max(1, int(np.ceil(len(yt) * 0.10)))), True),
        "ndcg_at_20pct": (lambda yt, yp: ndcg_at_k(yt, yp, max(1, int(np.ceil(len(yt) * 0.20)))), True),
    }
    rows = []
    for metric_name, (metric_fn, higher_is_better) in metric_fns.items():
        diff = paired_bootstrap_difference(
            task["y_true"],
            task["y_pred_a"],
            task["y_pred_b"],
            metric_fn=metric_fn,
            n_bootstrap=task["n_bootstrap"],
            seed=1234,
        )
        rows.append(
            {
                "split_type": task["split_type"],
                "seed": task["seed"],
                "model_a": task["model_a"],
                "model_b": task["model_b"],
                "metric": metric_name,
                "higher_is_better": higher_is_better,
                **diff,
            }
        )
    return rows


def _pivot_metrics(df: pd.DataFrame, metric: str, model_col: str = "model") -> pd.DataFrame:
    """Pivot metrics by split_type, showing mean +/- std."""
    if model_col not in df.columns:
        # Try 'ablation' or 'target'
        for col in ["ablation", "target"]:
            if col in df.columns:
                model_col = col
                break
        else:
            return pd.DataFrame()

    if metric not in df.columns:
        return pd.DataFrame()

    rows = []
    for model, grp in df.groupby(model_col):
        row: dict = {"model": model}
        for st, st_grp in grp.groupby("split_type"):
            vals = st_grp[metric].dropna()
            if len(vals) > 0:
                row[st] = f"{vals.mean():.4f} +/- {vals.std():.4f}"
        row["overall"] = f"{grp[metric].dropna().mean():.4f} +/- {grp[metric].dropna().std():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_metrics = _discover_metrics_files(results_dir)
    all_predictions = _discover_prediction_files(results_dir)
    all_runtime = _discover_runtime_files(results_dir)

    report_lines = [
        "# ResidualGxE-Former Results Report",
        "",
        f"Results directory: `{results_dir}`",
        "",
        "All formal tables in this report are generated from metrics and prediction files.",
        "Legacy hard-coded figure scripts are intentionally excluded.",
        "",
    ]

    if not all_metrics and not all_predictions and not all_runtime:
        report_lines.append("No metrics, prediction, or runtime files found. Run baselines and model training first.")
        report_path = out_dir / "results_report.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(report_path)
        return

    if all_metrics:
        report_lines.append(f"## Data sources ({len(all_metrics)} found)")
        for key in sorted(all_metrics.keys()):
            df = all_metrics[key]
            report_lines.append(f"- **{key}**: {len(df)} rows, columns={list(df.columns)}")
        report_lines.append("")
    else:
        report_lines.append("No metrics files found; prediction-level diagnostics only.")
        report_lines.append("")

    if all_runtime:
        runtime_summary = _write_runtime_summary(all_runtime, out_dir)
        report_lines.append("## Runtime And Resource Summary")
        report_lines.append("")
        report_lines.append(f"Saved: `runtime_resources.csv` ({len(runtime_summary)} rows).")
        if len(runtime_summary) > 0:
            cols = runtime_summary.columns.tolist()
            report_lines.append("| " + " | ".join(cols) + " |")
            report_lines.append("|" + "|".join(["---" for _ in cols]) + "|")
            for _, row in runtime_summary.head(30).iterrows():
                report_lines.append(
                    "| "
                    + " | ".join(
                        f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                        for c in cols
                    )
                    + " |"
                )
        report_lines.append("")

    # ---- Dataset summary ----
    pheno_path = args.data_dir / "phenotype.parquet" if args.data_dir else None
    pheno = None
    if pheno_path and pheno_path.exists():
        pheno = read_table(pheno_path)
        report_lines.append("## Dataset Summary")
        report_lines.append(f"- Samples: {len(pheno)}")
        report_lines.append(f"- Genotypes: {pheno['genotype_id'].nunique()}")
        report_lines.append(f"- Environments: {pheno['environment_id'].nunique()}")
        if "year" in pheno.columns:
            report_lines.append(f"- Years: {sorted(pheno['year'].dropna().unique().astype(int))}")
        report_lines.append(f"- Mean phenotype: {pheno['phenotype_value'].mean():.2f}")
        report_lines.append(f"- Std phenotype: {pheno['phenotype_value'].std():.2f}")
        report_lines.append("")

    if all_predictions:
        report_lines.append(f"## Prediction Files ({len(all_predictions)} found)")
        for key, df in sorted(all_predictions.items()):
            report_lines.append(f"- **{key}**: {len(df)} rows")
        report_lines.append("")

        bootstrap_df = _write_bootstrap_tables(
            all_predictions,
            out_dir,
            n_bootstrap=args.bootstrap_iterations,
            jobs=max(1, int(args.jobs)),
        )
        grouped_outputs = _write_grouped_diagnostics(all_predictions, pheno, out_dir)
        paired_df = _write_paired_comparisons(
            all_predictions,
            out_dir,
            n_bootstrap=args.bootstrap_iterations,
            jobs=max(1, int(args.jobs)),
        )

        if len(bootstrap_df) > 0:
            report_lines.append("## Bootstrap 95% Confidence Intervals")
            report_lines.append("")
            report_lines.append("Saved: `bootstrap_metric_ci.csv`")
            pearson_ci = bootstrap_df[bootstrap_df["metric"] == "pearson"]
            if len(pearson_ci) > 0:
                cols = ["prediction_key", "split_type", "seed", "estimate", "ci_low", "ci_high", "n"]
                report_lines.append("| " + " | ".join(cols) + " |")
                report_lines.append("|" + "|".join(["---" for _ in cols]) + "|")
                for _, row in pearson_ci.iterrows():
                    report_lines.append(
                        "| "
                        + " | ".join(
                            f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                            for c in cols
                        )
                        + " |"
                    )
                report_lines.append("")

        if grouped_outputs:
            report_lines.append("## Group-Wise Diagnostics")
            report_lines.append("")
            if "environment" in grouped_outputs:
                report_lines.append(f"- Environment-wise metrics saved: `environment_wise_metrics.csv` ({len(grouped_outputs['environment'])} rows)")
            if "genotype" in grouped_outputs:
                report_lines.append(f"- Genotype-wise metrics saved: `genotype_wise_metrics.csv` ({len(grouped_outputs['genotype'])} rows)")
            report_lines.append("")

        if len(paired_df) > 0:
            report_lines.append("## Paired Model Comparisons")
            report_lines.append("")
            report_lines.append("Saved: `paired_model_comparisons.csv`. Difference is `model_a - model_b`; use `higher_is_better` to interpret direction.")
            preview = paired_df.head(20)
            cols = ["split_type", "seed", "model_a", "model_b", "metric", "higher_is_better", "estimate", "ci_low", "ci_high", "p_two_sided", "n"]
            report_lines.append("| " + " | ".join(cols) + " |")
            report_lines.append("|" + "|".join(["---" for _ in cols]) + "|")
            for _, row in preview.iterrows():
                report_lines.append(
                    "| "
                    + " | ".join(
                        f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                        for c in cols
                    )
                    + " |"
                )
            report_lines.append("")

    # ---- Main results table ----
    report_lines.append("## Main Results: Pearson Correlation")
    report_lines.append("")

    for key, df in sorted(all_metrics.items()):
        table = _pivot_metrics(df, "pearson")
        if table.empty:
            continue
        report_lines.append(f"### {key}")
        report_lines.append("")
        # Render as markdown table
        cols = table.columns.tolist()
        report_lines.append("| " + " | ".join(cols) + " |")
        report_lines.append("|" + "|".join(["---" for _ in cols]) + "|")
        for _, row in table.iterrows():
            report_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        report_lines.append("")

        # Save as CSV for manuscript
        table_path = out_dir / f"{key.replace('/', '_')}_pearson.csv"
        write_table(table, table_path)

    # ---- RMSE table ----
    report_lines.append("## Main Results: RMSE")
    report_lines.append("")
    for key, df in sorted(all_metrics.items()):
        table = _pivot_metrics(df, "rmse")
        if table.empty:
            continue
        report_lines.append(f"### {key}")
        report_lines.append("")
        cols = table.columns.tolist()
        report_lines.append("| " + " | ".join(cols) + " |")
        report_lines.append("|" + "|".join(["---" for _ in cols]) + "|")
        for _, row in table.iterrows():
            report_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        report_lines.append("")

    # ---- Selection gain table ----
    report_lines.append("## Selection Utility")
    report_lines.append("")
    for key, df in sorted(all_metrics.items()):
        sel_cols = [c for c in df.columns if "selection_gain" in c]
        if not sel_cols:
            continue
        report_lines.append(f"### {key}")
        for sc in sel_cols:
            table = _pivot_metrics(df, sc)
            if table.empty:
                continue
            report_lines.append(f"**{sc}**")
            report_lines.append("")
            cols = table.columns.tolist()
            report_lines.append("| " + " | ".join(cols) + " |")
            report_lines.append("|" + "|".join(["---" for _ in cols]) + "|")
            for _, row in table.iterrows():
                report_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
            report_lines.append("")

    # ---- Weighted score ----
    report_lines.append("## Weighted Score")
    report_lines.append("")
    for key, df in sorted(all_metrics.items()):
        if "pearson" not in df.columns:
            continue
        model_col = "model" if "model" in df.columns else ("ablation" if "ablation" in df.columns else None)
        if model_col is None:
            continue
        scores = []
        for model, grp in df.groupby(model_col):
            avg_by_split: dict = {}
            for st, st_grp in grp.groupby("split_type"):
                avg_by_split[st] = float(st_grp["pearson"].mean())
            scores.append({"model": model, "weighted_score": weighted_score(avg_by_split), **avg_by_split})
        scores_df = pd.DataFrame(scores)
        report_lines.append(f"### {key}")
        report_lines.append("")
        cols = scores_df.columns.tolist()
        report_lines.append("| " + " | ".join(cols) + " |")
        report_lines.append("|" + "|".join(["---" for _ in cols]) + "|")
        for _, row in scores_df.iterrows():
            report_lines.append("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |")
        report_lines.append("")
        scores_path = out_dir / f"{key.replace('/', '_')}_weighted_scores.csv"
        write_table(scores_df, scores_path)

    report_path = out_dir / "results_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
