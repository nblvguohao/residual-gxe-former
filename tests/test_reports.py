from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


def _load_reports_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "07_make_reports.py"
    spec = importlib.util.spec_from_file_location("make_reports", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_frame(model: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(y_true))],
            "y_true": y_true,
            "y_pred": y_pred,
            "split_type": "leave_environment",
            "seed": 1234,
            "model": model,
            "target": "phenotype",
        }
    )


def test_paired_report_comparisons_cover_core_and_selection_metrics(tmp_path: Path):
    module = _load_reports_script()
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    predictions = {
        "model_a": _prediction_frame("model_a", y_true, y_true),
        "model_b": _prediction_frame("model_b", y_true, y_true[::-1]),
    }
    out = module._write_paired_comparisons(predictions, tmp_path, n_bootstrap=20)
    expected_metrics = {
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "selection_gain_at_5pct",
        "selection_gain_at_10pct",
        "selection_gain_at_20pct",
        "ndcg_at_5pct",
        "ndcg_at_10pct",
        "ndcg_at_20pct",
    }
    assert expected_metrics <= set(out["metric"])
    assert "higher_is_better" in out.columns
    assert (tmp_path / "paired_model_comparisons.csv").exists()


def test_runtime_report_summary_aggregates_formal_runtime_tables(tmp_path: Path):
    module = _load_reports_script()
    runtime_tables = {
        "baselines": pd.DataFrame(
            {
                "model": ["ridge", "ridge", "rf"],
                "split_type": ["random", "random", "leave_environment"],
                "seed": [1, 1, 1],
                "time_s": [1.0, 3.0, 5.0],
                "peak_memory_mb": [100.0, 120.0, 300.0],
            }
        )
    }
    out = module._write_runtime_summary(runtime_tables, tmp_path)
    ridge = out[(out["model"] == "ridge") & (out["split_type"] == "random")].iloc[0]
    assert ridge["n_runs"] == 2
    assert ridge["mean_time_s"] == 2.0
    assert "mean_peak_memory_mb" in out.columns
    assert (tmp_path / "runtime_resources.csv").exists()


def test_prediction_discovery_prefers_phenotype_columns_without_duplicate_metric_arrays(tmp_path: Path):
    module = _load_reports_script()
    pred_dir = tmp_path / "residual_gxe_g2f" / "random" / "predictions" / "random" / "seed1234"
    pred_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "y_true": [0.1, 0.2, 0.3],
            "y_pred": [0.0, 0.0, 0.0],
            "y_true_phenotype": [1.0, 2.0, 3.0],
            "y_pred_phenotype": [1.1, 1.9, 3.2],
            "split_type": ["random", "random", "random"],
            "seed": [1234, 1234, 1234],
            "model": ["ResidualGxEFormer"] * 3,
            "target": ["phenotype"] * 3,
        }
    )
    df.to_parquet(pred_dir / "predictions.parquet", index=False)

    predictions = module._discover_prediction_files(tmp_path)
    assert len(predictions) == 1
    out = next(iter(predictions.values()))
    assert out["y_true"].to_numpy().ndim == 1
    assert out["y_pred"].to_numpy().ndim == 1
    assert out["y_true"].tolist() == [1.0, 2.0, 3.0]
    assert out["y_pred"].tolist() == [1.1, 1.9, 3.2]

    ci = module._write_bootstrap_tables(predictions, tmp_path / "reports", n_bootstrap=5, jobs=1)
    assert set(ci["metric"]) >= {"pearson", "rmse"}


def test_report_bootstrap_and_pairwise_parallel_paths(tmp_path: Path):
    if sys.platform.startswith("win"):
        pytest.skip("ProcessPool report workers are exercised on Linux server runs.")
    module = _load_reports_script()
    y_true = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    predictions = {
        "model_a": _prediction_frame("model_a", y_true, y_true),
        "model_b": _prediction_frame("model_b", y_true, y_true[::-1]),
    }
    boot = module._write_bootstrap_tables(predictions, tmp_path / "boot", n_bootstrap=3, jobs=2)
    paired = module._write_paired_comparisons(predictions, tmp_path / "paired", n_bootstrap=3, jobs=2)
    assert len(boot) == 20
    assert set(paired["metric"]) >= {"pearson", "rmse", "selection_gain_at_10pct"}
