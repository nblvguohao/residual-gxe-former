from __future__ import annotations

import numpy as np
import pandas as pd

from residual_gxe.evaluation.metrics import (
    bootstrap_metrics_ci,
    grouped_metrics,
    ndcg_at_k,
    paired_bootstrap_difference,
    regression_metrics,
    selection_gain_at_fraction,
    safe_pearson,
)


def test_regression_metrics_perfect():
    y = np.array([1, 2, 3, 4], dtype=float)
    m = regression_metrics(y, y)
    assert m.pearson > 0.999
    assert m.spearman > 0.999
    assert m.rmse == 0
    assert m.mae == 0


def test_selection_gain_positive_for_good_ranking():
    y = np.array([0, 1, 2, 10], dtype=float)
    pred = np.array([0, 1, 2, 10], dtype=float)
    gain = selection_gain_at_fraction(y, pred, 0.25)
    assert gain > 0


def test_ndcg_perfect_ranking():
    y = np.array([3, 2, 1], dtype=float)
    pred = np.array([3, 2, 1], dtype=float)
    assert ndcg_at_k(y, pred, 3) == 1.0


def test_bootstrap_metrics_ci_contains_perfect_pearson():
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    ci = bootstrap_metrics_ci(y, y, n_bootstrap=50, seed=1)
    pearson = ci[ci["metric"] == "pearson"].iloc[0]
    assert pearson["estimate"] > 0.999
    assert pearson["ci_low"] > 0.999
    assert pearson["ci_high"] > 0.999
    assert {
        "selection_gain_at_5pct",
        "selection_gain_at_10pct",
        "selection_gain_at_20pct",
        "ndcg_at_5pct",
        "ndcg_at_10pct",
        "ndcg_at_20pct",
    } <= set(ci["metric"])


def test_paired_bootstrap_difference_favors_better_predictions():
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    good = y.copy()
    bad = y[::-1]
    diff = paired_bootstrap_difference(y, good, bad, metric_fn=safe_pearson, n_bootstrap=100, seed=2)
    assert diff["estimate"] > 1.5
    assert diff["ci_low"] > 0


def test_grouped_metrics_returns_nan_for_tiny_groups():
    df = pd.DataFrame(
        {
            "environment_id": ["e1", "e1", "e1", "e2"],
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = grouped_metrics(df, "environment_id", min_n=3)
    e1 = out[out["environment_id"] == "e1"].iloc[0]
    e2 = out[out["environment_id"] == "e2"].iloc[0]
    assert e1["pearson"] > 0.999
    assert np.isnan(e2["pearson"])
