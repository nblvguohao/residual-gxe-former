from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class RegressionMetrics:
    pearson: float
    spearman: float
    rmse: float
    mae: float


def safe_pearson(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def safe_spearman(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return RegressionMetrics(
        pearson=safe_pearson(y_true, y_pred),
        spearman=safe_spearman(y_true, y_pred),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
    )


def ndcg_at_k(y_true, y_score, k: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if k <= 0 or len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    ideal = np.argsort(-y_true)[:k]
    gains = y_true[order]
    ideal_gains = y_true[ideal]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    idcg = float(np.sum(ideal_gains * discounts))
    if idcg == 0:
        return float("nan")
    return dcg / idcg


def selection_gain_at_fraction(y_true, y_score, fraction: float = 0.10) -> float:
    """Return relative improvement in observed phenotype mean among selected samples.

    Selection gain = mean(true phenotype of top predicted fraction) - population mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    k = max(1, int(math.ceil(len(y_true) * fraction)))
    top_idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[top_idx]) - np.mean(y_true))


def grouped_pearson(df: pd.DataFrame, group_col: str, y_col: str = "y_true", pred_col: str = "y_pred") -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col):
        rows.append({
            group_col: group,
            "n": len(sub),
            "pearson": safe_pearson(sub[y_col], sub[pred_col]),
            "spearman": safe_spearman(sub[y_col], sub[pred_col]),
        })
    return pd.DataFrame(rows)


def grouped_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_col: str = "y_true",
    pred_col: str = "y_pred",
    min_n: int = 3,
) -> pd.DataFrame:
    """Compute regression metrics within each group."""
    rows = []
    for group, sub in df.groupby(group_col):
        row: dict[str, float | int | str] = {group_col: group, "n": int(len(sub))}
        if len(sub) >= min_n:
            row.update(metrics_dict(sub[y_col].to_numpy(), sub[pred_col].to_numpy()))
        else:
            row.update({
                "pearson": float("nan"),
                "spearman": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> dict[str, float]:
    """Bootstrap a scalar metric over paired samples."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    estimates = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        value = float(metric_fn(y_true[idx], y_pred[idx]))
        if value == value:
            estimates.append(value)

    estimate = float(metric_fn(y_true, y_pred))
    if not estimates:
        return {"estimate": estimate, "ci_low": float("nan"), "ci_high": float("nan"), "n": n}

    alpha = 1.0 - confidence
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(estimates, alpha / 2)),
        "ci_high": float(np.quantile(estimates, 1 - alpha / 2)),
        "n": n,
    }


def bootstrap_metrics_ci(
    y_true,
    y_pred,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for the core regression metrics."""
    metric_fns = {
        "pearson": safe_pearson,
        "spearman": safe_spearman,
        "rmse": lambda yt, yp: regression_metrics(yt, yp).rmse,
        "mae": lambda yt, yp: regression_metrics(yt, yp).mae,
        "selection_gain_at_5pct": lambda yt, yp: selection_gain_at_fraction(yt, yp, 0.05),
        "selection_gain_at_10pct": lambda yt, yp: selection_gain_at_fraction(yt, yp, 0.10),
        "selection_gain_at_20pct": lambda yt, yp: selection_gain_at_fraction(yt, yp, 0.20),
        "ndcg_at_5pct": lambda yt, yp: ndcg_at_k(yt, yp, max(1, int(math.ceil(len(yt) * 0.05)))),
        "ndcg_at_10pct": lambda yt, yp: ndcg_at_k(yt, yp, max(1, int(math.ceil(len(yt) * 0.10)))),
        "ndcg_at_20pct": lambda yt, yp: ndcg_at_k(yt, yp, max(1, int(math.ceil(len(yt) * 0.20)))),
    }
    rows = []
    for i, (name, fn) in enumerate(metric_fns.items()):
        row = bootstrap_metric_ci(
            y_true,
            y_pred,
            fn,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + i,
        )
        row["metric"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    y_true,
    y_pred_a,
    y_pred_b,
    metric_fn=safe_pearson,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> dict[str, float]:
    """Bootstrap paired metric difference: metric(a) - metric(b)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)
    if len(y_true) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_two_sided": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diff = float(metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx]))
        if diff == diff:
            diffs.append(diff)

    estimate = float(metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b))
    if not diffs:
        return {"estimate": estimate, "ci_low": float("nan"), "ci_high": float("nan"), "p_two_sided": float("nan"), "n": n}

    diffs_arr = np.asarray(diffs, dtype=float)
    alpha = 1.0 - confidence
    p_pos = float(np.mean(diffs_arr >= 0))
    p_neg = float(np.mean(diffs_arr <= 0))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(diffs_arr, alpha / 2)),
        "ci_high": float(np.quantile(diffs_arr, 1 - alpha / 2)),
        "p_two_sided": float(min(1.0, 2.0 * min(p_pos, p_neg))),
        "n": n,
    }


def metrics_dict(y_true, y_pred, topk_fractions=(0.05, 0.10, 0.20)) -> dict[str, float]:
    base = regression_metrics(y_true, y_pred)
    out = {
        "pearson": base.pearson,
        "spearman": base.spearman,
        "rmse": base.rmse,
        "mae": base.mae,
    }
    n = len(y_true)
    for frac in topk_fractions:
        pct = int(round(frac * 100))
        k = max(1, int(math.ceil(n * frac)))
        out[f"selection_gain_at_{pct}pct"] = selection_gain_at_fraction(y_true, y_pred, frac)
        out[f"ndcg_at_{pct}pct"] = ndcg_at_k(y_true, y_pred, k)
    return out
