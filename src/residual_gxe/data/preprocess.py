from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_train_apply(train: pd.Series, values: pd.Series) -> pd.Series:
    mean = train.mean()
    std = train.std(ddof=0)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (values - mean) / std


def basic_marker_filter(genotype_matrix: pd.DataFrame, missing_threshold: float = 0.20, maf_threshold: float = 0.01) -> pd.DataFrame:
    """Filter a wide genotype matrix with genotype_id as first column.

    Assumes marker dosages are numeric 0/1/2.
    """
    if "genotype_id" not in genotype_matrix.columns:
        raise ValueError("Expected genotype_id column")
    marker_cols = [c for c in genotype_matrix.columns if c != "genotype_id"]
    X = genotype_matrix[marker_cols]
    keep_missing = X.isna().mean(axis=0) <= missing_threshold
    X2 = X.loc[:, keep_missing]
    # Approximate MAF from dosage mean / 2.
    allele_freq = X2.mean(axis=0, skipna=True) / 2.0
    maf = np.minimum(allele_freq, 1.0 - allele_freq)
    keep_maf = maf >= maf_threshold
    kept_cols = list(X2.loc[:, keep_maf].columns)
    return genotype_matrix[["genotype_id"] + kept_cols].copy()


def mean_impute_markers(train_matrix: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    if "genotype_id" not in matrix.columns:
        raise ValueError("Expected genotype_id column")
    marker_cols = [c for c in matrix.columns if c != "genotype_id"]
    means = train_matrix[marker_cols].mean(axis=0)
    out = matrix.copy()
    out[marker_cols] = out[marker_cols].fillna(means)
    return out


def prune_ld(genotype_matrix: pd.DataFrame, window_size: int = 50, r2_threshold: float = 0.8) -> pd.DataFrame:
    """LD-based marker pruning: within a sliding window, keep the marker
    with highest MAF when r^2 exceeds threshold.

    This reduces multicollinearity while preserving informative markers.
    """
    if "genotype_id" not in genotype_matrix.columns:
        raise ValueError("Expected genotype_id column")
    marker_cols = [c for c in genotype_matrix.columns if c != "genotype_id"]
    X = genotype_matrix[marker_cols].fillna(0.0).to_numpy(dtype=np.float32)

    n_markers = len(marker_cols)
    kept = np.ones(n_markers, dtype=bool)
    # Compute MAF for tie-breaking
    allele_freq = X.mean(axis=0) / 2.0
    maf = np.minimum(allele_freq, 1.0 - allele_freq)
    # Standardize for correlation
    X_std = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-10)

    for i in range(n_markers):
        if not kept[i]:
            continue
        end = min(i + window_size, n_markers)
        if end - i <= 1:
            continue
        chunk = X_std[:, i + 1:end]
        target = X_std[:, i:i + 1]
        r2 = np.square(np.dot(target.T, chunk) / X_std.shape[0]).flatten()
        for j, r2_val in enumerate(r2):
            idx = i + 1 + j
            if r2_val > r2_threshold and kept[idx]:
                if maf[i] >= maf[idx]:
                    kept[idx] = False
                else:
                    kept[i] = False
                    break

    kept_cols = [c for c, k in zip(marker_cols, kept) if k]
    return genotype_matrix[["genotype_id"] + kept_cols].copy()


def select_markers_by_strategy(
    marker_cols: list,
    max_markers: int,
    strategy: str = "random",
    geno_wide: pd.DataFrame | None = None,
    window_size: int = 50,
    r2_threshold: float = 0.8,
) -> list:
    """Select markers by strategy: random, variance, ld_pruned, uniform_spacing.

    - random: random subsample
    - variance: top-N by per-marker variance
    - ld_pruned: LD-based pruning then random subsample to max_markers
    - uniform_spacing: evenly spaced across marker list
    """
    if len(marker_cols) <= max_markers:
        return list(marker_cols)

    rng = np.random.default_rng(42)

    if strategy == "variance" and geno_wide is not None:
        vars_ = geno_wide[marker_cols].var().sort_values(ascending=False)
        return list(vars_.index[:max_markers])

    if strategy == "ld_pruned" and geno_wide is not None:
        pruned = prune_ld(geno_wide[["genotype_id"] + marker_cols], window_size, r2_threshold)
        pruned_cols = [c for c in pruned.columns if c != "genotype_id"]
        if len(pruned_cols) > max_markers:
            return list(rng.choice(pruned_cols, size=max_markers, replace=False))
        return pruned_cols

    if strategy == "uniform_spacing":
        step = len(marker_cols) / max_markers
        return [marker_cols[int(i * step)] for i in range(max_markers)]

    # default: random
    return list(rng.choice(marker_cols, size=max_markers, replace=False))
