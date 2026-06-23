from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

SplitType = Literal["random", "leave_genotype", "leave_environment", "leave_year", "leave_ge"]


@dataclass(frozen=True)
class SplitConfig:
    split_type: SplitType
    seed: int = 1234
    val_fraction: float = 0.15
    test_fraction: float = 0.20


def _assign_train_val_test(ids: np.ndarray, seed: int, val_fraction: float, test_fraction: float):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(pd.unique(ids)))
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test_fraction))) if n > 2 else 1
    n_val = max(1, int(round(n * val_fraction))) if n > 3 else 1
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])
    train_ids = set(ids[n_test + n_val:])
    return train_ids, val_ids, test_ids


def make_split_table(pheno: pd.DataFrame, config: SplitConfig, fold: int = 0) -> pd.DataFrame:
    """Create deterministic split table from a phenotype dataframe.

    This function is intentionally conservative. It does not stratify by trait yet.
    Extend this for production use if stricter stratification is needed.
    """
    required = {"sample_id", "genotype_id", "environment_id", "year"}
    missing = required - set(pheno.columns)
    if missing:
        raise ValueError(f"Phenotype table missing required columns: {sorted(missing)}")

    df = pheno[["sample_id", "genotype_id", "environment_id", "year"]].copy()
    split_type = config.split_type

    if split_type == "random":
        train_ids, val_ids, test_ids = _assign_train_val_test(
            df["sample_id"].to_numpy(), config.seed, config.val_fraction, config.test_fraction
        )
        key_col = "sample_id"
    elif split_type == "leave_genotype":
        train_ids, val_ids, test_ids = _assign_train_val_test(
            df["genotype_id"].to_numpy(), config.seed, config.val_fraction, config.test_fraction
        )
        key_col = "genotype_id"
    elif split_type == "leave_environment":
        train_ids, val_ids, test_ids = _assign_train_val_test(
            df["environment_id"].to_numpy(), config.seed, config.val_fraction, config.test_fraction
        )
        key_col = "environment_id"
    elif split_type == "leave_year":
        years = np.array(sorted(pd.unique(df["year"])))
        if len(years) < 3:
            raise ValueError("leave_year split requires at least 3 years")
        test_ids = {years[-1]}
        val_ids = {years[-2]}
        train_ids = set(years[:-2])
        key_col = "year"
    elif split_type == "leave_ge":
        # Hold out both genotypes and environments; val is separate sets.
        g_train, g_val, g_test = _assign_train_val_test(
            df["genotype_id"].to_numpy(), config.seed, config.val_fraction, config.test_fraction
        )
        e_train, e_val, e_test = _assign_train_val_test(
            df["environment_id"].to_numpy(), config.seed + 17, config.val_fraction, config.test_fraction
        )
        labels = []
        for _, row in df.iterrows():
            g = row["genotype_id"]
            e = row["environment_id"]
            if g in g_test and e in e_test:
                labels.append("test")
            elif g in g_val and e in e_val:
                labels.append("val")
            elif g in g_train and e in e_train:
                labels.append("train")
            else:
                # Exclude ambiguous cross-region samples from training to avoid leakage.
                labels.append("unused")
        out = pd.DataFrame({
            "sample_id": df["sample_id"],
            "split": labels,
            "fold": fold,
            "reason": split_type,
        })
        return out[out["split"] != "unused"].reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported split type: {split_type}")

    def label(value):
        if value in test_ids:
            return "test"
        if value in val_ids:
            return "val"
        return "train"

    return pd.DataFrame({
        "sample_id": df["sample_id"],
        "split": df[key_col].map(label),
        "fold": fold,
        "reason": split_type,
    })


def assert_no_group_leakage(pheno: pd.DataFrame, split: pd.DataFrame, group_col: str) -> None:
    merged = pheno[["sample_id", group_col]].merge(split, on="sample_id", how="inner")
    train_groups = set(merged.loc[merged["split"] == "train", group_col])
    test_groups = set(merged.loc[merged["split"] == "test", group_col])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"Leakage in {group_col}: {sorted(list(overlap))[:10]}")


def load_split_table(split_dir: str | Path) -> pd.DataFrame:
    """Load repository split tables and normalize split_type/seed columns.

    G2F benchmark splits are stored as splits.parquet with split_type/seed.
    FIP1 official splits are stored as official_splits.parquet and are treated
    as one split configuration named "official" with seed 0.
    """
    split_dir = Path(split_dir)
    split_path = split_dir / "splits.parquet"
    official_path = split_dir / "official_splits.parquet"
    if split_path.exists():
        df = pd.read_parquet(split_path)
    elif official_path.exists():
        df = pd.read_parquet(official_path)
    else:
        raise FileNotFoundError(f"No split table found under {split_dir}")

    if "split_type" not in df.columns:
        df["split_type"] = "official"
    if "seed" not in df.columns:
        df["seed"] = 0
    return df
