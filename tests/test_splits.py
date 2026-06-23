from __future__ import annotations

import pandas as pd

from residual_gxe.data.splits import SplitConfig, assert_no_group_leakage, load_split_table, make_split_table


def pheno_df():
    rows = []
    for g in range(10):
        for e in range(5):
            rows.append({
                "sample_id": f"S{g}_{e}",
                "genotype_id": f"G{g}",
                "environment_id": f"E{e}",
                "year": 2020 + e % 3,
                "phenotype_value": float(g + e),
            })
    return pd.DataFrame(rows)


def test_leave_genotype_no_leakage():
    df = pheno_df()
    split = make_split_table(df, SplitConfig("leave_genotype", seed=1), fold=0)
    assert_no_group_leakage(df, split, "genotype_id")


def test_leave_environment_no_leakage():
    df = pheno_df()
    split = make_split_table(df, SplitConfig("leave_environment", seed=1), fold=0)
    assert_no_group_leakage(df, split, "environment_id")


def test_split_deterministic():
    df = pheno_df()
    a = make_split_table(df, SplitConfig("random", seed=1), fold=0)
    b = make_split_table(df, SplitConfig("random", seed=1), fold=0)
    assert a.equals(b)


def test_load_official_split_table_normalizes_columns(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "split": ["train", "test"],
            "fold": [0, 0],
            "reason": ["official_train", "official_test_environment"],
            "official_split": ["train", "test_environment"],
        }
    ).to_parquet(split_dir / "official_splits.parquet", index=False)

    out = load_split_table(split_dir)
    assert set(out["split_type"]) == {"official"}
    assert set(out["seed"]) == {0}
    assert "official_split" in out.columns
