from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from residual_gxe.data.manifest import (
    processed_dataset_summary,
    refreshed_manifest,
    refreshed_metadata,
)


def test_processed_dataset_summary_counts_long_marker_table(tmp_path: Path):
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "genotype_id": ["g1", "g2"],
            "environment_id": ["e1", "e1"],
            "year": [2020, 2020],
            "trait_id": ["yield", "yield"],
        }
    ).to_parquet(tmp_path / "phenotype.parquet")
    pd.DataFrame({"environment_id": ["e1"], "year": [2020]}).to_parquet(tmp_path / "environment.parquet")
    pd.DataFrame(
        {
            "genotype_id": ["g1", "g1", "g2", "g2"],
            "marker_id": ["m1", "m2", "m1", "m2"],
            "allele_dosage": [0, 1, 1, 2],
        }
    ).to_parquet(tmp_path / "genotype.parquet")
    pd.DataFrame({"environment_id": ["e1"], "date": ["2020-01-01"]}).to_parquet(tmp_path / "weather_daily.parquet")

    summary = processed_dataset_summary(tmp_path, dataset="g2f")
    assert summary["n_samples"] == 2
    assert summary["n_genotypes"] == 2
    assert summary["n_environments"] == 1
    assert summary["n_markers"] == 2
    assert summary["has_weather_daily"]


def test_processed_dataset_summary_counts_array_marker_table(tmp_path: Path):
    pd.DataFrame(
        {
            "sample_id": ["s1"],
            "genotype_id": ["g1"],
            "environment_id": ["e1"],
            "year": [2020],
            "trait_id": ["yield_adjusted"],
        }
    ).to_parquet(tmp_path / "phenotype.parquet")
    pd.DataFrame({"genotype_id": ["g1"], "marker_biallelic_codes": [np.array([0, 1, 2])] }).to_parquet(
        tmp_path / "genotype.parquet"
    )

    summary = processed_dataset_summary(tmp_path, dataset="fip1")
    assert summary["n_markers"] == 3
    assert not summary["has_weather_daily"]


def test_refreshed_manifest_and_metadata_preserve_target_trait():
    summary = {
        "dataset": "g2f",
        "n_samples": 10,
        "n_genotypes": 4,
        "n_environments": 2,
        "n_years": 1,
        "n_markers": 3,
        "has_weather_daily": True,
        "has_soil": False,
        "n_weather_environments": 2,
    }
    manifest = refreshed_manifest({"target_trait": "grain_yield", "notes": []}, summary)
    metadata = refreshed_metadata({}, summary)
    assert manifest["traits"] == ["grain_yield"]
    assert manifest["n_samples"] == 10
    assert manifest["missingness"]["weather"] is None
    assert manifest["missingness"]["soil"] == "soil.parquet missing"
    assert metadata["n_markers"] == 3
