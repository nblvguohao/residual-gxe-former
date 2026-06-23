from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from residual_gxe.data.fip1 import prepare_fip1_external, validate_complete_shards
from residual_gxe.data.schema import REQUIRED_PHENOTYPE_COLUMNS, REQUIRED_SPLIT_COLUMNS


def _write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _row(plot: str, genotype: str, env: str, value: float, year: int = 2021) -> dict:
    return {
        "plot_uid": plot,
        "yearsite_uid": env,
        "genotype_id": genotype,
        "crop_type": "winter_wheat",
        "experiment_number": 1,
        "plot_number": 10,
        "range": 1,
        "row": 2,
        "lot": 3,
        "latitude": 48.1,
        "longitude": 11.6,
        "sowing_date": "2020-10-01",
        "harvest_date": "2021-07-15",
        "harvest_year": year,
        "yield_adjusted_value": value,
        "yield_adjusted_trait_id": "yield_adjusted",
        "yield_adjusted_trait_name": "Adjusted yield",
        "yield_adjusted_si_unit": "dt/ha",
        "marker_biallelic_codes": [0, 1, 2],
        "marker_metadata_strings": ["m1|1|10", "m2|1|20", "m3|2|30"],
    }


def test_prepare_fip1_external_writes_unified_tables(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _write_shard(raw / "train-00000-of-00002.parquet", [_row("p1", "g1", "e1", 1.0)])
    _write_shard(raw / "train-00001-of-00002.parquet", [_row("p2", "g2", "e1", 2.0)])
    _write_shard(raw / "validation-00000-of-00001.parquet", [_row("p3", "g1", "e2", 3.0)])
    _write_shard(raw / "test_environment-00000-of-00001.parquet", [_row("p4", "g1", "e3", 4.0)])

    prepare_fip1_external(raw, out, target_trait="yield_adjusted")

    phenotype = pd.read_parquet(out / "phenotype.parquet")
    split = pd.read_parquet(out / "splits" / "official_splits.parquet")
    environment = pd.read_parquet(out / "environment.parquet")
    genotype = pd.read_parquet(out / "genotype.parquet")

    assert REQUIRED_PHENOTYPE_COLUMNS <= set(phenotype.columns)
    assert REQUIRED_SPLIT_COLUMNS <= set(split.columns)
    assert phenotype["sample_id"].tolist() == ["p1", "p2", "p3", "p4"]
    assert phenotype["environment_id"].tolist() == ["e1", "e1", "e2", "e3"]
    assert phenotype["phenotype_value"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert split[["sample_id", "split", "official_split", "reason"]].to_dict("records") == [
        {"sample_id": "p1", "split": "train", "official_split": "train", "reason": "official_train"},
        {"sample_id": "p2", "split": "train", "official_split": "train", "reason": "official_train"},
        {"sample_id": "p3", "split": "val", "official_split": "validation", "reason": "official_validation"},
        {"sample_id": "p4", "split": "test", "official_split": "test_environment", "reason": "official_test_environment"},
    ]
    assert set(environment["environment_id"]) == {"e1", "e2", "e3"}
    assert set(genotype["genotype_id"]) == {"g1", "g2"}
    assert (out / "metadata.json").exists()
    assert (out / "data_manifest.yaml").exists()


def test_validate_complete_shards_reports_missing_index(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_shard(raw / "train-00001-of-00002.parquet", [_row("p2", "g2", "e1", 2.0)])

    with pytest.raises(ValueError, match="train.*00000"):
        validate_complete_shards(sorted(raw.glob("*.parquet")))


def test_prepare_fip1_script_runs_from_repo_root(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _write_shard(raw / "train-00000-of-00001.parquet", [_row("p1", "g1", "e1", 1.0)])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/08_prepare_fip1_external.py",
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(out),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "phenotype.parquet").exists()
