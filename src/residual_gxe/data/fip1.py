from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq
import yaml


SHARD_RE = re.compile(r"(?P<split>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.parquet$")
SPLIT_ORDER = {
    "train": 0,
    "validation": 1,
    "test_plot": 2,
    "test_genotype": 3,
    "test_environment": 4,
    "test_genotype_environment": 5,
}


@dataclass(frozen=True)
class Fip1PrepareResult:
    phenotype_path: Path
    split_path: Path
    environment_path: Path
    genotype_path: Path
    metadata_path: Path
    manifest_path: Path


def infer_official_split(path: str | Path) -> str:
    match = SHARD_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"FIP1 shard name does not match expected pattern: {path}")
    return match.group("split")


def _shard_sort_key(path: Path) -> tuple[int, str, int]:
    match = SHARD_RE.match(path.name)
    if not match:
        return (999, path.name, 0)
    split = match.group("split")
    return (SPLIT_ORDER.get(split, 900), split, int(match.group("index")))


def list_fip1_parquet_shards(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    return sorted(raw_dir.rglob("*.parquet"), key=_shard_sort_key)


def validate_complete_shards(paths: Iterable[str | Path]) -> None:
    grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for path in paths:
        p = Path(path)
        match = SHARD_RE.match(p.name)
        if not match:
            raise ValueError(f"FIP1 shard name does not match expected pattern: {p.name}")
        grouped[match.group("split")].append(
            (int(match.group("index")), int(match.group("total")), p.name)
        )

    problems: list[str] = []
    for split, entries in sorted(grouped.items()):
        totals = {total for _, total, _ in entries}
        if len(totals) != 1:
            problems.append(f"{split}: inconsistent shard totals {sorted(totals)}")
            continue
        total = totals.pop()
        present = {index for index, _, _ in entries}
        missing = [f"{index:05d}" for index in range(total) if index not in present]
        if missing:
            problems.append(f"{split}: missing shard indexes {missing}")

    if problems:
        raise ValueError("; ".join(problems))


def _available_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _read_shard(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    available = _available_columns(path)
    selected = [col for col in columns if col in available]
    df = pd.read_parquet(path, columns=selected)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df["official_split"] = infer_official_split(path)
    return df


def _normalise_split(official_split: str) -> str:
    if official_split == "train":
        return "train"
    if official_split == "validation":
        return "val"
    return "test"


def _make_phenotype(df: pd.DataFrame, target_trait: str) -> tuple[pd.DataFrame, int]:
    value_col = f"{target_trait}_value"
    if value_col not in df.columns:
        raise ValueError(f"Missing target trait value column: {value_col}")

    trait_id_col = f"{target_trait}_trait_id"
    trait_name_col = f"{target_trait}_trait_name"
    unit_col = f"{target_trait}_si_unit"

    pheno = pd.DataFrame(
        {
            "sample_id": df["plot_uid"].astype(str),
            "genotype_id": df["genotype_id"].astype(str),
            "environment_id": df["yearsite_uid"].astype(str),
            "year": pd.to_numeric(df["harvest_year"], errors="coerce").astype("Int64"),
            "location_id": df["yearsite_uid"].astype(str),
            "trait_id": df.get(trait_id_col, target_trait).fillna(target_trait).astype(str),
            "trait_name": df.get(trait_name_col, target_trait).fillna(target_trait).astype(str),
            "trait_family": "yield" if "yield" in target_trait else "other",
            "phenotype_value": pd.to_numeric(df[value_col], errors="coerce"),
            "phenotype_unit": df.get(unit_col, "").fillna("").astype(str),
            "replicate_id": pd.NA,
            "block_id": pd.NA,
            "trial_id": df["experiment_number"].astype(str),
            "source_dataset": "fip1",
        }
    )
    before = len(pheno)
    pheno = pheno.dropna(subset=["sample_id", "genotype_id", "environment_id", "phenotype_value"])
    return pheno.reset_index(drop=True), before - len(pheno)


def _make_splits(df: pd.DataFrame, sample_ids: set[str]) -> pd.DataFrame:
    split = pd.DataFrame(
        {
            "sample_id": df["plot_uid"].astype(str),
            "split": df["official_split"].map(_normalise_split),
            "fold": 0,
            "reason": "official_" + df["official_split"].astype(str),
            "official_split": df["official_split"].astype(str),
        }
    )
    split = split[split["sample_id"].isin(sample_ids)]
    return split.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)


def _make_environment(df: pd.DataFrame) -> pd.DataFrame:
    env = df.copy()
    env["environment_id"] = env["yearsite_uid"].astype(str)
    env["year"] = pd.to_numeric(env["harvest_year"], errors="coerce").astype("Int64")
    env["location_id"] = env["yearsite_uid"].astype(str)
    grouped = env.groupby("environment_id", sort=True, dropna=False)
    return grouped.agg(
        year=("year", "first"),
        location_id=("location_id", "first"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        altitude=("plot_uid", lambda _: pd.NA),
        planting_date=("sowing_date", "first"),
        harvest_date=("harvest_date", "first"),
        management_notes=("crop_type", "first"),
        source_dataset=("plot_uid", lambda _: "fip1"),
    ).reset_index()


def _make_genotype(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["genotype_id"]
    for col in ["marker_biallelic_codes", "marker_metadata_strings"]:
        if col in df.columns:
            cols.append(col)
    geno = df[cols].drop_duplicates(subset=["genotype_id"]).reset_index(drop=True)
    geno["source_dataset"] = "fip1"
    return geno


def prepare_fip1_external(
    raw_dir: str | Path,
    out_dir: str | Path,
    target_trait: str = "yield_adjusted",
) -> Fip1PrepareResult:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    shards = list_fip1_parquet_shards(raw_dir)
    if not shards:
        raise FileNotFoundError(f"No FIP1 parquet shards found under {raw_dir}")
    validate_complete_shards(shards)

    requested_columns = [
        "plot_uid",
        "yearsite_uid",
        "genotype_id",
        "crop_type",
        "experiment_number",
        "latitude",
        "longitude",
        "sowing_date",
        "harvest_date",
        "harvest_year",
        f"{target_trait}_value",
        f"{target_trait}_trait_id",
        f"{target_trait}_trait_name",
        f"{target_trait}_si_unit",
        "marker_biallelic_codes",
        "marker_metadata_strings",
    ]
    raw = pd.concat([_read_shard(path, requested_columns) for path in shards], ignore_index=True)
    phenotype, dropped_phenotype_rows = _make_phenotype(raw, target_trait)
    sample_ids = set(phenotype["sample_id"])
    split = _make_splits(raw, sample_ids)
    environment = _make_environment(raw[raw["plot_uid"].astype(str).isin(sample_ids)])
    genotype = _make_genotype(raw[raw["plot_uid"].astype(str).isin(sample_ids)])

    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    phenotype_path = out_dir / "phenotype.parquet"
    split_path = split_dir / "official_splits.parquet"
    environment_path = out_dir / "environment.parquet"
    genotype_path = out_dir / "genotype.parquet"
    metadata_path = out_dir / "metadata.json"
    manifest_path = out_dir / "data_manifest.yaml"

    phenotype.to_parquet(phenotype_path, index=False)
    split.to_parquet(split_path, index=False)
    environment.to_parquet(environment_path, index=False)
    genotype.to_parquet(genotype_path, index=False)

    official_splits = sorted(raw["official_split"].unique(), key=lambda s: SPLIT_ORDER.get(s, 999))
    metadata = {
        "dataset": "fip1",
        "target_trait": target_trait,
        "source_file_count": len(shards),
        "official_splits": official_splits,
        "n_samples": int(len(phenotype)),
        "n_genotypes": int(phenotype["genotype_id"].nunique()),
        "n_environments": int(phenotype["environment_id"].nunique()),
        "n_years": int(phenotype["year"].nunique(dropna=True)),
        "dropped_rows_missing_target": int(dropped_phenotype_rows),
        "notes": [
            "FIP1 official split labels are preserved in splits/official_splits.parquet.",
            "split is normalized to train/val/test; official_split stores the original benchmark label.",
            "No preprocessing has been fit on validation or test data in this adapter.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "dataset": "fip1",
        "raw_data_path": None,
        "target_trait": target_trait,
        "n_samples": metadata["n_samples"],
        "n_genotypes": metadata["n_genotypes"],
        "n_environments": metadata["n_environments"],
        "n_years": metadata["n_years"],
        "traits": [target_trait],
        "missingness": {
            "genotype": None,
            "weather": None,
            "soil": None,
        },
        "notes": metadata["notes"],
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    return Fip1PrepareResult(
        phenotype_path=phenotype_path,
        split_path=split_path,
        environment_path=environment_path,
        genotype_path=genotype_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
    )
