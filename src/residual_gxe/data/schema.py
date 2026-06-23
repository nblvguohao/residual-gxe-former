from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_PHENOTYPE_COLUMNS = {
    "sample_id",
    "genotype_id",
    "environment_id",
    "year",
    "location_id",
    "trait_id",
    "trait_name",
    "phenotype_value",
}

REQUIRED_SPLIT_COLUMNS = {"sample_id", "split", "fold", "reason"}


@dataclass(frozen=True)
class SchemaCheckResult:
    ok: bool
    missing_columns: list[str]
    extra_message: str = ""


def check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> SchemaCheckResult:
    required_set = set(required)
    missing = sorted(required_set - set(df.columns))
    return SchemaCheckResult(ok=len(missing) == 0, missing_columns=missing)


def validate_phenotype_table(df: pd.DataFrame) -> SchemaCheckResult:
    result = check_required_columns(df, REQUIRED_PHENOTYPE_COLUMNS)
    if not result.ok:
        return result
    if df["sample_id"].duplicated().any():
        return SchemaCheckResult(False, [], "sample_id values must be unique")
    if df["phenotype_value"].isna().any():
        return SchemaCheckResult(False, [], "phenotype_value contains missing values")
    return SchemaCheckResult(True, [])


def validate_split_table(df: pd.DataFrame) -> SchemaCheckResult:
    result = check_required_columns(df, REQUIRED_SPLIT_COLUMNS)
    if not result.ok:
        return result
    valid_splits = {"train", "val", "test"}
    observed = set(df["split"].unique())
    invalid = observed - valid_splits
    if invalid:
        return SchemaCheckResult(False, [], f"Invalid split labels: {sorted(invalid)}")
    return SchemaCheckResult(True, [])
