from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _unique_count(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    return int(df[column].nunique(dropna=True))


def _marker_count(genotype: pd.DataFrame) -> int | None:
    if "marker_id" in genotype.columns:
        return int(genotype["marker_id"].nunique(dropna=True))
    if "marker_biallelic_codes" in genotype.columns and len(genotype) > 0:
        return int(len(genotype.iloc[0]["marker_biallelic_codes"]))
    marker_cols = [col for col in genotype.columns if str(col).startswith("marker_") and str(col)[7:].isdigit()]
    return int(len(marker_cols)) if marker_cols else None


def processed_dataset_summary(data_dir: str | Path, dataset: str | None = None) -> dict[str, Any]:
    """Summarize a processed dataset directly from parquet files."""
    data_dir = Path(data_dir)
    phenotype_path = data_dir / "phenotype.parquet"
    environment_path = data_dir / "environment.parquet"
    genotype_path = data_dir / "genotype.parquet"
    weather_path = data_dir / "weather_daily.parquet"
    soil_path = data_dir / "soil.parquet"

    summary: dict[str, Any] = {
        "dataset": dataset,
        "n_samples": None,
        "n_genotypes": None,
        "n_genotypes_in_phenotype": None,
        "n_environments": None,
        "n_years": None,
        "n_markers": None,
        "traits_from_phenotype": [],
        "has_weather_daily": bool(weather_path.exists()),
        "has_soil": bool(soil_path.exists()),
    }

    if phenotype_path.exists():
        phenotype = pd.read_parquet(phenotype_path)
        summary["n_samples"] = int(len(phenotype))
        summary["n_genotypes_in_phenotype"] = _unique_count(phenotype, "genotype_id")
        summary["n_environments"] = _unique_count(phenotype, "environment_id")
        summary["n_years"] = _unique_count(phenotype, "year")
        if "trait_id" in phenotype.columns:
            summary["traits_from_phenotype"] = sorted(map(str, phenotype["trait_id"].dropna().unique()))

    if environment_path.exists():
        environment = pd.read_parquet(environment_path)
        env_count = _unique_count(environment, "environment_id")
        if env_count is not None:
            summary["n_environments"] = env_count

    if genotype_path.exists():
        genotype = pd.read_parquet(genotype_path)
        genotype_count = _unique_count(genotype, "genotype_id")
        if genotype_count is not None:
            summary["n_genotypes"] = genotype_count
        summary["n_markers"] = _marker_count(genotype)

    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        weather_envs = _unique_count(weather, "environment_id")
        summary["n_weather_environments"] = weather_envs

    return summary


def refreshed_manifest(existing: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Return data_manifest.yaml content updated with parquet-derived counts."""
    out = dict(existing)
    if summary.get("dataset") and not out.get("dataset"):
        out["dataset"] = summary["dataset"]
    for key in ["n_samples", "n_genotypes", "n_environments", "n_years"]:
        if summary.get(key) is not None:
            out[key] = summary[key]

    target_trait = out.get("target_trait")
    if target_trait:
        out["traits"] = [target_trait]
    elif summary.get("traits_from_phenotype"):
        out["traits"] = summary["traits_from_phenotype"]

    out.setdefault("missingness", {})
    out["missingness"] = dict(out["missingness"] or {})
    out["missingness"]["weather"] = None if summary.get("has_weather_daily") else "weather_daily.parquet missing"
    out["missingness"]["soil"] = None if summary.get("has_soil") else "soil.parquet missing"
    out["missingness"].setdefault("genotype", None)

    notes = list(out.get("notes") or [])
    refresh_note = "Counts refreshed from processed parquet files."
    if refresh_note not in notes:
        notes.append(refresh_note)
    if summary.get("n_markers") is not None:
        marker_note = f"Detected {summary['n_markers']} genotype markers from processed genotype table."
        notes = [note for note in notes if not str(note).startswith("Detected ") or " genotype markers " not in str(note)]
        notes.append(marker_note)
    if summary.get("n_weather_environments") is not None:
        weather_note = f"Processed weather_daily covers {summary['n_weather_environments']} environments."
        notes = [note for note in notes if not str(note).startswith("Processed weather_daily covers ")]
        notes.append(weather_note)
    out["notes"] = notes
    return out


def refreshed_metadata(existing: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Return metadata.json content updated with parquet-derived counts."""
    out = dict(existing)
    for key in ["n_samples", "n_genotypes", "n_environments", "n_years", "n_markers"]:
        if summary.get(key) is not None:
            out[key] = summary[key]
    if summary.get("n_genotypes_in_phenotype") is not None:
        out["n_genotypes_in_phenotype"] = summary["n_genotypes_in_phenotype"]
    out["has_weather_daily"] = bool(summary.get("has_weather_daily"))
    out["has_soil"] = bool(summary.get("has_soil"))
    if summary.get("n_weather_environments") is not None:
        out["n_weather_environments"] = summary["n_weather_environments"]
    notes = list(out.get("notes") or [])
    refresh_note = "Counts refreshed from processed parquet files."
    if refresh_note not in notes:
        notes.append(refresh_note)
    out["notes"] = notes
    return out
