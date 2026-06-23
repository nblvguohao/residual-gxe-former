from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_prediction_frame(
    sample_ids,
    y_true,
    y_pred,
    split_type: str,
    seed: int,
    model: str,
    target: str = "phenotype",
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Create a registry-ready prediction table."""
    sample_ids = np.asarray(sample_ids)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not (len(sample_ids) == len(y_true) == len(y_pred)):
        raise ValueError("sample_ids, y_true, and y_pred must have the same length")

    df = pd.DataFrame({
        "sample_id": sample_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "split_type": split_type,
        "seed": int(seed),
        "model": model,
        "target": target,
    })
    if extra_columns:
        for key, value in extra_columns.items():
            if np.isscalar(value) or isinstance(value, str):
                df[key] = value
            else:
                if len(value) != len(df):
                    raise ValueError(f"extra column {key} has length {len(value)} but expected {len(df)}")
                df[key] = value
    return df


def write_prediction_artifact(predictions: pd.DataFrame, out_dir: str | Path) -> Path:
    """Write predictions.parquet in a formal artifact directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.parquet"
    predictions.to_parquet(path, index=False)
    return path


def write_metrics_artifact(metrics: pd.DataFrame, out_dir: str | Path, filename: str = "metrics.csv") -> Path:
    """Write a formal metrics CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    metrics.to_csv(path, index=False)
    return path


def build_runtime_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a registry-ready runtime/resource table."""
    df = pd.DataFrame(records)
    if len(df) == 0:
        return pd.DataFrame(columns=["model", "split_type", "seed", "time_s"])
    if "time_s" not in df.columns:
        raise ValueError("runtime records must include time_s")
    return df


def write_runtime_artifact(runtime: pd.DataFrame, out_dir: str | Path) -> Path:
    """Write runtime.csv in a formal artifact directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime.csv"
    runtime.to_csv(path, index=False)
    return path
