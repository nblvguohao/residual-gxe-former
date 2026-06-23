from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from residual_gxe.evaluation.artifacts import (
    build_prediction_frame,
    build_runtime_frame,
    write_metrics_artifact,
    write_prediction_artifact,
    write_runtime_artifact,
)


def test_build_prediction_frame_has_registry_columns():
    df = build_prediction_frame(
        sample_ids=["s1", "s2"],
        y_true=[1.0, 2.0],
        y_pred=[1.1, 1.9],
        split_type="random",
        seed=1234,
        model="ridge",
        target="phenotype",
        extra_columns={"fold": [0, 0]},
    )
    assert {"sample_id", "y_true", "y_pred", "split_type", "seed", "model", "target"} <= set(df.columns)
    assert df["model"].iloc[0] == "ridge"


def test_build_prediction_frame_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_prediction_frame(["s1"], [1.0, 2.0], [1.0], "random", 1, "ridge")


def test_write_prediction_and_metrics_artifacts(tmp_path: Path):
    pred = build_prediction_frame(["s1"], [1.0], [1.0], "random", 1, "ridge")
    pred_path = write_prediction_artifact(pred, tmp_path / "run")
    assert pred_path.name == "predictions.parquet"
    assert len(pd.read_parquet(pred_path)) == 1

    metrics = pd.DataFrame({"model": ["ridge"], "split_type": ["random"], "seed": [1], "pearson": [1.0]})
    metrics_path = write_metrics_artifact(metrics, tmp_path / "run")
    assert metrics_path.name == "metrics.csv"
    assert len(pd.read_csv(metrics_path)) == 1


def test_write_runtime_artifact(tmp_path: Path):
    runtime = build_runtime_frame([{"model": "ridge", "split_type": "random", "seed": 1, "time_s": 0.5}])
    runtime_path = write_runtime_artifact(runtime, tmp_path / "run")
    assert runtime_path.name == "runtime.csv"
    assert pd.read_csv(runtime_path).iloc[0]["time_s"] == 0.5


def test_runtime_artifact_requires_time_s():
    with pytest.raises(ValueError):
        build_runtime_frame([{"model": "ridge"}])
