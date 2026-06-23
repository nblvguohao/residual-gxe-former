from __future__ import annotations

import pandas as pd

from residual_gxe.evaluation.atlas import (
    compare_stage_weather_features,
    parse_stage_weather_feature,
    select_candidate_environments,
    select_candidate_genotypes,
    summarize_stage_variable_effects,
    summarize_stage_weather_stability,
)


def test_parse_stage_weather_feature():
    parsed = parse_stage_weather_feature("mid_tmax_mean")
    assert parsed is not None
    assert parsed.stage == "mid"
    assert parsed.variable == "tmax"
    assert parsed.aggregation == "mean"
    assert parse_stage_weather_feature("not_a_stage_feature") is None


def test_summarize_stage_weather_stability_ranks_consistent_effect():
    df = pd.DataFrame(
        {
            "feature": ["mid_tmax_mean", "mid_tmax_mean", "late_precipitation_sum"],
            "split_type": ["leave_year", "leave_environment", "leave_year"],
            "seed": [1, 2, 1],
            "pearson": [0.12, 0.10, -0.02],
            "spearman": [0.11, 0.09, -0.01],
            "n": [100, 120, 90],
        }
    )
    stable = summarize_stage_weather_stability(df, min_abs_pearson=0.05)
    assert stable.iloc[0]["feature"] == "mid_tmax_mean"
    assert bool(stable.iloc[0]["passes_min_effect"])
    stage = summarize_stage_variable_effects(stable)
    assert stage.iloc[0]["stage"] == "mid"
    assert stage.iloc[0]["weather_variable"] == "tmax"


def test_candidate_environment_selection_uses_abs_residual_and_runs():
    df = pd.DataFrame(
        {
            "environment_id": ["e1", "e1", "e2"],
            "split_type": ["leave_year", "leave_environment", "leave_year"],
            "seed": [1, 2, 1],
            "mean_residual": [2.0, 1.0, 1.5],
            "std_residual": [0.5, 0.6, 0.5],
            "n": [10, 10, 10],
        }
    )
    out = select_candidate_environments(df, top_n=1)
    assert out.iloc[0]["environment_id"] == "e1"


def test_candidate_genotypes_returns_positive_and_negative_sets():
    df = pd.DataFrame(
        {
            "genotype_id": ["g1", "g2"],
            "split_type": ["leave_year", "leave_year"],
            "seed": [1, 1],
            "mean_residual": [2.0, -2.0],
            "std_residual": [0.5, 0.5],
            "n": [10, 10],
        }
    )
    out = select_candidate_genotypes(df, top_n=1)
    assert set(out["candidate_type"]) == {"positive_residual_stability", "negative_residual_sensitivity"}
    assert set(out["genotype_id"]) == {"g1", "g2"}


def test_compare_stage_weather_features_reports_sign_agreement():
    left = pd.DataFrame(
        {
            "feature": ["mid_tmax_mean"],
            "mean_pearson": [0.1],
            "mean_abs_pearson": [0.1],
            "sign_consistency": [1.0],
            "n_runs": [3],
        }
    )
    right = pd.DataFrame(
        {
            "feature": ["mid_tmax_mean"],
            "mean_pearson": [0.2],
            "mean_abs_pearson": [0.2],
            "sign_consistency": [1.0],
            "n_runs": [2],
        }
    )
    out = compare_stage_weather_features(left, right, "g2f", "fip1")
    assert len(out) == 1
    assert bool(out.iloc[0]["sign_agreement"])
