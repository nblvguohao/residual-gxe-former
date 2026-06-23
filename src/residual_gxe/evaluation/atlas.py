from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_RE = re.compile(r"^(?P<stage>early|mid|late)_(?P<variable>.+)_(?P<aggregation>mean|sum)$")
STABLE_FEATURE_COLUMNS = [
    "feature", "stage", "weather_variable", "aggregation", "n_runs",
    "n_splits", "n_seeds", "mean_pearson", "median_pearson",
    "mean_abs_pearson", "max_abs_pearson", "mean_spearman", "total_n",
    "sign_consistency", "passes_min_effect", "discovery_scope",
]
STAGE_EFFECT_COLUMNS = [
    "stage", "weather_variable", "n_features", "n_runs", "mean_abs_pearson",
    "max_abs_pearson", "mean_signed_pearson", "mean_sign_consistency",
    "n_passing_features",
]


@dataclass(frozen=True)
class ParsedStageFeature:
    stage: str
    variable: str
    aggregation: str


def parse_stage_weather_feature(feature: str) -> ParsedStageFeature | None:
    match = FEATURE_RE.match(str(feature))
    if not match:
        return None
    return ParsedStageFeature(
        stage=match.group("stage"),
        variable=match.group("variable"),
        aggregation=match.group("aggregation"),
    )


def annotate_stage_weather_features(df: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    out = df.copy()
    parsed = out[feature_col].map(parse_stage_weather_feature)
    out["stage"] = parsed.map(lambda x: x.stage if x else None)
    out["weather_variable"] = parsed.map(lambda x: x.variable if x else None)
    out["aggregation"] = parsed.map(lambda x: x.aggregation if x else None)
    return out


def summarize_stage_weather_stability(
    associations: pd.DataFrame,
    min_abs_pearson: float = 0.05,
) -> pd.DataFrame:
    """Summarize residual-weather associations across split/seed runs."""
    required = {"feature", "split_type", "seed", "pearson", "spearman", "n"}
    missing = required - set(associations.columns)
    if missing:
        raise ValueError(f"Associations table missing required columns: {sorted(missing)}")
    if len(associations) == 0:
        return pd.DataFrame(columns=STABLE_FEATURE_COLUMNS)

    df = annotate_stage_weather_features(associations)
    df = df[np.isfinite(df["pearson"].astype(float))].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=STABLE_FEATURE_COLUMNS)
    rows = []
    for feature, sub in df.groupby("feature"):
        signs = np.sign(sub["pearson"].to_numpy(dtype=float))
        signs = signs[signs != 0]
        sign_consistency = float(max(np.mean(signs > 0), np.mean(signs < 0))) if len(signs) else float("nan")
        parsed = parse_stage_weather_feature(feature)
        rows.append({
            "feature": feature,
            "stage": parsed.stage if parsed else None,
            "weather_variable": parsed.variable if parsed else None,
            "aggregation": parsed.aggregation if parsed else None,
            "n_runs": int(len(sub)),
            "n_splits": int(sub["split_type"].nunique()),
            "n_seeds": int(sub["seed"].nunique()),
            "mean_pearson": float(sub["pearson"].mean()),
            "median_pearson": float(sub["pearson"].median()),
            "mean_abs_pearson": float(sub["pearson"].abs().mean()),
            "max_abs_pearson": float(sub["pearson"].abs().max()),
            "mean_spearman": float(sub["spearman"].mean()),
            "total_n": int(sub["n"].sum()),
            "sign_consistency": sign_consistency,
            "passes_min_effect": bool(sub["pearson"].abs().mean() >= min_abs_pearson),
            "discovery_scope": "training_fold_only",
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=STABLE_FEATURE_COLUMNS)
    return out.sort_values(
        ["passes_min_effect", "mean_abs_pearson", "sign_consistency", "n_runs"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def summarize_stage_variable_effects(stable_features: pd.DataFrame) -> pd.DataFrame:
    """Collapse feature stability table to stage x weather-variable effects."""
    if len(stable_features) == 0:
        return pd.DataFrame(columns=STAGE_EFFECT_COLUMNS)
    required = {"stage", "weather_variable", "mean_abs_pearson", "mean_pearson", "sign_consistency", "n_runs"}
    missing = required - set(stable_features.columns)
    if missing:
        raise ValueError(f"Stable feature table missing required columns: {sorted(missing)}")

    rows = []
    for (stage, variable), sub in stable_features.groupby(["stage", "weather_variable"], dropna=False):
        rows.append({
            "stage": stage,
            "weather_variable": variable,
            "n_features": int(len(sub)),
            "n_runs": int(sub["n_runs"].sum()),
            "mean_abs_pearson": float(sub["mean_abs_pearson"].mean()),
            "max_abs_pearson": float(sub["max_abs_pearson"].max()) if "max_abs_pearson" in sub.columns else float("nan"),
            "mean_signed_pearson": float(sub["mean_pearson"].mean()),
            "mean_sign_consistency": float(sub["sign_consistency"].mean()),
            "n_passing_features": int(sub["passes_min_effect"].sum()) if "passes_min_effect" in sub.columns else 0,
        })
    if not rows:
        return pd.DataFrame(columns=STAGE_EFFECT_COLUMNS)
    return pd.DataFrame(rows).sort_values(
        ["n_passing_features", "mean_abs_pearson", "mean_sign_consistency"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def select_candidate_environments(
    environment_profiles: pd.DataFrame,
    top_n: int = 20,
    min_n: int = 8,
    min_runs: int = 1,
) -> pd.DataFrame:
    required = {"environment_id", "split_type", "seed", "mean_residual", "std_residual", "n"}
    missing = required - set(environment_profiles.columns)
    if missing:
        raise ValueError(f"Environment profile table missing required columns: {sorted(missing)}")
    df = environment_profiles[environment_profiles["n"] >= min_n].copy()
    if len(df) == 0:
        return pd.DataFrame()
    summary = df.groupby("environment_id").agg(
        n_runs=("mean_residual", "size"),
        n_splits=("split_type", "nunique"),
        n_seeds=("seed", "nunique"),
        mean_residual=("mean_residual", "mean"),
        mean_abs_residual=("mean_residual", lambda x: float(np.mean(np.abs(x)))),
        mean_std_residual=("std_residual", "mean"),
        total_n=("n", "sum"),
    ).reset_index()
    summary = summary[summary["n_runs"] >= min_runs].copy()
    if len(summary) == 0:
        return pd.DataFrame()
    summary["candidate_score"] = summary["mean_abs_residual"] * np.sqrt(summary["n_runs"])
    return summary.sort_values(["candidate_score", "total_n"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def select_candidate_genotypes(
    genotype_stability: pd.DataFrame,
    top_n: int = 20,
    min_n: int = 8,
    min_runs: int = 1,
) -> pd.DataFrame:
    required = {"genotype_id", "split_type", "seed", "mean_residual", "std_residual", "n"}
    missing = required - set(genotype_stability.columns)
    if missing:
        raise ValueError(f"Genotype stability table missing required columns: {sorted(missing)}")
    df = genotype_stability[genotype_stability["n"] >= min_n].copy()
    if len(df) == 0:
        return pd.DataFrame()
    summary = df.groupby("genotype_id").agg(
        n_runs=("mean_residual", "size"),
        n_splits=("split_type", "nunique"),
        n_seeds=("seed", "nunique"),
        mean_residual=("mean_residual", "mean"),
        mean_abs_residual=("mean_residual", lambda x: float(np.mean(np.abs(x)))),
        mean_std_residual=("std_residual", "mean"),
        total_n=("n", "sum"),
    ).reset_index()
    summary = summary[summary["n_runs"] >= min_runs].copy()
    if len(summary) == 0:
        return pd.DataFrame()
    summary["stable_positive_score"] = summary["mean_residual"] / (summary["mean_std_residual"] + 1e-8)
    summary["stable_negative_score"] = -summary["mean_residual"] / (summary["mean_std_residual"] + 1e-8)
    positive = summary.sort_values(["stable_positive_score", "total_n"], ascending=[False, False]).head(top_n).copy()
    positive["candidate_type"] = "positive_residual_stability"
    negative = summary.sort_values(["stable_negative_score", "total_n"], ascending=[False, False]).head(top_n).copy()
    negative["candidate_type"] = "negative_residual_sensitivity"
    return pd.concat([positive, negative], ignore_index=True)


def compare_stage_weather_features(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    """Compare stable stage-weather features between two atlas summaries."""
    required = {"feature", "mean_pearson", "mean_abs_pearson", "sign_consistency", "n_runs"}
    if len(left) == 0 or len(right) == 0:
        return pd.DataFrame(columns=[
            "feature", "left_dataset", "right_dataset", "left_mean_pearson",
            "right_mean_pearson", "left_mean_abs_pearson", "right_mean_abs_pearson",
            "sign_agreement", "left_n_runs", "right_n_runs",
        ])
    missing_left = required - set(left.columns)
    missing_right = required - set(right.columns)
    if missing_left or missing_right:
        raise ValueError(f"Missing columns: left={sorted(missing_left)} right={sorted(missing_right)}")

    merged = left[list(required)].merge(
        right[list(required)],
        on="feature",
        how="inner",
        suffixes=("_left", "_right"),
    )
    if len(merged) == 0:
        return pd.DataFrame(columns=[
            "feature", "left_dataset", "right_dataset", "left_mean_pearson",
            "right_mean_pearson", "left_mean_abs_pearson", "right_mean_abs_pearson",
            "sign_agreement", "left_n_runs", "right_n_runs",
        ])
    out = pd.DataFrame({
        "feature": merged["feature"],
        "left_dataset": left_label,
        "right_dataset": right_label,
        "left_mean_pearson": merged["mean_pearson_left"],
        "right_mean_pearson": merged["mean_pearson_right"],
        "left_mean_abs_pearson": merged["mean_abs_pearson_left"],
        "right_mean_abs_pearson": merged["mean_abs_pearson_right"],
        "sign_agreement": np.sign(merged["mean_pearson_left"]) == np.sign(merged["mean_pearson_right"]),
        "left_n_runs": merged["n_runs_left"],
        "right_n_runs": merged["n_runs_right"],
    })
    return out.sort_values(["sign_agreement", "left_mean_abs_pearson"], ascending=[False, False]).reset_index(drop=True)
