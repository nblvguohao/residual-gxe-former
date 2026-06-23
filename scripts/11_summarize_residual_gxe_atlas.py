from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.atlas import (
    select_candidate_environments,
    select_candidate_genotypes,
    summarize_stage_variable_effects,
    summarize_stage_weather_stability,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize residual GxE atlas into candidate biological findings.")
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-abs-pearson", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-n", type=int, default=8)
    parser.add_argument("--min-runs", type=int, default=6)
    return parser.parse_args()


def _read_optional(path: Path) -> pd.DataFrame:
    return read_table(path) if path.exists() else pd.DataFrame()


def _write_markdown_summary(
    out_dir: Path,
    stable_features: pd.DataFrame,
    stage_effects: pd.DataFrame,
    candidate_envs: pd.DataFrame,
    candidate_genotypes: pd.DataFrame,
) -> None:
    lines = [
        "# Residual GxE Atlas Summary",
        "",
        "This summary is generated from machine-readable atlas tables.",
        "Associations are descriptive and must not be presented as causal mechanisms without validation.",
        "",
    ]

    lines.append("## Stable Stage-Weather Associations")
    if len(stable_features) == 0:
        lines.append("No stage-weather associations available.")
    else:
        cols = ["feature", "n_runs", "n_splits", "mean_pearson", "mean_abs_pearson", "sign_consistency"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---" for _ in cols]) + "|")
        for _, row in stable_features.head(10).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                    for c in cols
                )
                + " |"
            )
    lines.append("")

    lines.append("## Stage x Weather Variable Summary")
    if len(stage_effects) == 0:
        lines.append("No stage-level summary available.")
    else:
        cols = ["stage", "weather_variable", "n_features", "mean_abs_pearson", "mean_sign_consistency", "n_passing_features"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---" for _ in cols]) + "|")
        for _, row in stage_effects.head(10).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                    for c in cols
                )
                + " |"
            )
    lines.append("")

    lines.append("## Candidate Environments")
    if len(candidate_envs) == 0:
        lines.append("No candidate environments available.")
    else:
        cols = ["environment_id", "n_runs", "mean_residual", "mean_abs_residual", "candidate_score"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---" for _ in cols]) + "|")
        for _, row in candidate_envs.head(10).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                    for c in cols
                )
                + " |"
            )
    lines.append("")

    lines.append("## Candidate Genotypes")
    if len(candidate_genotypes) == 0:
        lines.append("No candidate genotypes available.")
    else:
        cols = ["genotype_id", "candidate_type", "n_runs", "mean_residual", "mean_std_residual"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---" for _ in cols]) + "|")
        for _, row in candidate_genotypes.head(20).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                    for c in cols
                )
                + " |"
            )

    (out_dir / "atlas_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    assoc = _read_optional(args.atlas_dir / "stage_weather_residual_associations.csv")
    geno = _read_optional(args.atlas_dir / "genotype_residual_stability.csv")
    env = _read_optional(args.atlas_dir / "environment_residual_profiles.csv")

    stable_features = summarize_stage_weather_stability(assoc, min_abs_pearson=args.min_abs_pearson) if len(assoc) else pd.DataFrame()
    stage_effects = summarize_stage_variable_effects(stable_features) if len(stable_features) else pd.DataFrame()
    candidate_envs = select_candidate_environments(env, top_n=args.top_n, min_n=args.min_n, min_runs=args.min_runs) if len(env) else pd.DataFrame()
    candidate_genotypes = select_candidate_genotypes(geno, top_n=args.top_n, min_n=args.min_n, min_runs=args.min_runs) if len(geno) else pd.DataFrame()

    write_table(stable_features, args.out_dir / "stable_stage_weather_features.csv")
    write_table(stage_effects, args.out_dir / "stage_weather_effect_summary.csv")
    write_table(candidate_envs, args.out_dir / "candidate_environments.csv")
    write_table(candidate_genotypes, args.out_dir / "candidate_genotypes.csv")
    _write_markdown_summary(args.out_dir, stable_features, stage_effects, candidate_envs, candidate_genotypes)

    manifest = {
        "schema_version": 1,
        "script": "scripts/11_summarize_residual_gxe_atlas.py",
        "atlas_dir": str(args.atlas_dir),
        "min_abs_pearson": float(args.min_abs_pearson),
        "top_n": int(args.top_n),
        "min_n": int(args.min_n),
        "min_runs": int(args.min_runs),
        "outputs": {
            "stable_stage_weather_features": str(args.out_dir / "stable_stage_weather_features.csv"),
            "stage_weather_effect_summary": str(args.out_dir / "stage_weather_effect_summary.csv"),
            "candidate_environments": str(args.out_dir / "candidate_environments.csv"),
            "candidate_genotypes": str(args.out_dir / "candidate_genotypes.csv"),
            "atlas_summary": str(args.out_dir / "atlas_summary.md"),
        },
        "notes": [
            "Rankings identify candidates for biological interpretation and validation.",
            "Do not claim causality from these descriptive residual associations alone.",
        ],
    }
    (args.out_dir / "atlas_summary_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
