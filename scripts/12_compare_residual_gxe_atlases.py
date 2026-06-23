from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table
from residual_gxe.evaluation.atlas import compare_stage_weather_features


def parse_args():
    parser = argparse.ArgumentParser(description="Compare residual GxE atlas summaries across datasets.")
    parser.add_argument("--left-summary-dir", type=Path, required=True)
    parser.add_argument("--right-summary-dir", type=Path, required=True)
    parser.add_argument("--left-label", type=str, default="g2f")
    parser.add_argument("--right-label", type=str, default="fip1")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_table(path)
    except EmptyDataError:
        return pd.DataFrame()


def _dataset_summary(summary_dir: Path, label: str) -> dict:
    stable = _read_csv_if_exists(summary_dir / "stable_stage_weather_features.csv")
    stage = _read_csv_if_exists(summary_dir / "stage_weather_effect_summary.csv")
    env = _read_csv_if_exists(summary_dir / "candidate_environments.csv")
    geno = _read_csv_if_exists(summary_dir / "candidate_genotypes.csv")
    return {
        "dataset": label,
        "n_stable_stage_weather_features": int(len(stable)),
        "n_stage_weather_effect_rows": int(len(stage)),
        "n_candidate_environments": int(len(env)),
        "n_candidate_genotypes": int(len(geno)),
        "has_stage_weather_evidence": bool(len(stable) > 0),
    }


def _write_report(
    out_dir: Path,
    left_label: str,
    right_label: str,
    summary_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> None:
    lines = [
        "# Cross-Dataset Residual GxE Atlas Comparison",
        "",
        "This report compares machine-generated atlas summaries.",
        "Absence of overlap can mean either biological disagreement or missing/unmatched weather features; inspect dataset summary before interpretation.",
        "",
        "## Dataset Summary",
        "",
    ]
    cols = summary_df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---" for _ in cols]) + "|")
    for _, row in summary_df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines.append("")

    lines.append("## Shared Stage-Weather Features")
    if len(overlap_df) == 0:
        lines.append(
            f"No shared stage-weather feature rows were available between {left_label} and {right_label}. "
            "This is expected when one dataset lacks processed weather_daily features."
        )
    else:
        cols = ["feature", "left_mean_pearson", "right_mean_pearson", "sign_agreement", "left_n_runs", "right_n_runs"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---" for _ in cols]) + "|")
        for _, row in overlap_df.head(30).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                    for c in cols
                )
                + " |"
            )
    lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("")
    lines.append("- Shared direction supports cross-dataset consistency, not causality.")
    lines.append("- Missing weather features in an external dataset prevent stage-weather validation.")
    lines.append("- Candidate genotype/environment lists are dataset-specific unless linked through shared germplasm or harmonized environments.")

    (out_dir / "cross_dataset_atlas_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    left_stable = _read_csv_if_exists(args.left_summary_dir / "stable_stage_weather_features.csv")
    right_stable = _read_csv_if_exists(args.right_summary_dir / "stable_stage_weather_features.csv")
    overlap = compare_stage_weather_features(left_stable, right_stable, args.left_label, args.right_label)
    write_table(overlap, args.out_dir / "shared_stage_weather_features.csv")

    summary = pd.DataFrame([
        _dataset_summary(args.left_summary_dir, args.left_label),
        _dataset_summary(args.right_summary_dir, args.right_label),
    ])
    write_table(summary, args.out_dir / "atlas_dataset_summary.csv")
    _write_report(args.out_dir, args.left_label, args.right_label, summary, overlap)

    manifest = {
        "schema_version": 1,
        "script": "scripts/12_compare_residual_gxe_atlases.py",
        "left_summary_dir": str(args.left_summary_dir),
        "right_summary_dir": str(args.right_summary_dir),
        "left_label": args.left_label,
        "right_label": args.right_label,
        "outputs": {
            "shared_stage_weather_features": str(args.out_dir / "shared_stage_weather_features.csv"),
            "atlas_dataset_summary": str(args.out_dir / "atlas_dataset_summary.csv"),
            "report": str(args.out_dir / "cross_dataset_atlas_comparison.md"),
        },
    }
    (args.out_dir / "cross_dataset_atlas_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
