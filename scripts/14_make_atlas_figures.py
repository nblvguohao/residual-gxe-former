from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_gxe.data.loaders import read_table, write_table


STAGE_ORDER = ["early", "mid", "late"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create figure-ready residual GxE atlas tables and plots from summary CSVs.")
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-label", type=str, default="g2f_maize")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return read_table(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _stage_weather_matrix(stage_effects: pd.DataFrame) -> pd.DataFrame:
    if len(stage_effects) == 0:
        return pd.DataFrame()
    required = {"stage", "weather_variable", "mean_abs_pearson"}
    missing = required - set(stage_effects.columns)
    if missing:
        raise ValueError(f"stage_weather_effect_summary missing columns: {sorted(missing)}")
    mat = stage_effects.pivot_table(
        index="stage",
        columns="weather_variable",
        values="mean_abs_pearson",
        aggfunc="mean",
    )
    ordered_index = [s for s in STAGE_ORDER if s in mat.index] + [s for s in mat.index if s not in STAGE_ORDER]
    return mat.loc[ordered_index].fillna(0.0)


def _plot_stage_weather_heatmap(matrix: pd.DataFrame, dataset_label: str, out_dir: Path) -> None:
    if matrix.empty:
        return
    fig_w = max(6.0, 0.55 * len(matrix.columns) + 2.0)
    fig_h = max(3.0, 0.7 * len(matrix.index) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Weather variable")
    ax.set_ylabel("Growth stage")
    ax.set_title(f"{dataset_label}: residual association by stage and weather")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean absolute Pearson r")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if value > 0:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > matrix.values.max() / 2 else "black", fontsize=7)
    fig.tight_layout()
    _save_fig(fig, out_dir / "fig_stage_weather_heatmap")


def _plot_candidate_bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    out_base: Path,
    top_n: int,
) -> None:
    if len(df) == 0 or label_col not in df.columns or value_col not in df.columns:
        return
    sub = df.head(top_n).copy()
    sub[label_col] = sub[label_col].astype(str)
    fig_h = max(4.0, 0.32 * len(sub) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    y = np.arange(len(sub))
    ax.barh(y, sub[value_col].to_numpy(dtype=float), color="#2E7D32")
    ax.set_yticks(y)
    ax.set_yticklabels(sub[label_col])
    ax.invert_yaxis()
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)
    fig.tight_layout()
    _save_fig(fig, out_base)


def _write_markdown_index(out_dir: Path, dataset_label: str, generated: dict[str, str]) -> None:
    lines = [
        "# Residual GxE Atlas Figure Outputs",
        "",
        f"Dataset: `{dataset_label}`",
        "",
        "All plots are rendered from figure-ready CSV files in this directory.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in generated.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("")
    lines.append("These figures show descriptive residual associations and ranked candidates. They are not causal evidence without independent validation.")
    (out_dir / "atlas_figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gap_report(out_dir: Path, dataset_label: str, reason: str) -> None:
    lines = [
        "# Residual GxE Atlas Figure Gap Report",
        "",
        f"Dataset: `{dataset_label}`",
        "",
        reason,
        "",
        "No hard-coded or synthetic figure data were generated.",
    ]
    (out_dir / "atlas_figure_gap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stable = _read_optional(args.summary_dir / "stable_stage_weather_features.csv")
    stage_effects = _read_optional(args.summary_dir / "stage_weather_effect_summary.csv")
    candidate_envs = _read_optional(args.summary_dir / "candidate_environments.csv")
    candidate_genotypes = _read_optional(args.summary_dir / "candidate_genotypes.csv")

    generated: dict[str, str] = {}

    if len(stable) > 0:
        top_features = stable.head(args.top_n).copy()
        write_table(top_features, args.out_dir / "table_top_stage_weather_features.csv")
        generated["table_top_stage_weather_features"] = str(args.out_dir / "table_top_stage_weather_features.csv")

    matrix = _stage_weather_matrix(stage_effects)
    if not matrix.empty:
        matrix_out = matrix.reset_index().rename(columns={"index": "stage"})
        write_table(matrix_out, args.out_dir / "matrix_stage_weather_mean_abs_pearson.csv")
        _plot_stage_weather_heatmap(matrix, args.dataset_label, args.out_dir)
        generated["matrix_stage_weather_mean_abs_pearson"] = str(args.out_dir / "matrix_stage_weather_mean_abs_pearson.csv")
        generated["fig_stage_weather_heatmap"] = str(args.out_dir / "fig_stage_weather_heatmap.png")

    if len(candidate_envs) > 0:
        env_table = candidate_envs.head(args.top_n).copy()
        write_table(env_table, args.out_dir / "table_candidate_environments.csv")
        _plot_candidate_bar(
            env_table,
            label_col="environment_id",
            value_col="candidate_score",
            title=f"{args.dataset_label}: candidate residual environments",
            out_base=args.out_dir / "fig_candidate_environments",
            top_n=args.top_n,
        )
        generated["table_candidate_environments"] = str(args.out_dir / "table_candidate_environments.csv")
        generated["fig_candidate_environments"] = str(args.out_dir / "fig_candidate_environments.png")

    if len(candidate_genotypes) > 0:
        genotype_table = candidate_genotypes.head(args.top_n).copy()
        write_table(genotype_table, args.out_dir / "table_candidate_genotypes.csv")
        score_col = "stable_positive_score" if "stable_positive_score" in genotype_table.columns else "mean_residual"
        _plot_candidate_bar(
            genotype_table,
            label_col="genotype_id",
            value_col=score_col,
            title=f"{args.dataset_label}: candidate residual genotypes",
            out_base=args.out_dir / "fig_candidate_genotypes",
            top_n=args.top_n,
        )
        generated["table_candidate_genotypes"] = str(args.out_dir / "table_candidate_genotypes.csv")
        generated["fig_candidate_genotypes"] = str(args.out_dir / "fig_candidate_genotypes.png")

    if generated:
        _write_markdown_index(args.out_dir, args.dataset_label, generated)
    else:
        _write_gap_report(args.out_dir, args.dataset_label, "No non-empty atlas summary tables were found.")

    manifest = {
        "schema_version": 1,
        "script": "scripts/14_make_atlas_figures.py",
        "summary_dir": str(args.summary_dir),
        "dataset_label": args.dataset_label,
        "top_n": int(args.top_n),
        "generated": generated,
        "notes": [
            "Figures are generated only from atlas summary CSV files.",
            "No hard-coded metrics are used.",
        ],
    }
    (args.out_dir / "atlas_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()

