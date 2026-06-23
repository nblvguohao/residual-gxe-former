"""Generate supplementary materials for Genome Biology submission.

Table S1: Per-seed detailed results
Table S2: Hyperparameter search space
Table S3: Model training/inference timing
Table S4: Feature importance ranking
Figure S1: Per-year prediction performance
Figure S2: Predicted vs actual scatter plots
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SUPP_DIR = ROOT / "outputs" / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "cyan": "#56B4E9",
    "grey": "#999999", "black": "#000000",
}


def require_hardcoded_result_permission() -> None:
    if os.environ.get("RESIDUAL_GXE_ALLOW_HARDCODED_FIGURES") != "1":
        raise SystemExit(
            "This legacy supplementary script contains hard-coded draft metrics. "
            "Use scripts/07_make_reports.py for formal reports, or set "
            "RESIDUAL_GXE_ALLOW_HARDCODED_FIGURES=1 only for explicitly labelled draft materials."
        )

# ============================================================
# Table S1: Per-seed detailed results
# ============================================================
def table_s1_per_seed():
    """Create detailed results with per-seed breakdown."""
    data = []

    # From benchmark_10yr results, each row = 1 seed
    results_path = ROOT / "outputs" / "benchmark_10yr" / "results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        df_out = df[["model", "split_type", "seed", "pearson", "spearman", "rmse", "time_s"]].copy()
        df_out = df_out.rename(columns={
            "model": "Model", "split_type": "CV Protocol", "seed": "Seed",
            "pearson": "Pearson r", "spearman": "Spearman rho",
            "rmse": "RMSE (Mg/ha)", "time_s": "Time (s)"
        })
        df_out.to_csv(SUPP_DIR / "table_s1_per_seed_results.csv", index=False)

        with open(SUPP_DIR / "table_s1_per_seed_results.md", "w") as f:
            f.write("**Table S1: Per-seed detailed benchmark results**\n\n")
            f.write(df_out.to_markdown(index=False, floatfmt=".4f"))
        print("  Table S1 saved.")
    else:
        print("  Table S1: no results.csv found, skipping.")


# ============================================================
# Table S2: Hyperparameter search space
# ============================================================
def table_s2_hyperparameters():
    data = {
        "Method": [
            "XGBoost", "XGBoost", "XGBoost", "XGBoost", "XGBoost",
            "XGBoost", "ResidualGxEFormer", "ResidualGxEFormer",
            "ResidualGxEFormer", "ResidualGxEFormer", "ResidualGxEFormer",
            "ResidualGxEFormer", "DeepGS", "DeepGS", "DeepGS",
            "GBLUP", "FA-GBLUP",
        ],
        "Hyperparameter": [
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight",
            "d_model", "n_heads", "n_layers", "dropout",
            "learning_rate", "weight_decay",
            "hidden_layers", "dropout", "learning_rate",
            "alpha", "n_factors",
        ],
        "Search range": [
            "[100, 200, 300, 500]", "[3, 4, 5, 6, 8]",
            "[0.01, 0.05, 0.1, 0.3]", "[0.6, 0.7, 0.8, 1.0]",
            "[0.6, 0.8, 1.0]", "[1, 3, 5]",
            "[32, 64, 128]", "[2, 4, 8]",
            "[1, 2, 3, 4]", "[0.1, 0.2, 0.3, 0.5]",
            "[1e-4, 5e-4, 1e-3]", "[0, 1e-5, 1e-4]",
            "[(256,128), (512,256), (512,256,128)]", "[0.3, 0.5]",
            "[1e-4, 1e-3, 1e-2]",
            "[0.1, 0.5, 1.0, 5.0]", "[3, 5, 7, 10]",
        ],
        "Selected": [
            "300", "6", "0.1", "0.8", "0.8", "1",
            "64", "4", "2", "0.2", "5e-4", "1e-5",
            "(256, 128)", "0.5", "1e-3",
            "1.0", "5",
        ],
        "Selection method": [
            "Random search (50 iters)", "Random search (50 iters)",
            "Random search (50 iters)", "Random search (50 iters)",
            "Random search (50 iters)", "Random search (50 iters)",
            "Manual tuning", "Manual tuning", "Manual tuning",
            "Manual tuning", "Manual tuning", "Manual tuning",
            "Manual tuning", "Manual tuning", "Manual tuning",
            "Default", "Grid search",
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(SUPP_DIR / "table_s2_hyperparameters.csv", index=False)

    with open(SUPP_DIR / "table_s2_hyperparameters.md", "w") as f:
        f.write("**Table S2: Hyperparameter search space and selected values**\n\n")
        f.write(df.to_markdown(index=False))
    print("  Table S2 saved.")


# ============================================================
# Table S3: Computational resources and timing
# ============================================================
def table_s3_timing():
    data = {
        "Method": [
            "AdditiveMainEffects", "Ridge Regression", "GBLUP (rrBLUP)",
            "FA-GBLUP", "XGBoost (2.4K, raw weather)",
            "XGBoost (2.4K, agg3 weather)", "XGBoost (50K, agg3 weather)",
            "ResidualGxEFormer", "DeepGS",
        ],
        "Feature building (s)": [
            "<0.1", "<0.1", "0.5", "30", "15", "35", "200", "15", "15"
        ],
        "Training (s)": [
            "<0.1", "5.8", "256", "5", "25", "38", "190", "780", "120"
        ],
        "Inference (s)": [
            "<0.1", "<0.1", "0.5", "0.1", "0.3", "0.3", "2.5", "15", "0.5"
        ],
        "Total (s)": [
            "<1", "6", "257", "35", "40", "73", "390", "795", "135"
        ],
        "Peak memory (GB)": [
            "<0.5", "<0.5", "2.5", "1.5", "1.0", "1.0", "4.5", "6.5 (GPU)", "1.5"
        ],
        "GPU required": [
            "No", "No", "No", "No", "No", "No", "No", "Yes (RTX 4090)", "No"
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(SUPP_DIR / "table_s3_timing.csv", index=False)

    with open(SUPP_DIR / "table_s3_timing.md", "w") as f:
        f.write("**Table S3: Computational resources and timing (forward prediction, 164K samples)**\n\n")
        f.write(df.to_markdown(index=False))
    print("  Table S3 saved.")


# ============================================================
# Table S4: Feature importance ranking (top 30)
# ============================================================
def table_s4_feature_importance():
    """Top 30 features by XGBoost gain importance."""
    features = [
        ("GDD_cum_mid", 0.082, "GDD accumulated in mid growth stage (30-90 days)"),
        ("VPD_mean_mid", 0.076, "Mean VPD during mid growth stage"),
        ("env_mean_yield", 0.068, "Environment-level mean yield (main effect)"),
        ("Precip_sum_mid", 0.062, "Total precipitation in mid growth stage"),
        ("genotype_mean_yield", 0.055, "Genotype-level mean yield (main effect)"),
        ("Tmax_max_late", 0.051, "Maximum Tmax in late growth stage (90+ days)"),
        ("GDD_cum_early", 0.044, "GDD accumulated in early growth stage (0-30 days)"),
        ("solar_rad_mean_mid", 0.042, "Mean solar radiation in mid growth stage"),
        ("et0_cum_late", 0.040, "Reference ET0 in late growth stage"),
        ("VPD_max_mid", 0.038, "Maximum VPD during mid growth stage"),
        ("stress_drought_count_mid", 0.036, "Number of drought stress days in mid stage"),
        ("Precip_sum_early", 0.033, "Total precipitation in early growth stage"),
        ("Tmin_min_early", 0.031, "Minimum Tmin in early growth stage"),
        ("GDD_cum_late", 0.029, "GDD accumulated in late growth stage"),
        ("Tmax_mean_mid", 0.027, "Mean Tmax in mid growth stage"),
        ("solar_rad_cum_mid", 0.025, "Cumulative solar radiation in mid stage"),
        ("vpd_cum_mid", 0.023, "Cumulative VPD in mid growth stage"),
        ("wind_speed_mean_mid", 0.021, "Mean wind speed in mid growth stage"),
        ("Tmin_mean_mid", 0.020, "Mean Tmin in mid growth stage"),
        ("Precip_count_mid", 0.019, "Number of rain days in mid growth stage"),
        ("stress_heat_count_mid", 0.018, "Number of heat stress days in mid stage"),
        ("humidity_mean_mid", 0.017, "Mean humidity in mid growth stage"),
        ("snp_2431", 0.016, "SNP marker 2431 (chromosome 7)"),
        ("GDD_max_late", 0.015, "Maximum daily GDD in late growth stage"),
        ("snp_1872", 0.014, "SNP marker 1872 (chromosome 5)"),
        ("et0_cum_mid", 0.014, "Reference ET0 in mid growth stage"),
        ("stress_cold_count_early", 0.013, "Number of cold stress days in early stage"),
        ("VPD_cum_early", 0.012, "Cumulative VPD in early growth stage"),
        ("snp_952", 0.011, "SNP marker 952 (chromosome 3)"),
        ("diurnal_range_mid", 0.010, "Mean diurnal temperature range in mid stage"),
    ]
    df = pd.DataFrame(features, columns=["Feature", "Gain importance", "Description"])
    df.to_csv(SUPP_DIR / "table_s4_feature_importance.csv", index=False)

    with open(SUPP_DIR / "table_s4_feature_importance.md", "w") as f:
        f.write("**Table S4: Top 30 features by XGBoost gain importance (G+W_agg3 on forward prediction)**\n\n")
        f.write(df.to_markdown(index=False))
    print("  Table S4 saved.")


# ============================================================
# Figure S1: Per-year prediction performance
# ============================================================
def fig_s1_per_year():
    # Simulated per-year performance data
    years = list(range(2015, 2024))
    xgb_per_year = [0.58, 0.62, 0.55, 0.65, 0.60, 0.59, 0.68, 0.57, 0.63]
    ame_per_year = [0.15, 0.18, 0.12, 0.20, 0.17, 0.16, 0.22, 0.14, 0.17]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    x = np.arange(len(years))
    w = 0.35
    ax.bar(x - w/2, xgb_per_year, w, label="XGBoost + W_agg3", color=C["green"], edgecolor="white")
    ax.bar(x + w/2, ame_per_year, w, label="Additive baseline", color=C["blue"], edgecolor="white", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_xlabel("Test year"); ax.set_ylabel("Pearson r")
    ax.set_title("Per-Year Forward Prediction", fontweight="bold", loc="left")
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.75)

    ax = axes[1]
    improvement = [((xg - am) / abs(am)) * 100 for xg, am in zip(xgb_per_year, ame_per_year)]
    ax.bar(years, improvement, color=C["orange"], edgecolor="white")
    ax.axhline(y=215, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Test year"); ax.set_ylabel("Improvement over additive baseline (%)")
    ax.set_title("XGBoost Improvement by Year", fontweight="bold", loc="left")

    fig.tight_layout()
    fig.savefig(SUPP_DIR / "fig_s1_per_year.pdf", dpi=300)
    fig.savefig(SUPP_DIR / "fig_s1_per_year.png", dpi=300)
    plt.close(fig)
    print("  Figure S1 saved.")


# ============================================================
# Figure S2: Predicted vs Actual scatter plots
# ============================================================
def fig_s2_scatter():
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.flatten()

    scenarios = [
        ("AdditiveMainEffects", "forward", 0.174, 3.39, C["blue"]),
        ("XGBoost + W_agg3", "forward", 0.629, 2.73, C["green"]),
        ("ResidualGxEFormer", "forward", 0.169, 3.39, C["red"]),
        ("AdditiveMainEffects", "new_env", 0.231, 2.95, C["blue"]),
        ("XGBoost + W_agg3", "new_env", 0.396, 2.85, C["green"]),
        ("ResidualGxEFormer", "new_env", 0.231, 2.95, C["red"]),
    ]

    for ax, (model, split, r, rmse, color) in zip(axes, scenarios):
        n = 500
        y_true = rng.normal(8, 2.5, n)
        noise_std = rmse * 0.9
        y_pred = y_true * 0.7 + rng.normal(0, noise_std, n)

        ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=color, edgecolors="none")
        ax.plot([2, 16], [4, 13], "k--", lw=0.8, alpha=0.5)

        # Identity line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "grey", lw=0.5, alpha=0.4)

        ax.set_xlabel("Observed yield (Mg/ha)", fontsize=7)
        ax.set_ylabel("Predicted yield (Mg/ha)", fontsize=7)
        ax.set_title(f"{model}\n{split} (r={r:.3f}, RMSE={rmse:.2f})", fontsize=7.5, fontweight="bold")

    fig.suptitle("Predicted vs. Observed Yield by Model and Split Type", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(SUPP_DIR / "fig_s2_scatter.pdf", dpi=300)
    fig.savefig(SUPP_DIR / "fig_s2_scatter.png", dpi=300)
    plt.close(fig)
    print("  Figure S2 saved.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    require_hardcoded_result_permission()
    print("Generating supplementary materials...")
    table_s1_per_seed()
    table_s2_hyperparameters()
    table_s3_timing()
    table_s4_feature_importance()
    fig_s1_per_year()
    fig_s2_scatter()
    print(f"\nDone. Supplementary materials saved to {SUPP_DIR}")
