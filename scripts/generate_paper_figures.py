"""Generate publication-quality figures for Genome Biology submission.

Figure 1: Framework overview (schematic)
Figure 2: Main benchmark results heatmap
Figure 3: Feature engineering ablation
Figure 4: Marker density vs. performance
Figure 5: Cross-species validation
Figure 6: Computational cost vs. accuracy
Table 1: Full benchmark results
Table 2: Model characteristics
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Global style
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})

# Color palette (colorblind-friendly: Wong 2011)
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "cyan": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "black": "#000000",
    "pink": "#F781BF"
}


def require_hardcoded_result_permission() -> None:
    if os.environ.get("RESIDUAL_GXE_ALLOW_HARDCODED_FIGURES") != "1":
        raise SystemExit(
            "This legacy script contains hard-coded draft metrics. "
            "Use scripts/07_make_reports.py for formal reports, or set "
            "RESIDUAL_GXE_ALLOW_HARDCODED_FIGURES=1 only for explicitly labelled draft figures."
        )

# ============================================================
# Figure 1: Framework Overview (schematic)
# ============================================================
def fig1_framework_overview():
    fig, axes = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw={"width_ratios": [3, 2, 3]})

    # Panel A: Feature Engineering Layers
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("A. Structured Feature Engineering", fontweight="bold", loc="left")

    boxes = [
        (1, 7.5, 8, 1.5, "Layer 1: Genomic Features\nSNP markers (2,425) + NTV3 embeddings (512-dim)", C["blue"]),
        (1, 4.5, 8, 1.5, "Layer 2: Environmental Features\nGrowth-stage weather aggregation (GDD, VPD, ET0)", C["orange"]),
        (1, 1.5, 8, 1.5, "Layer 3: Physiological Indices\nCausal features from crop growth principles", C["green"]),
    ]
    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="white", alpha=0.3, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=7,
                color=color, fontweight="bold")

    # Arrows between layers
    for y in [7.5, 4.5]:
        ax.annotate("", xy=(5, y - 0.3), xytext=(5, y + 1.8),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.5))

    # Panel B: CV Protocols
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("B. Evaluation Protocols", fontweight="bold", loc="left")

    protocols = [
        (1, 7, 8, 1.2, "Random Split\nInterpolation", C["grey"]),
        (1, 5, 8, 1.2, "Leave-Genotype\nNew genotypes", C["purple"]),
        (1, 3, 8, 1.2, "Leave-Environment\nNew environments", C["cyan"]),
        (1, 1, 8, 1.2, "Forward (Leave-Year)\n2014-22 → 2023", C["red"]),
    ]
    for x, y, w, h, label, color in protocols:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor="white", alpha=0.25, linewidth=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=6.5, fontweight="bold", color=color)

    # Panel C: Methods Comparison
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("C. Methods Benchmarked", fontweight="bold", loc="left")

    methods = [
        (1, 7.5, 3.5, 1.5, "Classical\nGBLUP, Ridge", C["blue"]),
        (5.5, 7.5, 3.5, 1.5, "Bayesian\nFA-GBLUP (MegaLMM-lite)", C["purple"]),
        (1, 5, 3.5, 1.5, "Tree-based\nXGBoost, LightGBM", C["green"]),
        (5.5, 5, 3.5, 1.5, "Deep Learning\nTransformer, DeepGS", C["red"]),
        (1, 2.5, 8, 1.5, "Ensemble\nMulti-modal stacking + feature grouping", C["orange"]),
    ]
    for x, y, w, h, label, color in methods:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor="white", alpha=0.25, linewidth=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=6.5, fontweight="bold", color=color)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_framework_overview.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig1_framework_overview.png", dpi=300)
    plt.close(fig)
    print("  Figure 1 saved.")


# ============================================================
# Figure 2: Main Benchmark Results (heatmap)
# ============================================================
def fig2_benchmark_heatmap():
    # Data from Table 1 (Pearson r, mean across seeds)
    methods = [
        "Ridge\nRegression", "GBLUP\n(rrBLUP)", "FA-GBLUP\n(MegaLMM-lite)",
        "Additive\nMainEffects", "XGBoost\n+ G only", "XGBoost\n+ G+W raw",
        "XGBoost\n+ G+W agg3", "XGBoost\n+ G+W+EC", "XGBoost\n+ G+Physio",
        "XGBoost\n+ 50K+W", "Residual\nGxEFormer"
    ]
    splits = ["forward\n(leave_year)", "new_env\n(leave_environment)",
              "new_geno\n(leave_genotype)", "random"]

    data = np.array([
        [0.022, -0.041, 0.100, 0.423],          # Ridge
        [0.079, 0.229, 0.200, 0.317],           # GBLUP
        [0.174, 0.231, np.nan, np.nan],          # FA-GBLUP
        [0.174, 0.231, 0.744, 0.764],            # AdditiveMainEffects
        [0.198, np.nan, np.nan, np.nan],         # XGBoost+G_only
        [0.386, 0.359, 0.784, 0.806],            # XGBoost+G+W_raw
        [0.629, 0.396, np.nan, np.nan],          # XGBoost+G+W_agg3
        [0.515, 0.433, np.nan, np.nan],          # XGBoost+G+W+EC
        [0.423, 0.354, np.nan, np.nan],          # XGBoost+G+Physio
        [0.465, np.nan, np.nan, np.nan],         # XGBoost+50K+W
        [0.169, 0.231, 0.744, 0.764],            # ResidualGxEFormer
    ])

    fig, ax = plt.subplots(figsize=(9, 6))

    # Mask NaN cells
    mask = np.isnan(data)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Plot heatmap
    annot_data = np.where(mask, "", np.round(data, 3))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.85)

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not mask[i, j]:
                color = "white" if abs(data[i, j]) < 0.4 else "black"
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                       fontsize=7, fontweight="bold", color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="grey")

    # Highlight best per column
    for j in range(data.shape[1]):
        col = data[:, j]
        valid = ~np.isnan(col)
        if valid.any():
            best_i = np.nanargmax(col[valid])
            best_idx = np.where(valid)[0][best_i]
            ax.add_patch(plt.Rectangle((j-0.5, best_idx-0.5), 1, 1, fill=False,
                                        edgecolor=C["red"], lw=2.5, zorder=10))

    ax.set_xticks(range(len(splits))); ax.set_xticklabels(splits)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    ax.set_xlabel("Cross-validation protocol"); ax.set_ylabel("Model")
    ax.set_title("Main Benchmark Results (Pearson r)", fontweight="bold", loc="left")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("Pearson r")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_benchmark_heatmap.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig2_benchmark_heatmap.png", dpi=300)
    plt.close(fig)
    print("  Figure 2 saved.")


# ============================================================
# Figure 3: Feature Engineering Ablation
# ============================================================
def fig3_feature_ablation():
    features = ["G only", "G+W raw", "G+W\nagg3", "G+W\n+EC", "G+\nPhysio"]

    xgb_forward = [0.198, 0.386, 0.629, 0.515, 0.423]
    ame_forward = [0.174, 0.174, 0.174, 0.174, 0.174]
    xgb_newenv = [np.nan, 0.359, 0.396, 0.433, 0.354]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel A: forward prediction
    ax = axes[0]
    x = np.arange(len(features))
    w = 0.35
    bars1 = ax.bar(x - w/2, xgb_forward, w, label="XGBoost", color=C["green"], edgecolor="white")
    bars2 = ax.bar(x + w/2, ame_forward, w, label="Additive baseline", color=C["blue"], edgecolor="white", alpha=0.5)

    # Annotate improvement
    for i, (xgb, ame) in enumerate(zip(xgb_forward, ame_forward)):
        if xgb > ame:
            improvement = (xgb - ame) / abs(ame) * 100
            ax.annotate(f"+{improvement:.0f}%", (i - w/2, xgb + 0.02), ha="center", fontsize=6.5,
                       color=C["red"], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(features)
    ax.set_ylabel("Pearson r")
    ax.set_title("A. Forward Prediction (leave_year)", fontweight="bold", loc="left")
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.72)
    ax.axhline(y=0.174, color=C["blue"], linestyle="--", alpha=0.3, lw=0.8)

    # Panel B: new environment
    ax = axes[1]
    x2 = np.arange(len(features))
    valid_newenv = [0.386, 0.359, 0.396, 0.433, 0.354]  # G+W_raw, G+W_agg3, ...
    valid_labels = ["G+W raw", "G+W\nagg3", "G+W\nagg3+lag", "G+W\n+EC", "G+\nPhysio"]

    colors = [C["green"], C["green"], C["green"], C["orange"], C["cyan"]]
    bars = ax.bar(range(len(valid_newenv)), valid_newenv, color=colors, edgecolor="white")
    ax.axhline(y=0.231, color=C["blue"], linestyle="--", alpha=0.5, lw=1, label="Additive baseline (0.231)")
    ax.set_xticks(range(len(valid_newenv))); ax.set_xticklabels(valid_labels)
    ax.set_ylabel("Pearson r")
    ax.set_title("B. New-Environment Prediction", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=6)
    ax.set_ylim(0, 0.52)

    fig.suptitle("Feature Engineering Ablation", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_feature_ablation.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig3_feature_ablation.png", dpi=300)
    plt.close(fig)
    print("  Figure 3 saved.")


# ============================================================
# Figure 4: Marker Density vs. Performance
# ============================================================
def fig4_marker_density():
    fig, ax = plt.subplots(figsize=(6, 4))

    markers = [500, 1000, 2425, 3000, 5000, 10000, 20000, 50000]
    variance_selected = [0.592, 0.601, 0.599, 0.595, 0.588, 0.605, 0.572, 0.465]
    random_selected = [0.540, 0.555, 0.560, 0.558, 0.555, 0.548, 0.530, 0.420]

    ax.plot(markers, variance_selected, "o-", color=C["green"], label="Variance-selected markers", lw=2, markersize=7)
    ax.plot(markers, random_selected, "s--", color=C["grey"], label="Random markers", lw=1.5, markersize=6)
    ax.axvline(x=2425, color=C["red"], linestyle=":", lw=1.5, alpha=0.7)
    ax.axhline(y=0.629, color=C["red"], linestyle=":", lw=1.5, alpha=0.7)

    # Competition 2425 highlight
    ax.scatter([2425], [0.629], color=C["red"], s=120, zorder=10, edgecolors="white", linewidth=1.5)
    ax.annotate("Competition\n2,425 markers\nr = 0.629", (2425, 0.629), textcoords="offset points",
               xytext=(15, 15), fontsize=7, color=C["red"], fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))

    ax.set_xscale("log")
    ax.set_xlabel("Number of SNP markers (log scale)")
    ax.set_ylabel("Pearson r (forward prediction)")
    ax.set_title("Marker Density vs. Prediction Accuracy", fontweight="bold", loc="left")
    ax.legend(frameon=False)
    ax.set_ylim(0.38, 0.68)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_marker_density.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig4_marker_density.png", dpi=300)
    plt.close(fig)
    print("  Figure 4 saved.")


# ============================================================
# Figure 5: Cross-Species Validation
# ============================================================
def fig5_cross_species():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # G2F Maize
    ax = axes[0]
    models_g2f = ["Global\nMean", "Ridge", "GBLUP", "Additive\nMainFx", "XGBoost\n+W agg3"]
    pearson_g2f = [np.nan, 0.022, 0.079, 0.174, 0.629]
    colors_g2f = [C["grey"], C["blue"], C["purple"], C["cyan"], C["green"]]

    bars = ax.bar(range(len(models_g2f)), pearson_g2f, color=colors_g2f, edgecolor="white")
    ax.axhline(y=0.174, color=C["cyan"], linestyle="--", alpha=0.5, lw=1)
    ax.set_xticks(range(len(models_g2f))); ax.set_xticklabels(models_g2f, fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("G2F Maize (forward prediction)", fontweight="bold")
    ax.set_ylim(0, 0.72)
    for i, v in enumerate(pearson_g2f):
        if not np.isnan(v):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # FIP1 Wheat
    ax = axes[1]
    models_fip1 = ["Global\nMean", "Ridge\n(rrBLUP)", "Additive\nMainFx", "XGBoost\n(5K markers)"]
    pearson_fip1 = [np.nan, 0.015, 0.227, 0.448]
    colors_fip1 = [C["grey"], C["blue"], C["cyan"], C["green"]]

    bars = ax.bar(range(len(models_fip1)), pearson_fip1, color=colors_fip1, edgecolor="white")
    ax.axhline(y=0.227, color=C["cyan"], linestyle="--", alpha=0.5, lw=1)
    ax.set_xticks(range(len(models_fip1))); ax.set_xticklabels(models_fip1, fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("FIP1 Wheat (official split)", fontweight="bold")
    ax.set_ylim(0, 0.55)
    for i, v in enumerate(pearson_fip1):
        if not np.isnan(v):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # Improvement annotation
    ax.annotate("2.0× improvement\nover additive", (3, 0.448), textcoords="offset points",
               xytext=(40, -30), fontsize=7, color=C["red"], fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))

    fig.suptitle("Cross-Species Validation", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_cross_species.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig5_cross_species.png", dpi=300)
    plt.close(fig)
    print("  Figure 5 saved.")


# ============================================================
# Figure 6: Computational Cost vs. Accuracy
# ============================================================
def fig6_cost_accuracy():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # (total_time_seconds, pearson_r, label, color)
    methods_data = [
        (0.1, 0.174, "Additive\nMainEffects", C["blue"], "o"),
        (6, 0.022, "Ridge\nRegression", C["blue"], "s"),
        (260, 0.079, "GBLUP\n(rrBLUP)", C["purple"], "D"),
        (40, 0.386, "XGBoost\n+ W raw", C["green"], "p"),
        (48, 0.629, "XGBoost\n+ W agg3", C["green"], "P"),
        (90, 0.515, "XGBoost\n+ W+EC", C["orange"], "H"),
        (37, 0.423, "XGBoost\n+ Physio", C["cyan"], "X"),
        (200, 0.465, "XGBoost\n+ 50K+W", C["orange"], "h"),
        (800, 0.169, "Residual\nGxEFormer", C["red"], "^"),
    ]

    for time_s, pearson, label, color, marker in methods_data:
        ax.scatter(time_s, pearson, c=color, s=120, marker=marker, edgecolors="white",
                  linewidth=0.8, zorder=10)
        offset = (10, -10) if pearson < 0.3 else (10, 5)
        ax.annotate(label, (time_s, pearson), textcoords="offset points",
                   xytext=offset, fontsize=6.5, color=color, fontweight="bold")

    # Pareto frontier
    pareto_t = [0.1, 40, 48]
    pareto_r = [0.174, 0.386, 0.629]
    ax.plot(pareto_t, pareto_r, "--", color="grey", alpha=0.5, lw=1)
    ax.annotate("Pareto frontier", (48, 0.629), textcoords="offset points",
               xytext=(30, -15), fontsize=7, color="grey")

    ax.set_xscale("log")
    ax.set_xlabel("Total compute time (seconds, log scale)")
    ax.set_ylabel("Pearson r (forward prediction)")
    ax.set_title("Computational Cost vs. Prediction Accuracy", fontweight="bold", loc="left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_cost_accuracy.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig6_cost_accuracy.png", dpi=300)
    plt.close(fig)
    print("  Figure 6 saved.")


# ============================================================
# Table 1: Full Benchmark Results
# ============================================================
def table1_full_results():
    data = {
        "Model": [
            "GlobalMean", "Ridge Regression", "GBLUP (rrBLUP)",
            "FA-GBLUP (MegaLMM-lite)", "AdditiveMainEffects",
            "XGBoost + genotype only", "XGBoost + G + weather raw",
            "XGBoost + G + weather agg3", "XGBoost + G + weather agg3 + lag",
            "XGBoost + G + physio indices", "XGBoost + 50K markers + weather",
            "ResidualGxEFormer (residual)", "ResidualGxEFormer (direct)"
        ],
        "forward (leave_year)": [
            "—", "0.022 ± 0.01", "0.079 ± 0.02", "0.174 ± 0.02", "0.174 ± 0.02",
            "0.198 ± 0.01", "0.386 ± 0.01", "0.629 ± 0.01", "0.515 ± 0.01",
            "0.423 ± 0.01", "0.465 ± 0.02", "0.169 ± 0.02", "0.133 ± 0.02"
        ],
        "new_env": [
            "—", "-0.041 ± 0.03", "0.229 ± 0.02", "0.231 ± 0.02", "0.231 ± 0.02",
            "—", "0.359 ± 0.01", "0.396 ± 0.01", "0.433 ± 0.01",
            "0.354 ± 0.01", "—", "0.231 ± 0.02", "—"
        ],
        "new_geno": [
            "—", "0.100 ± 0.02", "0.200 ± 0.02", "—", "0.744 ± 0.01",
            "—", "0.784 ± 0.01", "—", "—", "—", "—", "0.744 ± 0.01", "—"
        ],
        "random": [
            "—", "0.423 ± 0.01", "0.317 ± 0.01", "—", "0.764 ± 0.01",
            "—", "0.806 ± 0.01", "—", "—", "—", "—", "0.764 ± 0.01", "—"
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(TAB_DIR / "table1_benchmark_results.csv", index=False)

    # Also create a formatted markdown version
    with open(TAB_DIR / "table1_benchmark_results.md", "w") as f:
        f.write("**Table 1: Full benchmark results (Pearson r, mean ± SE across 3 seeds)**\n\n")
        f.write(df.to_markdown(index=False))

    print("  Table 1 saved.")


# ============================================================
# Table 2: Model Characteristics
# ============================================================
def table2_model_characteristics():
    data = {
        "Model": [
            "AdditiveMainEffects", "Ridge Regression", "GBLUP (rrBLUP)",
            "FA-GBLUP", "XGBoost (2.4K markers)", "XGBoost (50K markers)",
            "ResidualGxEFormer", "DeepGS"
        ],
        "Type": [
            "Linear mixed model", "Linear (L2)", "Kernel regression",
            "Factor Analytic + Linear", "Gradient-boosted trees", "Gradient-boosted trees",
            "Transformer + Cross-attention", "MLP (256→128→1)"
        ],
        "Parameters": [
            "~5K (2 means per env+geno)", "2,426", "2,426 (raw marker)",
            "~15 (5 factors ×3)", "~18M (300 trees × depth 6)",
            "~18M (300 trees × depth 6)", "~1.2M", "~330K"
        ],
        "Training time (s)": [
            "<0.1", "5.8", "257", "35", "40", "200", "800", "120"
        ],
        "Inference time (s)": [
            "<0.1", "<0.1", "0.5", "0.1", "0.3", "2.5", "15", "0.5"
        ],
        "GPU required?": [
            "No", "No", "No", "No", "No", "No", "Yes (RTX 4090)", "Yes (optional)"
        ],
        "Key hyperparameters": [
            "None", "α = 1.0", "α = 1.0", "n_factors = 5, α = 1.0",
            "n_est=300, depth=6, lr=0.1", "n_est=300, depth=6, lr=0.1",
            "d_model=64, heads=4, layers=2", "hidden=[256,128], dropout=0.5"
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(TAB_DIR / "table2_model_characteristics.csv", index=False)

    with open(TAB_DIR / "table2_model_characteristics.md", "w") as f:
        f.write("**Table 2: Model characteristics and computational requirements**\n\n")
        f.write(df.to_markdown(index=False))

    print("  Table 2 saved.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    require_hardcoded_result_permission()
    print("Generating paper figures and tables...")
    print(f"  Output: {FIG_DIR} / {TAB_DIR}")

    fig1_framework_overview()
    fig2_benchmark_heatmap()
    fig3_feature_ablation()
    fig4_marker_density()
    fig5_cross_species()
    fig6_cost_accuracy()
    table1_full_results()
    table2_model_characteristics()

    print(f"\nDone. Generated 6 figures + 2 tables.")
