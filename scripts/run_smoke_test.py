from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_gxe.data.schema import validate_phenotype_table
from residual_gxe.data.splits import SplitConfig, make_split_table
from residual_gxe.evaluation.metrics import metrics_dict
from residual_gxe.models.deep import ResidualGxEFormer
from residual_gxe.models.mixed_model import fit_additive_main_effects


def synthetic_pheno(n_genotypes=20, n_envs=8, years=(2020, 2021, 2022, 2023)):
    rows = []
    rng = np.random.default_rng(1234)
    for g in range(n_genotypes):
        for e in range(n_envs):
            year = years[e % len(years)]
            value = 10 + 0.1 * g - 0.2 * e + rng.normal(0, 0.5)
            rows.append({
                "sample_id": f"S{g}_{e}",
                "genotype_id": f"G{g}",
                "environment_id": f"E{e}",
                "year": year,
                "location_id": f"L{e % 4}",
                "trait_id": "yield",
                "trait_name": "synthetic_yield",
                "trait_family": "yield",
                "phenotype_value": value,
                "phenotype_unit": "synthetic_unit",
                "source_dataset": "synthetic_smoke_test",
            })
    return pd.DataFrame(rows)


def main():
    out_dir = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pheno = synthetic_pheno()
    schema = validate_phenotype_table(pheno)
    assert schema.ok, schema

    split = make_split_table(pheno, SplitConfig("leave_environment", seed=1234), fold=0)
    merged = pheno.merge(split, on="sample_id")
    train = merged[merged["split"] == "train"].copy()
    test = merged[merged["split"] == "test"].copy()

    effects = fit_additive_main_effects(train)
    test_pred_main = effects.predict_main_effects(test)
    m = metrics_dict(test["phenotype_value"], test_pred_main)

    model = ResidualGxEFormer(n_markers=128, weather_dim=6, static_env_dim=4, hidden_dim=32, patch_size=16)
    markers = torch.randn(5, 128)
    weather = torch.randn(5, 30, 6)
    static_env = torch.randn(5, 4)
    pred = model(markers, weather, static_env)
    assert pred.shape == (5,)

    report = [
        "# Synthetic Smoke Test Report",
        "",
        "This report uses synthetic data only. Do not report these values as scientific results.",
        "",
        f"n_samples: {len(pheno)}",
        f"test_metrics_main_effect_baseline: {m}",
        f"deep_model_output_shape: {tuple(pred.shape)}",
    ]
    path = out_dir / "smoke_test_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
