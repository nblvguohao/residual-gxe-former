from __future__ import annotations

import numpy as np
import pandas as pd

from residual_gxe.training.preprocessing import FoldPreprocessor, normalize_weather_dates


def test_marker_selection_uses_training_genotypes_only():
    train_pheno = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "genotype_id": ["g1", "g2", "g3"],
            "environment_id": ["e1", "e1", "e1"],
        }
    )
    test_pheno = pd.DataFrame(
        {
            "sample_id": ["s4", "s5"],
            "genotype_id": ["g4", "g5"],
            "environment_id": ["e1", "e1"],
        }
    )
    geno = pd.DataFrame(
        {
            "genotype_id": ["g1", "g2", "g3", "g4", "g5"],
            "train_signal": [0.0, 1.0, 2.0, 0.0, 0.0],
            "test_only_signal": [1.0, 1.0, 1.0, -100.0, 100.0],
        }
    )

    pre = FoldPreprocessor(max_markers=1, marker_strategy="variance")
    X_train, _Xw_train, _Xe_train = pre.fit_transform(train_pheno, geno)
    X_test, _Xw_test, _Xe_test = pre.transform(test_pheno, geno)

    assert pre.marker_cols == ["train_signal"]
    assert X_train.shape == (3, 1)
    assert X_test.shape == (2, 1)
    assert np.allclose(X_test[:, 0], [0.0, 0.0])


def test_marker_imputation_uses_training_mean_for_test_rows():
    train_pheno = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "genotype_id": ["g1", "g2"],
            "environment_id": ["e1", "e1"],
        }
    )
    test_pheno = pd.DataFrame(
        {
            "sample_id": ["s3"],
            "genotype_id": ["g3"],
            "environment_id": ["e1"],
        }
    )
    geno = pd.DataFrame(
        {
            "genotype_id": ["g1", "g2", "g3"],
            "m1": [0.0, 2.0, np.nan],
        }
    )

    pre = FoldPreprocessor(max_markers=10)
    pre.fit(train_pheno, geno)
    X_test, _Xw_test, _Xe_test = pre.transform(test_pheno, geno)

    assert pre.marker_fill_values["m1"] == 1.0
    assert X_test[0, 0] == 1.0


def test_weather_date_normalizer_recovers_yyyymmdd_from_epoch_nanoseconds():
    weather = pd.DataFrame(
        {
            "environment_id": ["e1", "e1"],
            "date": pd.to_datetime([20150102, 20150101]),
            "tmax": [20.0, 10.0],
        }
    )

    out = normalize_weather_dates(weather)

    assert out["_weather_date"].dt.strftime("%Y-%m-%d").tolist() == ["2015-01-02", "2015-01-01"]


def test_stage_summary_weather_uses_dap_windows_and_train_standardization():
    train_pheno = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "genotype_id": ["g1", "g2"],
            "environment_id": ["e1", "e2"],
        }
    )
    test_pheno = pd.DataFrame(
        {
            "sample_id": ["s3"],
            "genotype_id": ["g3"],
            "environment_id": ["e3"],
        }
    )
    geno = pd.DataFrame(
        {
            "genotype_id": ["g1", "g2", "g3"],
            "m1": [0.0, 1.0, 2.0],
        }
    )
    weather = pd.DataFrame(
        {
            "environment_id": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "day_after_planting": [5, 65, 10, 70, 15, 80],
            "tmax": [10.0, 30.0, 20.0, 40.0, 15.0, 35.0],
            "precipitation": [0.0, 2.0, 1.0, 0.0, 3.0, 0.0],
        }
    )

    pre = FoldPreprocessor(
        max_markers=10,
        weather_mode="stage_summary",
        weather_standardize=True,
        weather_feat_dim=2,
    )
    _Xg_train, Xw_train, _Xe_train = pre.fit_transform(train_pheno, geno, weather_data=weather)
    _Xg_test, Xw_test, _Xe_test = pre.transform(test_pheno, geno, weather_data=weather, main_effects=[1.5])

    assert Xw_train.shape == (2, 5, len(pre.weather_feature_names))
    assert Xw_test.shape == (1, 5, len(pre.weather_feature_names))
    assert pre.weather_center
    assert pre.weather_scale


def test_main_effect_input_is_appended_to_static_environment():
    train_pheno = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "genotype_id": ["g1"],
            "environment_id": ["e1"],
        }
    )
    geno = pd.DataFrame({"genotype_id": ["g1"], "m1": [1.0]})
    env = pd.DataFrame({"environment_id": ["e1"], "latitude": [42.0]})

    pre = FoldPreprocessor(max_markers=10)
    pre.fit(train_pheno, geno, env)
    _Xg, _Xw, Xe = pre.transform(train_pheno, geno, env, main_effects=[3.25])

    assert Xe.shape == (1, 2)
    assert Xe[0, -1] == 3.25
