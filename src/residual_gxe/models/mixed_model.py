from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import OneHotEncoder


@dataclass
class AdditiveMainEffects:
    global_mean: float
    genotype_effects: dict[str, float]
    environment_effects: dict[str, float]
    genotype_col: str = "genotype_id"
    environment_col: str = "environment_id"
    target_col: str = "phenotype_value"

    def predict_main_effects(self, df: pd.DataFrame) -> pd.Series:
        g = df[self.genotype_col].map(self.genotype_effects).fillna(0.0)
        e = df[self.environment_col].map(self.environment_effects).fillna(0.0)
        return self.global_mean + g + e

    def residuals(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target_col] - self.predict_main_effects(df)


def fit_additive_main_effects(
    train_df: pd.DataFrame,
    genotype_col: str = "genotype_id",
    environment_col: str = "environment_id",
    target_col: str = "phenotype_value",
) -> AdditiveMainEffects:
    """Simple fold-safe additive main-effect estimator.

    This is a stable fallback, not a replacement for a full mixed model.
    It estimates genotype and environment deviations from the training-fold global mean.
    """
    global_mean = float(train_df[target_col].mean())
    genotype_effects = (train_df.groupby(genotype_col)[target_col].mean() - global_mean).to_dict()
    environment_effects = (train_df.groupby(environment_col)[target_col].mean() - global_mean).to_dict()
    return AdditiveMainEffects(
        global_mean=global_mean,
        genotype_effects={str(k): float(v) for k, v in genotype_effects.items()},
        environment_effects={str(k): float(v) for k, v in environment_effects.items()},
        genotype_col=genotype_col,
        environment_col=environment_col,
        target_col=target_col,
    )


@dataclass
class RidgeMainEffects:
    """Regularized main-effect estimator using BayesianRidge on one-hot factors.

    Unlike AdditiveMainEffects, this estimator:
    - Uses L2 regularization to shrink rare-genotype estimates toward the mean
    - Naturally handles unseen genotypes/environments (coefficient → 0)
    - Captures implicit G+E structure through joint regularization
    """

    intercept: float
    genotype_coefficients: dict[str, float]
    environment_coefficients: dict[str, float]
    genotype_col: str = "genotype_id"
    environment_col: str = "environment_id"
    target_col: str = "phenotype_value"

    @property
    def global_mean(self):
        """Alias for backward compatibility with additive model API."""
        return self.intercept

    @property
    def genotype_effects(self):
        """Alias for backward compatibility with additive model API."""
        return self.genotype_coefficients

    @property
    def environment_effects(self):
        """Alias for backward compatibility with additive model API."""
        return self.environment_coefficients

    def predict_main_effects(self, df: pd.DataFrame) -> pd.Series:
        g = df[self.genotype_col].map(self.genotype_coefficients).fillna(0.0)
        e = df[self.environment_col].map(self.environment_coefficients).fillna(0.0)
        return pd.Series(self.intercept + g.values + e.values, index=df.index)

    def residuals(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target_col] - self.predict_main_effects(df)


def fit_ridge_main_effects(
    train_df: pd.DataFrame,
    genotype_col: str = "genotype_id",
    environment_col: str = "environment_id",
    target_col: str = "phenotype_value",
    alpha: float = 1.0,
    use_bayesian: bool = True,
) -> RidgeMainEffects:
    """Fit regularized main effects via BayesianRidge on one-hot encoded factors.

    This estimator one-hot encodes genotype_id and environment_id, fits a
    regularized linear model, then recovers per-factor coefficients.
    Unseen factors receive zero contribution (regularization toward intercept).

    Parameters
    ----------
    alpha : float
        Regularization strength (only used if use_bayesian=False for plain Ridge).
    use_bayesian : bool
        If True, use BayesianRidge (auto-tuned regularization). If False, use Ridge.
    """
    train_df = train_df.copy()
    train_df[genotype_col] = train_df[genotype_col].astype(str)
    train_df[environment_col] = train_df[environment_col].astype(str)

    y = train_df[target_col].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y))
    y_centered = y - global_mean

    # One-hot encode genotype and environment separately for coefficient recovery
    geno_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    env_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    X_geno = geno_enc.fit_transform(train_df[[genotype_col]])
    X_env = env_enc.fit_transform(train_df[[environment_col]])

    X = np.column_stack([X_geno, X_env])

    if use_bayesian:
        model = BayesianRidge(
            max_iter=300,
            tol=1e-3,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
        )
    else:
        model = Ridge(alpha=alpha, fit_intercept=False)

    model.fit(X, y_centered)

    coef = model.coef_
    n_geno = X_geno.shape[1]
    geno_coef = coef[:n_geno]
    env_coef = coef[n_geno:]

    genotype_coefficients = {
        str(cat): float(geno_coef[i])
        for i, cat in enumerate(geno_enc.categories_[0])
    }
    environment_coefficients = {
        str(cat): float(env_coef[i])
        for i, cat in enumerate(env_enc.categories_[0])
    }

    return RidgeMainEffects(
        intercept=global_mean,
        genotype_coefficients=genotype_coefficients,
        environment_coefficients=environment_coefficients,
        genotype_col=genotype_col,
        environment_col=environment_col,
        target_col=target_col,
    )
