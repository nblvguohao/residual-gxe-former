from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineResult:
    model_name: str
    predictions: np.ndarray
    metadata: dict[str, Any]


def fit_bayes_ridge(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> BaselineResult:
    model = make_pipeline(StandardScaler(with_mean=True), BayesianRidge())
    model.fit(X_train, y_train)
    return BaselineResult("bayes_ridge", model.predict(X_test), {})


def fit_ridge_rrblup_like(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> BaselineResult:
    model = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    return BaselineResult("ridge_rrblup_like", model.predict(X_test), {"alpha": alpha})


def fit_gblup_kernel_ridge(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> BaselineResult:
    scaler = StandardScaler(with_mean=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    K_train = linear_kernel(X_train_s, X_train_s) / max(1, X_train_s.shape[1])
    K_test = linear_kernel(X_test_s, X_train_s) / max(1, X_train_s.shape[1])
    model = KernelRidge(alpha=alpha, kernel="precomputed")
    model.fit(K_train, y_train)
    return BaselineResult("gblup_kernel_ridge", model.predict(K_test), {"alpha": alpha})


def fit_gblup_efficient(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> BaselineResult:
    """Efficient GBLUP via Ridge on marker matrix.

    Mathematically equivalent to GBLUP (VanRaden 2008): rrBLUP with Ridge
    regression on the raw marker matrix produces the same predictions as
    GBLUP with a linear kernel. This scales O(n*m) instead of O(n^2) and
    works with large training sets where KernelRidge runs out of memory.
    """
    scaler = StandardScaler(with_mean=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = Ridge(alpha=alpha, solver="sag", max_iter=2000)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    return BaselineResult("gblup_efficient", pred, {"alpha": alpha})


def fit_random_forest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int = 1234, n_estimators: int = 300) -> BaselineResult:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2)
    model.fit(X_train, y_train)
    return BaselineResult("random_forest", model.predict(X_test), {"n_estimators": 300})


def mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "phenotype_value") -> BaselineResult:
    pred = np.full(len(test_df), train_df[target_col].mean())
    return BaselineResult("global_mean", pred, {})


def fit_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    seed: int = 1234, n_estimators: int = 300,
) -> BaselineResult:
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is not installed. Install with: pip install xgboost")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1,
        max_depth=6, learning_rate=0.05, subsample=0.8,
    )
    model.fit(X_train, y_train)
    return BaselineResult("xgboost", model.predict(X_test), {"n_estimators": n_estimators})


def fit_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    seed: int = 1234, n_estimators: int = 300,
) -> BaselineResult:
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is not installed. Install with: pip install lightgbm")
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1,
        max_depth=8, learning_rate=0.05, subsample=0.8, verbose=-1,
    )
    model.fit(X_train, y_train)
    return BaselineResult("lightgbm", model.predict(X_test), {"n_estimators": n_estimators})


class DeepGS(nn.Module):
    """Deep Genomic Selection model (Ma et al. 2018).

    Architecture: input markers -> wide hidden layers (BatchNorm + ReLU + Dropout) -> output.
    Uses high dropout (0.5) and wide hidden dims (256) as described in the paper.
    """

    def __init__(self, n_markers: int, hidden_dims: tuple = (256, 128), dropout: float = 0.5):
        super().__init__()
        layers = []
        prev_dim = n_markers
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_deepgs(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    hidden_dims: tuple = (256, 128),
    dropout: float = 0.5,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 1234,
) -> BaselineResult:
    """Train a DeepGS model following Ma et al. 2018."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = DeepGS(n_markers=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    model.train()
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        scheduler.step(avg_loss)

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 40:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()

    return BaselineResult("deepgs", preds, {"epochs": epoch + 1, "hidden_dims": hidden_dims})


def fit_rkhs(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    gamma: float | None = None, alpha: float = 1.0,
) -> BaselineResult:
    """RKHS (Reproducing Kernel Hilbert Space) regression via RBF-kernel Ridge.

    This is the standard genomic prediction kernel method equivalent to
    GBLUP with a Gaussian kernel on markers (de los Campos et al., 2013;
    Morota & Gianola, 2014). The RBF kernel captures nonlinear marker
    relationships that the linear GBLUP kernel misses.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    scaler = StandardScaler(with_mean=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Default gamma: 1 / n_features (scikit-learn default)
    if gamma is None:
        gamma = 1.0 / max(1, X_train_s.shape[1])

    K_train = rbf_kernel(X_train_s, X_train_s, gamma=gamma)
    K_test = rbf_kernel(X_test_s, X_train_s, gamma=gamma)

    model = KernelRidge(alpha=alpha, kernel="precomputed")
    model.fit(K_train, y_train)
    pred = model.predict(K_test)
    return BaselineResult("rkhs", pred, {"gamma": gamma, "alpha": alpha})


def fit_rkhs_fast(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    gamma: float | None = None, alpha: float = 1.0,
) -> BaselineResult:
    """Fast RKHS via Nystroem approximation for large training sets.

    Uses the Nystroem method to approximate the RBF kernel with a
    low-rank decomposition, avoiding the O(n²) kernel matrix.
    Suitable when n_train > 20,000.
    """
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge

    scaler = StandardScaler(with_mean=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if gamma is None:
        gamma = 1.0 / max(1, X_train_s.shape[1])

    n_components = min(500, X_train_s.shape[0] // 2)
    nystroem = Nystroem(kernel="rbf", gamma=gamma, n_components=n_components, random_state=42)
    X_train_k = nystroem.fit_transform(X_train_s)
    X_test_k = nystroem.transform(X_test_s)

    model = Ridge(alpha=alpha)
    model.fit(X_train_k, y_train)
    pred = model.predict(X_test_k)
    return BaselineResult("rkhs_fast", pred, {"gamma": gamma, "alpha": alpha, "n_components": n_components})


# ---------------------------------------------------------------------------
# DNNGP — Deep Neural Network for Genomic Prediction
# Wang et al. (2023) "DNNGP: a deep neural network-based method for genomic
# prediction using high-dimensional SNP data"
# Key architecture: deep MLP with batch norm + residual skip connections
# ---------------------------------------------------------------------------


class DNNGP(nn.Module):
    """Deep Neural Network for Genomic Prediction (Wang et al. 2023).

    Architecture:
    - Input projection: n_markers → hidden_dim (with BN + GELU + Dropout)
    - Residual blocks × n_blocks: each is two Linear(hidden_dim, hidden_dim)
      layers with BN, GELU, Dropout, and a skip connection
    - Output head: hidden_dim → 1

    The constant-width residual design matches the DNNGP paper's approach
    of deep feature extraction with identity-preserving skip connections.
    """

    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_markers, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)  # standard residual skip connection
        return self.head(h).squeeze(-1)


def fit_dnngp(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    n_blocks: int = 3,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 1234,
) -> BaselineResult:
    """Train DNNGP following Wang et al. 2023.

    Includes NaN detection, progress logging, and early stopping to prevent
    the silent hangs observed in previous runs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    n_train = X_train_s.shape[0]
    n_markers = X_train_s.shape[1]
    print(f"  [DNNGP] n_train={n_train} n_markers={n_markers} hidden_dim={hidden_dim} "
          f"n_blocks={n_blocks} epochs={epochs} batch_size={batch_size} device={device}",
          flush=True)

    model = DNNGP(
        n_markers=n_markers,
        hidden_dim=hidden_dim,
        dropout=dropout,
        n_blocks=n_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [DNNGP] params={n_params:,}", flush=True)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    n_batches = len(train_loader)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    early_stop_patience = 25
    nan_streak = 0

    for epoch in range(epochs):
        total_loss = 0.0
        nan_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)

            # NaN detection — abort if loss is consistently NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                nan_streak += 1
                if nan_streak >= 3:
                    raise RuntimeError(
                        f"  [DNNGP] ABORT: loss is NaN for {nan_streak} consecutive batches "
                        f"at epoch {epoch+1}. Check learning rate, data scaling, or model architecture."
                    )
                continue  # skip this batch, optimizer step not called

            nan_streak = 0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if nan_batches > 0:
            print(f"  [DNNGP] WARNING epoch {epoch+1}: {nan_batches}/{n_batches} batches had NaN loss",
                  flush=True)

        avg_loss = total_loss / max(1, n_batches - nan_batches)
        scheduler.step()

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Progress logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [DNNGP] epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}  "
                  f"best={best_loss:.6f}  lr={lr_now:.2e}  patience={patience_counter}/{early_stop_patience}",
                  flush=True)

        if patience_counter >= early_stop_patience:
            print(f"  [DNNGP] early stop at epoch {epoch+1} (best_loss={best_loss:.6f})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print(f"  [DNNGP] WARNING: no best state saved (all epochs had NaN?)", flush=True)

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()

    # Final sanity check
    if np.any(np.isnan(preds)):
        print(f"  [DNNGP] WARNING: {np.isnan(preds).sum()} NaN predictions — replacing with train mean",
              flush=True)
        preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))

    return BaselineResult("dnngp", preds, {
        "epochs": epoch + 1,
        "hidden_dim": hidden_dim,
        "n_blocks": n_blocks,
    })


# ---------------------------------------------------------------------------
# Reaction Norm GBLUP — GBLUP extended with genotype × environment interaction
# Jarquín et al. (2014) "A reaction norm model for genomic selection using
# high-dimensional genomic and environmental data"
# ---------------------------------------------------------------------------


def fit_reaction_norm_gblup(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    env_index_train: np.ndarray | None = None,
    env_index_test: np.ndarray | None = None,
    alpha: float = 1.0,
) -> BaselineResult:
    """Reaction Norm GBLUP with environmental index (Jarquín et al., 2014).

    Extends GBLUP by adding a genotype × environment interaction term:
        y = G_main + G×E_interaction + ε

    The G×E covariance is modelled as:
        Z_g G Z_g' ⊙ Z_e Z_e'
    where G is the genomic relationship matrix (linear kernel on markers),
    Z_g is the genotype design matrix, and Z_e encodes the environmental index.

    For the simplified single-environment-index version implemented here:
    1. Compute G = XX' / p (genomic relationship from markers)
    2. Compute G×E = G ⊙ (e e') where e is the environmental index
    3. Predict with a kernel that combines both terms

    If env_index is not provided, it is estimated as the mean phenotype
    per environment from the training data.
    """
    from sklearn.metrics.pairwise import linear_kernel

    scaler = StandardScaler(with_mean=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    p = X_train_s.shape[1]

    # Genomic relationship matrix
    G_train = linear_kernel(X_train_s, X_train_s) / p
    G_test = linear_kernel(X_test_s, X_train_s) / p

    # Environmental index: if not provided, use a unit vector (no env info)
    # This falls back to standard GBLUP
    if env_index_train is None:
        K_train = G_train
        K_test = G_test
    else:
        e_train = np.asarray(env_index_train, dtype=float).ravel()
        e_test = np.asarray(env_index_test, dtype=float).ravel()

        # Standardize the environmental index
        e_mean = np.mean(e_train)
        e_std = np.std(e_train) or 1.0
        e_train_s = (e_train - e_mean) / e_std
        e_test_s = (e_test - e_mean) / e_std

        # Reaction norm kernel: G_main + G ⊙ (ee')
        # K_ij = G_ij + G_ij * e_i * e_j = G_ij * (1 + e_i * e_j)
        K_train = G_train * (1.0 + np.outer(e_train_s, e_train_s))
        K_test = G_test * (1.0 + np.outer(e_test_s, e_train_s))

    model = KernelRidge(alpha=alpha, kernel="precomputed")
    model.fit(K_train, y_train)
    pred = model.predict(K_test)

    has_env = env_index_train is not None
    return BaselineResult(
        "reaction_norm_gblup",
        pred,
        {"alpha": alpha, "has_environmental_index": has_env},
    )


def fit_reaction_norm_gblup_efficient(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    env_index_train: np.ndarray | None = None,
    env_index_test: np.ndarray | None = None,
    alpha: float = 1.0,
) -> BaselineResult:
    """Memory-efficient Reaction Norm GBLUP via Ridge on augmented markers.

    Instead of forming the n×n kernel matrix, this augments the marker
    matrix with marker × environmental_index interaction terms and fits
    Ridge regression directly. Mathematically equivalent to the kernel
    formulation when env_index exists, and equivalent to rrBLUP otherwise.
    Scales O(n×p) instead of O(n²), suitable for large training sets.
    """
    scaler = StandardScaler(with_mean=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if env_index_train is not None:
        e_train = np.asarray(env_index_train, dtype=float).ravel()
        e_test = np.asarray(env_index_test, dtype=float).ravel()
        e_mean = np.mean(e_train)
        e_std = np.std(e_train) or 1.0
        e_train_s = (e_train - e_mean) / e_std
        e_test_s = (e_test - e_mean) / e_std

        # Augment markers with marker × env_index
        X_train_aug = np.column_stack([
            X_train_s,
            X_train_s * e_train_s.reshape(-1, 1),
        ])
        X_test_aug = np.column_stack([
            X_test_s,
            X_test_s * e_test_s.reshape(-1, 1),
        ])
    else:
        X_train_aug = X_train_s
        X_test_aug = X_test_s

    model = Ridge(alpha=alpha, solver="cholesky")
    model.fit(X_train_aug, y_train)
    pred = model.predict(X_test_aug)

    has_env = env_index_train is not None
    return BaselineResult(
        "reaction_norm_gblup_efficient",
        pred,
        {"alpha": alpha, "has_environmental_index": has_env},
    )
