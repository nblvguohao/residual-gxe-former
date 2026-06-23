from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from residual_gxe.data.preprocess import select_markers_by_strategy
from residual_gxe.models.deep import ResidualGxEFormer
from residual_gxe.training.losses import (
    PairwiseRankLoss,
    multi_task_loss,
    residual_gxe_loss,
    weighted_huber_loss,
)


def resolve_torch_device(device: str = "auto", require_cuda: bool = False) -> torch.device:
    """Resolve a requested torch device and fail loudly when CUDA is requested but unavailable."""
    requested = str(device or "auto")
    resolved = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if requested == "auto" and not torch.cuda.is_available():
        resolved = "cpu"

    if resolved.startswith("cuda") and not torch.cuda.is_available():
        details = {
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "requested_device": requested,
        }
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Install a PyTorch CUDA build compatible with the server NVIDIA driver, "
            f"or use --device cpu. Details: {details}"
        )
    if require_cuda and not resolved.startswith("cuda"):
        raise RuntimeError(f"CUDA is required but resolved device is {resolved!r}.")
    return torch.device(resolved)


def torch_device_report(device: str = "auto") -> dict[str, Any]:
    """Return torch/CUDA diagnostics for run manifests and server preflight checks."""
    report: dict[str, Any] = {
        "requested_device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current)
        report.update(
            {
                "resolved_device": f"cuda:{current}",
                "device_name": torch.cuda.get_device_name(current),
                "device_capability": ".".join(map(str, torch.cuda.get_device_capability(current))),
                "total_memory_gb": round(props.total_memory / 1024**3, 3),
            }
        )
    else:
        report["resolved_device"] = "cpu"
    return report


class MultiModalDataset(Dataset):
    """Dataset for genotype + weather + static environment + target."""

    def __init__(
        self,
        markers: np.ndarray,
        weather: np.ndarray | None,
        static_env: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ):
        self.markers = torch.as_tensor(markers, dtype=torch.float32)
        self.weather = torch.as_tensor(weather, dtype=torch.float32) if weather is not None else None
        self.static_env = torch.as_tensor(static_env, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32) if sample_weights is not None else None

    def __len__(self):
        return len(self.markers)

    def __getitem__(self, idx):
        if self.weather is not None:
            data = (self.markers[idx], self.weather[idx], self.static_env[idx], self.targets[idx])
        else:
            data = (self.markers[idx], torch.zeros(1, 6), self.static_env[idx], self.targets[idx])
        if self.sample_weights is not None:
            data = data + (self.sample_weights[idx],)
        return data


WEATHER_SEQ_LEN = 30
WEATHER_FEAT_DIM = 6


def _build_feature_arrays(
    pheno: pd.DataFrame,
    geno_wide: pd.DataFrame,
    env_feats: pd.DataFrame | None,
    weather_data: pd.DataFrame | None,
    max_markers: int = 5000,
    weather_seq_len: int = WEATHER_SEQ_LEN,
    weather_feat_dim: int = WEATHER_FEAT_DIM,
    marker_strategy: str = "random",
    main_effects: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature arrays from dataframes. Always returns array (never None) for each modality."""
    n = len(pheno)

    # Genotype
    marker_cols = [c for c in geno_wide.columns if c != "genotype_id"]
    if len(marker_cols) > max_markers:
        marker_cols = select_markers_by_strategy(marker_cols, max_markers, strategy=marker_strategy, geno_wide=geno_wide)
    geno_indexed = pheno[["genotype_id"]].merge(
        geno_wide[["genotype_id"] + marker_cols], on="genotype_id", how="left"
    )
    X_geno = geno_indexed[marker_cols].fillna(0.0).to_numpy(dtype=np.float32)

    # Weather: always produce [n, weather_seq_len, weather_feat_dim]
    if weather_data is not None and len(weather_data) > 0:
        w_cols_available = [c for c in ["tmax", "tmin", "tmean", "precipitation", "solar_radiation", "relative_humidity"] if c in weather_data.columns]
        if w_cols_available:
            weather_feats = []
            for _, row in pheno.iterrows():
                env_id = row["environment_id"]
                env_w = weather_data[weather_data["environment_id"] == env_id][w_cols_available].fillna(0).to_numpy(dtype=np.float32)
                actual_len = min(len(env_w), weather_seq_len)
                seq = np.zeros((weather_seq_len, weather_feat_dim), dtype=np.float32)
                if actual_len > 0:
                    env_w = env_w[:actual_len, :weather_feat_dim]
                    seq[:actual_len, :env_w.shape[1]] = env_w
                weather_feats.append(seq)
            X_weather = np.stack(weather_feats, axis=0).astype(np.float32)
        else:
            X_weather = np.zeros((n, weather_seq_len, weather_feat_dim), dtype=np.float32)
    else:
        X_weather = np.zeros((n, weather_seq_len, weather_feat_dim), dtype=np.float32)

    # Static env
    static_dim = 4
    if env_feats is not None and len(env_feats) > 0:
        env_cols = [c for c in env_feats.columns if c != "environment_id" and env_feats[c].dtype in ("float64", "int64", "int32", "float32", "bool")]
        env_cols = env_cols[:20]
        if env_cols:
            env_indexed = pheno[["environment_id"]].merge(
                env_feats[["environment_id"] + env_cols], on="environment_id", how="left"
            )
            X_env = env_indexed[env_cols].fillna(0).to_numpy(dtype=np.float32)
            static_dim = X_env.shape[1]
        else:
            X_env = np.zeros((n, static_dim), dtype=np.float32)
    else:
        X_env = np.zeros((n, static_dim), dtype=np.float32)

    # Augment with explicit main effects if provided (genotype_effect + environment_effect)
    if main_effects is not None:
        me = np.asarray(main_effects, dtype=np.float32).reshape(-1, 1)
        X_env = np.column_stack([X_env, me])

    return X_geno, X_weather, X_env


def train_model(
    model: nn.Module,
    train_dataset: MultiModalDataset,
    val_dataset: MultiModalDataset | None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 15,
    gradient_clip_norm: float = 1.0,
    rank_weight: float = 0.05,
    device: str = "auto",
    phenotype_weight: float = 0.5,
    num_workers: int = 0,
    log_every: int = 20,
    amp: bool = False,
) -> dict[str, Any]:
    """Train a ResidualGxEFormer model and return training history."""
    device_obj = resolve_torch_device(device)
    model = model.to(device_obj)
    use_amp = bool(amp and device_obj.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pin_memory = device_obj.type == "cuda"
    print(f"  training_device={device_obj} amp={use_amp} num_workers={num_workers}", flush=True)

    use_multi_task = getattr(model, "multi_task", False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            has_weights = len(batch) == 5
            if has_weights:
                markers, weather, static_env, targets, sw = batch
                sw = sw.to(device_obj, non_blocking=pin_memory)
            else:
                markers, weather, static_env, targets = batch
                sw = None
            markers = markers.to(device_obj, non_blocking=pin_memory)
            weather = weather.to(device_obj, non_blocking=pin_memory)
            static_env = static_env.to(device_obj, non_blocking=pin_memory)
            targets = targets.to(device_obj, non_blocking=pin_memory)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_multi_task:
                    y_true_pheno = targets[:, 0]
                    y_true_resid = targets[:, 1]
                    y_resid, y_pheno = model(markers, weather, static_env)
                    loss = multi_task_loss(
                        y_resid, y_true_resid, y_pheno, y_true_pheno,
                        rank_weight=rank_weight, phenotype_weight=phenotype_weight,
                        sample_weights=sw,
                    )
                else:
                    preds = model(markers, weather, static_env)
                    if sw is not None:
                        loss = weighted_huber_loss(preds, targets, sw)
                        if rank_weight > 0:
                            loss = loss + rank_weight * PairwiseRankLoss()(preds, targets)
                    else:
                        loss = residual_gxe_loss(preds, targets, rank_weight=rank_weight)
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    n_items = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else batch[0].size(0)
                    if len(batch) == 5:
                        markers, weather, static_env, targets, _sw = batch
                    else:
                        markers, weather, static_env, targets = batch
                    markers = markers.to(device_obj, non_blocking=pin_memory)
                    weather = weather.to(device_obj, non_blocking=pin_memory)
                    static_env = static_env.to(device_obj, non_blocking=pin_memory)
                    targets = targets.to(device_obj, non_blocking=pin_memory)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        if use_multi_task:
                            y_true_pheno = targets[:, 0]
                            y_true_resid = targets[:, 1]
                            y_resid, y_pheno = model(markers, weather, static_env)
                            val_total += nn.functional.mse_loss(y_resid, y_true_resid).item() * len(markers)
                        else:
                            preds = model(markers, weather, static_env)
                            val_total += nn.functional.mse_loss(preds, targets).item() * len(markers)
                    val_n += len(markers)
            val_loss = val_total / max(1, val_n)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch - best_epoch >= early_stopping_patience:
                break

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if log_every > 0 and epoch % log_every == 0:
            vloss_str = f"{val_loss:.6f}" if val_loss is not None else "NA"
            print(f"  epoch {epoch:3d}: train_loss={train_loss:.6f} val_loss={vloss_str}", flush=True)

    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}


@torch.no_grad()
def predict(model: nn.Module, dataset: MultiModalDataset, batch_size: int = 128, device: str = "auto", return_phenotype: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    device_obj = resolve_torch_device(device)
    model = model.to(device_obj)
    model.eval()
    pin_memory = device_obj.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    preds = []
    pheno_preds = [] if return_phenotype else None
    for markers, weather, static_env, _targets in loader:
        markers = markers.to(device_obj, non_blocking=pin_memory)
        weather = weather.to(device_obj, non_blocking=pin_memory)
        static_env = static_env.to(device_obj, non_blocking=pin_memory)
        out = model(markers, weather, static_env)
        if return_phenotype and isinstance(out, tuple):
            resid, pheno = out
            preds.append(resid.cpu().numpy())
            pheno_preds.append(pheno.cpu().numpy())
        else:
            preds.append(out.cpu().numpy())
    if return_phenotype and pheno_preds is not None:
        return np.concatenate(preds, axis=0), np.concatenate(pheno_preds, axis=0)
    return np.concatenate(preds, axis=0)


def predict_with_uncertainty(
    model: nn.Module,
    dataset: MultiModalDataset,
    n_samples: int = 30,
    batch_size: int = 128,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """MC Dropout uncertainty estimation.

    Runs n_samples forward passes with dropout active (model.train()).
    Returns (mean, std) of the predictions.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.train()  # keep dropout active
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_samples = []
    for _ in range(n_samples):
        batch_preds = []
        for markers, weather, static_env, _targets in loader:
            markers = markers.to(device_obj)
            weather = weather.to(device_obj)
            static_env = static_env.to(device_obj)
            with torch.no_grad():
                out = model(markers, weather, static_env)
            batch_preds.append(out.cpu().numpy())
        all_samples.append(np.concatenate(batch_preds, axis=0))

    samples = np.stack(all_samples, axis=0)  # [n_samples, N]
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return mean, std


def train_ensemble(
    model_cls: type,
    model_kwargs: dict,
    train_dataset: MultiModalDataset,
    val_dataset: MultiModalDataset | None,
    n_models: int = 3,
    **train_kwargs,
) -> list[nn.Module]:
    """Train a small ensemble with different random seeds.

    Returns a list of trained model instances.
    """
    models = []
    for i in range(n_models):
        torch.manual_seed(i * 42 + 1234)
        np.random.seed(i * 42 + 1234)
        m = model_cls(**model_kwargs)
        result = train_model(m, train_dataset, val_dataset, **train_kwargs)
        models.append(m)
        print(f"  Ensemble model {i+1}/{n_models}: best_val_loss={result['best_val_loss']:.4f}")
    return models


def predict_ensemble(
    models: list[nn.Module],
    dataset: MultiModalDataset,
    batch_size: int = 128,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with an ensemble, returning mean and std across models."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    all_preds = []
    for model in models:
        model = model.to(device_obj)
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for markers, weather, static_env, _targets in loader:
                markers = markers.to(device_obj)
                weather = weather.to(device_obj)
                static_env = static_env.to(device_obj)
                out = model(markers, weather, static_env)
                preds.append(out.cpu().numpy())
        all_preds.append(np.concatenate(preds, axis=0))
    stacked = np.stack(all_preds, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)
