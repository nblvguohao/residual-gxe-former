from __future__ import annotations

import torch
from torch import nn


class PairwiseRankLoss(nn.Module):
    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.numel() < 2:
            return y_pred.new_tensor(0.0)
        diff_true = y_true.unsqueeze(0) - y_true.unsqueeze(1)
        diff_pred = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
        mask = diff_true > 0
        if not torch.any(mask):
            return y_pred.new_tensor(0.0)
        losses = torch.relu(self.margin - diff_pred[mask])
        return losses.mean()


def residual_gxe_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    huber_delta: float = 1.0,
    rank_weight: float = 0.0,
) -> torch.Tensor:
    huber = nn.functional.huber_loss(y_pred, y_true, delta=huber_delta)
    if rank_weight <= 0:
        return huber
    rank = PairwiseRankLoss()(y_pred, y_true)
    return huber + rank_weight * rank


def multi_task_loss(
    y_residual: torch.Tensor,
    y_true_residual: torch.Tensor,
    y_phenotype: torch.Tensor,
    y_true_phenotype: torch.Tensor,
    huber_delta: float = 1.0,
    rank_weight: float = 0.0,
    phenotype_weight: float = 0.5,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined loss for dual-head model.

    residual_loss: Huber on residual target (main task)
    phenotype_loss: MSE on direct phenotype prediction (auxiliary)
    rank_loss: pairwise ranking loss on residual (ordering signal)
    sample_weights: optional per-sample weight tensor
    """
    if sample_weights is not None:
        residual_loss = weighted_huber_loss(y_residual, y_true_residual, sample_weights, delta=huber_delta)
        phenotype_loss = weighted_mse_loss(y_phenotype, y_true_phenotype, sample_weights)
    else:
        residual_loss = nn.functional.huber_loss(y_residual, y_true_residual, delta=huber_delta)
        phenotype_loss = nn.functional.mse_loss(y_phenotype, y_true_phenotype)
    total = residual_loss + phenotype_weight * phenotype_loss
    if rank_weight > 0:
        rank = PairwiseRankLoss()(y_residual, y_true_residual)
        total = total + rank_weight * rank
    return total


def weighted_huber_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor,
    weights: torch.Tensor, delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss with per-sample weights."""
    diff = y_pred - y_true
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    return (huber * weights).mean()


def weighted_mse_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """MSE loss with per-sample weights."""
    sq = (y_pred - y_true) ** 2
    return (sq * weights).mean()
