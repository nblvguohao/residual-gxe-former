from __future__ import annotations

import torch

from residual_gxe.models.deep import ResidualGxEFormer


def test_residual_gxe_former_forward_and_backward_smoke_cpu():
    torch.manual_seed(1234)
    batch_size = 3
    n_markers = 10
    weather_steps = 5
    weather_dim = 4
    static_env_dim = 2
    model = ResidualGxEFormer(
        n_markers=n_markers,
        weather_dim=weather_dim,
        static_env_dim=static_env_dim,
        hidden_dim=16,
        patch_size=4,
        dropout=0.0,
    )

    markers = torch.randn(batch_size, n_markers)
    weather = torch.randn(batch_size, weather_steps, weather_dim)
    static_env = torch.randn(batch_size, static_env_dim)
    target = torch.randn(batch_size)

    output = model(markers, weather, static_env)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    assert output.shape == (batch_size,)
    assert torch.isfinite(output).all()
    assert all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters())


def test_residual_gxe_former_multitask_gated_forward_backward_cpu():
    torch.manual_seed(1234)
    model = ResidualGxEFormer(
        n_markers=12,
        weather_dim=6,
        static_env_dim=3,
        hidden_dim=16,
        patch_size=4,
        dropout=0.0,
        multi_task=True,
        gated_residual=True,
    )

    markers = torch.randn(4, 12)
    weather = torch.randn(4, 5, 6)
    static_env = torch.randn(4, 3)
    target_residual = torch.randn(4)
    target_phenotype = torch.randn(4)

    residual, phenotype = model(markers, weather, static_env)
    loss = torch.nn.functional.mse_loss(residual, target_residual)
    loss = loss + torch.nn.functional.mse_loss(phenotype, target_phenotype)
    loss.backward()

    assert residual.shape == (4,)
    assert phenotype.shape == (4,)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(phenotype).all()
