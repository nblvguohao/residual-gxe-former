from __future__ import annotations

import math
import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for weather time steps.

    Encodes day-of-growing-season so the model can distinguish early-season
    from late-season weather effects without consuming a feature dimension.
    """

    def __init__(self, d_model: int, max_len: int = 365):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, :x.shape[1], :]


class AttentionPooling(nn.Module):
    """Learned attention pooling over time steps.

    Computes attention weights for each time step and returns a weighted sum,
    giving the model the ability to focus on critical periods (e.g. flowering).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H] -> [B, H]
        scores = self.scorer(x).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)  # [B, H]


class GenotypeEncoder(nn.Module):
    def __init__(self, n_markers: int, patch_size: int = 64, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.n_markers = n_markers
        self.patch_size = patch_size
        self.n_patches = (n_markers + patch_size - 1) // patch_size
        self.patch_proj = nn.Linear(patch_size, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.transformer_batch_first = True
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
        except TypeError:
            self.transformer_batch_first = False
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M]
        b, m = x.shape
        pad = self.n_patches * self.patch_size - m
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        x = x.view(b, self.n_patches, self.patch_size)
        h = self.patch_proj(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        h = self.dropout(h)
        if self.transformer_batch_first:
            h = self.transformer(h)
        else:
            h = self.transformer(h.transpose(0, 1)).transpose(0, 1)
        return h.mean(dim=1)


class WeatherEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15, use_tcn: bool = False):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        self.use_tcn = use_tcn
        if use_tcn:
            self.tcn = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.tcn_norm = nn.LayerNorm(hidden_dim)
        else:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        # weather: [B, T, F]
        h = self.input_proj(weather)
        h = self.pos_encoding(h)
        if self.use_tcn:
            h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
            h = self.tcn_norm(h)
        else:
            h, _ = self.gru(h)
            h = self.out_proj(h)
        h = self.norm(self.dropout(h))
        # Return both token sequence for fusion AND pooled vector
        return h  # [B, T, H]


class StaticEnvEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_heads: int = 4, dropout: float = 0.15, use_film: bool = True):
        super().__init__()
        self.attn_batch_first = True
        try:
            self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        except TypeError:
            self.attn_batch_first = False
            self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout)
        self.use_film = use_film
        if use_film:
            self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, genotype_emb: torch.Tensor, weather_tokens: torch.Tensor, static_env_emb: torch.Tensor) -> torch.Tensor:
        # genotype_emb/static_env_emb: [B, H], weather_tokens: [B, T, H]
        query = genotype_emb.unsqueeze(1)
        key_value = weather_tokens + static_env_emb.unsqueeze(1)
        if self.attn_batch_first:
            attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
            fused = attn_out.squeeze(1)
        else:
            attn_out, _ = self.attn(
                query.transpose(0, 1),
                key_value.transpose(0, 1),
                key_value.transpose(0, 1),
                need_weights=False,
            )
            fused = attn_out.transpose(0, 1).squeeze(1)
        if self.use_film:
            gamma_beta = self.film(static_env_emb)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            fused = fused + (1.0 + torch.tanh(gamma)) * genotype_emb + beta
        return self.norm(fused)


# ---------------------------------------------------------------------------
# Ablation model variants — single-modality and partial-fusion architectures
# ---------------------------------------------------------------------------


class GenotypeOnlyModel(nn.Module):
    """Ablation: genotype markers → prediction (no environment or weather)."""

    def __init__(self, n_markers: int, hidden_dim: int = 128, patch_size: int = 64, dropout: float = 0.15):
        super().__init__()
        self.genotype_encoder = GenotypeEncoder(n_markers, patch_size=patch_size, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, markers: torch.Tensor, weather: torch.Tensor, static_env: torch.Tensor) -> torch.Tensor:
        g = self.genotype_encoder(markers)
        return self.head(g).squeeze(-1)


class WeatherOnlyModel(nn.Module):
    """Ablation: weather sequence → prediction (no genotype)."""

    def __init__(self, weather_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.weather_encoder = WeatherEncoder(weather_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, markers: torch.Tensor, weather: torch.Tensor, static_env: torch.Tensor) -> torch.Tensor:
        w = self.weather_encoder(weather)  # [B, T, H]
        w_pooled = self.attn_pool(w)  # [B, H]
        return self.head(w_pooled).squeeze(-1)


class StaticEnvOnlyModel(nn.Module):
    """Ablation: static environment features → prediction (no genotype or weather)."""

    def __init__(self, static_env_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.env_encoder = StaticEnvEncoder(static_env_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, markers: torch.Tensor, weather: torch.Tensor, static_env: torch.Tensor) -> torch.Tensor:
        e = self.env_encoder(static_env)
        return self.head(e).squeeze(-1)


class GenoEnvModel(nn.Module):
    """Ablation: genotype + static env (no weather). Simple concat fusion.

    This tests the marginal value of daily weather sequence encoding
    vs. just using genotype + static environment features.
    """

    def __init__(self, n_markers: int, static_env_dim: int, hidden_dim: int = 128, patch_size: int = 64, dropout: float = 0.15):
        super().__init__()
        self.genotype_encoder = GenotypeEncoder(n_markers, patch_size=patch_size, hidden_dim=hidden_dim, dropout=dropout)
        self.env_encoder = StaticEnvEncoder(static_env_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, markers: torch.Tensor, weather: torch.Tensor, static_env: torch.Tensor) -> torch.Tensor:
        g = self.genotype_encoder(markers)
        e = self.env_encoder(static_env)
        fused = self.norm(self.fusion(torch.cat([g, e], dim=-1)))
        return self.head(fused).squeeze(-1)


# ---------------------------------------------------------------------------
# L4 Transformer Baseline — Simplified Cropformer
# Based on: Cropformer uses multi-scale 1D convolutions on SNP markers
# followed by transformer blocks to capture local LD patterns.
# ---------------------------------------------------------------------------


class CropformerSimple(nn.Module):
    """Simplified Cropformer baseline for genomic prediction.

    Architecture (FIXED — patch-first design avoids O(n²) on 2425 markers):
    1. Patch markers (patch_size=64 → ~38 patches) and linear projection
    2. Multi-scale 1D conv on the 38-patch sequence (kernel sizes 3, 5)
    3. Transformer encoder (2 layers) on 38 positions
    4. Mean pooling + MLP head

    Environment features are concatenated at the final representation level.
    """

    def __init__(self, n_markers: int, env_dim: int = 0, hidden_dim: int = 128,
                 patch_size: int = 64, dropout: float = 0.15):
        super().__init__()
        self.n_markers = n_markers
        self.patch_size = patch_size
        self.n_patches = (n_markers + patch_size - 1) // patch_size

        # Patch embedding (same as GenotypeEncoder)
        self.patch_proj = nn.Linear(patch_size, hidden_dim)

        # Multi-scale conv on patch sequence (~38 positions)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder on ~38 positions (fast!)
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.transformer_batch_first = True
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu",
            )
            self.transformer_batch_first = False
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Head
        fusion_dim = hidden_dim + env_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, markers: torch.Tensor, weather: torch.Tensor, static_env: torch.Tensor) -> torch.Tensor:
        # markers: [B, M] → [B, P, patch_size] → [B, P, H]
        b, m = markers.shape
        pad = self.n_patches * self.patch_size - m
        if pad > 0:
            markers = torch.nn.functional.pad(markers, (0, pad))
        x = markers.view(b, self.n_patches, self.patch_size)
        h = self.patch_proj(x)  # [B, P, H]

        # Multi-scale conv on patch dimension
        h_t = h.transpose(1, 2)  # [B, H, P]
        c3 = self.conv3(h_t).transpose(1, 2)  # [B, P, H/2]
        c5 = self.conv5(h_t).transpose(1, 2)  # [B, P, H/2]
        h = torch.cat([c3, c5], dim=-1)  # [B, P, H]
        h = self.conv_norm(h)
        h = self.dropout(h)

        if self.transformer_batch_first:
            h = self.transformer(h)
        else:
            h = self.transformer(h.transpose(0, 1)).transpose(0, 1)
        h = h.mean(dim=1)  # [B, H]

        # Concatenate environment features
        if static_env.size(-1) > 0:
            h = torch.cat([h, static_env], dim=-1)
        return self.head(h).squeeze(-1)


class ResidualGxEFormer(nn.Module):
    def __init__(
        self,
        n_markers: int,
        weather_dim: int,
        static_env_dim: int,
        hidden_dim: int = 128,
        patch_size: int = 64,
        dropout: float = 0.15,
        multi_task: bool = False,
        gated_residual: bool = False,
        use_film: bool = True,
    ):
        super().__init__()
        self.multi_task = multi_task
        self.gated_residual = gated_residual
        self.genotype_encoder = GenotypeEncoder(n_markers, patch_size=patch_size, hidden_dim=hidden_dim, dropout=dropout)
        self.weather_encoder = WeatherEncoder(weather_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.static_env_encoder = StaticEnvEncoder(static_env_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.fusion = CrossAttentionFusion(hidden_dim=hidden_dim, dropout=dropout, use_film=use_film)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if gated_residual:
            self.gate_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        if multi_task:
            self.phenotype_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, markers: torch.Tensor, weather: torch.Tensor, static_env: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        g = self.genotype_encoder(markers)
        w = self.weather_encoder(weather)
        e = self.static_env_encoder(static_env)
        fused = self.fusion(g, w, e)
        residual = self.head(fused).squeeze(-1)
        if self.gated_residual:
            residual = residual * self.gate_head(fused).squeeze(-1)
        if self.multi_task:
            phenotype = self.phenotype_head(fused).squeeze(-1)
            return residual, phenotype
        return residual
