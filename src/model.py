from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class MultimodalFusionClassifier(nn.Module):
    """Fuse image, text, and tabular embeddings with optional modality masks."""

    def __init__(
        self,
        image_dim: int = 256,
        text_dim: int = 256,
        tabular_dim: int = 16,
        hidden: int = 64,
        classes: int = 2,
    ) -> None:
        super().__init__()
        self.image_encoder = MLPEncoder(image_dim, hidden)
        self.text_encoder = MLPEncoder(text_dim, hidden)
        self.tabular_encoder = MLPEncoder(tabular_dim, hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, classes))

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        tabular: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = torch.stack(
            (
                self.image_encoder(image),
                self.text_encoder(text),
                self.tabular_encoder(tabular),
            ),
            dim=1,
        )
        if modality_mask is not None:
            encoded = encoded * modality_mask.to(encoded.dtype).unsqueeze(-1)
        return self.head(encoded.flatten(1))
