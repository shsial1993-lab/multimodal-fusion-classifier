from __future__ import annotations

import torch

from .model import MultimodalFusionClassifier


def main() -> None:
    torch.manual_seed(7)
    model = MultimodalFusionClassifier(image_dim=32, text_dim=24, tabular_dim=8)
    image = torch.randn(4, 32)
    text = torch.randn(4, 24)
    tabular = torch.randn(4, 8)
    mask = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    logits = model(image, text, tabular, mask)
    print('logits:', tuple(logits.shape))


if __name__ == '__main__':
    main()
