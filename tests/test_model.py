import torch

from src.model import MultimodalFusionClassifier


def test_fusion_output_shape() -> None:
    model = MultimodalFusionClassifier(image_dim=8, text_dim=6, tabular_dim=4)
    logits = model(torch.randn(2, 8), torch.randn(2, 6), torch.randn(2, 4))
    assert logits.shape == (2, 2)
