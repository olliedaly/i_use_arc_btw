"""Classification heads, used only during training.

Each head maps a batch of embeddings (and their labels) to logits, which then go
into the same F.cross_entropy. The heads differ only in how the logits are
computed - the softmax / cross-entropy part is identical.
"""

import torch
from torch import nn


class SoftmaxHead(nn.Module):
    """Baseline: a plain linear classifier, logit_j = W_j . x"""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        # One weight row per identity. No bias: a per-class constant doesn't depend
        # on the face, and dropping it makes logit_j exactly |W_j| |x| cos(theta_j).
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # labels are unused here; ArcFace needs them to know which logit gets the margin.
        return self.fc(emb)
