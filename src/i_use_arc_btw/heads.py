"""Classification heads, used only during training.

Each head maps a batch of embeddings (and their labels) to logits, which then go
into the same F.cross_entropy. The heads differ only in how the logits are
computed - the softmax / cross-entropy part is identical.

Call heads in fp32, outside autocast: ArcFace works with cosines near 1, and
bf16 can't tell an angle of 0 degrees from one of 5.
"""

import math

import torch
import torch.nn.functional as F
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


class ArcFaceHead(nn.Module):
    """ArcFace: logit_y = s * cos(theta_y + m) for the correct identity y,
    logit_j = s * cos(theta_j) for every other identity."""

    def __init__(self, embed_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        # Same layer as SoftmaxHead (same shape, same init) - only how it's used differs.
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)
        self.s = s
        self.m = m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1 + 2. Normalise embeddings and identity rows, so every logit is cos(theta_j).
        cos = F.linear(F.normalize(emb, dim=1), F.normalize(self.fc.weight, dim=1))

        # 3. Angle to the correct identity only. Clamp so acos never sees exactly
        #    +-1, where its gradient is infinite.
        cos_y = cos.gather(1, labels[:, None])
        theta_y = torch.acos(cos_y.clamp(-1 + 1e-7, 1 - 1e-7))

        # 4. Add the margin to that angle.
        target = torch.cos(theta_y + self.m)
        # Past theta = pi - m, cos(theta + m) turns back upwards, so a *worse* angle
        # would get a higher logit. There, fall back to a fixed subtractive margin.
        fallback = cos_y - self.m * math.sin(self.m)
        target = torch.where(theta_y + self.m < math.pi, target, fallback)

        # Put the penalised value back in the correct column, then scale.
        return self.s * cos.scatter(1, labels[:, None], target)
