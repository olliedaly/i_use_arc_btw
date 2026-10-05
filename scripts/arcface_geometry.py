"""Step 4: what the ArcFace head changes compared to softmax.

Run with:  uv run scripts/arcface_geometry.py
"""

import math

import torch
import torch.nn.functional as F

from i_use_arc_btw.heads import ArcFaceHead

torch.manual_seed(0)
head = ArcFaceHead(embed_dim=384, num_classes=10)
x = torch.randn(4, 384)  # 4 fake embeddings
labels = torch.tensor([0, 1, 2, 3])

# 1. The length shortcut from softmax_geometry.py is gone: x is normalised
#    inside the head, so scaling it changes nothing.
for scale in (1, 2, 4, 8):
    loss = F.cross_entropy(head(scale * x, labels), labels)
    print(f"|x| x{scale}: loss {loss:.4f}")

# 2. The correct identity's logit (divided by s) as its angle theta grows.
#    Build embeddings at exact angles to identity 0's row, then ask the head.
with torch.no_grad():
    w0 = F.normalize(head.fc.weight[0], dim=0)
    r = torch.randn(384)
    u = F.normalize(r - (r @ w0) * w0, dim=0)  # unit vector at 90 degrees to w0
    degrees = torch.tensor([0, 30, 60, 90, 120, 150, 160, 170, 180.0])
    theta = torch.deg2rad(degrees)
    emb = torch.cos(theta)[:, None] * w0 + torch.sin(theta)[:, None] * u
    arcface = head(emb, torch.zeros(len(theta), dtype=torch.long))[:, 0] / head.s

print(f"\nmargin m = {head.m} rad = {math.degrees(head.m):.1f} degrees")
print(f"{'theta':>6} {'cos(theta)':>11} {'cos(theta+m)':>13} {'ArcFaceHead':>12}")
for d, t, a in zip(degrees, theta, arcface):
    print(f"{d:>5.0f}° {math.cos(t):>11.3f} {math.cos(t + head.m):>13.3f} {a:>12.3f}")
