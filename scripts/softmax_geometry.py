"""Step 3: what a softmax head's logits actually measure.

Run with:  uv run scripts/softmax_geometry.py
"""

import torch
import torch.nn.functional as F

from i_use_arc_btw.heads import SoftmaxHead

torch.manual_seed(0)
head = SoftmaxHead(embed_dim=384, num_classes=10)
x = torch.randn(4, 384)  # 4 fake embeddings
logits = head(x, labels=None)

# 1. Each logit is a dot product with one identity's weight row, and a dot
#    product splits into  length * length * cos(angle).
W = head.fc.weight  # (10, 384): one row per identity
cos = F.normalize(x, dim=1) @ F.normalize(W, dim=1).T
rebuilt = x.norm(dim=1, keepdim=True) * W.norm(dim=1) * cos
print(f"logits == |x| * |W_j| * cos(theta_j): {torch.allclose(logits, rebuilt, atol=1e-5)}")
print(f"|W_j| per identity: {[round(n, 2) for n in W.norm(dim=1).tolist()]}")

# 2. Pretend the head already classifies these 4 correctly. Scaling x changes
#    no angle (so no cosine similarity at test time) but still lowers the loss.
labels = logits.argmax(dim=1)
print()
for scale in (1, 2, 4, 8):
    loss = F.cross_entropy(head(scale * x, labels), labels)
    print(f"|x| x{scale}: loss {loss:.4f}")
