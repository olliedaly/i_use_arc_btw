"""Step 5: softmax vs ArcFace on MNIST with a 2-D embedding, so we can plot it.

Same tiny network, data order and optimiser for both runs; only the head changes.

Run with:  uv run scripts/toy_2d.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import MNIST

from i_use_arc_btw.heads import ArcFaceHead, SoftmaxHead

DEVICE = "cuda"
EPOCHS = 30
RUNS = [
    ("Softmax", SoftmaxHead, {}),
    # The paper's settings, tuned for ~85k identities in 512-D. Here they collapse.
    ("ArcFace s=64 m=0.5", ArcFaceHead, {"s": 64.0, "m": 0.5}),
    ("ArcFace s=16 m=0.5", ArcFaceHead, {"s": 16.0, "m": 0.5}),
]


def load(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    # MNIST is tiny (60k 28x28 images), so keep it all on the GPU - no DataLoader.
    ds = MNIST("data", train=train, download=True)
    x = (ds.data.float() / 255).unsqueeze(1)  # (N, 1, 28, 28)
    return x.to(DEVICE), ds.targets.to(DEVICE)


def make_embedder() -> nn.Module:
    # A tiny CNN standing in for DINOv2, then the same neck as FaceEmbedder -
    # but out to 2 dimensions so the embedding can be plotted directly.
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 14x14
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 7x7
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 2, bias=False),
        nn.BatchNorm1d(2),
    )


def train(head_cls, x, y, **head_kwargs):
    torch.manual_seed(0)  # same init and data order for both heads
    embedder = make_embedder().to(DEVICE)
    head = head_cls(embed_dim=2, num_classes=10, **head_kwargs).to(DEVICE)
    opt = torch.optim.Adam([*embedder.parameters(), *head.parameters()], lr=1e-3)

    for epoch in range(EPOCHS):
        for idx in torch.randperm(len(x), device=DEVICE).split(256):
            loss = F.cross_entropy(head(embedder(x[idx]), y[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:2d}  loss {loss.item():.3f}")
    return embedder.eval(), head


def predict(head, emb):
    if isinstance(head, ArcFaceHead):  # trained on angles, so decide on angle alone
        return (F.normalize(emb) @ F.normalize(head.fc.weight).T).argmax(1)
    return head.fc(emb).argmax(1)


def angle_stats(emb, y):
    """Mean angle from each point to its class centre, and the smallest angle
    between two class centres - the same quantities verification depends on."""
    u = F.normalize(emb)
    centres = F.normalize(torch.stack([u[y == c].mean(0) for c in range(10)]))
    within = torch.rad2deg(torch.acos((u * centres[y]).sum(1).clamp(-1, 1))).mean()
    between = torch.rad2deg(torch.acos((centres @ centres.T).clamp(-1, 1)))
    return within.item(), between.fill_diagonal_(360).min().item()


x_train, y_train = load(train=True)
x_test, y_test = load(train=False)

fig, axes = plt.subplots(1, len(RUNS), figsize=(6 * len(RUNS), 6.5))
for ax, (name, head_cls, head_kwargs) in zip(axes, RUNS):
    print(name)
    embedder, head = train(head_cls, x_train, y_train, **head_kwargs)
    with torch.no_grad():
        emb = embedder(x_test)
        acc = (predict(head, emb) == y_test).float().mean().item()
    within, between = angle_stats(emb, y_test)
    print(f"  test acc {acc:.1%}, within-class {within:.1f} deg, closest classes {between:.1f} deg")

    emb, labels = emb.cpu(), y_test.cpu()
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=2, alpha=0.5)
    # Each identity's weight row W_j, drawn as a direction from the origin.
    reach = emb.norm(dim=1).max()
    for c, w in enumerate(F.normalize(head.fc.weight.detach().cpu())):
        ax.plot([0, w[0] * reach], [0, w[1] * reach], color=plt.cm.tab10(c), lw=1)
        ax.annotate(str(c), w * reach * 1.05, ha="center", va="center")
    ax.set_title(f"{name}\nacc {acc:.1%}, within-class {within:.1f}°, closest classes {between:.1f}°")
    ax.set_aspect("equal")

out = Path("outputs/toy_2d.png")
out.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"saved {out}")
