"""ArcFace's low-quality direction (see docs/low_quality_direction.md).

Uses the step-7 checkpoints on the same val faces as evaluate.py.

Run with:  uv run scripts/low_quality_direction.py
Writes outputs/low_quality_pairs.png. It isn't committed: the faces are real
people from a research-only dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evaluate import MODELS, load
from i_use_arc_btw.data import MEAN, STD, FaceDataset, prepare
from i_use_arc_btw.verify import embed, pair_histograms, pick_faces, summarise

DEVICE = "cuda"
N_HUB_PAIRS = 450  # FAR = 1e-5 of val's 45M impostor pairs
LAPLACIAN = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]], device=DEVICE)[None, None]


@torch.no_grad()
def raw_embed_and_sharpness(model, ds, faces):
    """Embeddings before normalisation, and a model-free quality score per face.

    Sharpness is the variance of the Laplacian of the grey image: blur removes the
    fine detail the Laplacian responds to, so blurred faces score low.
    """
    model.eval()
    embs, sharp = [], []
    for x, _ in DataLoader(Subset(ds, faces), batch_size=512, num_workers=8):
        x = x.to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            embs.append(model(x).float())
        grey = (x * STD.to(DEVICE) + MEAN.to(DEVICE)).mean(1, keepdim=True) * 255
        sharp.append(F.conv2d(grey, LAPLACIAN).flatten(1).var(1))
    return torch.cat(embs), torch.cat(sharp)


def top_impostor_pairs(e, labels, n):
    """(i, j, cosine) of the n highest-scoring pairs of different people."""
    sims = e @ e.T
    sims[(labels[:, None] == labels[None]) | ~torch.ones_like(sims, dtype=torch.bool).triu(1)] = -2
    top = sims.flatten().topk(n)
    return top.indices // len(e), top.indices % len(e), top.values


def hubs(e, labels, n):
    i, j, _ = top_impostor_pairs(e, labels, n)
    h = torch.zeros(len(e), dtype=torch.bool, device=e.device)
    h[i] = h[j] = True
    return h


def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def remove(e, d):
    """Project direction d out of every embedding, then renormalise."""
    return F.normalize(e - (e @ d)[:, None] * d, dim=1)


def tar(e, labels):
    r = summarise(*pair_histograms(e, labels))
    return " ".join(f"{r[f'TAR@FAR={f:g}']:6.1%}" for f in (1e-3, 1e-4, 1e-5))


def main():
    prepare()
    ds = FaceDataset("val")
    faces = pick_faces(ds)

    arc_model = load(MODELS["ArcFace"])
    raw, sharp = raw_embed_and_sharpness(arc_model, ds, faces)
    e = F.normalize(raw, dim=1)
    labels = torch.tensor([ds.samples[i][1] for i in faces], device=DEVICE)
    hub = hubs(e, labels, N_HUB_PAIRS)

    print(f"1. ArcFace's {N_HUB_PAIRS} highest-scoring impostor pairs involve {int(hub.sum())} images")
    print(f"   sharpness (median): those images {sharp[hub].median():.0f}, other faces {sharp[~hub].median():.0f}")

    print("\n2. Mean cosine between those images (different people only), against any two different people:")
    diff = labels[:, None] != labels[None]
    for name, checkpoint in MODELS.items():
        en = e if name == "ArcFace" else embed(load(checkpoint), ds, faces, DEVICE)[0]
        s = en @ en.T
        print(f"   {name:26s} {s[hub[:, None] & hub[None] & diff].mean():5.2f}   {s[diff].mean():5.2f}")

    print("\n3. Fit a direction on half of val's identities, remove it from the other half.")
    print("   TAR@FAR 1e-3 / 1e-4 / 1e-5 on the held-out half:")
    for fit in (0, 1):
        A, B = labels % 2 == fit, labels % 2 != fit
        blurry = sharp[A] <= sharp[A].quantile(0.05)
        d_blur = F.normalize(e[A][blurry].mean(0) - e[A].mean(0), dim=0)
        d_hub = F.normalize(e[A][hubs(e[A], labels[A], N_HUB_PAIRS // 2)].mean(0), dim=0)
        hub_B = hubs(e[B], labels[B], N_HUB_PAIRS // 2)
        print(f"   fit on half {fit}, test on half {1 - fit} (cos between the two directions {d_blur @ d_hub:.2f})")
        for tag, d in (("blur", d_blur), ("hub", d_hub)):
            c = e[B] @ d
            print(f"     {tag} direction: rank correlation with sharpness {spearman(c, sharp[B]):+.2f}, "
                  f"cos to it: held-out hubs {c[hub_B].mean():.2f}, other faces {c[~hub_B].mean():.2f}")
        print(f"     as is                    {tar(e[B], labels[B])}")
        print(f"     blur direction removed   {tar(remove(e[B], d_blur), labels[B])}")
        print(f"     hub direction removed    {tar(remove(e[B], d_hub), labels[B])}")

    print("\n4. Explanations that don't hold")
    rows = F.normalize(torch.load(MODELS["ArcFace"])["head"]["fc.weight"].to(DEVICE), dim=1)
    away = -F.normalize(rows.mean(0), dim=0)
    angle = torch.rad2deg(torch.acos((e @ rows.T).clamp(-1, 1))).mean(1)
    print(f"   a) 'pushed away from every identity': cos(hub direction, away from the average "
          f"identity row) {F.normalize(e[hub].mean(0), dim=0) @ away:.2f}")
    print(f"      mean angle to the 9,072 training identities: hubs {angle[hub].mean():.1f}, "
          f"others {angle[~hub].mean():.1f} deg;  TAR with it removed {tar(remove(e, away), labels)}")
    beta = arc_model.neck[1].bias.detach()
    face_part = (raw - beta).norm(dim=1)  # BatchNorm output minus its shift = gamma * normalised input
    print(f"   b) 'collapsed onto BatchNorm's shift beta': |beta| {beta.norm():.1f}, typical "
          f"|embedding| {raw.norm(dim=1).median():.1f}")
    print(f"      face-specific part |emb - beta|: hubs {face_part[hub].median():.1f}, "
          f"others {face_part[~hub].median():.1f};  TAR with beta subtracted "
          f"{tar(F.normalize(raw - beta, dim=1), labels)}")
    print(f"   as is: {tar(e, labels)}")

    i, j, cos = top_impostor_pairs(e, labels, 12)
    fig, axes = plt.subplots(2, 12, figsize=(18, 3.6))
    for col, pair in enumerate(zip(i.tolist(), j.tolist())):
        for row, k in enumerate(pair):
            img = ds[faces[k]][0] * STD + MEAN
            axes[row, col].imshow(img.clamp(0, 1).permute(1, 2, 0))
            axes[row, col].axis("off")
        axes[0, col].set_title(f"cos {cos[col]:.2f}", fontsize=8)
    fig.suptitle("ArcFace's 12 highest-scoring pairs of different people (val): top row vs bottom row")
    fig.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/low_quality_pairs.png", dpi=100)
    print("\nwrote outputs/low_quality_pairs.png")


if __name__ == "__main__":
    main()
