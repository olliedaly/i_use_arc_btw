"""Step 8: verification on unseen identities - raw DINOv2, softmax and ArcFace.

Every model embeds the same faces (up to 20 per identity), and every pair of
faces is scored by cosine similarity. The val split is for development; test is
kept for the final comparison in step 9.

Run with:  uv run scripts/evaluate.py
Writes outputs/verification_val.png.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from i_use_arc_btw.data import FaceDataset, prepare
from i_use_arc_btw.model import FaceEmbedder
from i_use_arc_btw.verify import BINS, embed, pair_histograms, pick_faces, summarise

DEVICE = "cuda"
# The strictest FAR is fragile here: val has 45M impostor pairs, so FAR=1e-5 lets
# only 450 through, and a few dozen bad images can decide which 450. For ArcFace
# on val they come from ~50 blurred, side-on or badly cropped faces that it embeds
# close together, whoever they are.
FARS = (1e-3, 1e-4, 1e-5)
MODELS = {
    "DINOv2, no face training": None,
    "softmax": "outputs/runs/softmax/checkpoint.pt",
    "ArcFace": "outputs/runs/arcface/checkpoint.pt",
}


def load(checkpoint: str | None) -> nn.Module:
    if checkpoint is None:
        model = FaceEmbedder()
        # The neck is random until trained, so leave it out: this baseline is
        # DINOv2's own CLS feature, exactly as it comes from pretraining.
        model.neck = nn.Identity()
    else:
        model = FaceEmbedder(pretrained=False)  # every weight comes from the checkpoint
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["embedder"])
    return model.to(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    prepare()
    ds = FaceDataset(args.split)
    faces = pick_faces(ds)

    results = {}
    for name, checkpoint in MODELS.items():
        emb, labels = embed(load(checkpoint), ds, faces, DEVICE)
        results[name] = summarise(*pair_histograms(emb, labels), fars=FARS)

    r = next(iter(results.values()))
    print(f"{args.split}: {ds.num_classes} identities, {len(faces)} faces, "
          f"{r['genuine pairs']:,} genuine pairs, {r['impostor pairs']:,} impostor pairs\n")
    print(f"{'':26s}" + "".join(f"{'TAR@FAR=' + f'{f:g}':>15s}" for f in FARS)
          + f"{'best acc':>10s}{'threshold':>11s}{'mean cos: same':>16s}{'different':>11s}")
    for name, r in results.items():
        print(f"{name:26s}" + "".join(f"{r[f'TAR@FAR={f:g}']:15.1%}" for f in FARS)
              + f"{r['best accuracy']:10.2%}{r['threshold']:11.3f}"
              f"{r['mean genuine cos']:16.3f}{r['mean impostor cos']:11.3f}")

    # Left: ROC. Right: the two score distributions per model, with the threshold
    # that gives the best accuracy. Verification works when the two humps separate.
    fig, axes = plt.subplots(1, 1 + len(results), figsize=(5.5 * (1 + len(results)), 4.5))
    ax = axes[0]
    for name, r in results.items():
        ax.plot(r["far"], r["tar"], label=name)
    for f in FARS:
        ax.axvline(f, color="grey", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FAR: different-person pairs accepted")
    ax.set_ylabel("TAR: same-person pairs accepted")
    ax.set_title(f"ROC on {args.split} (unseen identities)")
    ax.legend(loc="lower right")

    merge = 100  # 20,000 bins -> 200 for plotting
    edges = torch.linspace(-1, 1, BINS // merge + 1)
    for ax, (name, r) in zip(axes[1:], results.items()):
        g, i = (r[k].view(-1, merge).sum(1).double() for k in ("genuine", "impostor"))
        ax.stairs(g / g.sum(), edges, label="same person", fill=True, alpha=0.5)
        ax.stairs(i / i.sum(), edges, label="different people", fill=True, alpha=0.5)
        ax.axvline(r["threshold"], color="black", lw=1, ls="--", label=f"threshold {r['threshold']:.2f}")
        ax.set_xlim(-0.5, 1)
        ax.set_xlabel("cosine similarity of the pair")
        ax.set_title(f"{name}: best acc {r['best accuracy']:.1%}")
        ax.legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel("fraction of pairs")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"outputs/verification_{args.split}.png", dpi=110)
    print(f"\nwrote outputs/verification_{args.split}.png")


if __name__ == "__main__":
    main()
