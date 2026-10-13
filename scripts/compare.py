"""Step 9: softmax vs ArcFace on the test split, which is used once.

Thresholds are chosen on val and carried over to test. A deployed system has to
work the same way: it sets its threshold before it meets the people it will see.
A paired bootstrap over identities then puts an interval on ArcFace - softmax.

Run with:  uv run scripts/compare.py
           uv run scripts/compare.py --eval-split val   (dry run: val's thresholds on val)
Writes three figures: to docs/img/ for test (they illustrate docs/softmax_vs_arcface.md),
to outputs/ for the dry run.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from evaluate import MODELS, load
from i_use_arc_btw.data import FaceDataset, prepare
from i_use_arc_btw.verify import BINS, embed, pair_histograms, pick_faces, summarise

DEVICE = "cuda"
FARS = (1e-3, 1e-4, 1e-5)
N_BOOT = 1000  # resamples; each one rescores every pair for both heads
METRICS = [f"TAR @ FAR 1e{round(math.log10(f))}" for f in FARS] + ["accuracy at\nval's threshold"]
COLOURS = {"DINOv2, no face training": "#6f6f6f", "softmax": "#e8803a", "ArcFace": "#2a9d8f"}


def threshold_index(r: dict, far: float) -> int:
    """Index of the lowest threshold whose FAR is within budget (as in summarise)."""
    return int((r["far"] <= far).nonzero()[0])


def best_acc_index(r: dict) -> int:
    return int(((r["tar"] + 1 - r["far"]) / 2).argmax())


def balanced_acc(r: dict, k: int) -> float:
    return float((r["tar"][k] + 1 - r["far"][k]) / 2)


def paired_bootstrap(embs: dict, k_acc: dict, n_ids: int):
    """ArcFace - softmax on each metric, with a 95% interval over identities.

    Resample people with replacement, score both heads on the same resample, and
    keep the difference. This covers which people happen to be in the split, not
    the randomness of training itself.

    TAR@FAR sets its threshold again on every resample, at exactly that FAR, so
    both heads are compared at the same FAR. Val's thresholds would put them at
    slightly different FARs on another split. Accuracy uses val's threshold (k_acc).
    """

    def scores(name, weights=None):
        r = summarise(*pair_histograms(*embs[name], weights), fars=FARS)
        values = [r[f"TAR@FAR={f:g}"] for f in FARS] + [balanced_acc(r, k_acc[name])]
        return torch.tensor(values, dtype=torch.float64)

    point = scores("ArcFace") - scores("softmax")
    gen = torch.Generator().manual_seed(0)
    boot = []
    for _ in range(N_BOOT):
        drawn = torch.randint(n_ids, (n_ids,), generator=gen)
        w = torch.bincount(drawn, minlength=n_ids).to(DEVICE)
        boot.append(scores("ArcFace", w) - scores("softmax", w))
    lo, hi = torch.stack(boot).quantile(torch.tensor([0.025, 0.975]).double(), dim=0)
    return point, lo, hi


@torch.no_grad()
def images_behind_top_pairs(emb: torch.Tensor, labels: torch.Tensor, far: float) -> tuple[int, int]:
    """How many distinct images the highest-scoring impostor pairs at this FAR involve.

    Far fewer images than pairs means a few faces each match many others
    (see docs/low_quality_direction.md).
    """
    n = len(emb)
    per_identity = torch.bincount(labels).double()
    k = round(far * (n * (n - 1) / 2 - float((per_identity * (per_identity - 1) / 2).sum())))
    values, pairs = [], []
    for start in range(0, n, 256):
        rows = torch.arange(start, min(start + 256, n), device=emb.device)
        sims = emb[rows] @ emb.T
        skip = (labels[rows, None] == labels[None]) | (torch.arange(n, device=emb.device)[None] <= rows[:, None])
        sims[skip] = -2
        v, j = sims.flatten().topk(k)  # the overall top k can't hold more than k from one chunk
        values.append(v)
        pairs.append(torch.stack([rows[j // n], j % n], 1))
    top = torch.cat(values).topk(k).indices
    return k, len(torch.cat(pairs)[top].unique())


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def plot_roc(on_val, on_ev, split, path):
    """ROC, with dots where the thresholds chosen on val land. A dot right of its
    dotted line would mean the false-accept budget set on val was exceeded."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, r in on_ev.items():
        ax.plot(r["far"], r["tar"], color=COLOURS[name], lw=2.2, label=name)
        ks = [threshold_index(on_val[name], f) for f in FARS]
        ax.plot([r["far"][k] for k in ks], [r["tar"][k] for k in ks], "o",
                color=COLOURS[name], ms=7, mec="white", mew=1.5, zorder=3)
        if name != "DINOv2, no face training":
            k = ks[0]
            ax.annotate(f"{r['tar'][k]:.0%}", (r["far"][k], r["tar"][k]), xytext=(8, -4),
                        textcoords="offset points", color=COLOURS[name], fontweight="bold")
    for f in FARS:
        ax.axvline(f, color="grey", lw=0.7, ls=":")
    r = on_ev["ArcFace"]
    ax.annotate("ArcFace's low-quality\nface cluster", xy=(1e-5, r["TAR@FAR=1e-05"]),
                xytext=(1.3e-6, 0.55), color=COLOURS["ArcFace"], fontsize=9,
                arrowprops={"arrowstyle": "->", "color": COLOURS["ArcFace"]})
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FAR: share of different-person pairs accepted")
    ax.set_ylabel("TAR: share of same-person pairs accepted")
    ax.set_title(f"Verification on people never seen in training ({split})\n"
                 "dots: thresholds chosen on val for FAR 1e-3, 1e-4, 1e-5", fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_scores(on_val, on_ev, split, path):
    """Cosine of same-person and different-person pairs, one panel per model."""
    fig, axes = plt.subplots(len(on_ev), 1, figsize=(7.5, 6.2), sharex=True)
    merge = 100  # 20,000 bins -> 200
    edges = torch.linspace(-1, 1, BINS // merge + 1)
    for ax, (name, r) in zip(axes, on_ev.items()):
        g, i = (r[k].view(-1, merge).sum(1) for k in ("genuine", "impostor"))
        ax.stairs(i / i.sum(), edges, fill=True, color="#d9d9d9", label="different people")
        ax.stairs(g / g.sum(), edges, fill=True, color=COLOURS[name], alpha=0.75, label="same person")
        t = on_val[name]["thresholds"][threshold_index(on_val[name], 1e-3)]
        ax.axvline(t, color="black", lw=1, ls="--")
        ax.text(t + 0.01, 0.9, "threshold for\nFAR 1e-3", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8)
        ax.set_title(name, loc="left", fontsize=10, color=COLOURS[name], fontweight="bold")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    axes[-1].set_xlim(-0.4, 1)
    axes[-1].set_xlabel("cosine similarity of a pair of faces")
    fig.suptitle(f"How each model scores pairs of faces ({split})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_differences(diffs, path):
    """ArcFace - softmax with 95% intervals: val (where the predictions came from)
    next to the evaluation split."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    styles = {"val": ("o", "white", "#7a7a7a", -0.14, "val (the prediction)"),
              "test": ("o", "#264653", "#264653", 0.14, "test (the result)")}
    y = torch.arange(len(METRICS)).double()
    for split, (point, lo, hi) in diffs.items():
        marker, face, edge, offset, label = styles[split]
        if len(diffs) == 1:
            offset = 0
        ax.errorbar(100 * point, y + offset, xerr=[100 * (point - lo), 100 * (hi - point)],
                    fmt=marker, mfc=face, mec=edge, ecolor=edge, ms=7, capsize=4, label=label)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(y.tolist(), METRICS)
    ax.set_ylim(len(METRICS) - 0.5, -0.5)
    ax.set_xlabel("ArcFace - softmax (percentage points)")
    ax.set_title("ArcFace - softmax, with 95% intervals over people", fontsize=11)
    ax.text(0.99, 0.02, "right of 0: ArcFace better", transform=ax.transAxes, ha="right",
            fontsize=8, color="grey")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    split = args.eval_split

    prepare()
    val, ev = FaceDataset("val"), FaceDataset(split)
    val_faces, ev_faces = pick_faces(val), pick_faces(ev)
    on_val, on_ev, emb_val, emb_ev = {}, {}, {}, {}
    for name, checkpoint in MODELS.items():
        model = load(checkpoint)
        emb_val[name] = embed(model, val, val_faces, DEVICE)
        on_val[name] = summarise(*pair_histograms(*emb_val[name]), fars=FARS)
        emb_ev[name] = embed(model, ev, ev_faces, DEVICE)
        on_ev[name] = summarise(*pair_histograms(*emb_ev[name]), fars=FARS)
    r = on_ev["ArcFace"]
    print(f"{split}: {ev.num_classes} identities, {len(ev_faces)} faces, "
          f"{r['genuine pairs']:,.0f} genuine pairs, {r['impostor pairs']:,.0f} impostor pairs")

    # 1. The deployment view: thresholds fixed on val, applied to people never seen.
    print(f"\n1. Thresholds chosen on val, applied to {split}")
    print(f"{'':26s}{'chosen for':>14s}{'threshold':>11s}{f'{split} FAR':>12s}{f'{split} TAR':>10s}")
    for name in MODELS:
        for f in FARS:
            k = threshold_index(on_val[name], f)
            print(f"{name:26s}{f'FAR={f:g}':>14s}{on_val[name]['thresholds'][k]:11.3f}"
                  f"{on_ev[name]['far'][k]:12.1e}{on_ev[name]['tar'][k]:10.1%}")
        k = best_acc_index(on_val[name])
        print(f"{'':26s}{'best accuracy':>14s}{on_val[name]['thresholds'][k]:11.3f}"
              f"{'accuracy':>12s}{balanced_acc(on_ev[name], k):10.2%}")

    # 2. The split's own ROC: each FAR met exactly, as step 8 reported for val.
    print(f"\n2. {split}'s own ROC")
    print(f"{'':26s}" + "".join(f"{f'TAR@FAR={f:g}':>15s}" for f in FARS) + f"{'best acc':>10s}")
    for name, r in on_ev.items():
        print(f"{name:26s}" + "".join(f"{r[f'TAR@FAR={f:g}']:15.1%}" for f in FARS)
              + f"{r['best accuracy']:10.2%}")

    # 3. ArcFace - softmax, on this split and (for the predictions) on val.
    k_acc = {h: best_acc_index(on_val[h]) for h in ("softmax", "ArcFace")}
    diffs = {split: paired_bootstrap(emb_ev, k_acc, ev.num_classes)}
    if split != "val":
        diffs = {"val": paired_bootstrap(emb_val, k_acc, val.num_classes), **diffs}
    for s, (point, lo, hi) in diffs.items():
        print(f"\n3. ArcFace - softmax on {s}, with a 95% bootstrap interval ({N_BOOT} resamples of identities)")
        for m, p, a, b in zip(METRICS, point, lo, hi):
            print(f"   {m.replace(chr(10), ' '):28s}{100 * p:+6.1f} points   [{100 * a:+5.1f}, {100 * b:+5.1f}]")

    # 4. Are the false accepts at the strictest FAR spread out, or a few faces matching many?
    print(f"\n4. Distinct images behind the highest-scoring impostor pairs at FAR = {FARS[-1]:g}")
    for name in ("softmax", "ArcFace"):
        k, n_images = images_behind_top_pairs(*emb_ev[name], FARS[-1])
        print(f"   {name:8s} {k:,} pairs involve {n_images:,} images")

    out = Path("docs/img" if split == "test" else "outputs")
    out.mkdir(parents=True, exist_ok=True)
    plot_roc(on_val, on_ev, split, out / f"{split}_roc.png")
    plot_scores(on_val, on_ev, split, out / f"{split}_scores.png")
    plot_differences(diffs, out / f"{split}_differences.png")
    print(f"\nwrote {out}/{split}_roc.png, {split}_scores.png, {split}_differences.png")


if __name__ == "__main__":
    main()
