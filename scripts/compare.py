"""Step 9: softmax vs ArcFace on the test split, which is used once.

Thresholds are chosen on val and carried over to test. A deployed system has to
work the same way: it sets its threshold before it meets the people it will see.
A paired bootstrap over test identities then puts an interval on ArcFace - softmax.

Run with:  uv run scripts/compare.py
           uv run scripts/compare.py --eval-split val   (dry run: val's thresholds on val)
Writes outputs/compare_<split>.png.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from evaluate import MODELS, load
from i_use_arc_btw.data import FaceDataset, prepare
from i_use_arc_btw.verify import embed, pair_histograms, pick_faces, summarise

DEVICE = "cuda"
FARS = (1e-3, 1e-4, 1e-5)
N_BOOT = 1000  # resamples; each one rescores every pair for both heads


def threshold_index(r: dict, far: float) -> int:
    """Index of the lowest threshold whose FAR is within budget (as in summarise)."""
    return int((r["far"] <= far).nonzero()[0])


def balanced_acc(r: dict, k: int) -> float:
    return float((r["tar"][k] + 1 - r["far"][k]) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    split = args.eval_split

    prepare()
    val, ev = FaceDataset("val"), FaceDataset(split)
    val_faces, ev_faces = pick_faces(val), pick_faces(ev)
    on_val, on_ev, emb_ev = {}, {}, {}
    for name, checkpoint in MODELS.items():
        model = load(checkpoint)
        on_val[name] = summarise(*pair_histograms(*embed(model, val, val_faces, DEVICE)), fars=FARS)
        emb_ev[name] = embed(model, ev, ev_faces, DEVICE)
        on_ev[name] = summarise(*pair_histograms(*emb_ev[name]), fars=FARS)
    labels = emb_ev["ArcFace"][1]
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
        k = int(((on_val[name]["tar"] + 1 - on_val[name]["far"]) / 2).argmax())
        print(f"{'':26s}{'best accuracy':>14s}{on_val[name]['thresholds'][k]:11.3f}"
              f"{'accuracy':>12s}{balanced_acc(on_ev[name], k):10.2%}")

    # 2. The split's own ROC: each FAR met exactly, as step 8 reported for val.
    print(f"\n2. {split}'s own ROC")
    print(f"{'':26s}" + "".join(f"{f'TAR@FAR={f:g}':>15s}" for f in FARS) + f"{'best acc':>10s}")
    for name, r in on_ev.items():
        print(f"{name:26s}" + "".join(f"{r[f'TAR@FAR={f:g}']:15.1%}" for f in FARS)
              + f"{r['best accuracy']:10.2%}")

    # 3. Paired bootstrap over identities: resample people with replacement, score
    #    both heads on the same resample, and keep the difference. This covers which
    #    people happen to be in the split, not the randomness of training itself.
    metrics = [f"TAR@FAR={f:g}" for f in FARS] + ["accuracy at val's threshold"]
    k_acc = {h: int(((on_val[h]["tar"] + 1 - on_val[h]["far"]) / 2).argmax()) for h in ("softmax", "ArcFace")}

    def scores(name, weights=None):
        r = summarise(*pair_histograms(*emb_ev[name], weights), fars=FARS)
        values = [r[f"TAR@FAR={f:g}"] for f in FARS] + [balanced_acc(r, k_acc[name])]
        return torch.tensor(values, dtype=torch.float64)

    point = scores("ArcFace") - scores("softmax")
    gen = torch.Generator().manual_seed(0)
    boot = []
    for _ in range(N_BOOT):
        drawn = torch.randint(ev.num_classes, (ev.num_classes,), generator=gen)
        w = torch.bincount(drawn, minlength=ev.num_classes).to(DEVICE)
        boot.append(scores("ArcFace", w) - scores("softmax", w))
    lo, hi = torch.stack(boot).quantile(torch.tensor([0.025, 0.975]).double(), dim=0)
    print(f"\n3. ArcFace - softmax on {split}, with a 95% bootstrap interval ({N_BOOT} resamples of identities)")
    for m, p, a, b in zip(metrics, point, lo, hi):
        print(f"   {m:28s}{100 * p:+6.1f} points   [{100 * a:+5.1f}, {100 * b:+5.1f}]")

    # Left: ROC, with dots where val's FAR thresholds actually land. A dot right of
    # its dotted line means the false-accept budget set on val was exceeded.
    # Right: the differences and their intervals.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), width_ratios=[1.3, 1])
    for name, r in on_ev.items():
        line, = ax.plot(r["far"], r["tar"], label=name)
        ks = [threshold_index(on_val[name], f) for f in FARS]
        ax.plot([r["far"][k] for k in ks], [r["tar"][k] for k in ks], "o", color=line.get_color(), ms=5)
    for f in FARS:
        ax.axvline(f, color="grey", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FAR: different-person pairs accepted")
    ax.set_ylabel("TAR: same-person pairs accepted")
    ax.set_title(f"ROC on {split}; dots = thresholds chosen on val")
    ax.legend(loc="lower right")

    y = range(len(metrics))
    ax2.errorbar(100 * point, y, xerr=[100 * (point - lo), 100 * (hi - point)], fmt="o", capsize=4)
    ax2.axvline(0, color="grey", lw=0.8)
    ax2.set_yticks(list(y), metrics)
    ax2.invert_yaxis()
    ax2.set_xlabel("ArcFace - softmax (percentage points)")
    ax2.set_title("Difference, with 95% bootstrap interval")

    Path("outputs").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"outputs/compare_{split}.png", dpi=110)
    print(f"\nwrote outputs/compare_{split}.png")


if __name__ == "__main__":
    main()
