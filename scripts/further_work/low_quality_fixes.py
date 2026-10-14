"""Further work, beyond the roadmap: two fixes for ArcFace's false matches at FAR 1e-5.

At the strictest setting, FAR 1e-5, softmax beat ArcFace (docs/softmax_vs_arcface.md).
The cause: ArcFace puts unreadable photos of different people in the same place, so
they match each other (docs/low_quality_direction.md). Neither fix retrains the
model. Both change only how a pair's cosine becomes a yes/no answer.

1. Unsure rule (Deng et al., WACV 2023). Ruin training photos (8x8 pixels, heavy
   blur, or a grey block over half the face) and average where the model puts
   them: the model's direction for "can't tell who this is". A photo whose cosine
   with that direction is above a cutoff is "unsure", and any pair containing it
   is answered "different".
2. Cohort normalisation (AS-norm, from speaker verification). Compare each photo
   with a fixed set of training photos, the cohort. The mean and spread of its 300
   highest cosines say how freely that photo matches strangers. Each pair's cosine
   is rescaled by both photos' mean and spread.

Both are built from training photos only (people in neither val nor test). The
unsure cutoff and every threshold are chosen on val and carried over unchanged,
as in step 9.

Run with:  uv run scripts/further_work/low_quality_fixes.py --eval-split val   (development)
           uv run scripts/further_work/low_quality_fixes.py                    (test, once)
Writes three figures per split to docs/img/further_work/, named after the split.
They illustrate docs/further_work/low_quality_fixes.md.
"""

import argparse
import sys
from pathlib import Path

# The step scripts this builds on are one folder up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import gaussian_blur

from compare import COLOURS, threshold_index, tidy
from evaluate import MODELS, load
from i_use_arc_btw.data import FaceDataset, prepare
from i_use_arc_btw.verify import BINS, embed, pick_faces, summarise

DEVICE = "cuda"
FARS = (1e-3, 1e-4, 1e-5)
HEADS = ("softmax", "ArcFace")
FIXES = ("as is", "unsure rule", "cohort normalisation")
UNSURE_SHARE = 0.01  # the cutoff is set so that 1% of val photos are unsure
K = 300  # cohort normalisation uses each photo's 300 highest cosines with the cohort
N_BOOT = 1000
# Each fix's scores are counted into BINS bins over this range. Cohort-normalised
# scores aren't cosines, so they get a wider one; histograms() checks that no
# score falls off either end.
RANGES = {"as is": (-1.0, 1.0), "unsure rule": (-1.0, 1.0), "cohort normalisation": (-60.0, 60.0)}
FIX_COLOURS = {"as is": COLOURS["ArcFace"], "unsure rule": "#2a78d6", "cohort normalisation": "#4a3aa7"}


def ruin(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """One of three corruptions per photo, each strong enough to remove who it is:
    8x8 pixels, Gaussian blur (sigma 6), or an 80x80 grey block (half the face)."""
    n, size = len(x), x.shape[-1]
    kind = torch.randint(3, (n,), generator=g, device=x.device)[:, None, None, None]
    low = F.interpolate(F.interpolate(x, size=8, mode="area"), size=size, mode="bilinear")
    blur = gaussian_blur(x, kernel_size=31, sigma=6.0)
    top, left = torch.randint(size - 80 + 1, (2, n), generator=g, device=x.device)
    p = torch.arange(size, device=x.device)
    in_y = (p[None] >= top[:, None]) & (p[None] < top[:, None] + 80)
    in_x = (p[None] >= left[:, None]) & (p[None] < left[:, None] + 80)
    # Inputs are normalised, so 0 is the dataset's mean colour, a mid grey.
    block = x.masked_fill((in_y[:, :, None] & in_x[:, None, :])[:, None], 0.0)
    return torch.where(kind == 0, low, torch.where(kind == 1, blur, block))


@torch.no_grad()
def embed_ruined(model, ds: FaceDataset, indices: list[int]) -> torch.Tensor:
    """Unit-length embeddings of ruined copies of ds[indices]."""
    model.eval()
    g = torch.Generator(device=DEVICE).manual_seed(0)
    embs = []
    for x, _ in DataLoader(Subset(ds, indices), batch_size=512, num_workers=8):
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            e = model(ruin(x.to(DEVICE), g))
        embs.append(F.normalize(e.float(), dim=1))
    return torch.cat(embs)


@torch.no_grad()
def cohort_stats(emb: torch.Tensor, cohort: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and spread (standard deviation) of each photo's K highest cosines with the cohort."""
    mus, sds = [], []
    for start in range(0, len(emb), 1024):
        top = (emb[start : start + 1024] @ cohort.T).topk(K, dim=1).values
        mus.append(top.mean(1))
        sds.append(top.std(1))
    return torch.cat(mus), torch.cat(sds)


def scorer(fix: str, d: dict, cutoff: float):
    """The function that turns a block of pair cosines into each fix's scores.
    rows are the block's photos; its columns are every photo."""
    if fix == "as is":
        return lambda rows, s: s
    if fix == "unsure rule":
        unsure = d["near"] >= cutoff
        # Pairs with an unsure photo get the lowest score, so no threshold accepts them.
        return lambda rows, s: s.masked_fill(unsure[rows, None] | unsure[None], -1.0)
    mu, sd = d["mu"], d["sd"]
    # How many spreads the cosine sits above each photo's usual score with
    # strangers, averaged over the two photos.
    return lambda rows, s: 0.5 * ((s - mu[rows, None]) / sd[rows, None] + (s - mu[None]) / sd[None])


@torch.no_grad()
def histograms(d: dict, fix: str, cutoff: float, identity_weights=None):
    """verify.pair_histograms, with each pair scored by the fix instead of by its cosine."""
    emb, labels = d["emb"], d["labels"]
    score, (lo, hi) = scorer(fix, d, cutoff), RANGES[fix]
    n = len(emb)
    if identity_weights is None:
        identity_weights = torch.ones(int(labels.max()) + 1, device=DEVICE)
    w = identity_weights.double()[labels]
    genuine = torch.zeros(BINS, dtype=torch.float64, device=DEVICE)
    impostor = torch.zeros_like(genuine)
    for start in range(0, n, 256):
        rows = torch.arange(start, min(start + 256, n), device=DEVICE)
        s = score(rows, emb[rows] @ emb.T)
        bins = ((s - lo) / (hi - lo) * BINS).long().clamp(0, BINS - 1)
        later = torch.arange(n, device=DEVICE)[None] > rows[:, None]
        same = labels[rows, None] == labels[None]
        g, i = later & same, later & ~same
        genuine += torch.bincount(bins[g], weights=w[rows, None].expand_as(s)[g], minlength=BINS)
        impostor += torch.bincount(bins[i], weights=(w[rows, None] * w[None])[i], minlength=BINS)
    if fix == "cohort normalisation":
        assert genuine[[0, -1]].sum() + impostor[[0, -1]].sum() == 0, "scores ran off RANGES"
    return genuine.cpu(), impostor.cpu()


def threshold_value(fix: str, k: int) -> float:
    lo, hi = RANGES[fix]
    return lo + k * (hi - lo) / BINS


def tars(d, fix, cutoff, weights=None) -> torch.Tensor:
    r = summarise(*histograms(d, fix, cutoff, weights), fars=FARS)
    return torch.tensor([r[f"TAR@FAR={f:g}"] for f in FARS], dtype=torch.float64)


def paired_bootstrap(data: dict, cutoffs: dict, split: str, comparisons: dict, n_ids: int) -> dict:
    """Each comparison is (head, fix) minus (head, fix), with a 95% interval over people.

    As in compare.py: resample identities, score both sides on the same resample, and
    set each TAR@FAR threshold again on every resample, so both sides are compared at
    exactly the same FAR.
    """
    sides = {side for pair in comparisons.values() for side in pair}

    def scores(side, w=None):
        head, fix = side
        return tars(data[head, split], fix, cutoffs[head], w)

    point = {side: scores(side) for side in sides}
    gen = torch.Generator().manual_seed(0)
    boot = {side: [] for side in sides}
    for _ in range(N_BOOT):
        drawn = torch.randint(n_ids, (n_ids,), generator=gen)
        w = torch.bincount(drawn, minlength=n_ids).to(DEVICE)
        for side in sides:
            boot[side].append(scores(side, w))
    out = {}
    for name, (a, b) in comparisons.items():
        d = torch.stack(boot[a]) - torch.stack(boot[b])
        lo, hi = d.quantile(torch.tensor([0.025, 0.975]).double(), dim=0)
        out[name] = (point[a] - point[b], lo, hi)
    return out


@torch.no_grad()
def pairs_above(d: dict, same_person: bool, keep) -> tuple[torch.Tensor, torch.Tensor]:
    """Every pair (i < j) of the given kind whose cosine passes keep(cosines)."""
    emb, labels = d["emb"], d["labels"]
    n, out = len(emb), []
    for start in range(0, n, 256):
        rows = torch.arange(start, min(start + 256, n), device=DEVICE)
        s = emb[rows] @ emb.T
        later = torch.arange(n, device=DEVICE)[None] > rows[:, None]
        kind = (labels[rows, None] == labels[None]) == same_person
        i, j = (later & kind & keep(s)).nonzero().T
        out.append(torch.stack([rows[i], j]))
    return tuple(torch.cat(out, 1))


def walk_through(d: dict, a: int, b: int, cutoff: float, t_raw: float, t_unsure: float, t_cohort: float):
    """One pair through both fixes, with every number shown."""
    s = float(d["emb"][a] @ d["emb"][b])
    verdict = lambda ok: "accepted" if ok else "rejected"
    print(f"     cosine {s:.3f}: as is, {verdict(s >= t_raw)} (threshold {t_raw:.3f})")
    na, nb = float(d["near"][a]), float(d["near"][b])
    unsure = na >= cutoff or nb >= cutoff
    print(f"     unsure rule: cosines with the \"can't tell who this is\" direction {na:.2f} and {nb:.2f} "
          f"(cutoff {cutoff:.2f}) -> {'unsure, so rejected' if unsure else verdict(s >= t_unsure)}")
    z = []
    for photo in (a, b):
        mu, sd = float(d["mu"][photo]), float(d["sd"][photo])
        z.append((s - mu) / sd)
        print(f"     cohort normalisation: photo with top-{K} mean {mu:.3f}, spread {sd:.3f}: "
              f"({s:.3f} - {mu:.3f}) / {sd:.3f} = {z[-1]:+.2f}")
    score = sum(z) / 2
    print(f"       average {score:+.2f} -> {verdict(score >= t_cohort)} (threshold {t_cohort:.2f})")


def plot_centre(near: dict, ruined: dict, cutoffs: dict, split: str, path: Path):
    """Each photo's cosine with the "can't tell who this is" direction, per model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    bins = torch.linspace(-0.4, 1, 71).numpy()
    for ax, head in zip(axes, HEADS):
        # Log scale, so a handful of photos in a bin still shows: the few val photos
        # that land among the ruined ones are the point of the figure. They are an
        # outline drawn on top, so the filled bars can't hide them.
        x = ruined[head].cpu().numpy()
        ax.hist(x, bins=bins, weights=[1 / len(x)] * len(x), color="#4a3aa7", alpha=0.45,
                label="ruined training photos (they define the direction)")
        x = near[head].cpu().numpy()
        ax.hist(x, bins=bins, weights=[1 / len(x)] * len(x), histtype="step", color="#2b2b2b",
                lw=1.4, label=f"{split} photos")
        ax.set_yscale("log")
        ax.set_ylim(0.5 / len(near[head]), 1)
        ax.axvline(cutoffs[head], color="black", lw=1.2, ls="--")
        ax.text(cutoffs[head] + 0.02, 0.97, f"unsure above {cutoffs[head]:.2f}\n(1% of val photos)",
                transform=ax.get_xaxis_transform(), va="top", fontsize=8)
        ax.set_title(head, loc="left", fontsize=10, color=COLOURS[head], fontweight="bold")
        ax.set_xlabel('cosine with the "can\'t tell who this is" direction')
        tidy(ax)
    axes[0].set_ylabel("share of photos per bin (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Where each model puts photos it can't read", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)


def plot_roc(on_ev: dict, split: str, path: Path):
    """ArcFace with each fix, against softmax as is, on the evaluation split."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    lines = [("softmax", "as is", COLOURS["softmax"], "--")]
    lines += [("ArcFace", fix, FIX_COLOURS[fix], "-") for fix in FIXES]
    for head, fix, colour, ls in lines:
        r = on_ev[head, fix]
        t = r["TAR@FAR=1e-05"]
        # The TAR at FAR 1e-5 goes in the legend: the dots are too close to label.
        ax.plot(r["far"], r["tar"], color=colour, lw=2, ls=ls, label=f"{head}, {fix}  ({t:.0%})")
        ax.plot(1e-5, t, "o", color=colour, ms=7, mec="white", mew=1.5, zorder=3)
    ax.axvline(1e-5, color="grey", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FAR: share of different-person pairs accepted")
    ax.set_ylabel("TAR: share of same-person pairs accepted")
    ax.set_title(f"ArcFace with each fix ({split})\ndots and (%): TAR at FAR 1e-5", fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_differences(diffs: dict, path: Path):
    """Two questions, with 95% intervals: does each fix help ArcFace, and with both
    heads fixed, is ArcFace ahead of softmax? val hollow, evaluation split filled."""
    groups = {
        "Does the fix help ArcFace?\n(ArcFace with the fix - ArcFace as is)": "help",
        "Is ArcFace now ahead of softmax?\n(both with the same fix)": "vs softmax",
    }
    rows = [f"{fix}\nFAR {f:g}" for fix in FIXES[1:] for f in FARS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    styles = {"val": ("white", "#7a7a7a", -0.14, "val (the prediction)"),
              "test": ("#264653", "#264653", 0.14, "test (the result)")}
    y = torch.arange(len(rows)).double()
    for ax, (title, group) in zip(axes, groups.items()):
        for split, by_name in diffs.items():
            face, edge, offset, label = styles[split]
            if len(diffs) == 1:
                offset = 0
            pts = [by_name[f"{group}: {fix}"] for fix in FIXES[1:]]
            point = torch.cat([p[0] for p in pts])
            lo = torch.cat([p[1] for p in pts])
            hi = torch.cat([p[2] for p in pts])
            ax.errorbar(100 * point, y + offset, xerr=[100 * (point - lo), 100 * (hi - point)],
                        fmt="o", mfc=face, mec=edge, ecolor=edge, ms=7, capsize=4, label=label)
        ax.axvline(0, color="grey", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("difference in TAR (percentage points)")
        tidy(ax)
    axes[0].set_yticks(y.tolist(), rows, fontsize=8)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    split = args.eval_split
    splits = ["val"] if split == "val" else ["val", split]

    prepare()
    ds = {s: FaceDataset(s) for s in splits}
    train = FaceDataset("train")
    train.flip = False  # the cohort is fixed: no random mirroring
    cohort_faces = pick_faces(train, per_identity=2, seed=0)  # 18,144 photos
    ruined_faces = pick_faces(train, per_identity=1, seed=1)  # 9,072 photos

    data, cutoffs, ruined_near = {}, {}, {}
    for head in HEADS:
        model = load(MODELS[head])
        cohort = embed(model, train, cohort_faces, DEVICE)[0]
        ruined = embed_ruined(model, train, ruined_faces)
        centre = F.normalize(ruined.mean(0), dim=0)  # "can't tell who this is"
        ruined_near[head] = ruined @ centre
        for s in splits:
            emb, labels = embed(model, ds[s], pick_faces(ds[s]), DEVICE)
            mu, sd = cohort_stats(emb, cohort)
            data[head, s] = {"emb": emb, "labels": labels, "near": emb @ centre, "mu": mu, "sd": sd}
        cutoffs[head] = float(data[head, "val"]["near"].quantile(1 - UNSURE_SHARE))

    print("1. The unsure rule, fitted on training photos and val")
    for head in HEADS:
        share = (data[head, split]["near"] >= cutoffs[head]).float().mean()
        print(f"   {head:8s} ruined training photos: median cosine with their direction "
              f"{ruined_near[head].median():.2f}.  cutoff {cutoffs[head]:.3f};  "
              f"{split} photos unsure: {share:.1%}")

    on = {(head, fix, s): summarise(*histograms(data[head, s], fix, cutoffs[head]), fars=FARS)
          for head in HEADS for fix in FIXES for s in splits}

    print(f"\n2. Thresholds chosen on val, applied to {split}")
    print(f"   {'':32s}{'chosen for':>12s}{'threshold':>11s}{f'{split} FAR':>11s}{f'{split} TAR':>10s}")
    for head in HEADS:
        for fix in FIXES:
            for f in FARS:
                k = threshold_index(on[head, fix, "val"], f)
                r = on[head, fix, split]
                print(f"   {f'{head}, {fix}':32s}{f'FAR {f:g}':>12s}{threshold_value(fix, k):11.3f}"
                      f"{r['far'][k]:11.1e}{r['tar'][k]:10.1%}")

    print(f"\n3. {split}'s own ROC: TAR at each FAR")
    print(f"   {'':32s}" + "".join(f"{f'FAR {f:g}':>11s}" for f in FARS))
    for head in HEADS:
        for fix in FIXES:
            r = on[head, fix, split]
            print(f"   {f'{head}, {fix}':32s}" + "".join(f"{r[f'TAR@FAR={f:g}']:11.1%}" for f in FARS))

    comparisons = {}
    for fix in FIXES[1:]:
        comparisons[f"help: {fix}"] = (("ArcFace", fix), ("ArcFace", "as is"))
        comparisons[f"vs softmax: {fix}"] = (("ArcFace", fix), ("softmax", fix))
    diffs = {s: paired_bootstrap(data, cutoffs, s, comparisons, ds[s].num_classes) for s in splits}
    for s, by_name in diffs.items():
        print(f"\n4. Differences on {s}, with a 95% bootstrap interval ({N_BOOT} resamples of identities)")
        for name, (point, lo, hi) in by_name.items():
            for f, p, a, b in zip(FARS, point, lo, hi):
                print(f"   {name:34s} FAR {f:<7g}{100 * p:+6.1f} points   [{100 * a:+5.1f}, {100 * b:+5.1f}]")

    # 5. One real pair of each kind through both fixes: ArcFace, thresholds for FAR 1e-5 from val.
    d = data["ArcFace", split]
    t = {fix: threshold_value(fix, threshold_index(on["ArcFace", fix, "val"], 1e-5)) for fix in FIXES}
    args5 = (cutoffs["ArcFace"], t["as is"], t["unsure rule"], t["cohort normalisation"])
    print(f"\n5. ArcFace on {split}, at FAR 1e-5 with val's thresholds: two pairs worked through")
    i, j = pairs_above(d, False, lambda s: s >= t["as is"])
    worst = int(((d["emb"][i] * d["emb"][j]).sum(1)).argmax())
    print(f"   a) the highest-scoring pair of different people")
    walk_through(d, int(i[worst]), int(j[worst]), *args5)
    i, j = pairs_above(d, True, lambda s: s < t["as is"])
    cos = (d["emb"][i] * d["emb"][j]).sum(1)
    z = 0.5 * ((cos - d["mu"][i]) / d["sd"][i] + (cos - d["mu"][j]) / d["sd"][j])
    rescued = (z >= t["cohort normalisation"]).nonzero().flatten()
    pick = rescued[cos[rescued].argsort()[len(rescued) // 2]]
    print(f"   b) a same-person pair rejected as is but accepted after cohort normalisation "
          f"({len(rescued):,} such pairs; this one has their median cosine)")
    walk_through(d, int(i[pick]), int(j[pick]), *args5)

    out = Path("docs/img/further_work")
    out.mkdir(parents=True, exist_ok=True)
    plot_centre({h: data[h, split]["near"] for h in HEADS}, ruined_near, cutoffs, split,
                out / f"{split}_unsure_direction.png")
    plot_roc({(h, fix): on[h, fix, split] for h in HEADS for fix in FIXES}, split, out / f"{split}_roc.png")
    plot_differences(diffs, out / f"{split}_differences.png")
    print(f"\nwrote {out}/{split}_unsure_direction.png, {split}_roc.png, {split}_differences.png")


if __name__ == "__main__":
    main()
