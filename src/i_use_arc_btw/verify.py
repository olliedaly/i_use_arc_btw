"""Face verification on unseen identities: are two faces the same person?

No head and no training identities here. Embed every face, compare every pair of
faces by cosine similarity, and measure how well one threshold separates
same-person pairs (genuine) from different-person pairs (impostor). A deployed
system works the same way: it compares a new face with an enrolled one, and has
never seen either person.
"""

import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from i_use_arc_btw.data import FaceDataset

# Pair scores are counted into bins instead of stored (val alone has 45M pairs).
# 20,000 bins over the cosine range [-1, 1] puts candidate thresholds 1e-4 apart.
BINS = 20_000


def pick_faces(ds: FaceDataset, per_identity: int = 20, seed: int = 0) -> list[int]:
    """Up to `per_identity` random images of each identity, as indices into ds.

    A person with n photos gives n(n-1)/2 genuine pairs, so without a cap the few
    people with hundreds of photos would dominate the score: in val, the 10 largest
    of 500 identities supply 42% of all genuine pairs. With a cap of 20, every
    identity counts roughly equally.
    """
    by_label = defaultdict(list)
    for i, (_, label) in enumerate(ds.samples):
        by_label[label].append(i)
    rng = random.Random(seed)
    return sorted(
        i for idx in by_label.values() for i in rng.sample(idx, min(per_identity, len(idx)))
    )


@torch.no_grad()
def embed(
    model: nn.Module, ds: FaceDataset, indices: list[int], device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-length embeddings and identity labels for ds[indices]."""
    # eval() makes BatchNorm use the running statistics it collected in training.
    # In train mode it normalises with the current batch's statistics, so a face's
    # embedding would depend on which other faces happen to share its batch.
    model.eval()
    loader = DataLoader(Subset(ds, indices), batch_size=512, num_workers=8)
    embs, labels = [], []
    for x, y in loader:
        with torch.autocast(device, dtype=torch.bfloat16):
            e = model(x.to(device, non_blocking=True))
        embs.append(F.normalize(e.float(), dim=1))
        labels.append(y)
    return torch.cat(embs), torch.cat(labels).to(device)


@torch.no_grad()
def pair_histograms(
    emb: torch.Tensor, labels: torch.Tensor, identity_weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Histograms of cosine similarity over every genuine and every impostor pair.

    Each pair is counted once (i < j). Rows are processed in chunks so the full
    n x n similarity matrix never has to sit in memory.

    identity_weights (one per label) resample identities for a bootstrap: an
    identity drawn twice counts its genuine pairs twice, and a pair of identities
    a, b counts w[a] * w[b] times. By default every identity counts once.
    """
    n = len(emb)
    if identity_weights is None:
        identity_weights = torch.ones(int(labels.max()) + 1, device=emb.device)
    w = identity_weights.double()[labels]  # each face carries its identity's weight
    genuine = torch.zeros(BINS, dtype=torch.float64, device=emb.device)
    impostor = torch.zeros_like(genuine)
    # 256 rows per chunk keeps each GPU kernel short. The driver kills kernels that
    # run too long on a GPU that also drives a display, and 1,024 rows of test's
    # 18,814 faces hit that limit.
    for start in range(0, n, 256):
        rows = torch.arange(start, min(start + 256, n), device=emb.device)
        sims = emb[rows] @ emb.T
        bins = ((sims + 1) / 2 * BINS).long().clamp(0, BINS - 1)
        later = torch.arange(n, device=emb.device)[None] > rows[:, None]
        same = labels[rows, None] == labels[None]
        g, i = later & same, later & ~same
        genuine += torch.bincount(bins[g], weights=w[rows, None].expand_as(sims)[g], minlength=BINS)
        impostor += torch.bincount(bins[i], weights=(w[rows, None] * w[None])[i], minlength=BINS)
    return genuine.cpu(), impostor.cpu()


def summarise(genuine: torch.Tensor, impostor: torch.Tensor, fars=(1e-3, 1e-4, 1e-5)) -> dict:
    """ROC numbers from the two histograms.

    A threshold t accepts a pair as "same person" when its cosine is >= t.
      TAR (true accept rate):  fraction of genuine pairs accepted
      FAR (false accept rate): fraction of impostor pairs accepted
    """
    thresholds = torch.linspace(-1, 1, BINS + 1, dtype=torch.float64)[:-1]  # bins' lower edges
    # At threshold k, every pair in bin k or above is accepted: a cumulative sum from the top.
    tar = genuine.flip(0).cumsum(0).flip(0).double() / genuine.sum()
    far = impostor.flip(0).cumsum(0).flip(0).double() / impostor.sum()
    centres = thresholds + 1 / BINS
    out = {
        "genuine pairs": int(genuine.sum()),
        "impostor pairs": int(impostor.sum()),
        "mean genuine cos": float((genuine * centres).sum() / genuine.sum()),
        "mean impostor cos": float((impostor * centres).sum() / impostor.sum()),
        "genuine": genuine,
        "impostor": impostor,
        "thresholds": thresholds,
        "tar": tar,
        "far": far,
    }
    for f in fars:
        # Raising the threshold lowers both rates. The lowest threshold that keeps FAR
        # within budget accepts the most genuine pairs.
        k = int((far <= f).nonzero()[0])
        out[f"TAR@FAR={f:g}"] = float(tar[k])
    # Best accuracy on a balanced set (as many genuine as impostor pairs, like LFW's
    # 3,000 + 3,000): the average of TAR and the true reject rate, 1 - FAR.
    acc = (tar + 1 - far) / 2
    k = int(acc.argmax())
    out["best accuracy"] = float(acc[k])
    out["threshold"] = float(thresholds[k])
    return out
