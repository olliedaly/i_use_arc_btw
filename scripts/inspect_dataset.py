"""Step 6: download CASIA-WebFace, split it by identity, and look at what the model gets.

Run with:  uv run scripts/inspect_dataset.py
Writes outputs/dataset.png.
"""

import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from i_use_arc_btw.data import MEAN, STD, FaceDataset, prepare

prepare()
splits = {name: FaceDataset(name) for name in ("train", "val", "test")}

# 1. No person in two splits. Labels restart at 0 in every split, so compare the
#    folder names instead - those are the original CASIA identity ids.
people = {name: {Path(p).parent.name for p, _ in ds.samples} for name, ds in splits.items()}
assert not people["train"] & people["val"]
assert not people["train"] & people["test"]
assert not people["val"] & people["test"]
print("no identity appears in two splits\n")

per_identity = {}
for name, ds in splits.items():
    counts = sorted(Counter(label for _, label in ds.samples).values())
    per_identity[name] = counts
    print(
        f"{name:5s}  {ds.num_classes:5d} identities  {len(ds):6d} images   images per identity: "
        f"min {counts[0]}, median {median(counts):.0f}, max {counts[-1]}"
    )

x, y = splits["train"][0]
print(f"\none sample: {tuple(x.shape)} {x.dtype}, range [{x.min():.2f}, {x.max():.2f}], label {y}")

# 2. Can the loader keep up? The GPU trains at ~2,000 images/s (batch 256, bf16).
#    The first pass reads from disk; after that the files sit in the OS page cache.
loader = DataLoader(splits["train"], batch_size=256, shuffle=True, num_workers=8)
batches = iter(loader)
next(batches)  # worker start-up
start = time.time()
for _ in range(50):
    next(batches)
print(f"DataLoader, 8 workers: {50 * 256 / (time.time() - start):.0f} images/s")

# 3. Plot. Left: 6 training identities x 8 images, exactly as the model receives
#    them (random flips included), with the normalisation undone for display.
#    Right: images per identity in each split.
fig = plt.figure(figsize=(15, 6.5))
left, right = fig.subfigures(1, 2, width_ratios=[1.3, 1])

ds = splits["train"]
by_label = defaultdict(list)
for i, (_, label) in enumerate(ds.samples):
    by_label[label].append(i)
labels = torch.randperm(ds.num_classes, generator=torch.Generator().manual_seed(0))[:6]

axes = left.subplots(6, 8)
for row, label in zip(axes, labels.tolist()):
    for ax, i in zip(row, by_label[label][:8]):
        img, _ = ds[i]
        ax.imshow((img * STD + MEAN).clamp(0, 1).permute(1, 2, 0))
    for ax in row:
        ax.axis("off")
    row[0].set_title(f"train label {label}", fontsize=8, loc="left")
left.suptitle("Each row is one person (train split)")

ax = right.subplots()
bins = torch.logspace(0, 3, 40).tolist()
for name, counts in per_identity.items():
    ax.hist(counts, bins=bins, histtype="step", density=True, linewidth=1.5,
            label=f"{name} ({len(counts)} identities)")
ax.set_xscale("log")
ax.set_xlabel("images per identity")
ax.set_ylabel("density")
ax.set_title("Same shape in every split: identities were assigned at random")
ax.legend()

Path("outputs").mkdir(exist_ok=True)
fig.savefig("outputs/dataset.png", dpi=110)
print("wrote outputs/dataset.png")
