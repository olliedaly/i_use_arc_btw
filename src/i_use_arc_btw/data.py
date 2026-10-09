"""CASIA-WebFace: download, identity-disjoint split, and a PyTorch Dataset.

The faces are already aligned (5 facial landmarks warped onto fixed positions in
a 112x112 crop), the same preprocessing the ArcFace paper trains on, so there is
no face detection or alignment here.
"""

import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image, read_file

# A third-party re-upload of InsightFace's 112x112 CASIA-WebFace, as parquet shards.
# Non-commercial research use only.
REPO = "SaffalPoosh/casia_web_face"
ROOT = Path("data/casia_webface")

# DINOv2 was pretrained on inputs normalised with ImageNet statistics (its timm
# config lists exactly these values), so its weights expect the same here.
MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)


def prepare(root: Path = ROOT) -> None:
    """Download the parquet shards and unpack them to root/<identity>/<n>.png, once."""
    if (root / ".complete").exists():
        return
    parquet_dir = Path(
        snapshot_download(
            REPO,
            repo_type="dataset",
            local_dir=root.with_name(root.name + "_parquet"),
            allow_patterns=["data/*.parquet"],
        )
    )
    n = 0
    for shard in sorted(parquet_dir.glob("data/*.parquet")):
        table = pq.read_table(shard)
        images = table["image"].combine_chunks().field("bytes").to_pylist()
        for img, label in zip(images, table["label"].to_pylist()):
            folder = root / f"{label:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            # The images are stored as PNG (lossless), so writing the bytes as-is
            # keeps every pixel exactly. Re-encoding to JPEG would save disk but
            # add compression artefacts.
            (folder / f"{n:06d}.png").write_bytes(img)
            n += 1
        print(f"  unpacked {shard.name} ({n} images so far)")
    (root / ".complete").touch()


def split_identities(
    root: Path = ROOT, n_val: int = 500, n_test: int = 1000, seed: int = 0
) -> dict[str, list[str]]:
    """Assign whole identities to train / val / test, so no person is in two splits.

    Splitting images instead would put different photos of the same person in
    train and test, and verification would then measure recognising people the
    model was trained on - not the unseen-identity case face recognition is for.
    """
    identities = sorted(p.name for p in root.iterdir() if p.is_dir())
    shuffled = random.Random(seed).sample(identities, len(identities))
    return {
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test : n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val :]),
    }


class FaceDataset(Dataset):
    """Aligned 112x112 faces from one split, as normalised (3, 112, 112) tensors.

    Labels run 0..num_classes-1 *within the split*: label 5 in train and label 5
    in test are different people. Training needs contiguous labels because the
    head has one weight row per identity. Val and test never touch the head -
    they only ask whether two faces are the same person.
    """

    def __init__(self, split: str, root: Path = ROOT):
        identities = split_identities(root)[split]
        self.samples = [
            (str(path), label)
            for label, identity in enumerate(identities)
            for path in sorted((root / identity).iterdir())
        ]
        self.num_classes = len(identities)
        # Horizontal flip is the only augmentation, as in InsightFace's training
        # code. The alignment template is left-right symmetric (eye midpoint, nose
        # and mouth midpoint all at x ~= 56 of 112), so a mirrored face keeps its
        # landmarks where they were. Random crops or rotations would move them away
        # from where every test face has them.
        self.flip = split == "train"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        img = decode_image(read_file(path), mode=ImageReadMode.RGB)  # uint8 (3, 112, 112)
        if self.flip and torch.rand(()) < 0.5:
            img = img.flip(-1)
        return (img.float() / 255 - MEAN) / STD, label
