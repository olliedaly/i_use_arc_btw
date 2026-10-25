"""Face embedding model: DINOv2 backbone + a small embedding neck.

Maps an aligned face crop to an embedding vector. It knows nothing about
identities - the softmax / ArcFace heads sit on top of it during training and
are thrown away at test time.
"""

import timm
import torch
from torch import nn

BACKBONE = "vit_small_patch14_dinov2.lvd142m"
# timm downloads these weights from Hugging Face. Pinning the commit our results
# used means a fresh download gets the same file even if the repo is updated.
BACKBONE_HF = "timm/vit_small_patch14_dinov2.lvd142m@4610ca143709d58a633b6397a74412c2c3842454"


class FaceEmbedder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        pool: str = "cls",
        img_size: int = 112,
        pretrained: bool = True,
    ):
        super().__init__()
        if pool not in ("cls", "flatten"):
            raise ValueError(f"pool must be 'cls' or 'flatten', got {pool!r}")
        self.pool = pool

        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=pretrained,
            pretrained_cfg_overlay={"hf_hub_id": BACKBONE_HF},
            img_size=img_size,
        )
        width = self.backbone.embed_dim  # 384 for ViT-S

        if pool == "cls":
            in_features = width
        else:
            in_features = self.backbone.patch_embed.num_patches * width  # 64 * 384

        # No bias on the Linear: the BatchNorm right after it subtracts the batch
        # mean, which cancels any bias anyway (BN has its own learnable shift).
        self.neck = nn.Sequential(
            nn.Linear(in_features, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, 1 CLS + patches, width), already through DINOv2's final LayerNorm.
        tokens = self.backbone.forward_features(x)
        if self.pool == "cls":
            feats = tokens[:, 0]
        else:
            feats = tokens[:, self.backbone.num_prefix_tokens :].flatten(1)
        return self.neck(feats)
