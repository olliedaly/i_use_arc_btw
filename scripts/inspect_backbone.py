"""Step 1: load DINOv2 ViT-S/14 and look at what comes out of it.

Run with:  uv run scripts/inspect_backbone.py
"""

import timm
import torch

IMG_SIZE = 112  # standard size for aligned face crops

model = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",
    pretrained=True,
    # DINOv2 was pretrained at 518x518. Asking for 112 makes timm resample the
    # learned position embeddings from a 37x37 grid down to an 8x8 grid.
    img_size=IMG_SIZE,
).eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters        : {n_params / 1e6:.1f}M")
print(f"patch embedding   : {model.patch_embed.proj}")
print(f"patch grid        : {model.patch_embed.grid_size}")
print(f"transformer blocks: {len(model.blocks)}")
print(f"position embedding: {tuple(model.pos_embed.shape)}")
print(f"classifier head   : {model.head}")

# A fake batch of 4 RGB images, just to see shapes.
x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)

with torch.no_grad():
    tokens = model.forward_features(x)  # every token after the last block
    pooled = model(x)                   # what the model returns by default

print(f"\ninput             : {tuple(x.shape)}")
print(f"forward_features  : {tuple(tokens.shape)}   (batch, 1 CLS + patches, dim)")
print(f"model(x)          : {tuple(pooled.shape)}   (batch, dim)")
print(f"model(x) is the CLS token: {torch.equal(pooled, tokens[:, 0])}")
