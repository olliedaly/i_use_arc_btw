"""Step 7: fine-tune DINOv2 on CASIA-WebFace with a softmax or an ArcFace head.

The two runs differ only in the head: same seed, initial weights, data order,
optimiser and schedule.

Run with:  uv run scripts/train.py --head arcface
           uv run scripts/train.py --head softmax
Writes outputs/runs/<name>/log.csv and checkpoint.pt (saved every epoch).
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from i_use_arc_btw.data import FaceDataset, prepare
from i_use_arc_btw.heads import ArcFaceHead, SoftmaxHead
from i_use_arc_btw.model import FaceEmbedder

DEVICE = "cuda"
SEED = 0
EMBED_DIM = 384
BATCH_SIZE = 256
# The backbone already knows useful features, so it moves slowly; the neck and head
# start random and need to move fast. InsightFace uses 1e-3 when training a ViT
# from scratch - that's our rate for the fresh layers.
LR_BACKBONE = 1e-4
LR_FRESH = 1e-3
WEIGHT_DECAY = 0.05
LOG_EVERY = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=["softmax", "arcface"], required=True)
    parser.add_argument("--s", type=float, default=64.0)
    parser.add_argument("--m", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, help="stop early, for smoke tests")
    parser.add_argument("--name", help="run folder name (default: the head)")
    args = parser.parse_args()
    out = Path("outputs/runs") / (args.name or args.head)
    out.mkdir(parents=True, exist_ok=True)

    prepare()
    train = FaceDataset("train")
    loader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),  # same order (and flips) for both heads
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        # A smaller last batch would give BatchNorm noisier statistics; just skip it.
        drop_last=True,
    )

    torch.manual_seed(SEED)  # same neck init, and same head init: both heads are one nn.Linear
    embedder = FaceEmbedder(embed_dim=EMBED_DIM).to(DEVICE).train()
    if args.head == "arcface":
        head = ArcFaceHead(EMBED_DIM, train.num_classes, s=args.s, m=args.m)
    else:
        head = SoftmaxHead(EMBED_DIM, train.num_classes)
    head = head.to(DEVICE)

    opt = torch.optim.AdamW(
        [
            {"params": embedder.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": [*embedder.neck.parameters(), *head.parameters()], "lr": LR_FRESH},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    # Linear warm-up over the first 10% of steps, then linear decay to 0 (as InsightFace).
    # Warm-up matters here: the random head's first gradients are noise, and at full
    # learning rate they would scramble the pretrained backbone before the head has
    # learned anything.
    # For ArcFace it also decides whether training escapes a trap. Every angle starts
    # near 90 deg, and above (pi - m) / 2 = 75.7 deg the margin makes the loss *fall*
    # when faces and identity rows all drift apart. A 600-step test (60-step warm-up)
    # went that way, theta_y 89 -> 96 deg; with this warm-up it is below 75 deg within
    # the first epoch.
    total = args.max_steps or args.epochs * len(loader)
    warmup = total // 10
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min((step + 1) / warmup, (total - step) / (total - warmup))
    )

    print(f"{args.head}: {train.num_classes} identities, {len(loader)} steps/epoch, {total} steps")
    log_file = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "epoch", "loss", "theta_y_deg", "emb_norm", "grad_norm", "lr_backbone"])

    step, running_loss, tick = 0, torch.zeros((), device=DEVICE), time.time()
    for epoch in range(args.epochs):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                emb = embedder(x)
            # Head outside autocast, in fp32: bf16 can't resolve angles below ~5 degrees.
            loss = F.cross_entropy(head(emb.float(), y), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Rescale the backbone gradient to norm <= 5 (as InsightFace). Adam divides by
            # the gradient's running size, so a steady rescale changes nothing; what this
            # stops is one unusually large gradient dominating Adam's running averages.
            # ArcFace's s = 64 makes its gradients large, so it is clipped on most steps.
            grad_norm = torch.nn.utils.clip_grad_norm_(embedder.backbone.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            running_loss += loss.detach()  # .item() every step would stall the GPU

            if step % LOG_EVERY == 0:
                with torch.no_grad():
                    # Same diagnostics for both heads: the angle between each face and
                    # its own identity's weight row (what ArcFace adds the margin to),
                    # and the embedding length (what softmax can inflate instead).
                    cos_y = F.cosine_similarity(emb.float(), head.fc.weight[y], dim=1)
                    theta_y = torch.rad2deg(torch.acos(cos_y.clamp(-1, 1))).mean().item()
                    emb_norm = emb.float().norm(dim=1).mean().item()
                row = [step, epoch, running_loss.item() / LOG_EVERY, theta_y, emb_norm,
                       grad_norm.item(), sched.get_last_lr()[0]]
                log.writerow(row)
                log_file.flush()  # readable while training runs
                speed = LOG_EVERY * BATCH_SIZE / (time.time() - tick)
                print(f"step {step:6d}  epoch {epoch}  loss {row[2]:6.3f}  theta_y {theta_y:5.1f} deg  "
                      f"|emb| {emb_norm:5.1f}  grad {row[5]:5.2f}  {speed:5.0f} img/s", flush=True)
                running_loss.zero_()
                tick = time.time()

            if step == total:
                break

        torch.save(
            {"embedder": embedder.state_dict(), "head": head.state_dict(),
             "args": vars(args), "epoch": epoch},
            out / "checkpoint.pt",
        )
        if step == total:
            break


if __name__ == "__main__":
    main()
