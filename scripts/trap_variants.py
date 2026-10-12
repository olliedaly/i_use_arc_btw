"""The ArcFace drift trap, and changes that avoid it (see docs/arcface_trap.md).

Reruns the setting that fell into the trap - a 600-step ArcFace run with
train.py's schedule squeezed to fit, so a 60-step warm-up - with one change each.
Everything else is train.py unchanged: same data order, init and optimiser.

Run with:  uv run scripts/trap_variants.py <variant>
           (baseline, dim64, m0.2, easy, ramp, centred, cosface)
Writes outputs/runs/trap_<variant>/ and prints one RESULT line.
"""

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
import train  # noqa: E402

from i_use_arc_btw.heads import ArcFaceHead  # noqa: E402

VARIANTS = ["baseline", "dim64", "m0.2", "easy", "ramp", "centred", "cosface"]
STEPS = 600
RAMP_STEPS = 300  # "ramp": the margin grows from 0 to m over the first half of the run


class VariantHead(ArcFaceHead):
    """ArcFaceHead.forward with the variant's change, plus per-step diagnostics."""

    variant = "baseline"
    stats: list[torch.Tensor] = []

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.fc.weight, dim=1)
        if self.variant == "centred":
            # Subtract the average identity row, so the rows can't all move to one side.
            w = F.normalize(w - w.mean(0), dim=1)
        e = F.normalize(emb, dim=1)
        cos = F.linear(e, w)
        cos_y = cos.gather(1, labels[:, None])
        theta_y = torch.acos(cos_y.clamp(-1 + 1e-7, 1 - 1e-7))

        m = self.m
        if self.variant == "ramp":
            self.calls = getattr(self, "calls", 0) + 1
            m = self.m * min(1.0, self.calls / RAMP_STEPS)
        if self.variant == "cosface":
            target = cos_y - m  # same penalty at every angle, so no trap zone
        else:
            target = torch.cos(theta_y + m)
            target = torch.where(theta_y + m < math.pi, target, cos_y - m * math.sin(m))
            if self.variant == "easy":
                # No margin while the face is more than 90 deg from its own centre.
                target = torch.where(cos_y > 0, target, cos_y)

        with torch.no_grad():
            other = torch.ones_like(cos, dtype=torch.bool).scatter(1, labels[:, None], False)
            different = labels[:, None] != labels[None]
            self.stats.append(torch.stack([
                torch.rad2deg(theta_y).mean(),                            # own centre
                torch.rad2deg(torch.acos(cos.clamp(-1, 1)))[other].mean(),  # other centres
                (e @ e.T)[different].mean(),                    # faces of different people
                F.cosine_similarity(e.mean(0), w.mean(0), dim=0),  # average face vs average row
            ]))
        return self.s * cos.scatter(1, labels[:, None], target)


def main():
    variant = sys.argv[1]
    if variant not in VARIANTS:
        sys.exit(f"variant must be one of {VARIANTS}")
    VariantHead.variant = variant
    train.ArcFaceHead = VariantHead  # train.py builds whatever this name points to

    argv = ["--head", "arcface", "--max-steps", str(STEPS), "--name", f"trap_{variant}"]
    if variant == "dim64":
        train.EMBED_DIM = 64
    if variant == "m0.2":
        argv += ["--m", "0.2"]
    if variant == "cosface":
        argv += ["--m", "0.4"]  # InsightFace's CosFace margin
    sys.argv = ["train.py", *argv]
    train.main()

    s = torch.stack(VariantHead.stats).cpu()
    last = s[-100:].mean(0)
    print(f"RESULT {variant:8s}  theta_y per 100 steps: "
          + " ".join(f"{t:5.1f}" for t in s[:, 0].view(-1, 100).mean(1).tolist())
          + f"  | last 100 steps: own {last[0]:5.1f}  others {last[1]:5.1f}  "
          f"face-face {last[2]:5.2f}  face-row {last[3]:5.2f}")


if __name__ == "__main__":
    main()
