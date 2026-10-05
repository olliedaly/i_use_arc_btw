# i_use_arc_btw

A from-scratch learning project for understanding **ArcFace**.

A DINOv2 ViT-S/14 backbone is fine-tuned on face data twice — once with a plain
softmax classification head, once with an ArcFace (additive angular margin)
head — and both are evaluated on face verification for identities never seen
during training. Everything except the head is held fixed, so any difference
comes from the training signal alone.

The code is built up in small steps, one commit per step, so the history reads
as a log of what was added and why.

## Roadmap

- [x] 0. Environment: PyTorch on an RTX 5060 Ti (Blackwell, `sm_120`)
- [ ] 1. Load DINOv2 and inspect its outputs
- [ ] 2. Embedding neck on top of the backbone
- [ ] 3. Softmax head (baseline)
- [ ] 4. ArcFace head
- [ ] 5. Toy 2D experiment: softmax vs ArcFace feature geometry
- [ ] 6. Face dataset and identity-disjoint split
- [ ] 7. Training loop
- [ ] 8. Verification evaluation on unseen identities
- [ ] 9. Compare softmax vs ArcFace

## Setup

```bash
uv sync
uv run scripts/check_env.py
```
