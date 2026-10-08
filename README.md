# I Use Arc BTW
Exploration of how ArcFace is implementated and a comparison of ArcFace classifier vs softmax on a small training run using DINOv2 S backbone.

<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/9c139b84-78e1-425a-a59c-2315acc393e1" />


## Roadmap

- [x] 0. Environment: PyTorch on an RTX 5060 Ti (Blackwell, `sm_120`)
- [x] 1. Load DINOv2 and inspect its outputs
- [x] 2. Embedding neck on top of the backbone
- [x] 3. Softmax head (baseline)
- [x] 4. ArcFace head
- [x] 5. Toy 2D experiment: softmax vs ArcFace feature geometry
- [ ] 6. Face dataset and identity-disjoint split
- [ ] 7. Training loop
- [ ] 8. Verification evaluation on unseen identities
- [ ] 9. Compare softmax vs ArcFace

## Setup

```bash
uv sync
uv run scripts/check_env.py
```
