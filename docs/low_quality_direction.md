# ArcFace's low-quality direction

On val, ArcFace beats softmax at every operating point except the strictest one,
TAR@FAR = 1e-5 (step 8, `scripts/evaluate.py`):

| | TAR@FAR 1e-3 | 1e-4 | 1e-5 | best accuracy |
|---|---|---|---|---|
| softmax | 60.4% | 45.4% | **32.3%** | 90.2% |
| ArcFace | **70.2%** | **55.9%** | 16.9% | **91.3%** |

The cause is that ArcFace training learned a direction shared by low-quality
faces: blurred, side-on, looking up, partly hidden. Low-quality faces of
*different* people point along it together, so they match each other. To
ArcFace, an unreadable face isn't "nobody"; it is "the unreadable person".

## Where it shows up

FAR = 1e-5 lets through only 450 of val's 45M impostor pairs, so a handful of
images can decide that column. ArcFace's 450 highest-scoring impostor pairs
involve just **51 images** ("hubs"), each matching many others.

The hubs are low quality by a measure that uses no model at all. Their sharpness
(the variance of the Laplacian of the grey image, which blur suppresses) has a
median of 47, against 301 for the other faces. They are blurred greyscale faces,
profiles, a face looking up under a hat, faces behind flowers or in the glare off
glasses. They look nothing alike, yet ArcFace scores pairs of them at 0.91 to
0.96 cosine.

## Training made the cluster, not DINOv2

Mean cosine between the hubs (pairs of different people only), against any two
different people:

| | hub vs hub | any two different people |
|---|---|---|
| DINOv2, no face training | 0.31 | 0.35 |
| softmax | 0.05 | 0.00 |
| ArcFace | **0.43** | 0.00 |

To the raw backbone, the hubs are no more alike than any two faces. ArcFace
training pulled them together; softmax training barely did.

## It is mostly one direction

To test that without fitting to the faces being scored, val's identities are
split into two halves. A direction is fitted on one half and removed from the
other half's embeddings (project it out, renormalise), then the held-out half is
scored. Then the halves swap. Two directions are tried:

- **blur direction**: the average embedding of the blurriest 5% of faces (by
  sharpness), minus the average of all faces. It knows nothing about hubs.
- **hub direction**: the average embedding of the fitting half's own hub images.

TAR@FAR = 1e-5 on the held-out half:

| held out | as is | blur direction removed | hub direction removed |
|---|---|---|---|
| half 1 | 15.6% | 25.3% | **40.2%** |
| half 0 | 18.5% | 32.8% | **42.2%** |

- Even the blur direction, which knows nothing about hubs, finds them. The
  held-out hubs lean along it (cosine 0.20 to 0.21, against -0.02 for other
  faces), and blurrier faces lean along it more (rank correlation with sharpness
  -0.21 to -0.29).
- The hub direction works better because the hubs aren't only blurred: pose and
  occlusion count too. The two directions overlap only partly (cosine 0.44 to 0.55).
- Removing either direction costs nothing elsewhere. TAR@FAR 1e-3 is within
  0.1 points, and 1e-4 rises by up to 2 points.

## Explanations that don't hold

Two mechanisms were tested first and ruled out.

- **"Pushed away from every identity"**, the drift trap
  ([arcface_trap.md](arcface_trap.md)) acting on faces that can't be pulled in.
  If so, the hubs would sit opposite the average identity row of the head. They
  don't: the hub direction has cosine 0.13 with that direction. The hubs' mean
  angle to the 9,072 training identities is 90.1 deg, against 89.9 deg for other
  faces. Removing the direction leaves TAR@FAR 1e-5 at 15.2%.
- **"Collapsed onto BatchNorm's shift"**. The neck ends in BatchNorm,
  output = gamma * normalised input + beta. A face with nothing distinctive
  would give an input near the average and land on beta. But beta is tiny
  (length 1.0, against 12.2 for a typical embedding). The hubs' face-specific
  part, |embedding - beta|, is *larger* than typical (13.5 against 12.2).
  Subtracting beta moves TAR@FAR 1e-5 only from 16.9% to 19.0%.

Why ArcFace builds this direction and softmax barely does is still open.

## Why it matters

- A system that compares every face it's given will match unreadable faces of
  different people. That's what face quality assessment is for: it rejects
  unreadable faces before any comparison. MagFace and AdaFace build a quality
  signal into the training loss itself.
- In this repo, the FAR = 1e-5 column on val is decided by about 50 images, so
  treat it with care (a comment in `scripts/evaluate.py` says the same).
  Removing the direction is a fix fitted to val, not part of ArcFace, and
  nothing in the evaluation uses it.

## Limits

- Each half has about 11M impostor pairs, so FAR = 1e-5 there is about 110
  pairs and noisy. The effect is large and shows up both ways round.
- Sharpness only measures blur, not pose or occlusion.
- Everything is on val, with one training run per head.

## Reproduce

```bash
uv run scripts/low_quality_direction.py
```

It takes about 10 seconds. It also writes `outputs/low_quality_pairs.png`,
ArcFace's 12 highest-scoring pairs of different people. The image isn't
committed, because the faces are real people from a dataset licensed for
non-commercial research only.
