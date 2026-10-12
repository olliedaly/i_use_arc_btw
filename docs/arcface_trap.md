# The ArcFace drift trap

With ArcFace's usual margin (m = 0.5), training can start by pushing every face
*away from every identity at once*, instead of pulling each face towards its own
identity. The loss falls, so it looks like progress, but the embedding collapses:
faces of different people end up pointing the same way.

In this repo it happened in a 600-step test of `scripts/train.py`. The full
10-epoch run escaped, and that run is the one evaluated from step 8 on.

## Where it comes from

For each face, ArcFace scores its own identity y and every other identity j as

```
own identity:     s * cos(theta_y + m)
other identities: s * cos(theta_j)
```

and the loss only depends on the *gap* between the own score and the others.
There are two ways to widen that gap:

- **pull**: move the face towards its own identity, so theta_y falls while the
  others stay put. This is the intended route.
- **push**: move the face away from every identity at once, so all angles grow together.

The push pays off because cosine is steepest at 90 deg and flattens towards 180 deg.
In a 384-D embedding every angle starts near 90 deg: random directions in d
dimensions are 90 deg apart, give or take about 1/sqrt(d) radians (about 3 deg here).
So the other identities sit on the steep part of the curve, while the own
identity is scored at 90 + 28.6 = 118.6 deg, where the curve is already flatter.
Push every angle 10 deg further (s = 64):

| | score before | 10 deg further | change |
|---|---|---|---|
| each other identity (90 -> 100 deg) | 0 | -11.1 | -11.1 |
| own identity (scored at 118.6 -> 128.6 deg) | -30.7 | -40.0 | -9.3 |

The gap grew by 1.8 without the face getting any closer to its own identity.
Pushing apart lowers the loss whenever the own score falls more slowly than the
others, sin(theta + m) < sin(theta). That holds above (pi - m) / 2 = **75.7 deg**.
Below it, pushing apart costs more than it gains.

Loss along the push path, with every angle equal (the real `ArcFaceHead`, 9,072
identities as in training):

| every angle at | 60 | 75.7 | 90 | 105 | 120 | 150 | 151.3 | beyond 151.35 |
|---|---|---|---|---|---|---|---|---|
| m = 0.5 | 39.6 | **40.8** | 39.8 | 36.7 | 31.8 | 17.7 | **17.0** | 24.5 |
| m = 0 | 9.1 | 9.1 | 9.1 | 9.1 | 9.1 | 9.1 | 9.1 | 9.1 |

- Without a margin there is no trap: the loss is flat at ln(9,072) = 9.1, so
  pushing everything apart changes nothing.
- With m = 0.5, the path runs downhill from 75.7 deg to 151.35 deg, where the own
  identity is scored at 180 deg. Past that, the head's fallback takes over (a
  constant penalty, as in InsightFace), and the loss steps up to 24.5 and stays flat.
- Training starts at about 40, so the bottom of the trap, 17.0, looks like great progress.

To push every face away from every identity, the network doesn't need to learn
anything about faces. It moves all faces towards one shared direction and the
identity rows to the opposite side. That is one move shared by all 9,072
identities, against 9,072 separate pulls.

## What it looks like

`scripts/trap_variants.py baseline` reruns the test that fell in: `train.py
--max-steps 600`. That squeezes train.py's schedule into 600 steps, so a 60-step
warm-up and a learning rate back at 0 by step 600. Over the last 100 steps:

- theta_y has risen from 89.5 deg to 95.2 deg, and the other identities moved even
  further away, to 97.3 deg.
- Faces of different people have an average cosine of 0.70. They started at 0.00,
  so they are collapsing to one direction.
- The average face and the average identity row point opposite ways (cosine -0.96).
  The script prints this cosine for every variant, but it only means something
  when faces have collapsed. When they are spread out, the average face is close
  to zero length, so its direction says little.

The full run (`train.py --head arcface`, 10 epochs, 1,648-step warm-up) went the
other way. theta_y fell from the start, was first logged below 75.7 deg at step
1,300 (74.5 deg), and finished at 36.2 deg. Why the longer warm-up tipped it is not
established: warm-up length hasn't been tested on its own.

## Fewer dimensions don't fix it

It is tempting to think a smaller embedding leaves less room to drift. It doesn't.

- The push needs no spare room. It only needs all faces on one side and all
  identity rows on the other, and that fits in any number of dimensions, even 2.
- The dimension count only changes how widely the starting angles spread: about
  3 deg either side of 90 in 384-D, 7 deg in 64-D. Either way almost every face
  starts above 75.7 deg.

A 64-D embedding fell in too, and collapsed further (table below).

What does fence the angles in is *where the identity rows sit*. Take a unit face x
and unit identity rows w_j. If the rows average to zero, then
sum_j cos(theta_j) = x . sum_j w_j = 0 for every face. The face's average cosine to
the identities is then exactly 0, so pushing some angles above 90 deg forces
others below. That is the "centred rows" variant below.

## Changes tested

Each row is the 600-step test with one change, with everything else as in
train.py. Angles are averaged over the first and last 100 steps. "Face-face" is
the average cosine between faces of different people in a batch: 0 means
spread out, 1 means collapsed.

| change | theta_y | other identities (end) | face-face (end) | |
|---|---|---|---|---|
| none (m = 0.5) | 89.5 -> 95.2 | 97.3 | 0.70 | falls in |
| 64-D embedding | 89.9 -> 91.6 | 92.2 | 0.87 | falls in |
| m = 0.2 | 89.3 -> 85.1 | 90.2 | 0.03 | pulls |
| CosFace margin, m = 0.4 | 89.3 -> 85.3 | 90.2 | 0.03 | pulls |
| centred identity rows | 89.3 -> 86.2 | 90.0 | 0.02 | pulls |
| easy margin | 89.4 -> 87.6 | 90.7 | 0.08 | pulls, slower |
| margin ramp, 0 -> 0.5 over 300 steps | 89.5 -> 87.4 | 90.6 | 0.06 | pulls, slower |

- **m = 0.2**: the trap is still there (from 84.3 deg up) but much weaker. At 90 deg,
  pushing apart gains at a rate proportional to 1 - cos(m): 0.02 for m = 0.2,
  against 0.12 for m = 0.5. The cost is a smaller margin for all of training, and
  the margin is what sets the empty zone between identities.
- **CosFace**: the own identity's score is s * (cos(theta_y) - m). The same amount
  is taken off at every angle, so pushing apart lowers the own and other scores
  equally, and there is no trap zone at all. It is the cleanest fix, but it is a
  different loss from the one this repo studies. InsightFace's ViT configs use
  CosFace with m = 0.4; whether this trap is the reason isn't known here.
- **Centred identity rows**: the head subtracts the average row before
  normalising, so the rows can't all move to one side. This is an improvised fix,
  not one from the papers.
- **Easy margin**: no margin while a face is more than 90 deg from its own identity
  (`easy_margin` in [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py)).
  Faces closer than 90 deg get the full m = 0.5.
- **Margin ramp**: m grows linearly from 0 to 0.5 over the first 300 steps. In this
  test the learning rate is already decaying by the time m reaches 0.5, so it
  doesn't show whether the ramp holds up at full learning rate.

Limits of the test:

- 600 steps is about 4% of a full run, so this only tests the start. Every
  variant is still above 75.7 deg at step 600. They are heading the right way,
  not finished, and whether they end as well as the full run is untested.
- There is one run (one seed) per variant.
- Don't compare loss values between variants: each margin scores faces differently.

## What this repo does

Nothing changed. train.py's warm-up (10% of steps) escaped in the full run. The
signs of the trap are theta_y rising during warm-up and faces of different people
becoming similar. If they show up, use the margin ramp or the easy margin: both
keep m = 0.5 once faces are near their own identity, so the loss is still ArcFace.

Reproduce the table (about 12 minutes on an RTX 5060 Ti):

```bash
for v in baseline dim64 m0.2 cosface centred easy ramp; do uv run scripts/trap_variants.py $v; done
```
