# Softmax vs ArcFace

The two models differ only in the training head. Everything else is the same:
DINOv2 ViT-S/14 plus a Linear + BatchNorm neck, CASIA-WebFace (9,072 training
identities), 10 epochs, and the same seed, data order, optimiser and schedule
(`scripts/train.py`). Both are judged on verification of people neither has seen.

## How they are compared

- **Test is used once.** It holds 1,000 identities that were never used for
  training or for any decision. Everything was developed on val first, with
  `scripts/compare.py --eval-split val`.
- **Thresholds come from val.** A deployed system fixes its threshold before
  meeting the people it will see. So each threshold is chosen on val (for
  FAR = 1e-3, 1e-4 and 1e-5, and for best accuracy) and then applied to test,
  which shows whether the promised false-accept rate holds on new people.
- **Test's own ROC** is reported too (TAR at exactly FAR = 1e-3 and so on), so
  it can be compared directly with step 8's val numbers.
- **A paired bootstrap** over test identities puts a 95% interval on
  ArcFace - softmax: resample people with replacement, score both models on the
  same resample, keep the difference, 1,000 times. That covers which people
  happen to be in test, but not training randomness: there is one training run
  per head.
- Faces are capped at 20 per identity and every pair is scored, as in step 8.

## Predictions, written from val before test was used

1. ArcFace beats softmax at TAR@FAR = 1e-3 and 1e-4 by about 10 points, with
   intervals that exclude 0. On val: +9.8 [+9.0, +10.6] and +10.5 [+8.4, +12.2].
2. ArcFace beats softmax on accuracy at val's threshold by about 1 point.
   On val: +1.1 [+0.9, +1.3].
3. At FAR = 1e-5, ArcFace is not clearly better: the difference is negative or its
   interval includes 0. On val: -15.5 [-27.4, +3.2]. The reason is ArcFace's
   cluster of low-quality faces ([low_quality_direction.md](low_quality_direction.md)).
4. Val's thresholds carry over. At FAR = 1e-3 and 1e-4, test's FAR is within a
   factor of 2 of the target. At 1e-5 it may be off by more.
5. Raw DINOv2 stays far below both, under 10% TAR at FAR = 1e-3 (3.4% on val).
