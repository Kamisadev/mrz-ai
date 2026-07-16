# Phase 2 — the recognizer

Reads one MRZ line. **This is not stock PARSeq**, and every deviation was forced by a
measurement rather than chosen by taste.

## The input contract, and why nothing about it is a default

Stock PARSeq is built for scene text: 32×128 images (4:1), at most 25 characters, a
36-character lowercase alphabet. A TD3 line is 44 characters at roughly 24:1 over a
37-character alphabet. **Every axis of the stock config is wrong for us**, and the width is
not close — 44 characters in 128 pixels leaves 2.9 pixels each, which is not legible to
anything.

| | stock PARSeq | here |
| --- | --- | --- |
| input | 32×128 (4:1) | **32×704** (22:1) |
| patch | (4, 8) → 128 tokens | **(8, 8)** → 352 tokens |
| px/char | 2.9 | **16** |
| encoder | ViT-Small: 384-dim, 12 layers | **ViT-tiny: 192-dim, 6 layers** |
| decoder | autoregressive, 25 steps | **one shot, 44 positions** |
| label length | 25, with [EOS] | **44, fixed** |
| classes | 36 | **37** (`A-Z`, `0-9`, `<`) |

**The width follows from the alphabet**: 44 × 16 = 704, so a character is exactly 16 pixels
and exactly two patches. One patch per character is cheaper (176 tokens, 4.4ms) but the
crop jitter a real detector produces would misalign the grid by up to half a character and
smear every glyph across a patch boundary. Two patches per character absorbs that.

**The depth follows from the latency target.** Measured, single line, CPU, FP32:

| encoder | tokens | ms/line | two lines |
| --- | --- | --- | --- |
| ViT-Small @ 32×704, patch (4,8) | 704 | 75.5 | **151ms — over budget before detection runs** |
| ViT-tiny @ 32×704, patch (4,8) | 704 | 18.1 | 36ms |
| **ViT-tiny @ 32×704, patch (8,8)** | 352 | 7.6 | **15ms** |
| full model incl. decoder + head | 352 | **10.1** | **20ms** |

The blueprint targets under 100ms on CPU for the *whole pipeline*, covering detection plus
two line reads. Stock PARSeq's encoder would spend 151ms on the lines alone, so ViT-Small
is not an option here however well it reads. The full model at 20ms leaves ~80ms for
detection, before INT8.

## Why the decoder predicts all 44 characters at once

The blueprint says PARSeq; PARSeq is autoregressive. We are not, for three reasons in
descending order of weight:

1. **It emits what Phase 4 consumes.** The candidate decoder's plan is "top-K per position
   → ICAO validate". Independent positions give 44 marginals directly — that *is* the
   input. Autoregression gives a conditional factorization where top-K at position *t*
   depends on what was chosen before it: the wrong output *shape* for the pipeline, not
   merely a slower one.
2. **The context autoregression exists to learn is not there.** PARSeq's permutation
   language modelling buys an implicit language model. MRZ names are arbitrary
   transliterations and document numbers arbitrary alphanumerics — no prior to learn. The
   one real cross-position structure is the check digits, and Phase 4 enforces those
   *exactly* rather than approximately in weights.
3. **CPU budget.** Forty-four sequential decoder passes against one.

The blueprint asks for an "MRZ-aware decoder" with "top-K hypotheses". A fixed-length head
emitting per-position marginals reads that more faithfully than stock PARSeq would.

**Output contract, which Phase 4 depends on:** `(batch, 44, 37)` logits, one distribution
per character position, reading order. Pinned by test.

## Smaller decisions that carry weight

**The decoder uses learned position queries, not column pooling.** Pooling the 88 token
columns down to 44 and classifying is tempting — the pitch is fixed, after all. It is
wrong: the crop jitter, offset, residual skew and the 11–39% horizontal squeeze from
resizing a variable-aspect crop to a fixed 704 all mean the 44 characters do *not* land at
predictable pixels. Pooling would assume exactly the alignment the data pipeline
deliberately breaks. The queries find their characters by attention.

**No [EOS], [PAD] or [BOS] class.** A TD3 line is always 44 characters and `<` is a real
character, not padding. An [EOS] class would hand the model a way to emit a line that
cannot exist.

**Cross-entropy per position; CTC is deliberately dropped.** CTC exists to solve unknown
alignment and unknown length. We engineered both away — the length is fixed at 44 and the
queries handle alignment. It was a blueprint option, not a requirement.

**The resize is anisotropic.** Preserving aspect and padding would put characters at a
different scale in every sample depending only on how loosely the detector framed them.

## Evidence it works

The architecture **fully overfits 8 samples**: 100% character accuracy, 8/8 lines exact by
step 400, loss 0.0027 at step 700. At 200 steps it was dropping characters (`YAN<` →
`YA<<`), which is the failure mode a non-autoregressive head is supposed to be prone to —
but it was undertraining, not an alignment bug. The capacity and the plumbing are sound.

That is a statement about the architecture, not about accuracy. Real numbers need the full
curriculum on a GPU.

## Reading the training output

`char` is per-character accuracy; `line` is the fraction of lines correct in **all 44**
positions. Watch `line`. Per-character accuracy flatters badly at this length: 99.5% per
character is only ~80% of lines correct, and one wrong character is a wrong document.

Validation uses a disjoint generator seed at full severity, so there is no leakage — but it
is still synthetic grading synthetic. The blueprint's fifth curriculum stage is "fine-tune
on real images"; it is absent because there are no real images, and nothing here
substitutes for it.
