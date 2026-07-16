# The web reader

Upload a passport, drag a box around the MRZ, get the fields. Three modules:
`crop` (an image and a box become two line crops), `payload` (a reading becomes
data for the page), `api` (HTTP). Only `api` imports the web framework, and
nothing here imports torch except through `MRZReader`.

## The box is drawn by a person

Phase 3 — the detector that would find the MRZ by itself — is not built, and this
is not a stub standing in until it is.

The synthetic engine draws bare MRZ strips on blank paper. It has never drawn a
passport page: no photograph, no visual inspection zone, no border, no laminate.
A detector trained on that data would learn "find the text on the empty page",
which is not a skill, and would meet its first real passport in production with
no measurement anywhere in between. Hand-tuned morphology over a whole photograph
has the same problem from the other end: it would be confidently wrong on exactly
the real pages there is no way to check it against.

Inside a box a person has already drawn, the problem is different. It is
one-dimensional, the answer is checkable against the renderer's own ground-truth
line boxes, and it is tested that way in `tests/serve/test_crop.py`.

When there are real pages to train and measure a detector on, it belongs here.
Until then, asking is the honest version.

## Why the box is not simply halved

The recognizer resizes its crop anisotropically to exactly 32x704 — 44
characters, 16 pixels each. Framing is therefore part of the input, not a detail
of it: a crop carrying one extra character-width of blank paper shifts every
character out of the cell the model expects. Training drew crops with at most 0.3
line-heights of padding per edge, so the tolerance is roughly half a character.
Nobody can drag to half a character.

Measured on synthetic pages, reading the document from a box drawn `slack` proud
of the true MRZ zone on every side:

| slack | halve the box | find the ink first |
| --- | --- | --- |
| 0% | 100% | 100% |
| 15% | 85% | 100% |
| 30% | 10% | 100% |
| 60% | — | 100% |

(severity 0.3, 40 documents, real checkpoint.) The second column is the whole
reason `crop.py` is more than two lines of slicing. Through the shipped path,
accuracy is 100% up to severity 0.6 and independent of the drag; at severity 1.0
— the deliberately brutal end of the synthetic range — it falls to ~78%.

The mechanism: threshold the selection with Otsu, project ink onto the row axis,
take the two bands, then re-threshold each band alone and take its ink bounding
box. Both axes matter — fixing only the vertical split left accuracy at 20%,
because the horizontal framing is what the fixed-width resize is sensitive to.

Otsu rather than an adaptive threshold, which is not the obvious choice: a local
threshold handles uneven lighting better in general and read *worse* here (72% vs
85% on heavily degraded pages). An MRZ band is mostly paper, and a local window
inside one finds contrast in the grain when no character is nearby to anchor it.

When the ink does not resolve into exactly two bands — a shadow merging the lines,
a thumb on the page — the region is halved and each half tightened on its own. A
worse guess about *where* the lines are, still a good one about how each is framed.

## Clipping is reported, not swallowed

A box that cuts through a line is the one failure the reading cannot report on
its own: the model returns a confident guess at the characters it was shown, and
nothing about the result says the rest were never in the crop. It reads as a
broken model.

So `locate_lines` returns `Line.clipped` — the ink ran to the edge of the
selection — and the page says so above the fields. It is a suspicion, not a
finding: an MRZ genuinely at the edge of a photograph trips it too. That is the
right trade, because the cost of a false warning is a sentence and the cost of a
silent one is a wrong passport number.

This came out of using the page. A default box whose left edge fell inside the
MRZ read `TKMJURTA` for `TKLGUPTA` and offered no hint why. The default box is
now deliberately generous — slack is free, clipping is not, and the two errors
should not be treated as symmetric.

## Verified means checked, not correct

Every field carries what guards it:

| guard | fields | what it catches |
| --- | --- | --- |
| `checksum` | document number, dates, optional data | most misreads |
| `allowlist` | issuing state, nationality, sex, document code | substitutions onto codes that do not exist |
| `none` | surname, given names | nothing |

The names are the point. No check digit covers them, so `ERIKSSQN` is exactly as
valid an MRZ as `ERIKSSON`, and a document with a misread surname validates
perfectly. A page showing one green tick for the document would be making a claim
the standard does not support, and would be most convincing exactly when it was
wrong. So names are `unverified` — always, even when the document is valid.

**A failed composite withdraws line 2's ticks.** The composite digit is a
checksum over every checksummed field on line 2 at once. When it disagrees, the
line is internally inconsistent, and a field's own check digit passing no longer
means much — a single digit catches most single-character errors, not all.

This one is observed, not theorised. The page read document number `1028225<<` as
`13X8225<<`. Both satisfy check digit `8`. Every individual check passed, only the
composite objected, and the field was displayed as verified while being wrong. A
tick beside a wrong number is the one thing this page must not draw.

For the same reason the legend says a check "accepted" a field rather than
"confirmed" it.

## Privacy

The image is decoded in memory and never written to disk, logged, or sent
anywhere. A local model is the only reason that is possible; a demo that quietly
kept the uploads would be teaching the opposite lesson about what this is for.

## The caveat that outlives all of it

Every number here is measured on images the engine drew itself. Blueprint stage 5
— fine-tune on real images — has no data and has never run. None of this is a
claim about real passports.
