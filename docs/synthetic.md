# Phase 1 — the synthetic engine

There are no real passport images in this project. Everything the recognizer learns
about what a character looks like, it learns here.

```
random_identity  ->  serialize  ->  render_mrz  ->  degrade  ->  extract_line
   (identity.py)     (parser)      (render.py)   (degrade.py)   (dataset.py)
```

`MRZLineDataset` runs that chain per sample, online: nothing is cached, and the epoch is
mixed into the seed so sample *i* of epoch 2 is a different document from sample *i* of
epoch 1. The training set is effectively infinite, which is the whole reason to generate
rather than collect.

Like `parser`, this package does not import torch. Phase 2 wraps `MRZLineDataset` in a
`torch.utils.data.Dataset` in three lines.

## Decisions worth knowing

**The renderer places every glyph itself, on a 2.54mm grid** (10 characters per inch per
ICAO — exactly 30px at 300dpi). The bundled OCR-B claims to be monospaced and is not: `5`
is 782 units against 884 for the other digits and 886 for the letters. Letting PIL lay the
text out would drift off a real passport's grid *and* give `5` a gap no other glyph has —
a cue the model could learn instead of the shape, which would vanish on real documents.

**Severity is one number, not four pipelines.** The blueprint's curriculum (clean → blur →
reflection → heavy) is a magnitude, so every effect scales with `severity` in [0, 1] and
the curriculum is a sweep. `DatasetConfig.severity_range` samples a range rather than a
point, so easy documents stay in the mix and the model does not forget how to read a clean
one.

**Effects are numpy and OpenCV, not Albumentations.** Albumentations draws from its own
global RNG, which would quietly break the per-seed reproducibility everything else here
guarantees. Augraphy's paper and ink simulation is better than ours but costs 100ms–seconds
per sample; it belongs in an offline tier, not on a dataloader hot path where it would
starve the GPU it is meant to feed.

**Effects apply in physical order** — printed, posed, imaged through a lens, lit, sampled
by a sensor, compressed — because that is the order that composes correctly. Glare on top
of noise looks wrong; noise on top of glare looks like a photograph.

**Identity and camera draw from separate RNG streams.** Sharing one made *who the document
belonged to* depend on how many numbers the camera happened to draw first. The camera
stream is keyed to the document (both lines were in one photograph, so they share its
severity and blur); the crop stream is keyed to the line (a detector frames each line
separately).

**Consecutive indices are the two lines of one document.** Seeding from the index directly
would pair line 1 of one passport with line 2 of another, and nothing would catch it —
ICAO puts no check digit on line 1 to tie it to line 2. See `docs/parser.md`.

## Four things found by looking at the output

None of these came from reasoning about the code. They came from rendering it and looking.

**Digits were being clipped.** Ground-truth boxes were sized to the cap height of `H`, but
OCR-B's digits overshoot its capitals by 3px at 300dpi. Since those boxes are what the
recognizer crops with, every digit was losing its top. Boxes now span the measured ink
extent of the whole alphabet — and are uniform, so box height cannot leak a character's
identity.

**A crop showed two lines while the label named one.** An MRZ line is ~44:1, so rotating a
1320px line by 3° drags its ends 34px vertically — most of the 38px line height. The
upright box of one line grows tall enough to contain its neighbour. "Mild rotation" is not
mild at that aspect ratio. Lines are now extracted by mapping their quadrilateral onto a
rectangle, which yields exactly one line and mirrors production, where detection deskews
before recognition sees anything. `extract_line` does this; `DegradeResult.locate_quad`
supplies the corners.

**Resolution loss was missing entirely.** We render 30px characters; a phone photographing
a passport yields 8–12. A model that has only seen crisp glyphs has never seen its actual
input, and no amount of blur substitutes for absent resolution. `resample` throws
resolution away and restores the size.

**Severity 1.0 was far milder than advertised.** Drawing magnitudes uniformly from
`[0, max]` makes the average sample half-strength. `_scaled` now biases the floor upward,
so severity means what it says.

## Speed

About 156 samples/s per core (2.9ms render + 4.0ms degrade), so ~1250/s across 8 workers —
enough to keep a GPU fed. Shadows are built at quarter resolution and scaled up: a soft
shadow has no high-frequency content, and blurring at full size cost 2.8ms of a 4.3ms
budget for an identical result.

If a pod pairs a fast GPU with few vCPUs, lower `dpi` first (it scales everything) before
reaching for a pre-rendered cache.

## One external check, and it passes

Almost every check here grades the generator against its own definition of correct. The
exception is worth its weight: **PassportEye** — a stock MRZ reader built for real
passports, which knows nothing about this code — parses our renders.

| severity | MRZ found | avg valid_score | field accuracy |
| --- | --- | --- | --- |
| 0.0 | 12/12 | 81.1 | 91.7% |
| 0.2 | 12/12 | 92.0 | 98.6% |
| 0.4 | 12/12 | 89.0 | 97.2% |

A reader trained on real documents reading our synthetic ones at 97%+ is real evidence
that the glyph shapes, the 2.54mm pitch and the contrast are passport-like rather than
merely self-consistent. Had it failed on the pristine render, there would have been a
rendering flaw silently capping Phase 2 accuracy that no internal test could ever surface.

Note the shape of that table: **severity 0.0 reads *worse* than 0.2.** The pristine
render — flat 255 paper, uniform ink, no texture — is harder for a real reader than a
mildly degraded one. That is a small, direct confirmation that a flat background is the
tell of a synthetic sample, and a reason not to treat severity 0 as the "realistic" case.

Reproduce it with the optional last cell of the preview notebook. It needs tesseract
(`brew install tesseract`), so it is not part of the test suite.

## The limit that remains

PassportEye tells us the render is legible to something built for real passports. It does
not tell us the *degradations* resemble real photographs — glare, wear and phone optics are
still our invention, and they are most of what severity does.

The gate is your eyes on the contact sheet, and eventually 50–100 real images held out for
measurement only. Until then, Phase 2's accuracy numbers are unvalidated.
