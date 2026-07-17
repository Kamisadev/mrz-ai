# mrz-ai

TD3 passport MRZ recognition (ICAO 9303, two lines of 44 characters), trained
entirely on synthetic data and targeting <100ms CPU inference.

Scope is deliberately narrow: TD3 only, no TD1/TD2/visa. The fixed geometry means
every field is a constant slice and the charset is 37 characters.

## Status

| Phase | What | State |
| --- | --- | --- |
| 0 | ICAO engine — serialize / parse / validate | **done** |
| 1 | Synthetic engine — OCR-B render + degradation | **done** |
| 2 | Recognition — ViT-tiny encoder + fixed-length head | trained, **needs retraining**: the first run saw one font |
| 3 | Detection — MRZ zone crop | not started; the user draws the box instead |
| 4 | Candidate decoder — exact k-best × ICAO validation | **done** |
| 5 | Export — ONNX / INT8, CLI | |
| — | Web reader — upload, select, read | **done** (`mrz_ai.serve`) |
| — | Training dashboard — progress, health, real set | **done** (`mrz_ai.serve.dashboard`) |

## Layout

```text
src/mrz_ai/
├── parser/       # ICAO 9303 TD3. Pure Python, no ML deps.
├── synthetic/    # identity -> render -> degrade -> line crops. No torch.
├── recognition/  # 32x704 -> (44, 37) logits. ViT-tiny.
├── detection/    # phase 3
├── inference/    # phase 4 — exact k-best, ICAO ranking, MRZReader
├── evaluation/   # the real set — the one grader that is not the generator
├── serve/        # the web reader, and the training dashboard
└── training/     # entrypoints the notebooks call
notebooks/        # one self-contained cell per training job (RunPod)
configs/
docs/
tests/
```

## Training

Training runs on RunPod. Each notebook in `notebooks/` is a single
self-contained cell: it installs dependencies, imports the package and calls one
entrypoint from `mrz_ai.training`. All the real logic lives in the package, so
the cell stays paste-and-run and the same code path is testable locally.

## Reading a passport

```bash
uv pip install -e ".[serve]"
.venv/bin/python -m uvicorn mrz_ai.serve.api:app --port 8000
```

Open `http://localhost:8000`, drop in a passport image, drag a box around both
MRZ lines, read. The image is decoded in memory and never written to disk,
logged, or sent anywhere — a local model is the only reason that is possible, and
a passport is not a file to be casual with.

Point it at a different checkpoint with `MRZReader.from_checkpoint`; the default
is `recognition_model/recognition.pt`.

**There is no detection stage, so you draw the box.** That is not laziness — the
synthetic engine only ever draws bare MRZ strips, never a whole passport page, so
a detector trained on it would learn to find text on blank paper and would meet
its first real page in production. Inside a box you have drawn, the problem is
small enough to solve honestly, and `serve/crop.py` solves it: it finds the two
lines by their ink and reframes them to what the recognizer was trained on.

That reframing is load-bearing, not a nicety. The recognizer resizes its crop to
a fixed 32x704, so blank paper inside the crop shifts every character out of the
cell the model expects. Measured on synthetic pages, halving a box drawn 30%
loose reads **10%** of documents correctly; finding the ink first reads **100%**,
and stops depending on how carefully anyone dragged. The box being loose is free.
The box clipping a character is not — so the page says when it thinks that
happened, rather than quietly returning a confident misreading.

The box being *tilted* used to cost everything, and now costs nothing. ICAO sets
the two lines 4.23mm apart with a cap height of 3.2mm: 1.03mm of paper separates
them, across a line 111.76mm long. Past `asin(1.03/111.76)` = **0.53 degrees** the
far end of line 1 sinks into the rows of line 2, and no row projection can tell
them apart again. Measured, a clean render read 100% of documents level and **0%
at 5 degrees** — and the two-band search had actually stopped working at 0.75,
with everything in between being the halving fallback getting lucky and the
recognizer's 1.5 degrees of trained skew tolerance absorbing the rest. A
photograph taken by hand is not level, so `serve/crop.py` now measures the skew
and rotates it out first: **100% at 8 degrees**, 24ms on a 12MP photo.

Note where that hole came from. The training pipeline warps a quadrilateral onto
a rectangle, so it was never sensitive to tilt — but the quad came from ground
truth. Serving had no quad and no detector to produce one, so it passed an
axis-aligned box to code that assumed level text, and the assumption was never
written down anywhere to be doubted.

## The known risk

There are no real passport images in this project. Everything the model learns
comes from the synthetic engine, so the synthetic-to-real gap is the dominant
risk and almost no internal metric can measure it — synthetic evaluation mostly
grades the generator against itself.

It is not hypothetical. The first trained checkpoint scored 99.5% line accuracy
at full severity and misread real passports badly. The cause was one line of the
generator nobody had thought of as a parameter: **it used a single font.**

"OCR-B" is not one typeface. ICAO 9303 names a face; the cuts of it differ, and
a model trained on exactly one reads another at **72% of documents on a clean,
undegraded render** — confusing `0` with `O`, `J` with `U`, `G` with `S`. It had
not learned what a `0` is. It had learned what one vendor's `0` is, and a real
passport is printed with whichever cut its issuer bought. `tools/font_gap.py` is
that experiment, kept so it can be rerun.

The generator now randomizes the *printing* — the cut, and the weight of the ink
— not just the photographing. Both are drawn independently of severity, since a
crisp scan of a heavily-inked passport is a clean sample and not a degraded one.
Note the shape of the miss: everything about the camera was randomized carefully
and nothing about the press was, because the camera is what "degradation" sounds
like it means.

Two cuts is not many, and this is not declared fixed. A third cut would be worth
having — and having put both of ours into training, we no longer hold one out to
measure generalization with. The remaining honest test is a real passport.

One external check does exist and passes: PassportEye, a stock MRZ reader built
for real passports, reads our synthetic renders at 12/12 detection and ~97% field
accuracy (`docs/synthetic.md`). That says the glyphs and pitch are passport-like.
It does not say our glare, wear and blur resemble real photographs.

Two mitigations: over-randomize the domain rather than under, and acquire a
held-out set of 50–100 real images for measurement only, never for training.
Until that set exists, treat every accuracy number as unvalidated.

That set goes in `real/`, which git ignores as a whole directory — this
repository is public, and a test fails if anything under it is ever tracked. See
`docs/real.md` for what it needs to be, and for why training on it would destroy
the only thing it is good for.

```bash
.venv/bin/python tools/measure_real.py
```

Specimen passports are what belongs there: real printing, real typeface, invented
identities. The issuer's choice of OCR-B cut is the axis that broke the last
checkpoint, and a specimen carries it without carrying anybody's passport number.

Give each data page a `box` in `truth.json`. `default_box` guesses one by
arithmetic — ICAO fixes the TD3 page at 125x88mm and puts the zone 74.07mm down it,
so the MRZ is the bottom 16% of any data page — and on a real passport that guess
is not enough. The region does contain the MRZ. It also contains the authority
line and the holder's signature, which sit above the MRZ on every issued document,
so the ink resolves into three or four bands and the two-band search halves the
region instead. Measured on 15 specimen data pages: **0 of 15 crop**. An
already-cropped MRZ strip needs nothing.

That number was 4 of 4 on synthetic pages, which is how the default came to be
believed. The synthetic pages were wrong: the generator draws bare strips on blank
paper, so its "full page" has nothing above the MRZ to be confused by. The default
was validated against the one input that could not falsify it — the generator
grading its own exam, one level up from the fonts.

The failure is loud, which is the part that worked. Handed something with ink all
over it, `crop.py`'s two-band search gives up and halves the region, and a whole
page passed in as-is reads **0 of 4 documents at 8.8% of characters** — a cropping
failure in a recognizer's clothes. That is exactly the confusion this set exists to
prevent, so it is counted (`not_located`) and reported before any score. It is what
caught this.

## Watching a run

```bash
.venv/bin/python -m mrz_ai.serve.dashboard --dir checkpoints/recognition --port 8080
```

Progress, health, the synthetic curve, and — where a real set exists — how many
documents read exactly, with the confusion pairs. It reads a `status.json` the
trainer writes and can do nothing else: a 45-minute rented pod is no place to put
a web server inside the training process, so if the dashboard dies the run does
not notice, and there is no route that writes.

Two panels because they answer different questions. The synthetic one says the
run is learning. Only the real one says a passport will read, and every look at
it is a decision made on the test set — `docs/real.md` is explicit that this makes
it a dev set. The dashboard's loudest state is neither: it is the banner that
appears when the run has one cut of OCR-B, because that run is worthless and its
loss curve looks perfect.

A run that stops writing is reported stale rather than left showing its last step
forever. A pod that is OOM-killed runs no handler, so the clock is the only honest
signal there is.

## Training on RunPod

This job is **CPU-bound, not GPU-bound**: the model is 3.4M parameters, but every sample is
rendered and degraded from scratch on the CPU. Rent vCPUs, not a big GPU — an A100 would
need ~30 cores to keep it fed and no pod offers that pairing.

Roughly **45 minutes on a 16-vCPU pod**, ~20 on a 32-vCPU one. See `docs/runpod.md`.

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
.venv/bin/pytest tests -q
```

904 tests, `mypy --strict` clean.

See `docs/parser.md` for the ICAO engine's decisions and its two validation blind spots,
and `docs/synthetic.md` for the generator's — including four bugs that only showed up by
rendering the output and looking at it.

Run `notebooks/01_synthetic_preview.ipynb` before any training: it checks the labels,
charset coverage, throughput and reproducibility, and prints a contact sheet across the
severity ramp. Your eyes on that sheet are the only real check that exists.

## License

MIT — see `LICENSE`. Neither bundled font is covered by it, and they are not under
the same terms as each other:

- `OCR-B.ttf` / `OCR-B.otf` — SIL OFL 1.1, see `assets/fonts/OCR-B-LICENSE.md`.
- `OCRB.ttf` — Skala's conversion of Schwarz's Metafont OCR-B, see
  `assets/fonts/README.md`. Freely distributable; read its provenance section
  before shipping commercially.
