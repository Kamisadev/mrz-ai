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
| 2 | Recognition — ViT-tiny encoder + fixed-length head | **trained** — 99.5% line accuracy at full severity |
| 3 | Detection — MRZ zone crop | not started; the user draws the box instead |
| 4 | Candidate decoder — exact k-best × ICAO validation | **done** |
| 5 | Export — ONNX / INT8, CLI | |
| — | Web reader — upload, select, read | **done** (`mrz_ai.serve`) |

## Layout

```text
src/mrz_ai/
├── parser/       # ICAO 9303 TD3. Pure Python, no ML deps.
├── synthetic/    # identity -> render -> degrade -> line crops. No torch.
├── recognition/  # 32x704 -> (44, 37) logits. ViT-tiny.
├── detection/    # phase 3
├── inference/    # phase 4 — exact k-best, ICAO ranking, MRZReader
├── serve/        # the web reader: crop -> read -> page
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

## The known risk

There are no real passport images in this project. Everything the model learns
comes from the synthetic engine, so the synthetic-to-real gap is the dominant
risk and almost no internal metric can measure it — synthetic evaluation mostly
grades the generator against itself.

One external check does exist and passes: PassportEye, a stock MRZ reader built
for real passports, reads our synthetic renders at 12/12 detection and ~97% field
accuracy (`docs/synthetic.md`). That says the glyphs and pitch are passport-like.
It does not say our glare, wear and blur resemble real photographs.

Two mitigations: over-randomize the domain rather than under, and acquire a
held-out set of 50–100 real images for measurement only, never for training.
Until that set exists, treat every accuracy number as unvalidated.

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

847 tests, `mypy --strict` clean.

See `docs/parser.md` for the ICAO engine's decisions and its two validation blind spots,
and `docs/synthetic.md` for the generator's — including four bugs that only showed up by
rendering the output and looking at it.

Run `notebooks/01_synthetic_preview.ipynb` before any training: it checks the labels,
charset coverage, throughput and reproducibility, and prints a contact sheet across the
severity ramp. Your eyes on that sheet are the only real check that exists.

## License

MIT — see `LICENSE`. The bundled OCR-B font is SIL OFL 1.1 and is not covered by it;
see `assets/fonts/OCR-B-LICENSE.md`.
