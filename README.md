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
| 2 | Recognition — PARSeq, 37-char head | next |
| 3 | Detection — MRZ zone crop | |
| 4 | Candidate decoder — beam search × ICAO validation | |
| 5 | Export — ONNX / INT8, API, CLI | |

## Layout

```text
src/mrz_ai/
├── parser/       # ICAO 9303 TD3. Pure Python, no ML deps.
├── synthetic/    # identity -> render -> degrade -> line crops. No torch.
├── recognition/  # phase 2
├── detection/    # phase 3
├── inference/    # phase 4
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

## The known risk

There are no real passport images in this project. Everything the model learns
comes from the synthetic engine, so the synthetic-to-real gap is the dominant
risk and no internal metric can measure it — synthetic evaluation only grades
the generator against itself.

Two mitigations: over-randomize the domain rather than under, and acquire a
held-out set of 50–100 real images for measurement only, never for training.
Until that set exists, treat every accuracy number as unvalidated.

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
.venv/bin/pytest tests -q
```

663 tests, `mypy --strict` clean.

See `docs/parser.md` for the ICAO engine's decisions and its two validation blind spots,
and `docs/synthetic.md` for the generator's — including four bugs that only showed up by
rendering the output and looking at it.

Run `notebooks/01_synthetic_preview.ipynb` before any training: it checks the labels,
charset coverage, throughput and reproducibility, and prints a contact sheet across the
severity ramp. Your eyes on that sheet are the only real check that exists.
