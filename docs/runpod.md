# Training on RunPod

## The one thing that decides your pod

**This job is CPU-bound, not GPU-bound.** The recognizer is 3.4M parameters — trivial for
any modern GPU — while every training sample is drawn, rendered and degraded from scratch
on the CPU. Renting a bigger GPU buys nothing; it will sit idle waiting for data.

Measured generator throughput, per core. `dpi` is the knob, and it is set by a hard
constraint rather than taste: every crop is normalized to 32px tall, so a line rendered
shorter than that gets **upscaled**, and an upscaled sample is blurry however clean its
severity claims to be. A line's ink is ~3.5mm, which only reaches 32px above ~232dpi.

| dpi | source line height | resize | samples/s/core | verdict |
| --- | --- | --- | --- | --- |
| 150 | 21px | **1.52× upscale** | 181 | model never sees a sharp glyph |
| 200 | 28px | 1.14× upscale | 153 | still upscaling |
| **250** | **35px** | **0.91× downscale** | **118** | **default: cheapest dpi that invents nothing** |
| 300 | 41px | 0.78× | 80 | correct, 30% more CPU for nothing |

Width is *not* the constraint — the crop's aspect ratio fixes it near 758px whatever the
dpi. Only height matters.

Cores needed to keep a GPU busy at batch 128, dpi 250:

| GPU speed | samples/s needed | cores needed |
| --- | --- | --- |
| 10 it/s | 1,280 | 11 |
| 20 it/s | 2,560 | 22 |
| 40 it/s | 5,120 | 43 |
| 60 it/s | 7,680 | 65 |

A 3.4M-parameter model runs 40–60+ it/s on an A100, which would need ~30–40 cores to feed.
No pod pairs an A100 with 40 vCPUs, so **an A100 would spend most of its life waiting.**

## What to rent

**Pick the pod with the most vCPUs, not the best GPU.**

| | recommendation |
| --- | --- |
| GPU | RTX A4000 / A5000 / 3090 — anything ≥8GB. The model is 3.4M params. |
| vCPU | **16 or more.** This is the number that sets your wall-clock. |
| RAM | 16GB+ |
| disk | 20GB (the container, plus checkpoints of ~14MB) |
| template | RunPod PyTorch 2.x |

## How long, and what it costs

38,000 steps × batch 128 = **4.86M samples**. Wall-clock is set by the generator:

| vCPUs | samples/s | effective it/s | wall-clock |
| --- | --- | --- | --- |
| 8 | 944 | 7.4 | ~86 min |
| **16** | **1,888** | **14.8** | **~43 min** |
| 32 | 3,776 | 30 | ~21 min |
| 64 | 7,552 | 59 | ~11 min (the GPU may finally be the limit) |

So: **roughly 45 minutes on a 16-vCPU pod**, or ~20 on a 32-vCPU one. Cost is small either
way — well under a dollar. Spend it on cores, not on the GPU.

## Running it

```bash
# on the pod
git clone https://github.com/Kamisadev/mrz-ai.git
cd mrz-ai
pip install -q pillow numpy opencv-python-headless matplotlib
```

Then open the notebooks in JupyterLab and run the single cell in each:

1. **`notebooks/01_synthetic_preview.ipynb`** — run this first, every time. It checks the
   labels, the charset, the throughput and reproducibility, and prints a contact sheet.
   **Look at the contact sheet.** There are no real passport images in this project, so
   your eyes on those samples are the main check that exists.
2. **`notebooks/02_train_recognition.ipynb`** — training. One cell.

Each notebook clones the repo itself if it is not already there, and installs only what is
missing, so you can also just paste the cell into a fresh notebook.

## Watching it

```
step   1000  loss 1.8423  lr 7.00e-04  22.4 it/s
  eval  loss 1.2011  char 61.20%  line 0.00%
```

- **Watch `line`, not `char`.** A line is correct only if all 44 characters are. At that
  length, 99.5% per character is still only ~80% of lines correct — and one wrong character
  is a wrong document.
- **Watch `it/s` against the table above.** If `it/s × 128` is close to `vCPUs × 118`, you
  are CPU-bound: raise `num_workers`. Do **not** lower `dpi` below 250 to buy speed — that
  trades away real resolution and every sample becomes an upscale.
- `line` staying at 0% through the first stage is normal — 44 characters all have to land.

## Checkpoints

Saved to `/workspace/checkpoints/recognition/` when `/workspace` exists, which is RunPod's
persistent volume — a stopped pod keeps them. Anywhere else and they die with the pod.

The checkpoint carries both raw and EMA weights, plus the geometry it was trained at, so it
cannot be loaded into a mismatched model by accident.

## What the numbers will not tell you

Validation runs on a disjoint generator seed, so there is no leakage — but it is still
synthetic grading synthetic. Whatever line accuracy this reports is a claim about the
generator, not about passports. The blueprint's fifth curriculum stage is "fine-tune on
real images"; it is absent because there are no real images.

The one external check that exists is in notebook 01: PassportEye, a stock MRZ reader built
for real documents, reads our renders at ~97% field accuracy (`docs/synthetic.md`). That
validates the glyphs and the pitch. It says nothing about whether our glare, wear and blur
resemble real photographs.
