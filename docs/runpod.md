# Training on RunPod

## The one thing that decides your pod

**This job is CPU-bound, not GPU-bound.** The recognizer is 3.4M parameters — trivial for
any modern GPU — while every training sample is drawn, rendered and degraded from scratch
on the CPU. Renting a bigger GPU buys nothing; it will sit idle waiting for data.

Measured generator throughput, per core:

| dpi | samples/s/core | source px/char | verdict |
| --- | --- | --- | --- |
| 120 | 234 | 12 | too low — upscales to 16 and invents detail |
| **150** | **181** | **15** | **default: matches the model's 16px/char, almost no wasted work** |
| 200 | 171 | 20 | renders detail then discards it |
| 300 | 98 | 30 | wasteful |

Cores needed to keep a GPU busy at batch 128, dpi 150:

| GPU speed | samples/s needed | cores needed |
| --- | --- | --- |
| 10 it/s | 1,280 | 7 |
| 20 it/s | 2,560 | 14 |
| 40 it/s | 5,120 | 28 |
| 60 it/s | 7,680 | 42 |

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
| 8 | 1,450 | 11 | ~56 min |
| **16** | **2,900** | **23** | **~28 min** |
| 32 | 5,800 | 45 | ~14 min (GPU may now be the limit) |

So: **roughly 30 minutes on a 16-vCPU pod**, for something like $0.20–0.60 depending on the
GPU you happen to get. If the GPU is the cheap part, spend the money on cores.

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
- **Watch `it/s` against the table above.** If `it/s × 128` is close to `vCPUs × 181`, you
  are CPU-bound: raise `num_workers`, or lower `dpi` (but not below 150).
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
