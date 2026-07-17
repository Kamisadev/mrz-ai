# The real set

Put real passport photographs in `real/`. Nothing else in this repository has
ever seen one, and that is the whole reason this directory matters.

These are **specimen** passports: real documents, real printing, real typeface,
invented identities. That combination is what makes the set both worth having and
safe to have. The cut of OCR-B is whatever the issuer's press actually used —
which is the axis that broke the last checkpoint — while the name and number
belong to nobody, so the set can travel to a rented pod without a passport
holder's data going with it. Genuine passports would buy nothing this does not,
and would cost a great deal more to be wrong about.

```text
real/
├── images/
│   ├── 001_pass.jpg
│   └── 002_pass.png
└── truth.json
```

```json
{
  "001_pass.jpg": {
    "line1": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "line2": "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
  }
}
```

Both lines, 44 characters each, `<` for every filler. A short line scores as a
misreading forever, so the loader refuses one rather than letting a transcription
slip turn into a permanent point of accuracy. Every image needs truth and every
truth an image: an unmeasured image would drop out of the denominator and quietly
improve the score.

### The box, which you probably do not have to give

No box needed for either a data page or an already-cropped MRZ strip. Both read
4 of 4 in testing with nothing in `truth.json` but the two lines.

That is not detection. ICAO fixes the TD3 page at 125x88mm and puts the zone
74.07mm down it, so **the MRZ is the bottom 16% of any data page** — a constant,
not something to find. `default_box` takes the bottom third to be generous, and
tells a page from a strip by shape: a page is 1.4:1 and two MRZ lines are nearer
12:1, with nothing in between. Being loose costs nothing, because `crop.py` finds
the ink inside whatever it is given and levels it first.

Give a box when the arithmetic does not hold — a page photographed with the desk
around it, an unusual crop:

```json
{ "001_pass.jpg": { "line1": "...", "line2": "...", "box": [0, 400, 1130, 130] } }
```

**You will be told when it is needed.** `crop.py` looks for two dense bands of ink
on pale paper. Given something that is not that, the two-band search fails and the
region is halved as a guess — measured on whole pages passed in as-is, that read
**0 of 4 documents at 8.8% of characters**, which from the outside is a recognizer
that cannot read. So the fallback is counted: `not_located`, reported by the tool
and the dashboard *before* any score, because a score with that number above zero
is about the boxes and says nothing about the model.

Real detection — finding an MRZ in a photograph of a passport on a desk — is phase
3, and it is blocked on the Passport Generator rather than on effort. Training a
detector needs whole pages; the synthetic engine draws bare MRZ strips; and the
specimens here cannot be trained on without destroying what they are for.

Measure a checkpoint against the set:

```bash
.venv/bin/python tools/measure_real.py
```

Or have training read it as it goes — `TrainConfig(real_dir=Path("real"))`, shown
on the dashboard. See "Watching it" below for what that costs.

`real/` is ignored by git as a whole directory, with no negated pattern inside
it. This repository is public. `tests/test_real_data_stays_out_of_git.py` fails
if anything in there is ever tracked, because `.gitignore` is a request and
`git add -f` is not obliged to honour it — and a passport pushed here is on
GitHub permanently and in strangers' clones within minutes.

## Measurement only. Never training.

This is the rule the set exists to serve, and it is not a precaution — it is the
only thing that makes the set worth having.

Every number this project reports is currently graded by the same engine that
drew the exam. `README.md` records what that is worth: a checkpoint scored 99.5%
line accuracy at full severity and misread real passports badly, because the
generator used a single font and nothing that measured it could see outside the
generator. A held-out set of real images is the first thing here that can.

Train on it and it stops being able to. Not gradually — immediately, and without
saying so. The accuracy it reports afterwards will be excellent and will mean
exactly what the 99.5% meant.

So: no fine-tuning on these, no "just the ones it got wrong", no using them to
pick a checkpoint or an epoch or a threshold. When a real passport reads badly,
the fix belongs in the generator — find the axis it never randomized, add it,
retrain on synthetic data, and come back to measure. That is what happened with
the fonts, and it recovered 28% of documents at once rather than one image at a
time.

## Watching it during a run

`TrainConfig(real_dir=...)` reads the set every `real_every` steps and puts the
count on the dashboard. Useful, and not free, and the cost is worth naming.

A number you look at is a number you act on. Killing a run because this panel
dipped is selection on the test set, done by hand and one decision at a time. The
textbook answer is to split the set — a dev half you may watch and burn, a test
half touched once — and with a handful of documents there is nothing to split.

So this set is a **dev set**, and the panel says so. That is not a licence, it is
an admission: after the first run it steers, its number is no longer an unbiased
estimate of anything, and the sealed 50–100 that `README.md` asks for remains a
separate thing that does not exist. `real_every` defaults to 5,000 steps rather
than the synthetic eval's 1,000 for this reason and no other — it costs a second
to run.

The panel shows counts, per-document ticks and confusion pairs. Never the decoded
MRZ: these identities are invented, but the payload is served over a port, and
the day somebody points the same panel at a genuine passport is not the day to
find out that the page prints its number.

## What the set needs to be

**Ground truth.** An image without its true MRZ text measures nothing. Both lines,
all 44 characters each, transcribed from the document by eye — not by this model,
which would only be asking it to grade itself with extra steps.

**Enough of it.** Fewer than about 30 documents cannot separate a good model from
a mediocre one: at 10 documents, 9 correct and 10 correct are the same
measurement. `README.md` asks for 50–100. Ten is worth having as a smoke test and
is not worth quoting a percentage from.

**Spread.** Different issuing countries above all, since the issuer is who chose
the typeface — the failure this set exists to catch. Then different cameras,
lighting, angles, and wear.

## What it is not

It is not a corpus to grow forever, and it is not a place for genuine passports.
Specimens carry the one thing that matters here — the issuer's press, its cut of
OCR-B, its security print — and carry nobody's identity with it. A real passport
would buy nothing this does not and would put a real number, name and date of
birth in a directory, in the clear, on whatever machine measures. If one ever
does end up here, it is PII: it does not go on a pod, `not_located` and the
confusion pairs are still all the dashboard may show of it, and it goes when the
measuring does.
