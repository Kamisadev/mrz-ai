# The real set

Put real passport photographs in `real/`. Nothing else in this repository has
ever seen one, and that is the whole reason this directory matters.

```text
real/
├── images/
│   ├── 001.jpg
│   └── 002.jpg
└── truth.json     {"001.jpg": {"line1": "P<UTO...", "line2": "L898902C3..."}}
```

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

It is not a corpus to grow forever. These are identity documents belonging to
people: passport number, full name, date of birth, nationality, in one directory,
in the clear. Hold what is needed to measure and no more, know whose they are and
that they agreed, and delete them when the measuring is done.
