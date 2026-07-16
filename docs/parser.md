# Phase 0 — the ICAO TD3 engine

`mrz_ai.parser` is pure Python with no ML dependencies. The synthetic dataloader
imports it inside worker processes and the RunPod notebooks import it without
dragging in torch, so it must stay that way.

Three entry points:

| Function | Direction | Used by |
| --- | --- | --- |
| `serialize(fields)` | fields → MRZ string | Phase 1, to build training labels |
| `parse(mrz)` | MRZ string → fields + check digits | Phase 4, at inference |
| `validate(document, reference_year=...)` | document → list of issues | Phase 4, to rank hypotheses |

`serialize` is the load-bearing one. Every synthetic sample's label comes from
it, so a bug there silently teaches the model wrong data. It is fuzzed over 200
random identities against `parse` and `validate`.

## Decisions worth knowing

**Check digits are derived, never stored.** `TD3Fields` holds only the
human-meaningful content. There is no way to construct a sample whose check
digits disagree with its fields.

**The century is policy, not data.** `YYMMDD` carries no century, so `validate`
takes an explicit `reference_year` rather than reading the clock — the parser
stays pure and tests do not rot. Birth years use a sliding pivot, since nobody
is born in the future. Expiry years have no such direction available — an
expiry may legitimately sit either side of today — so they simply anchor to the
reference century. That is wrong near a century boundary (a 2103 expiry read in
2098 resolves to 2003), but the alternative misreads every passport that expired
more than ten years ago, which is an ordinary document to be handed today. We
took the error that is 70 years away over the one that is here now.

**Country codes are an allowlist, not a shape check.** ISO 3166-1 alpha-3 plus
the ICAO specials (`XXA`, `XXB`, `XXC`, `XXX`, `UNO`, `UNK`, `EUE`, `RKS`, the
`GB*` subsets, and Germany's `D<<`). `UTO` — fictional Utopia — is included
because the ICAO specimen uses it. This is not cosmetic: see below.

**The unused optional-data check digit may be `<` or `0`.** ICAO allows either.
`serialize` emits `<` by default (`filler_optional_cd=False` for `0`), and
`validate` accepts both. This is safe because the two characters have the same
numeric value, so the composite digit is identical either way — the classic
composite-mismatch bug cannot occur here.

## What validation cannot do

Both of these were found by fuzzing and are properties of the spec, not bugs.
The candidate decoder in Phase 4 must not assume them away.

**1. Nationality and sex have no checksum protection at all.** The composite
digit spans only line 2 positions `[0:10]`, `[13:20]` and `[21:43]`, which skips
nationality (10–12) and sex (20). See `fields.UNPROTECTED_LINE2`.

The specimen proves this rather than merely suggesting it: its real composite
digit is `0`, and our payload (which excludes sex) computes `0`. Were sex
included, `F` (value 15) would shift the sum by 15 × {7,3,1} ≡ 5 mod 10 in any
position, giving `5`. The spec and the specimen agree.

Consequences:

- **Sex is unrecoverable.** `M` misread as `F` yields a completely valid MRZ.
  Nothing can catch it, and the API must not imply a passing checksum vouches
  for that field.
- **Nationality is guarded only by the allowlist.** `AUS` → `AUZ` is caught
  because no such country exists — no check digit moved. But `AUS` → `AUT`
  turns Australia into Austria silently, and passes everything. That residual
  hole is the decoder's to cover with model confidence.

**2. Check digits are blind to characters whose values agree mod 10.** A
substitution moves a check digit by `delta * weight`. The 7-3-1 weights are all
coprime with 10, so the digit is unchanged exactly when `delta` is a multiple of
10. That makes `<`, `0`, `A`, `K`, `U` mutually invisible — as are `1/B/L/V`,
`2/C/M/W`, and so on for all ten classes.

The saving grace is that these classes are not visually confusable in OCR-B, and
the date fields reject letters outright. The pairs OCR actually confuses —
`0/O`, `1/I`, `8/B`, `5/S`, `2/Z` — all fall in different classes and *are*
caught. The exposed spot is the document number, where any class-mate
substitution is undetectable.

## Running it

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
.venv/bin/pytest tests -q      # 603 tests
.venv/bin/mypy src             # strict
.venv/bin/ruff check src tests
```
