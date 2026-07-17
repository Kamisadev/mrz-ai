# OCRB.ttf — the face the recognizer must be trained on

Real MRZs are printed in OCR-B. Training on an ordinary monospace stand-in (Menlo,
DejaVu, Courier) produces a model that validates the *pipeline* and is not
deployable: it learns the wrong glyph shapes for exactly the characters the
corrector exists to disambiguate. `8`/`S`, `6`/`G`, `0`/`O` and `1`/`L` are the
substitutions the ICAO check digits are blind to, and they are the ones whose OCR-B
forms differ most from a generic mono face. So the font is not a detail — it is the
difference between a demo and a reader.

Vendored here rather than left as a setup step because it is 19 KB, freely
distributable, and a missing font is otherwise a silent, expensive failure: a bare
GPU container ships no fonts at all. `render._discover_fonts` finds this file
automatically, and `render._font` raises rather than falling back to a bitmap face.

## Provenance

`OCRB.ttf` is from Matthew Skala's OCR-A/OCR-B package, version 0.3.1
(<https://tsukurimashou.org/ocr.php.en>, downloaded 2026-07-14). It descends from Norbert
Schwarz's (Ruhr-Universität Bochum) Metafont definitions, dated 1986–2010, which
were **originally distributed under a "non-commercial use only" restriction but have
since been released for unrestricted use and distribution**. Skala converted them to
TrueType and makes no copyright claim on the result.

`OCRB-provenance.pdf` is the package's own documentation, kept verbatim. Section 3
is the OCR-B licensing discussion; read it before shipping commercially. Skala is
explicit that he can only speak for his own contributions and declines to give legal
advice, so if a commercial deployment turns on this, get your own opinion.

Note the **OCR-A** font in the same upstream package carries a "may be used freely,
but cannot be distributed for profit" claim from a different author. That is why only
OCR-B is vendored here.

## Only `OCRB.ttf`

The upstream package also ships `OCRBE`/`OCRBF`/`OCRBL`/`OCRBS`/`OCRBX` — sharp-ends,
outline and reverse-video variants. Its documentation says plainly that they "aren't
suitable for actual OCR use." Do not train on them.

## Verified

All 37 MRZ characters (`0-9`, `A-Z`, `<`) render as non-blank glyphs at a single
advance width — it is genuinely monospaced, and it has the `<` chevron, which is
most of an MRZ by character count and which some OCR-B knock-offs omit.

## Re-deriving it

    curl -sSLO https://tsukurimashou.org/files/ocr-0.3.1.zip
    unzip ocr-0.3.1.zip && cp ocr-0.3.1/OCRB.ttf fonts/

    # ocr-0.3.1.zip  sha256 58136fccfdee0923cc83a20996a067b98bae054570ee41bf896d7ca8224399bf
    # OCRB.ttf       sha256 67b11c470222c7bb4550e7d4c216fd06145a939208af77e5f946bcee53e70868

Note the download link printed on the upstream page is relative and 404s if resolved
against the wrong host — the working URL is the one above, on `tsukurimashou.org`.
