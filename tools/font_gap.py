"""Can the recognizer read a cut of OCR-B it was not trained on?

This is the experiment that found the defect, kept so it can be rerun. Render
clean, undegraded MRZs — same text, same geometry, no noise, no blur, nothing but
the typeface changing — and read them. Any accuracy lost is lost to the glyphs
alone, because nothing else moved.

The first run, against a model trained on `OCR-B.ttf` only:

    OCR-B.ttf (trained on):  document 100.0%  line 100.0%  char 100.00%
    OCRB.ttf  (Skala cut):   document  72.0%  line  86.0%  char  99.57%
        top confusions: 0->O x9  J->U x4  J->C x3  J->O x2  G->S x1

28% of documents lost on a perfect image. The model had not learned what a '0'
is; it had learned what one vendor's '0' is, and a real passport is printed with
whichever cut its issuer bought. Note how char accuracy flatters: 99.57% per
character over 88 characters is 72% per document.

A METHODOLOGICAL WARNING, because this script will lie to you otherwise.

Once a font is in `DatasetConfig.fonts`, reading it well proves nothing about
unseen cuts — it is training accuracy. To measure *generalization* you must hold
a cut out of training and test on it, which costs you the ability to train on it.
Both cuts we have are now in the default config, so a rerun measures in-domain
performance and nothing more. To get a real number again you need either a third
cut held out, or a real passport.

    python tools/font_gap.py
    python tools/font_gap.py --fonts assets/fonts/OCRB.ttf   # the held-out one
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrz_ai.inference.pipeline import MRZReader  # noqa: E402
from mrz_ai.parser import serialize  # noqa: E402
from mrz_ai.synthetic.identity import IdentityConfig, random_identity  # noqa: E402
from mrz_ai.synthetic.render import available_fonts, render_mrz  # noqa: E402


def measure(reader: MRZReader, font: str, documents: int, weight: float) -> dict[str, object]:
    """Read ``documents`` clean renders in one cut. No degradation whatsoever."""
    lines_ok = chars_ok = chars_total = docs_ok = 0
    confusions: dict[tuple[str, str], int] = {}

    for index in range(documents):
        mrz = serialize(random_identity(random.Random(2000 + index), IdentityConfig()))
        rendered = render_mrz(mrz, dpi=300.0, font_path=font, ink_weight=weight)
        page = np.asarray(rendered.image)
        crops = [page[top:bottom, left:right] for left, top, right, bottom in rendered.line_boxes]

        reading = reader.read(crops[0], crops[1], reference_year=2026)
        truth = mrz.split("\n")
        for got, want in zip((reading.line1, reading.line2), truth):
            lines_ok += got == want
            for g, w in zip(got, want):
                chars_ok += g == w
                chars_total += 1
                if g != w:
                    confusions[(w, g)] = confusions.get((w, g), 0) + 1
        docs_ok += reading.line1 == truth[0] and reading.line2 == truth[1]

    return {
        "document": docs_ok / documents,
        "line": lines_ok / (2 * documents),
        "char": chars_ok / chars_total,
        "confusions": sorted(confusions.items(), key=lambda kv: -kv[1])[:8],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonts", nargs="*", default=None, help="defaults to every cut on disk")
    parser.add_argument("--documents", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=Path("recognition_model/recognition.pt"))
    parser.add_argument(
        "--ink-weight", type=float, default=0.0, help="thin (-1) to thick (+1) strokes"
    )
    args = parser.parse_args()

    fonts = args.fonts or list(available_fonts())
    reader = MRZReader.from_checkpoint(args.checkpoint, k=8)

    print(f"{args.documents} clean renders per cut, no degradation, ink weight {args.ink_weight}\n")
    for font in fonts:
        result = measure(reader, font, args.documents, args.ink_weight)
        print(
            f"{Path(font).name:12s}  document {result['document']:6.1%}  "
            f"line {result['line']:6.1%}  char {result['char']:7.2%}"
        )
        confusions = result["confusions"]
        assert isinstance(confusions, list)
        if confusions:
            shown = "  ".join(f"{want}->{got} x{count}" for (want, got), count in confusions)
            print(f"{'':12s}  confusions: {shown}")

    print(
        "\nA cut listed in DatasetConfig.fonts is training data: reading it well is"
        "\nnot evidence about unseen cuts. See this file's docstring."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
