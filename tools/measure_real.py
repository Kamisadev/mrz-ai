"""Read the real set with a trained checkpoint, and say what it got.

The only number this project produces that its own generator did not grade. See
`docs/real.md` — including why fitting anything to this set destroys the one
thing it is for.

    python tools/measure_real.py
    python tools/measure_real.py --checkpoint checkpoints/recognition/recognition.pt
    python tools/measure_real.py --raw          # skip EMA, match the dashboard

Prints the misread documents by name. That is the point of running it by hand
rather than reading the panel: the confusion pairs say which axis the generator is
missing, and a pair is a hypothesis to test with `tools/font_gap.py` or a
controlled render — not an instruction to go and fine-tune on the image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrz_ai.evaluation.real import REAL_DIR, load_real_set, measure_real  # noqa: E402
from mrz_ai.inference.pipeline import MRZReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REAL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=Path("recognition_model/recognition.pt"))
    parser.add_argument("--reference-year", type=int, default=None)
    parser.add_argument(
        "--raw", action="store_true",
        help="use the raw weights instead of the EMA, which is what the dashboard shows",
    )
    args = parser.parse_args()

    documents = load_real_set(args.dir)
    reader = MRZReader.from_checkpoint(args.checkpoint, use_ema=not args.raw, k=8)
    result = measure_real(reader, documents, reference_year=args.reference_year)

    weights = "raw" if args.raw else "EMA"
    print(f"{args.checkpoint} ({weights} weights) on {len(documents)} document(s)\n")
    print(f"  documents  {result.documents_read}/{result.documents}")
    print(f"  lines      {result.lines_read}/{2 * result.documents}")
    print(f"  characters {result.chars_read}/{result.chars_total}  ({result.char_rate:.2%})")

    if result.not_located:
        # Said before anything else, because it invalidates everything after it.
        print(
            f"\n  {result.not_located} of {result.documents} could not be cropped: the ink "
            f"did not\n  resolve into two lines, so the region was halved as a guess. That is "
            f"the\n  box being wrong, not the model. A whole passport page has ink "
            f"everywhere —\n  give each entry a \"box\": [x, y, width, height] around the MRZ, "
            f"or crop the\n  images to it. See docs/real.md."
        )

    missed = [name for name, ok in result.per_document if not ok]
    if missed:
        print(f"\n  misread: {', '.join(missed)}")
    if result.confusions:
        print("\n  confusions (what the generator never drew anything to separate):")
        for (want, got), count in result.confusions[:8]:
            print(f"    {want} -> {got}   x{count}")

    if result.documents < 30:
        # Said every time, because the number above is about to be quoted.
        print(
            f"\n  {result.documents} documents cannot separate a good model from a mediocre "
            f"one.\n  Worth having as a smoke test; not worth quoting a percentage from."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
