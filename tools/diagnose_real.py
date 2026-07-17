"""Why did the model misread this passport? Look at what it was actually shown.

Everything the recognizer has ever seen came out of the synthetic engine. When it
misreads a real page, the question is which of three different things went wrong,
and they need three different fixes:

1. The crop is crisp, readable OCR-B, and the model still misreads it.
   -> A genuine glyph gap. The generator uses one font at one weight, forever.
      Fix: randomize the printing (fonts, stroke weight, jitter) and retrain.

2. The crop is washed out, tinted, or has security print showing through.
   -> A photometric gap. The generator's paper is plain; a real data page has
      guilloche under the MRZ. Font randomization would not touch this.

3. The crop is misframed, the wrong scale, or cut off.
   -> `serve/crop.py` misbehaving on a real background — most likely Otsu
      latching onto security print instead of ink. Not a retrain at all.

Only the crop can tell them apart, so this dumps it: what the model was fed, next
to what training looked like for the very same characters, at the same size.

Nothing leaves your machine. The image is read, compared and written back to a
file you choose; it is not uploaded, and the passport's contents are printed to
your terminal only.

    python tools/diagnose_real.py passport.jpg
    python tools/diagnose_real.py passport.jpg --box 120 880 1400 190 --out look.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrz_ai.inference.beam import log_softmax  # noqa: E402
from mrz_ai.inference.pipeline import MRZReader  # noqa: E402
from mrz_ai.recognition.geometry import INPUT  # noqa: E402
from mrz_ai.recognition.preprocess import prepare  # noqa: E402
from mrz_ai.serve.crop import Box, decode_image, locate_lines  # noqa: E402
from mrz_ai.synthetic.render import render_mrz  # noqa: E402

Array = np.ndarray

#: What "certain" actually looks like. Label smoothing at 0.1 spreads a tenth of
#: every target across the other 36 classes, so the model is trained never to be
#: sure: a position it has no doubt about tops out around here, not at 1.0.
_SURE_CEILING = 0.91
#: Below this the model is genuinely hesitating rather than merely smoothed.
_SURE = 0.85


def as_model_input(crop: Array) -> Array:
    """The exact 32x704 the recognizer is handed, as something you can look at."""
    return (prepare(crop, INPUT)[0] * 255.0).astype(np.uint8)


def synthetic_twin(mrz: str) -> tuple[Array, Array]:
    """What training looked like for this text: the same lines, rendered clean.

    No degradation. The point is to compare the glyphs themselves, and a blurred
    reference would only make an honest difference harder to see.

    Put through `locate_lines` rather than cropped at the renderer's own line
    boxes, even though the truth is right there. The two paths do not frame
    identically — locate_lines pads by 0.15 of a line — and a reference framed
    differently from the sample would show a scale difference that is this
    script's artifact rather than the model's problem. A diagnostic that invents
    its own discrepancy is worse than none, since it sends you to fix the wrong
    thing.
    """
    rendered = render_mrz(mrz, dpi=300.0)
    page = np.asarray(rendered.image)
    left, top, right, bottom = rendered.mrz_box
    slack = 0.3 * (bottom - top)
    box = Box(
        x=left - slack, y=top - slack,
        width=(right - left) + 2 * slack, height=(bottom - top) + 2 * slack,
    )
    first, second = locate_lines(page, box)
    return tuple(as_model_input(_cut(page, line.box)) for line in (first, second))


def _cut(image: Array, box: Box) -> Array:
    return image[
        int(box.y) : int(box.y + box.height), int(box.x) : int(box.x + box.width)
    ]


def contrast_of(crop: Array) -> tuple[float, float]:
    """Ink and paper levels, as the 5th and 95th percentiles.

    A real data page prints the MRZ over security patterns, which lifts the ink
    and drags the paper down. Training saw ink near 0-60 on paper near 220-255.
    """
    return float(np.percentile(crop, 5)), float(np.percentile(crop, 95))


def stack(panels: list[tuple[str, Array]], zoom: int = 2) -> Array:
    """Lay the crops out one above another, captioned, for a human to compare."""
    import cv2

    rows = []
    for caption, panel in panels:
        big = cv2.resize(
            panel, (panel.shape[1] * zoom, panel.shape[0] * zoom), interpolation=cv2.INTER_NEAREST
        )
        label = np.full((22, big.shape[1]), 255, dtype=np.uint8)
        cv2.putText(label, caption, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)
        rows.append(label)
        rows.append(big)
        rows.append(np.full((10, big.shape[1]), 255, dtype=np.uint8))
    return np.vstack(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="the MRZ zone in image pixels. Defaults to the bottom third.",
    )
    parser.add_argument("--out", type=Path, default=Path("diagnosis.png"))
    parser.add_argument("--checkpoint", type=Path, default=Path("recognition_model/recognition.pt"))
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    picture = decode_image(args.image.read_bytes())
    height, width = picture.shape[:2]
    box = Box(*args.box) if args.box else Box(x=0, y=height * 0.66, width=width, height=height * 0.34)

    first, second = locate_lines(picture, box)
    if first.clipped or second.clipped:
        print("WARNING: the box cuts through a line. Pass a wider --box.\n")

    crops = [_cut(picture, line.box) for line in (first, second)]

    reader = MRZReader.from_checkpoint(args.checkpoint, k=8)
    reading = reader.read(crops[0], crops[1], reference_year=args.year)
    logits = reader.logits_for(crops)

    print(f"read: {reading.line1}")
    print(f"      {reading.line2}")
    print(f"valid: {reading.is_valid}\n")

    # Where the model hesitated. Read the numbers against the ceiling, not
    # against 1.0: the model was trained with label smoothing at 0.1, so a
    # position it is completely sure of tops out near 0.91 and never approaches
    # 1.0. Measured on a clean synthetic page, every position sits at 0.909-0.926;
    # on a crop with its first characters cut off, the doubt fell to 0.26.
    for index, line in enumerate((reading.line1, reading.line2), start=1):
        probs = np.exp(log_softmax(logits[index - 1]))
        best = probs.max(axis=1)
        weak = [(position, line[position], float(best[position]))
                for position in range(len(line)) if best[position] < _SURE]
        print(f"line {index}: median {np.median(best):.3f} (ceiling {_SURE_CEILING:.2f}), "
              f"lowest {best.min():.3f}, {len(weak)} of 44 positions unsure")
        for position, char, value in weak[:12]:
            print(f"    position {position:2d}  read {char!r} at {value:.2f}")

    print()
    real = [as_model_input(crop) for crop in crops]
    twin = synthetic_twin(reading.mrz)

    for index in range(2):
        ink, paper = contrast_of(real[index])
        t_ink, t_paper = contrast_of(twin[index])
        print(f"line {index + 1} contrast  yours: ink {ink:5.1f} paper {paper:5.1f}   "
              f"training: ink {t_ink:5.1f} paper {t_paper:5.1f}")

    panels = [
        ("YOUR PASSPORT - line 1, exactly as the model sees it (32x704)", real[0]),
        ("TRAINING - the same characters, rendered by our generator", twin[0]),
        ("YOUR PASSPORT - line 2", real[1]),
        ("TRAINING - line 2", twin[1]),
    ]
    import cv2

    cv2.imwrite(str(args.out), stack(panels))
    print(f"\nwrote {args.out}. Open it and compare the pairs:")
    print("  glyphs differ in shape or thickness  -> font gap, retrain the generator")
    print("  yours is washed out / patterned      -> background gap, retrain the paper")
    print("  yours is misframed or wrong scale    -> crop.py bug, no retrain needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
