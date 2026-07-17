"""Render an MRZ string as a clean, fixed-pitch image.

The MRZ is fixed-pitch: every character occupies an identical cell, whatever its
natural width. This matters more than it sounds. The bundled OCR-B is not
actually monospaced — '5' is 782 units against 884 for the other digits and 886
for the letters — so laying the text out with PIL's own advances would both drift
away from a real passport's grid and give '5' a width no other glyph has. A model
trained on that could learn to spot a '5' by the gap around it rather than by its
shape, and the cue would vanish on real documents. Placing each glyph on the grid
ourselves removes the leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..parser import fields as F
from ..parser.charset import ALPHABET, is_mrz_char
from .geometry import PageGeometry, mm_to_px

FONTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
FONT_PATH = FONTS_DIR / "OCR-B.ttf"

#: Every cut of OCR-B we can train on, and the reason there is more than one.
#:
#: ICAO 9303 names a typeface, not an outline. The cuts of OCR-B differ — enough
#: that a model trained on exactly one of them reads the other at 72% of
#: documents on a *clean, undegraded* render, with confusions on 0->O, J->U and
#: G->S: the shapes a reader must tell apart. That model had not learned what a
#: '0' is; it had learned what this vendor's '0' is, and a real passport is
#: printed with whichever cut its issuer bought.
#:
#: So the font is randomized like any other nuisance parameter. Two is not many —
#: more cuts would be better — but the difference between one and two is the
#: difference between memorizing an outline and having to find the letter in it.
MRZ_FONTS: tuple[Path, ...] = (
    FONTS_DIR / "OCR-B.ttf",   # SIL OFL 1.1; see OCR-B-LICENSE.md
    FONTS_DIR / "OCRB.ttf",    # Skala / Schwarz; see README.md
)


def available_fonts() -> tuple[str, ...]:
    """The cuts of OCR-B present on disk, best-effort.

    Missing a font is not fatal — a checkout without the assets should still
    render — but training on one when two were intended is a silent loss of
    variety, so the caller is told what it actually got.
    """
    return tuple(str(path) for path in MRZ_FONTS if path.exists())

#: Ink and paper as 8-bit greyscale. Real MRZ ink is not pure black, and the
#: degradation stage moves both of these around.
INK = 0
PAPER = 255


@dataclass(frozen=True)
class RenderResult:
    """A rendered MRZ and where its characters landed.

    ``char_boxes`` gives the pixel cell of every character in reading order,
    which is what lets the detection stage derive a ground-truth MRZ box and the
    recognition stage crop exact lines without guessing.
    """

    image: Image.Image
    char_boxes: tuple[tuple[int, int, int, int], ...]
    line_boxes: tuple[tuple[int, int, int, int], ...]

    @property
    def mrz_box(self) -> tuple[int, int, int, int]:
        left = min(b[0] for b in self.line_boxes)
        top = min(b[1] for b in self.line_boxes)
        right = max(b[2] for b in self.line_boxes)
        bottom = max(b[3] for b in self.line_boxes)
        return left, top, right, bottom


@lru_cache(maxsize=16)
def load_font(size_px: int, path: str = str(FONT_PATH)) -> ImageFont.FreeTypeFont:
    """Load OCR-B at a pixel size, cached because this is on the hot path."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"OCR-B not found at {path}. It ships in assets/fonts; see docs/synthetic.md."
        )
    return ImageFont.truetype(path, size_px)


def font_size_for_cap_height(cap_height_px: float, path: str = str(FONT_PATH)) -> int:
    """Find the font size whose capital letters are ``cap_height_px`` tall.

    PIL sizes a font by em, not by cap height, and the relationship depends on
    the font's own metrics. Measuring a real glyph is more reliable than scaling
    by unitsPerEm.
    """
    probe = 100
    font = load_font(probe, path)
    top, bottom = font.getbbox("H")[1], font.getbbox("H")[3]
    measured = bottom - top
    if measured <= 0:  # pragma: no cover - would mean a broken font
        raise ValueError("could not measure the font's cap height")
    return max(1, round(probe * cap_height_px / measured))


@lru_cache(maxsize=16)
def ink_extent(size_px: int, path: str = str(FONT_PATH)) -> tuple[float, float]:
    """How far the alphabet's ink reaches above and below the baseline, in pixels.

    Cap height is not the answer. OCR-B's digits are taller than its capitals —
    '0' and '8' overshoot 'H' by three pixels at 300dpi — so a box sized to the
    cap height clips the top off every digit. Since these boxes are the ground
    truth the recognizer crops with, that would train it on mutilated glyphs.
    Measure the real extent across every character we can render instead.
    """
    font = load_font(size_px, path)
    boxes = [font.getbbox(char) for char in ALPHABET]
    baseline = font.getbbox("H")[3]  # PIL reports bboxes from a common origin
    above = max(baseline - box[1] for box in boxes)
    below = max(box[3] - baseline for box in boxes)
    return float(above), float(below)


def render_mrz(
    mrz: str,
    *,
    dpi: float = 300.0,
    geometry: PageGeometry | None = None,
    ink: int = INK,
    paper: int = PAPER,
    padding_mm: float = 1.5,
    font_path: str = str(FONT_PATH),
    ink_weight: float = 0.0,
) -> RenderResult:
    """Render a two-line MRZ on a transparent-free greyscale strip.

    The result is just the zone, not a whole page: composing it onto a document
    background is a separate concern.

    ``font_path`` is a parameter because "OCR-B" is not one typeface. The spec
    names a face; the cuts of it differ, and a model trained on exactly one of
    them has no way to know which parts of a glyph are the letter and which are
    this particular vendor's idea of it.

    ``ink_weight`` in [-1, 1] thins or thickens the strokes, standing in for the
    press. Both of these belong here rather than in `degrade` on purpose: they
    are properties of how the document was *printed*, not of how it was
    photographed. `degrade` scales everything by severity, so putting them there
    would give the clean end of the curriculum exactly one typeface at exactly
    one weight — which is the hole this is closing.
    """
    geometry = geometry or PageGeometry()
    lines = mrz.split("\n")
    if len(lines) != F.LINE_COUNT:
        raise ValueError(f"expected {F.LINE_COUNT} lines, got {len(lines)}")
    for line in lines:
        if len(line) != F.LINE_LENGTH:
            raise ValueError(f"expected {F.LINE_LENGTH} characters, got {len(line)}")
        for char in line:
            if not is_mrz_char(char):
                raise ValueError(f"{char!r} is not an MRZ character")

    pitch = mm_to_px(geometry.char_pitch_mm, dpi)
    line_pitch = mm_to_px(geometry.line_pitch_mm, dpi)
    cap_height = mm_to_px(geometry.cap_height_mm, dpi)
    padding = mm_to_px(padding_mm, dpi)

    font_size = font_size_for_cap_height(cap_height, font_path)
    font = load_font(font_size, font_path)
    above, below = ink_extent(font_size, font_path)

    width = round(F.LINE_LENGTH * pitch + 2 * padding)
    height = round(above + below + line_pitch + 2 * padding)

    image = Image.new("L", (width, height), paper)
    draw = ImageDraw.Draw(image)

    char_boxes: list[tuple[int, int, int, int]] = []
    line_boxes: list[tuple[int, int, int, int]] = []

    for line_index, line in enumerate(lines):
        baseline = padding + above + line_index * line_pitch
        for char_index, char in enumerate(line):
            cell_left = padding + char_index * pitch
            # Centre the glyph in its cell rather than trusting its advance: the
            # cell is the grid, the glyph is only what fills it.
            glyph_left, glyph_top, glyph_right, glyph_bottom = font.getbbox(char)
            glyph_width = glyph_right - glyph_left
            x = cell_left + (pitch - glyph_width) / 2.0 - glyph_left
            draw.text((x, baseline), char, font=font, fill=ink, anchor="ls")
            # The box spans the alphabet's full ink extent, not this glyph's, so
            # that every cell is the same height and no digit is clipped.
            char_boxes.append(
                (
                    round(cell_left),
                    round(baseline - above),
                    round(cell_left + pitch),
                    round(baseline + below),
                )
            )
        line_boxes.append(
            (
                round(padding),
                round(baseline - above),
                round(padding + F.LINE_LENGTH * pitch),
                round(baseline + below),
            )
        )

    if ink_weight:
        image = weigh_ink(image, ink_weight)

    return RenderResult(image=image, char_boxes=tuple(char_boxes), line_boxes=tuple(line_boxes))


def weigh_ink(image: Image.Image, weight: float) -> Image.Image:
    """Thin or thicken the strokes: ``weight`` -1 is lightest, +1 heaviest.

    A press is not a plotter. The same typeface comes out heavier or lighter
    depending on the printer, the ink and the paper, and the strokes of an MRZ
    are only a few pixels wide at the size the recognizer sees — so a fraction of
    a pixel either way changes which glyphs are confusable.

    Fractional, by blending against the morphed image: a whole-pixel step at
    250dpi is a large change in weight, and stepping in whole pixels would train
    the model on three weights rather than a range.
    """
    array = np.asarray(image).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # The ink is dark, so eroding the image spreads the ink and dilating eats it.
    morphed = (cv2.erode if weight > 0 else cv2.dilate)(array, kernel)
    blended = array * (1.0 - abs(weight)) + morphed * abs(weight)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
