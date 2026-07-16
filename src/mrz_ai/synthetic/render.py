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

from PIL import Image, ImageDraw, ImageFont

from ..parser import fields as F
from ..parser.charset import is_mrz_char
from .geometry import PageGeometry, mm_to_px

FONT_PATH = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "OCR-B.ttf"

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


def render_mrz(
    mrz: str,
    *,
    dpi: float = 300.0,
    geometry: PageGeometry | None = None,
    ink: int = INK,
    paper: int = PAPER,
    padding_mm: float = 1.5,
) -> RenderResult:
    """Render a two-line MRZ on a transparent-free greyscale strip.

    The result is just the zone, not a whole page: composing it onto a document
    background is a separate concern.
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

    width = round(F.LINE_LENGTH * pitch + 2 * padding)
    height = round(cap_height + line_pitch + 2 * padding)

    image = Image.new("L", (width, height), paper)
    draw = ImageDraw.Draw(image)
    font = load_font(font_size_for_cap_height(cap_height))

    char_boxes: list[tuple[int, int, int, int]] = []
    line_boxes: list[tuple[int, int, int, int]] = []

    for line_index, line in enumerate(lines):
        baseline = padding + cap_height + line_index * line_pitch
        for char_index, char in enumerate(line):
            cell_left = padding + char_index * pitch
            # Centre the glyph in its cell rather than trusting its advance: the
            # cell is the grid, the glyph is only what fills it.
            glyph_left, glyph_top, glyph_right, glyph_bottom = font.getbbox(char)
            glyph_width = glyph_right - glyph_left
            x = cell_left + (pitch - glyph_width) / 2.0 - glyph_left
            draw.text((x, baseline), char, font=font, fill=ink, anchor="ls")
            char_boxes.append(
                (
                    round(cell_left),
                    round(baseline - cap_height),
                    round(cell_left + pitch),
                    round(baseline),
                )
            )
        line_boxes.append(
            (
                round(padding),
                round(baseline - cap_height),
                round(padding + F.LINE_LENGTH * pitch),
                round(baseline),
            )
        )

    return RenderResult(image=image, char_boxes=tuple(char_boxes), line_boxes=tuple(line_boxes))
