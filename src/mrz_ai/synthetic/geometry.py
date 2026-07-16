"""Physical dimensions of a TD3 passport data page.

Everything is held in millimetres and converted to pixels at render time, so a
sample can be produced at any DPI without the layout drifting. The numbers marked
"spec" come from ICAO 9303; the rest are plausible values chosen to look like a
real page and are free to be tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class PageGeometry:
    """The data page and the placement of the MRZ on it."""

    #: spec: ID-3 documents are 125mm x 88mm.
    page_width_mm: float = 125.0
    page_height_mm: float = 88.0

    #: spec: the MRZ is set in OCR-B at 10 characters per inch.
    char_pitch_mm: float = MM_PER_INCH / 10.0  # 2.54

    #: spec: OCR-B size 1 has a nominal cap height of 3.2mm.
    cap_height_mm: float = 3.2

    #: Baseline-to-baseline distance between the two MRZ lines. Approximate.
    line_pitch_mm: float = 4.23

    #: Distance from the bottom of the page to the baseline of the second line.
    #: Approximate.
    mrz_bottom_margin_mm: float = 6.5

    @property
    def mrz_width_mm(self) -> float:
        """44 characters at the spec pitch: 111.76mm inside a 125mm page."""
        from ..parser import fields as F

        return F.LINE_LENGTH * self.char_pitch_mm

    @property
    def mrz_left_margin_mm(self) -> float:
        """Centre the zone horizontally, leaving ~6.6mm either side."""
        return (self.page_width_mm - self.mrz_width_mm) / 2.0


def mm_to_px(mm: float, dpi: float) -> float:
    return mm * dpi / MM_PER_INCH


def px_to_mm(px: float, dpi: float) -> float:
    return px * MM_PER_INCH / dpi
