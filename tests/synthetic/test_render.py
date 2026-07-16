"""Rendering the MRZ onto a fixed-pitch grid."""

from __future__ import annotations

import random

import numpy
import pytest

from mrz_ai.parser import serialize
from mrz_ai.parser import fields as F
from mrz_ai.synthetic.geometry import PageGeometry
from mrz_ai.synthetic.identity import random_identity
from mrz_ai.synthetic.render import PAPER, render_mrz

SPECIMEN = (
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
)


def test_a_cell_is_produced_for_every_character() -> None:
    result = render_mrz(SPECIMEN)
    assert len(result.char_boxes) == F.LINE_COUNT * F.LINE_LENGTH
    assert len(result.line_boxes) == F.LINE_COUNT


def test_the_pitch_matches_the_spec() -> None:
    # ICAO sets the MRZ in OCR-B at ten characters per inch, so at 300dpi a cell
    # is exactly 30 pixels wide.
    result = render_mrz(SPECIMEN, dpi=300)
    assert result.char_boxes[1][0] - result.char_boxes[0][0] == 30


def test_every_cell_has_the_same_width() -> None:
    result = render_mrz(SPECIMEN)
    widths = {box[2] - box[0] for box in result.char_boxes}
    assert len(widths) == 1


def test_cells_are_evenly_spaced_across_a_line() -> None:
    result = render_mrz(SPECIMEN)
    line = result.char_boxes[: F.LINE_LENGTH]
    steps = {b[0] - a[0] for a, b in zip(line, line[1:])}
    assert len(steps) == 1


def test_a_glyphs_own_width_never_moves_the_grid() -> None:
    """The layout must not depend on which characters are being drawn.

    The bundled OCR-B is not truly monospaced: '5' is 100 units narrower than
    the other digits. Laying the line out with the font's advances would give
    '5' a distinctive gap, and a model could learn to spot it by the spacing
    rather than the shape — a cue that would not survive contact with a real
    passport. The cells must be identical whatever the text.
    """
    fives = render_mrz("5" * 44 + "\n" + "5" * 44)
    letters = render_mrz("A" * 44 + "\n" + "A" * 44)
    assert fives.char_boxes == letters.char_boxes


@pytest.mark.parametrize(
    ("label", "mrz"),
    [
        ("digits", "0" * 44 + "\n" + "8" * 44),
        ("letters", "H" * 44 + "\n" + "H" * 44),
        ("fillers", "<" * 44 + "\n" + "<" * 44),
    ],
)
def test_no_glyph_escapes_its_line_box(label: str, mrz: str) -> None:
    """Every pixel of ink must sit inside the ground-truth boxes.

    Sizing the box to the cap height of 'H' clipped three pixels off the top of
    every digit, because OCR-B's digits overshoot its capitals. The boxes are
    what the recognizer crops with, so that silently fed it mutilated glyphs.
    """
    result = render_mrz(mrz, dpi=300)
    ink_rows = numpy.where((numpy.asarray(result.image) < 128).any(axis=1))[0]

    assert ink_rows.min() >= result.line_boxes[0][1], f"{label} escape above the box"
    assert ink_rows.max() <= result.line_boxes[1][3], f"{label} escape below the box"


def test_digits_and_letters_share_a_box_height() -> None:
    # A per-glyph box would leak the character's identity through its geometry.
    result = render_mrz(SPECIMEN)
    assert len({box[3] - box[1] for box in result.char_boxes}) == 1


def test_the_zone_is_centred_within_the_page_width() -> None:
    geometry = PageGeometry()
    # 44 characters at 2.54mm is 111.76mm inside a 125mm page.
    assert geometry.mrz_width_mm == pytest.approx(111.76, abs=0.01)
    assert geometry.mrz_left_margin_mm == pytest.approx(6.62, abs=0.01)


def test_the_render_is_deterministic() -> None:
    assert render_mrz(SPECIMEN).image.tobytes() == render_mrz(SPECIMEN).image.tobytes()


def test_ink_is_actually_laid_down() -> None:
    result = render_mrz(SPECIMEN)
    assert numpy.asarray(result.image).min() < 128, "the image is blank"


def test_a_blank_line_leaves_the_paper_clean() -> None:
    result = render_mrz("<" * 44 + "\n" + "<" * 44)
    # Fillers are real glyphs, so even this draws ink; the point is that the
    # margins do not.
    assert set(numpy.asarray(result.image)[0].tolist()) == {PAPER}


def test_higher_dpi_scales_the_image() -> None:
    low = render_mrz(SPECIMEN, dpi=150)
    high = render_mrz(SPECIMEN, dpi=300)
    assert high.image.width == pytest.approx(low.image.width * 2, abs=2)


def test_the_mrz_box_covers_both_lines() -> None:
    result = render_mrz(SPECIMEN)
    left, top, right, bottom = result.mrz_box
    assert left <= min(b[0] for b in result.char_boxes)
    assert right >= max(b[2] for b in result.char_boxes)
    assert bottom - top > 0


@pytest.mark.parametrize(
    "bad",
    [
        "only one line",
        "P<UTO\nL898902C3",  # right shape, wrong length
        SPECIMEN.replace("P", "!"),  # not an MRZ character
    ],
)
def test_malformed_input_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        render_mrz(bad)


def test_a_generated_identity_renders() -> None:
    mrz = serialize(random_identity(random.Random(0)))
    assert render_mrz(mrz).image.size[0] > 0
