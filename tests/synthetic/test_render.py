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


# --------------------------------------------------------------------------
# the printing, as opposed to the photographing
# --------------------------------------------------------------------------


def test_more_than_one_cut_of_ocr_b_is_available() -> None:
    """The whole point. One font is how the last model learned outlines, not letters.

    Trained on a single cut, the recognizer read a different cut of OCR-B at 72%
    of documents — clean, undegraded, same geometry — confusing 0 with O and J
    with U. If this ever drops back to one, that regression returns silently and
    only a real passport would reveal it.
    """
    from mrz_ai.synthetic.render import available_fonts

    assert len(available_fonts()) >= 2, "the generator is back to a single typeface"


def test_the_cuts_are_actually_different() -> None:
    """Two files are not two typefaces if they are the same outlines."""
    from mrz_ai.synthetic.render import available_fonts

    first, second = available_fonts()[:2]
    a = numpy.asarray(render_mrz(SPECIMEN, dpi=150.0, font_path=first).image)
    b = numpy.asarray(render_mrz(SPECIMEN, dpi=150.0, font_path=second).image)

    # The strips are not the same height, and that is itself the point: a cut's
    # ink extent is its own, so even the box around the text moves. Compare where
    # they overlap.
    rows, columns = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    differing = float((a[:rows, :columns] != b[:rows, :columns]).mean())
    assert differing > 0.01, f"the two fonts are the same outlines ({differing:.3%} differ)"


def test_ink_weight_thickens_and_thins_the_strokes() -> None:
    light = numpy.asarray(render_mrz(SPECIMEN, dpi=150.0, ink_weight=-0.8).image)
    plain = numpy.asarray(render_mrz(SPECIMEN, dpi=150.0).image)
    heavy = numpy.asarray(render_mrz(SPECIMEN, dpi=150.0, ink_weight=0.8).image)

    # Ink is dark, so more ink means a lower mean.
    assert heavy.mean() < plain.mean() < light.mean()


def test_ink_weight_leaves_the_glyphs_where_they_were() -> None:
    """Weight is a property of the press, not of the layout.

    If it moved the text, the char_boxes handed to the recognizer as ground
    truth would no longer say where the characters are.
    """
    plain = render_mrz(SPECIMEN, dpi=150.0)
    heavy = render_mrz(SPECIMEN, dpi=150.0, ink_weight=0.8)

    assert heavy.char_boxes == plain.char_boxes
    assert heavy.line_boxes == plain.line_boxes
    assert numpy.asarray(heavy.image).shape == numpy.asarray(plain.image).shape


def test_zero_ink_weight_changes_nothing() -> None:
    plain = numpy.asarray(render_mrz(SPECIMEN, dpi=150.0).image)
    explicit = numpy.asarray(render_mrz(SPECIMEN, dpi=150.0, ink_weight=0.0).image)

    assert numpy.array_equal(plain, explicit)
