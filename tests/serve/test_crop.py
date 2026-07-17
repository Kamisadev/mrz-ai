"""Turning an upload and a dragged box into the two crops the model expects.

The load-bearing test here is `test_a_loose_drag_finds_the_same_lines_as_a_tight_one`.
The recognizer resizes its crop to a fixed 32x704, so blank paper inside the crop
is not ignored — it shifts every character out of the cell the model expects. A
box drawn 30% loose and simply halved reads 10% of documents correctly. Everything
in this module exists to make the drawn box's tightness stop mattering, and these
tests are how that claim is checked rather than asserted.

Ground truth comes from the synthetic renderer, which reports where it put each
line, so the located boxes are compared against where the text actually is.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

from mrz_ai.parser import serialize
from mrz_ai.synthetic.degrade import DegradeConfig, degrade
from mrz_ai.synthetic.identity import IdentityConfig, random_identity
from mrz_ai.synthetic.render import render_mrz
from mrz_ai.serve.crop import (
    Box,
    decode_image,
    deskew,
    find_lines,
    locate_lines,
    skew_angle,
    split_lines,
)


def boxes(image, box):
    """Just the geometry, for tests that do not care about clipping."""
    first, second = locate_lines(image, box)
    return first.box, second.box


def an_image(width: int = 400, height: int = 100) -> np.ndarray:
    """Pale paper with two dark bands where the MRZ lines would be."""
    image = np.full((height, width), 255, dtype=np.uint8)
    image[10:40, 20:380] = 0
    image[60:90, 20:380] = 0
    return image


def a_page(seed: int = 1000, severity: float = 0.2):
    """A rendered, degraded MRZ page, with the truth about where its lines are."""
    mrz = serialize(random_identity(random.Random(seed), IdentityConfig()))
    rendered = render_mrz(mrz, dpi=300.0)
    result = degrade(
        np.asarray(rendered.image), np.random.default_rng(seed),
        severity=severity, config=DegradeConfig(),
    )
    image = result.image
    if image.ndim == 3:
        image = image[..., 0]
    lines = tuple(result.locate(line) for line in rendered.line_boxes)
    return mrz, image, result.locate(rendered.mrz_box), lines


def test_a_png_upload_decodes_to_greyscale() -> None:
    ok, buffer = cv2.imencode(".png", an_image())
    assert ok

    image = decode_image(buffer.tobytes())

    assert image.ndim == 2, "the recognizer reads greyscale"
    assert image.shape == (100, 400)


def test_a_jpeg_upload_decodes() -> None:
    """Phone cameras produce JPEG; it is the format a real passport arrives in."""
    ok, buffer = cv2.imencode(".jpg", an_image())
    assert ok

    assert decode_image(buffer.tobytes()).shape == (100, 400)


def test_a_colour_upload_is_flattened() -> None:
    colour = cv2.cvtColor(an_image(), cv2.COLOR_GRAY2BGR)
    ok, buffer = cv2.imencode(".png", colour)
    assert ok

    assert decode_image(buffer.tobytes()).ndim == 2


def test_a_file_that_is_not_an_image_is_refused() -> None:
    """An upload is untrusted input: it must fail as an error, not a crash."""
    with pytest.raises(ValueError, match="image"):
        decode_image(b"this is not a picture")


def test_an_empty_upload_is_refused() -> None:
    with pytest.raises(ValueError, match="image"):
        decode_image(b"")


def test_two_crops_come_back_one_per_line() -> None:
    top, bottom = split_lines(an_image(), Box(x=0, y=0, width=400, height=100))

    assert top.ndim == 2 and bottom.ndim == 2
    assert top.min() == 0, "the first crop holds no ink"
    assert bottom.min() == 0, "the second crop holds no ink"


def test_the_lines_are_found_where_the_ink_is_not_where_the_box_was() -> None:
    """The box is a hint about which text to read, not about how to frame it."""
    first, second = boxes(an_image(), Box(x=0, y=0, width=400, height=100))

    # The bands sit at rows 10-40 and 60-90, columns 20-380. The located boxes
    # should hug them, padded by ~15% of the band's height.
    assert first.y == pytest.approx(10, abs=6)
    assert first.y + first.height == pytest.approx(40, abs=6)
    assert second.y == pytest.approx(60, abs=6)
    assert second.x == pytest.approx(20, abs=6)
    assert second.x + second.width == pytest.approx(380, abs=6)


def test_a_loose_drag_finds_the_same_lines_as_a_tight_one() -> None:
    """The point of the whole module.

    A person cannot drag a box to within half a character, and the recognizer's
    fixed-size input means half a character is the tolerance. If this ever fails,
    the page's accuracy silently becomes a function of how carefully the user
    dragged — which reads as a broken model and is not one.
    """
    image = an_image()
    tight = boxes(image, Box(x=20, y=10, width=360, height=80))
    loose = boxes(image, Box(x=0, y=0, width=400, height=100))

    # Both drags must land on the ink itself — bands at rows 10-40 and 60-90,
    # columns 20-380 — give or take the padding deliberately added around it.
    # The tight drag cannot pad outward past its own edges, which is the only
    # licensed difference between the two, and is why the tolerance is the pad.
    pad = 0.15 * 30
    for found in (tight, loose):
        first, second = found
        assert first.x == pytest.approx(20, abs=pad + 1)
        assert first.x + first.width == pytest.approx(380, abs=pad + 1)
        assert first.y == pytest.approx(10, abs=pad + 1)
        assert second.y == pytest.approx(60, abs=pad + 1)
        assert second.y + second.height == pytest.approx(90, abs=pad + 1)


def test_the_located_lines_match_the_renderers_own_ground_truth() -> None:
    """Checked against where the synthetic engine says it drew the text.

    This is the only test here that can catch the projection being subtly wrong
    on real-looking text rather than on painted rectangles.
    """
    _, image, (left, top, right, bottom), lines = a_page()
    # A deliberately loose drag, 30% of the zone's height proud on every side.
    slack = 0.3 * (bottom - top)
    box = Box(x=left - slack, y=top - slack,
              width=(right - left) + 2 * slack, height=(bottom - top) + 2 * slack)

    first, second = boxes(image, box)

    for found, (true_left, true_top, true_right, true_bottom) in zip((first, second), lines):
        line_height = true_bottom - true_top
        # Padding means the crop is a little proud of the text on every side, so
        # the bound is one-sided: it may not cut into the characters.
        assert found.y <= true_top, "the crop clipped the top of the line"
        assert found.y + found.height >= true_bottom, "the crop clipped the bottom"
        assert found.x <= true_left + 1, "the crop lost characters at the left"
        assert found.x + found.width >= true_right - 1, "the crop lost characters at the right"
        # And it must not wander: padding is 0.15 of a line, and the recognizer
        # was trained to tolerate 0.3.
        assert found.y > true_top - 0.3 * line_height
        assert found.y + found.height < true_bottom + 0.3 * line_height

    # Neither crop may contain the other line's characters. A few pixels of
    # overlap in the blank gap is fine and expected — the training crops padded
    # into it too — but a crop showing two lines is one the model cannot read.
    assert first.y + first.height < lines[1][1], "line 1's crop reaches line 2's text"
    assert second.y > lines[0][3], "line 2's crop reaches line 1's text"


def test_a_box_that_cuts_through_the_text_is_reported_as_clipped() -> None:
    """The one failure the reading itself cannot report.

    A crop that lost its first characters still produces a confident reading of
    the ones that remain, and nothing about that reading says the box was wrong.
    Observed rather than imagined: a default box whose left edge fell inside the
    MRZ read 'TKMJURTA' for 'TKLGUPTA' and offered no hint why.
    """
    image = an_image()  # ink spans columns 20-380
    # A box starting at column 60 slices the first characters off both lines.
    first, second = locate_lines(image, Box(x=60, y=0, width=340, height=100))

    assert first.clipped is True
    assert second.clipped is True


def test_a_box_that_contains_the_text_is_not_reported_as_clipped() -> None:
    """The warning has to stay quiet when nothing is wrong, or it is noise."""
    first, second = locate_lines(an_image(), Box(x=0, y=0, width=400, height=100))

    assert first.clipped is False
    assert second.clipped is False


def test_clipping_is_detected_on_a_rendered_page() -> None:
    """Painted rectangles cannot catch a threshold that is wrong on real text."""
    _, image, (left, top, right, bottom), _ = a_page()

    generous = Box(x=left - 20, y=top - 20, width=(right - left) + 40, height=(bottom - top) + 40)
    assert not any(line.clipped for line in locate_lines(image, generous))

    # Bite 40 pixels — more than a character — off the left of the zone.
    biting = Box(x=left + 40, y=top - 20, width=(right - left) - 40, height=(bottom - top) + 40)
    assert any(line.clipped for line in locate_lines(image, biting))


def test_a_box_is_clamped_to_the_image() -> None:
    """A drag that ran off the edge of the picture is normal, not an error."""
    first, second = boxes(an_image(), Box(x=-50, y=-20, width=900, height=400))

    assert first.x >= 0 and first.y >= 0
    assert second.x + second.width <= 400
    assert second.y + second.height <= 100


def test_a_box_outside_the_image_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        split_lines(an_image(), Box(x=500, y=500, width=100, height=100))


def test_a_box_too_small_to_split_is_refused() -> None:
    """A few pixels cannot become two lines, and returning junk silently is worse."""
    with pytest.raises(ValueError, match="small"):
        split_lines(an_image(), Box(x=0, y=0, width=400, height=2))


def test_a_zero_area_box_is_refused() -> None:
    with pytest.raises(ValueError, match="small"):
        split_lines(an_image(), Box(x=10, y=10, width=0, height=50))


def test_blank_paper_does_not_crash() -> None:
    """No ink anywhere: the fallback must return something, not raise."""
    blank = np.full((100, 400), 255, dtype=np.uint8)

    top, bottom = split_lines(blank, Box(x=0, y=0, width=400, height=100))

    assert top.size > 0 and bottom.size > 0


def test_one_band_falls_back_to_halving() -> None:
    """A shadow can merge the lines. A worse guess must still be a usable one."""
    merged = np.full((100, 400), 255, dtype=np.uint8)
    merged[10:90, 20:380] = 0  # one solid block, not two bands

    first, second = boxes(merged, Box(x=0, y=0, width=400, height=100))

    # Halving cannot know where the lines really are, so the two crops may share
    # a few rows once each is padded. What it must still get right is the order
    # and the framing: an upper crop, a lower one, both usable.
    assert first.y < second.y
    assert first.height > 0 and second.height > 0


def test_fractional_coordinates_from_the_browser_are_accepted() -> None:
    """A canvas drag produces floats; the page must not have to round them."""
    top, bottom = split_lines(an_image(), Box(x=0.4, y=0.6, width=399.2, height=98.7))

    assert top.size > 0 and bottom.size > 0


def test_crop_does_not_need_torch() -> None:
    import subprocess
    import sys

    program = "import sys; sys.modules['torch'] = None\nimport mrz_ai.serve.crop\nprint('ok')"
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, f"importing without torch failed:\n{result.stderr}"


# --------------------------------------------------------------------------
# levelling the text before anything looks for a band
# --------------------------------------------------------------------------


def a_tilted_page(degrees: float, seed: int = 1000):
    """A rendered MRZ turned by ``degrees``, and a loose box around where it is.

    Bordered generously before turning. The renderer draws a bare MRZ strip barely
    taller than its two lines, and rotating that in place swings the ends of a
    111mm line straight out of the picture — which looks like a skew estimator
    that degrades with angle, and is nothing of the sort. That artefact cost real
    time to see through once already.
    """
    mrz = serialize(random_identity(random.Random(seed), IdentityConfig()))
    rendered = render_mrz(mrz, dpi=250.0)
    margin = 200
    page = cv2.copyMakeBorder(
        np.asarray(rendered.image), margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=255,
    )
    left, top, right, bottom = (value + margin for value in rendered.mrz_box)

    height, width = page.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    tilted = cv2.warpAffine(
        page, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )

    slack = 0.3 * (bottom - top)
    box = Box(
        x=float(left - slack), y=float(top - slack),
        width=float(right - left + 2 * slack), height=float(bottom - top + 2 * slack),
    )
    return tilted, box


@pytest.mark.parametrize("degrees", [-8.0, -5.0, -2.0, -0.8, 0.8, 2.0, 5.0, 8.0])
def test_the_tilt_is_measured_back_out(degrees: float) -> None:
    """Turning the page by a known angle and asking what it is.

    Sign included: an estimator that is confidently backwards would double the
    tilt instead of removing it, and an unsigned check would pass.
    """
    tilted, box = a_tilted_page(degrees)
    # cv2 turns anticlockwise for a positive angle; the module calls a line that
    # runs downhill to the right positive. They are opposites.
    assert skew_angle(tilted, box) == pytest.approx(-degrees, abs=0.5)


def test_level_text_is_called_level() -> None:
    tilted, box = a_tilted_page(0.0)
    assert skew_angle(tilted, box) == 0.0


def test_deskewing_a_level_page_does_not_touch_its_pixels() -> None:
    """Not an optimisation — the correctness case.

    A rotation resamples every pixel. Glyphs reach the recognizer 16px wide, so
    re-rendering a straight scan to take out a hundredth of a degree returns it
    very slightly softer and no straighter.
    """
    tilted, box = a_tilted_page(0.0)
    leveled, angle = deskew(tilted, box)
    assert angle == 0.0
    assert leveled is tilted


@pytest.mark.parametrize("degrees", [-8.0, -3.0, 3.0, 8.0])
def test_deskewing_leaves_the_text_level(degrees: float) -> None:
    tilted, box = a_tilted_page(degrees)
    leveled, angle = deskew(tilted, box)
    assert angle != 0.0
    assert abs(skew_angle(leveled, box)) < 0.5


def test_paper_with_no_ink_on_it_claims_no_tilt() -> None:
    """Zero here means "no evidence", not "measured level"."""
    blank = np.full((200, 800), 250, dtype=np.uint8)
    assert skew_angle(blank, Box(0.0, 0.0, 800.0, 200.0)) == 0.0


def test_the_sweep_does_not_run_to_its_own_edge() -> None:
    """A regression on the bug that made the first version useless.

    Scoring the sheared rows by clamping them into the region's height stacks
    everything that overflows against row 0 — and a pile of ink scores exactly
    like a sharply-stacked line. Written that way this returned the widest angle
    in the sweep for every input, level pages included. Any estimate sitting on
    the limit is that bug, whatever the picture.
    """
    for degrees in (0.0, 1.0, -1.0, 4.0):
        tilted, box = a_tilted_page(degrees)
        assert abs(skew_angle(tilted, box)) < 14.0


def test_the_angle_survives_being_measured_at_a_smaller_size() -> None:
    """The estimate is made on a shrunk copy when the box is wide. An angle is
    scale-invariant, so this must not change the answer — and would silently, if
    the shrink were ever made non-uniform."""
    tilted, box = a_tilted_page(4.0)
    big = cv2.resize(tilted, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    scaled = Box(box.x * 2, box.y * 2, box.width * 2, box.height * 2)
    assert skew_angle(big, scaled) == pytest.approx(skew_angle(tilted, box), abs=0.3)


def test_half_a_degree_of_tilt_hides_one_line_inside_the_other() -> None:
    """The measurement the whole thing is built on, kept honest.

    ICAO's two lines are 4.23mm apart with a cap height of 3.2mm, so 1.03mm of
    paper separates them across a line 111.76mm long: past asin(1.03/111.76) =
    0.53 degrees, the far end of line 1 lands in the rows of line 2 and a row
    profile cannot tell them apart. Below that the search sees two bands; above,
    one. If this ever stops failing, the geometry moved and the deskew's whole
    reason went with it.
    """
    level, box = a_tilted_page(0.0)
    assert len(_bands(level, box)) == 2

    tilted, box = a_tilted_page(1.5)
    assert len(_bands(tilted, box)) == 1, "the bands stayed apart; re-check the arithmetic"

    leveled, _ = deskew(tilted, box)
    assert len(_bands(leveled, box)) == 2, "deskewing did not bring the two bands back"


def _bands(image, box):
    """What the two-band search sees, without the halving fallback hiding it."""
    from mrz_ai.serve.crop import _ink, _runs, _MIN_BAND, _ROW_INK, _clamped

    left, top, right, bottom = _clamped(image, box)
    profile = _ink(image[top:bottom, left:right]).sum(axis=1) / 255.0
    return _runs(profile > profile.max() * _ROW_INK, _MIN_BAND) if profile.max() > 0 else []


def test_find_lines_hands_back_the_image_its_boxes_belong_to() -> None:
    """Cutting the boxes out of the original photograph would miss by the tilt."""
    tilted, box = a_tilted_page(5.0)
    found = find_lines(tilted, box)
    assert found.skew_deg != 0.0
    assert found.image is not tilted
    assert len(found.lines) == 2


def test_a_tilted_page_still_yields_two_line_crops() -> None:
    tilted, box = a_tilted_page(6.0)
    first, second = split_lines(tilted, box)
    for crop in (first, second):
        assert crop.size > 0
        assert crop.shape[1] > crop.shape[0] * 10, "an MRZ line is about 44:1"
