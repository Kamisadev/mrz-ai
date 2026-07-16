"""Degradation: reproducibility, severity, and geometry tracking."""

from __future__ import annotations

import numpy as np
import pytest

from mrz_ai.synthetic.degrade import (
    DegradeConfig,
    compose_ink,
    degrade,
    paper_background,
    transform_box,
    transform_quad,
    warp,
)
from mrz_ai.synthetic.render import render_mrz

SPECIMEN = (
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
)


@pytest.fixture(scope="module")
def clean() -> np.ndarray:
    return np.asarray(render_mrz(SPECIMEN, dpi=150).image)


def test_a_seed_pins_the_output(clean: np.ndarray) -> None:
    """The whole package promises reproducibility from a seed; this is the stage
    most likely to break it.

    Albumentations and OpenCV draw from their own global RNG, which is why the
    effects here are written against a Generator the caller owns instead.
    """
    first = degrade(clean, np.random.default_rng(42), severity=1.0)
    second = degrade(clean, np.random.default_rng(42), severity=1.0)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.transform, second.transform)


def test_different_seeds_give_different_images(clean: np.ndarray) -> None:
    first = degrade(clean, np.random.default_rng(1), severity=1.0)
    second = degrade(clean, np.random.default_rng(2), severity=1.0)
    assert not np.array_equal(first.image, second.image)


def test_the_shape_and_dtype_survive(clean: np.ndarray) -> None:
    result = degrade(clean, np.random.default_rng(0), severity=1.0)
    assert result.image.shape == clean.shape
    assert result.image.dtype == np.uint8


def test_zero_severity_leaves_the_render_alone(clean: np.ndarray) -> None:
    # The first curriculum stage wants genuinely clean documents.
    result = degrade(clean, np.random.default_rng(0), severity=0.0)
    assert np.abs(result.image.astype(int) - clean.astype(int)).mean() < 1.0


def test_severity_raises_the_damage(clean: np.ndarray) -> None:
    """More severity must mean a bigger departure from the clean render.

    Measured over many seeds because any single sample is a lottery: a
    high-severity draw can happen to miss every effect.
    """

    def damage(severity: float) -> float:
        return float(
            np.mean(
                [
                    np.abs(
                        degrade(clean, np.random.default_rng(s), severity=severity).image.astype(int)
                        - clean.astype(int)
                    ).mean()
                    for s in range(12)
                ]
            )
        )

    assert damage(0.0) < damage(0.25) < damage(0.6) < damage(1.0)


def test_the_ink_is_never_pure_black(clean: np.ndarray) -> None:
    """Real MRZ ink is a dark grey that varies between printers.

    Checked on the printing stage rather than the finished image: noise, dirt
    and occlusion legitimately drive individual pixels to zero later on.
    """
    for seed in range(10):
        rng = np.random.default_rng(seed)
        paper = paper_background(clean.shape, rng, 1.0)
        printed = compose_ink(clean, paper, rng, 1.0)
        assert printed.min() > 0


def test_the_paper_is_never_a_flat_white(clean: np.ndarray) -> None:
    # A perfectly uniform background is the giveaway of a synthetic sample.
    paper = paper_background(clean.shape, np.random.default_rng(0), 1.0)
    assert paper.std() > 1.0


# --- geometry ---------------------------------------------------------------

def test_an_identity_transform_leaves_a_box_alone() -> None:
    box = (10, 20, 100, 40)
    assert transform_box(box, np.eye(3, dtype=np.float32)) == box


def test_a_quad_keeps_its_corners_in_order() -> None:
    quad = transform_quad((0, 0, 10, 4), np.eye(3, dtype=np.float32))
    assert quad.shape == (4, 2)
    np.testing.assert_allclose(quad, [[0, 0], [10, 0], [10, 4], [0, 4]])


def test_rotation_moves_a_long_line_far_more_than_its_angle_suggests(
    clean: np.ndarray,
) -> None:
    """Why the transform has to be tracked at all.

    An MRZ line is roughly 44:1. Rotating it by a few degrees drags its ends
    vertically by most of a line height, so an upright box around the line's
    original position would cut the ends off.
    """
    config = DegradeConfig(p_rotate=1.0, p_perspective=0.0)
    _, transform = warp(clean.astype(np.float32), np.random.default_rng(5), 1.0, config)

    line = (0, 0, clean.shape[1], 20)
    before = 20
    after = transform_box(line, transform)
    assert after[3] - after[1] > before * 1.5


def test_the_tracked_quad_follows_the_ink(clean: np.ndarray) -> None:
    """The tracked corners must actually land on the text after a warp."""
    rendered = render_mrz(SPECIMEN, dpi=150)
    config = DegradeConfig(
        p_rotate=1.0,
        p_perspective=1.0,
        p_motion_blur=0.0,
        p_defocus=0.0,
        p_resample=0.0,
        p_shadow=0.0,
        p_glare=0.0,
        p_dirt=0.0,
        p_scratches=0.0,
        p_occlusion=0.0,
        p_jpeg=0.0,
    )
    result = degrade(
        np.asarray(rendered.image), np.random.default_rng(7), severity=1.0, config=config
    )

    quad = result.locate_quad(rendered.line_boxes[0])
    # The quad should enclose dark pixels: sample its centre and check there is
    # ink nearby rather than blank paper.
    centre = quad.mean(axis=0).astype(int)
    window = result.image[
        max(centre[1] - 8, 0) : centre[1] + 8, max(centre[0] - 8, 0) : centre[0] + 8
    ]
    assert window.min() < 160, "the tracked quad does not sit on any text"


def test_an_untracked_box_would_swallow_the_neighbouring_line() -> None:
    """The bug that forced a deskewing crop rather than a padded box.

    Once the sheet is rotated, the upright box of one line grows tall enough to
    overlap the other, so a crop taken from it would show two lines while the
    label named one.
    """
    rendered = render_mrz(SPECIMEN, dpi=150)
    config = DegradeConfig(p_rotate=1.0, p_perspective=0.0)
    _, transform = warp(
        np.asarray(rendered.image).astype(np.float32), np.random.default_rng(5), 1.0, config
    )

    first = transform_box(rendered.line_boxes[0], transform)
    second = transform_box(rendered.line_boxes[1], transform)

    overlap = min(first[3], second[3]) - max(first[1], second[1])
    assert overlap > 0, "expected the upright boxes to overlap after rotation"
