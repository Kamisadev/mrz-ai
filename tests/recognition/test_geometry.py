"""The input contract, pinned so it cannot drift."""

from __future__ import annotations

import pytest

from mrz_ai.parser import fields as F
from mrz_ai.parser.charset import ALPHABET_SIZE
from mrz_ai.recognition.geometry import INPUT, MODEL, InputGeometry


def test_the_width_is_a_whole_number_of_characters() -> None:
    # 44 x 16 = 704. Keeping this exact is what aligns the patch grid to cells.
    assert INPUT.width == F.LINE_LENGTH * 16
    assert INPUT.pixels_per_char == 16


def test_a_character_spans_two_patches() -> None:
    """One patch per character would be cheaper but fragile.

    Crop jitter can misalign the grid by half a character, which with a
    character-wide patch would smear every glyph across a patch boundary.
    """
    assert INPUT.patches_per_char == 2


def test_the_patch_grid_divides_the_image() -> None:
    assert INPUT.grid == (4, 88)
    assert INPUT.num_tokens == 352


def test_the_geometry_is_far_from_the_scene_text_default() -> None:
    """Stock PARSeq is 32x128 at 4:1 for 25 characters; an MRZ line is nothing
    like that, and putting 44 characters in 128 pixels leaves 2.9 each."""
    assert INPUT.aspect_ratio == 22.0
    assert 128 / F.LINE_LENGTH < 3


def test_there_is_one_output_per_character() -> None:
    assert MODEL.max_label_length == F.LINE_LENGTH == 44


def test_the_classes_are_the_mrz_alphabet() -> None:
    assert MODEL.num_classes == ALPHABET_SIZE == 37


def test_a_width_that_is_not_a_whole_number_of_characters_is_rejected() -> None:
    with pytest.raises(ValueError, match="multiple"):
        InputGeometry(width=700)


def test_a_patch_grid_that_does_not_divide_the_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="divide"):
        InputGeometry(height=32, width=704, patch_height=5)
