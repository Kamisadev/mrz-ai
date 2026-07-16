"""Crop to tensor."""

from __future__ import annotations

import numpy as np
import pytest

from mrz_ai.recognition.geometry import INPUT
from mrz_ai.recognition.preprocess import prepare, prepare_batch
from mrz_ai.synthetic.dataset import DatasetConfig, MRZLineDataset


def test_any_crop_becomes_the_model_input() -> None:
    for width in (400, 704, 1200):
        assert prepare(np.full((32, width), 128, np.uint8)).shape == (1, 32, 704)


def test_a_variable_width_crop_from_the_dataset_fits() -> None:
    # The dataset's crops vary in width by design, because a real detector's
    # framing does.
    dataset = MRZLineDataset(DatasetConfig(dpi=150))
    for index in range(8):
        assert prepare(dataset[index].image).shape == (1, INPUT.height, INPUT.width)


def test_pixels_land_in_the_unit_interval() -> None:
    prepared = prepare(np.random.randint(0, 256, (32, 800), dtype=np.uint8))
    assert prepared.dtype == np.float32
    assert 0.0 <= prepared.min() and prepared.max() <= 1.0


def test_white_stays_white_and_black_stays_black() -> None:
    assert prepare(np.full((32, 704), 255, np.uint8)).max() == pytest.approx(1.0)
    assert prepare(np.full((32, 704), 0, np.uint8)).min() == pytest.approx(0.0)


def test_the_resize_is_anisotropic_by_design() -> None:
    """Width is squeezed to fit; the aspect ratio is not preserved.

    Padding to preserve it would put the characters at a different scale in every
    sample, depending only on how loosely the detector happened to frame them.
    """
    tall = prepare(np.full((32, 2000), 200, np.uint8))
    wide = prepare(np.full((32, 300), 200, np.uint8))
    assert tall.shape == wide.shape == (1, 32, 704)


def test_a_colour_image_is_accepted() -> None:
    assert prepare(np.zeros((32, 704, 3), np.uint8)).shape == (1, 32, 704)


def test_an_empty_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare(np.zeros((0, 0), np.uint8))


def test_a_batch_stacks() -> None:
    batch = prepare_batch([np.full((32, w), 128, np.uint8) for w in (400, 704, 900)])
    assert batch.shape == (3, 1, 32, 704)


def test_an_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="no images"):
        prepare_batch([])
