"""The online dataset: what the recognizer will actually be fed."""

from __future__ import annotations

import random

import numpy as np
import pytest

from mrz_ai.parser import fields as F
from mrz_ai.parser import parse, serialize, validate
from mrz_ai.synthetic.dataset import DatasetConfig, MRZLineDataset, extract_line
from mrz_ai.synthetic.identity import random_identity


@pytest.fixture(scope="module")
def dataset() -> MRZLineDataset:
    return MRZLineDataset(DatasetConfig(dpi=150))


def test_a_sample_carries_a_full_line_of_text(dataset: MRZLineDataset) -> None:
    for index in range(20):
        assert len(dataset[index].text) == F.LINE_LENGTH


def test_both_lines_are_produced(dataset: MRZLineDataset) -> None:
    indices = {dataset[i].line_index for i in range(10)}
    assert indices == {0, 1}


def test_the_first_line_is_always_a_passport(dataset: MRZLineDataset) -> None:
    for index in range(0, 20, 2):
        sample = dataset[index]
        assert sample.line_index == 0
        assert sample.text.startswith("P")


def test_the_crop_has_the_model_input_height() -> None:
    dataset = MRZLineDataset(DatasetConfig(target_height=32, dpi=150))
    for index in range(10):
        assert dataset[index].image.shape[0] == 32


def test_the_crop_keeps_a_line_shaped_aspect_ratio(dataset: MRZLineDataset) -> None:
    # 44 characters on one line is a long, thin strip; anything close to square
    # means the crop has gone wrong.
    for index in range(20):
        image = dataset[index].image
        assert 12 < image.shape[1] / image.shape[0] < 45


def test_a_sample_is_reproducible_from_its_index() -> None:
    first = MRZLineDataset(DatasetConfig(dpi=150))[5]
    second = MRZLineDataset(DatasetConfig(dpi=150))[5]

    assert np.array_equal(first.image, second.image)
    assert first.text == second.text


def test_workers_need_no_shared_state() -> None:
    # Every sample derives from its index alone, so a dataloader can fan out
    # without coordinating.
    dataset = MRZLineDataset(DatasetConfig(dpi=150))
    assert dataset[3].text == MRZLineDataset(DatasetConfig(dpi=150))[3].text


def test_a_new_epoch_yields_new_documents() -> None:
    """Online generation is the point: the model must never see a sample twice.

    Without the epoch in the seed, every epoch would replay the same documents
    and the model could memorise them.
    """
    first = MRZLineDataset(DatasetConfig(dpi=150))
    second = MRZLineDataset(DatasetConfig(dpi=150))
    second.set_epoch(1)

    assert sum(first[i].text != second[i].text for i in range(20)) >= 19


def test_a_different_seed_yields_a_different_stream() -> None:
    first = MRZLineDataset(DatasetConfig(seed=0, dpi=150))
    second = MRZLineDataset(DatasetConfig(seed=1, dpi=150))
    assert sum(first[i].text != second[i].text for i in range(20)) >= 19


def test_consecutive_samples_are_the_two_lines_of_one_document() -> None:
    """Samples are emitted a line at a time, so a mismatched pair is easy to
    produce and impossible to spot by validation.

    Checking `validate` on the reassembled pair would prove nothing: ICAO puts
    no check digit on line 1, so line 1 of one passport and line 2 of another
    validate perfectly happily. The only honest check is to rebuild the document
    the sample claims to come from and compare.
    """
    config = DatasetConfig(dpi=150)
    dataset = MRZLineDataset(config)

    for document_index in range(20):
        expected = serialize(
            random_identity(random.Random(dataset._seed_for(document_index)), config.identity)
        ).split("\n")
        first, second = dataset[document_index * 2], dataset[document_index * 2 + 1]

        assert [first.text, second.text] == expected


def test_every_label_belongs_to_a_valid_mrz() -> None:
    dataset = MRZLineDataset(DatasetConfig(dpi=150))
    for index in range(0, 40, 2):
        document = parse(f"{dataset[index].text}\n{dataset[index + 1].text}")
        assert validate(document, reference_year=2026).is_valid


def test_severity_stays_within_its_range() -> None:
    dataset = MRZLineDataset(DatasetConfig(severity_range=(0.2, 0.4), dpi=150))
    for index in range(30):
        assert 0.2 <= dataset[index].severity <= 0.4


def test_the_curriculum_can_ask_for_clean_documents() -> None:
    dataset = MRZLineDataset(DatasetConfig(severity_range=(0.0, 0.0), dpi=150))
    assert all(dataset[i].severity == 0.0 for i in range(5))


def test_the_length_is_the_configured_epoch_size() -> None:
    assert len(MRZLineDataset(DatasetConfig(epoch_size=1234))) == 1234


# --- extract_line -----------------------------------------------------------

def test_extract_line_straightens_a_tilted_quad() -> None:
    """Cropping by upright bounding box would include the neighbouring line.

    Mapping the quadrilateral onto a rectangle is what keeps one line in the
    crop, and it is what production does: detection locates the line and
    straightens it before recognition sees it.
    """
    image = np.full((100, 400), 255, np.uint8)
    image[48:52, :] = 0  # a horizontal bar standing in for a line of text

    tilted = np.float32([[0, 40], [400, 60], [400, 80], [0, 60]])
    crop = extract_line(image, tilted, target_height=32, padding=(0, 0, 0, 0), skew_deg=0)

    assert crop.shape[0] == 32
    assert crop.min() == 0, "the bar should have been captured"


def test_extract_line_handles_a_quad_running_off_the_image() -> None:
    # A rotated line's corners can land outside the frame; the crop must not
    # come back short.
    image = np.full((50, 200), 255, np.uint8)
    quad = np.float32([[-20, -10], [220, -10], [220, 30], [-20, 30]])
    crop = extract_line(image, quad, target_height=16, padding=(0, 0, 0, 0), skew_deg=0)
    assert crop.shape[0] == 16 and crop.shape[1] > 0


def test_padding_widens_the_crop() -> None:
    image = np.full((100, 400), 255, np.uint8)
    quad = np.float32([[0, 40], [400, 40], [400, 60], [0, 60]])

    tight = extract_line(image, quad, target_height=32, padding=(0, 0, 0, 0), skew_deg=0)
    loose = extract_line(image, quad, target_height=32, padding=(1, 0, 1, 0), skew_deg=0)

    assert loose.shape[1] > tight.shape[1]


def test_both_lines_of_a_document_share_its_photograph() -> None:
    # One photo, one exposure: the two lines must agree on severity.
    dataset = MRZLineDataset(DatasetConfig(dpi=150))
    for document in range(6):
        assert dataset[document * 2].severity == dataset[document * 2 + 1].severity


def test_each_line_is_framed_independently() -> None:
    """A detector crops each line on its own, so the jitter must differ.

    Keying the crop stream to the document rather than the line gave both lines
    identical framing, which no real detector would produce.
    """
    dataset = MRZLineDataset(DatasetConfig(dpi=150))
    widths = [(dataset[i * 2].image.shape[1], dataset[i * 2 + 1].image.shape[1]) for i in range(8)]
    assert any(first != second for first, second in widths)


def test_the_default_dpi_renders_lines_taller_than_the_crop_they_become() -> None:
    """A crop must never be upscaled, or every sample is blurry by construction.

    `extract_line` normalizes each crop to `target_height`, so if the source line
    is shorter than that, the crop is an upscale and the model never sees a sharp
    glyph — not even at severity 0. A line's ink is ~3.5mm, which only reaches
    32px above ~232dpi. The default was briefly 150 (a 1.52x upscale) after
    reasoning about character *width*, which is not what drives the resize:
    aspect ratio fixes the crop's width near 758px whatever the dpi.
    """
    from mrz_ai.synthetic.render import render_mrz

    config = DatasetConfig()
    mrz = serialize(random_identity(random.Random(0)))
    rendered = render_mrz(mrz, dpi=config.dpi)
    left, top, right, bottom = rendered.line_boxes[0]

    assert bottom - top >= config.target_height, (
        f"at {config.dpi}dpi a line is {bottom - top}px tall but crops are "
        f"{config.target_height}px: every sample would be upscaled"
    )


def test_caching_a_document_does_not_change_the_samples_it_yields() -> None:
    """The document cache must be invisible in the output.

    Rendering and degrading a document is the generator's dominant cost, and both
    of its lines need the same one — so the second line reuses what the first
    built. That is a 2x saving on the bottleneck of a training pod, and it is
    also the kind of optimization that quietly changes the data: serve a stale
    entry and a line gets cropped out of the wrong passport, with the right label
    still attached. Nothing downstream could catch that, so it is pinned here.
    """
    config = DatasetConfig(severity_range=(0.0, 1.0))
    sequential = MRZLineDataset(config)

    for index in range(6):
        # A fresh dataset per index never hits the cache, so this is the
        # uncached path by construction.
        cached, uncached = sequential[index], MRZLineDataset(config)[index]
        assert cached.text == uncached.text
        assert cached.severity == uncached.severity
        assert np.array_equal(cached.image, uncached.image), (
            f"index {index}: the cached document yielded different pixels"
        )


def test_a_new_epoch_is_not_served_from_the_cache() -> None:
    """The cache is keyed by epoch, because the same index means a new document."""
    dataset = MRZLineDataset(DatasetConfig())
    dataset.set_epoch(0)
    first = dataset[0]
    dataset.set_epoch(1)

    assert dataset[0].text != first.text, "the epoch's document was served stale"
